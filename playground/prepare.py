"""Build the input tables and the page data of every demo.

Each data group (imdb, bio, compaction) becomes one Arrow IPC file per
table, named by the hash of its bytes as Quail Server names uploads,
plus one JSON file with what the page draws: labels, text heads, token
counts. ``manifest.json`` lists the files and their content ids. The
tables are what the servers run on; the JSON never reaches a server.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from playground import compaction
from playground.demos import GROUPS, tables_of
from quail.server.artifacts import write_ipc_file
from quail.server.inputs import file_digest

IMDB_DATASET = "stanfordnlp/imdb"
IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
REVIEWS_PER_LABEL = 5_000
BIO_SCALE_FACTOR = 0.1
COMPACTION_LIMIT = 100
COMPACTION_SEED = 42
HEAD_CHARS = 240


def head(text: str, limit: int = HEAD_CHARS) -> str:
    """The first characters of a text for display, with a marker when cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def select_reviews(imdb: pa.Table) -> tuple[pa.Table, list]:
    """Take the first 5,000 negative and 5,000 positive reviews.

    Args:
        imdb: The IMDB rows with ``text`` and ``label`` (0 negative,
            1 positive).

    Returns:
        The reviews table (review_id, review) with the negative reviews
        first, and the page records with each review's label and head.

    """
    parts = []
    for label in (0, 1):
        rows = imdb.filter(pc.equal(imdb.column("label"), label))
        if rows.num_rows < REVIEWS_PER_LABEL:
            raise ValueError(
                f"IMDB has {rows.num_rows} reviews with label {label}; "
                f"{REVIEWS_PER_LABEL} are needed")
        parts.append(rows.slice(0, REVIEWS_PER_LABEL))
    chosen = pa.concat_tables(parts)
    texts = chosen.column("text").to_pylist()
    labels = chosen.column("label").to_pylist()
    reviews = pa.table({
        "review_id": pa.array([f"review-{i}" for i in range(len(texts))]),
        "review": pa.array(texts, pa.string()),
    })
    records = [{"id": f"review-{i}", "label": int(label), "head": head(text)}
               for i, (text, label) in enumerate(zip(texts, labels))]
    return reviews, records


def load_imdb() -> pa.Table:
    """The IMDB training split, pinned to one dataset commit."""
    from datasets import load_dataset

    split = load_dataset(IMDB_DATASET, split="train", revision=IMDB_REVISION)
    return split.data.table.select(["text", "label"])


def bio_records(reports: pa.Table, terms: pa.Table, token_lengths) -> dict:
    """The page's view of the BIO tables.

    Args:
        reports: The reports table (id, report).
        terms: The terms table (id, term).
        token_lengths: Callable(list of texts) -> token counts.

    """
    texts = reports.column("report").to_pylist()
    lengths = token_lengths(texts)
    return {
        "reports": [{"id": row_id, "head": head(text), "tokens": int(length)}
                    for row_id, text, length in zip(
                        reports.column("id").to_pylist(), texts, lengths)],
        "terms": [{"id": row_id, "term": term}
                  for row_id, term in zip(terms.column("id").to_pylist(),
                                          terms.column("term").to_pylist())],
    }


def load_bio() -> tuple[pa.Table, pa.Table]:
    """The quail-bench BIO tables at scale factor 0.1, from the public bucket."""
    from quail_b.data import load_table

    reports = load_table("reports", scale_factor=BIO_SCALE_FACTOR)
    terms = load_table("terms", scale_factor=BIO_SCALE_FACTOR)
    return (pa.table({"id": pc.cast(reports.column("id"), pa.string()),
                      "report": reports.column("report")}),
            pa.table({"id": pc.cast(terms.column("id"), pa.string()),
                      "term": terms.column("term")}))


def compaction_records(conversations: list[dict], questions: pa.Table) -> dict:
    """The page's view of the trajectories and their retention questions.

    Args:
        conversations: Rows with ``id``, ``instance_id`` and the original
            ``messages`` as JSON text.
        questions: The tool_questions table, in the order the server
            reads it.

    """
    calls_by_id = {}
    records = []
    for row in conversations:
        calls = compaction.collect_tool_calls(json.loads(row["messages"]))
        entries = []
        for call in calls:
            tokens = compaction.estimate_tokens(call.result)
            kept = compaction.estimate_tokens(compaction.truncated_result(call))
            entries.append({"id": call.id, "tool": call.tool, "tokens": tokens,
                            "truncated_tokens": min(kept, tokens),
                            "pinned": call.pinned})
            calls_by_id[(row["id"], call.source_id)] = call.id
        records.append({"id": row["id"], "name": row["instance_id"],
                        "calls": entries,
                        "tokens": sum(entry["tokens"] for entry in entries)})
    conversation_row = {row["id"]: index
                        for index, row in enumerate(conversations)}
    question_records = []
    for conversation_id, tool_call_id, kind in zip(
            questions.column("conversation_id").to_pylist(),
            questions.column("tool_call_id").to_pylist(),
            questions.column("kind").to_pylist()):
        question_records.append([conversation_row[conversation_id],
                                 calls_by_id[(conversation_id, tool_call_id)],
                                 kind])
    return {"conversations": records, "questions": question_records}


def prepare_compaction(workdir: Path, limit: int, seed: int) -> dict:
    """Sample trajectories and build the two compaction tables.

    Returns the tables the server reads and the page records.
    """
    rows = list(compaction.source_rows(limit, seed))
    conversations, questions, originals = [], [], []
    seen_issues = set()
    for row in rows:
        conv_id = row["trajectory_id"]
        if row["instance_id"] in seen_issues:
            raise ValueError(f"duplicate issue: {row['instance_id']}")
        seen_issues.add(row["instance_id"])
        messages = row["trajectory"]
        calls = compaction.collect_tool_calls(messages)
        candidates = [call for call in calls if not call.pinned]
        state, _tokens, _stage = (compaction.build_state(messages, calls)
                                  if candidates else (None, 0, "no candidates"))
        conversations.append({"id": conv_id,
                              "state": compaction.json_text(state)})
        originals.append({"id": conv_id, "instance_id": row["instance_id"],
                          "messages": json.dumps(messages)})
        for question in (item for call in candidates
                         for item in compaction.retention_questions(call)):
            questions.append({"id": f"{conv_id}:{question['key']}",
                              "conversation_id": conv_id, **question})
    schema = pa.schema([(name, pa.string()) for name in (
        "id", "conversation_id", "key", "tool_call_id", "kind", "statement")])
    tables = {
        "conversations": pa.Table.from_pylist(
            conversations, schema=pa.schema([("id", pa.string()),
                                             ("state", pa.string())])),
        "tool_questions": pa.Table.from_pylist(questions, schema=schema),
    }
    workdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(originals),
                   workdir / "messages.parquet")
    return {"tables": tables,
            "page": compaction_records(originals, tables["tool_questions"]),
            "metadata": {"dataset": compaction.DATASET,
                         "dataset_revision": compaction.DATASET_REVISION,
                         "seed": seed, "issue_limit": limit}}


def write_group(root: Path, group: str, tables: dict, page: dict,
                metadata: dict | None = None) -> dict:
    """Save one group's Arrow files and page JSON; return its manifest entry."""
    expected = tables_of(group)
    missing = set(expected) - set(tables)
    if missing:
        raise ValueError(f"group {group!r} lacks tables {sorted(missing)}")
    directory = root / group
    directory.mkdir(parents=True, exist_ok=True)
    entry = {"tables": {}, "metadata": metadata or {}}
    for name, table in tables.items():
        spec = expected[name]
        table = table.select(list(spec.columns))
        staging = directory / f"{name}.staging.arrow"
        rows = write_ipc_file(staging, table)
        content_id = file_digest(staging)
        path = directory / f"{name}.arrow"
        os.replace(staging, path)
        entry["tables"][name] = {
            "content_id": content_id, "rows": rows,
            "file": str(path.relative_to(root)), "id_col": spec.id_col,
            "columns": list(spec.columns)}
    page_path = root / f"{group}.json"
    temporary = page_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(page, separators=(",", ":")),
                         encoding="utf-8")
    os.replace(temporary, page_path)
    entry["page"] = page_path.name
    return entry


def uploads_for(item, entries: dict, root: Path) -> list:
    """The demo's tables as quail-server uploads, one per table.

    Each item is a ``PreparedInput`` whose ``spec`` is the submission's
    input description; ``upload_path`` is the Arrow file in the image.
    Tables the manifest lacks are left out.

    Args:
        item: The demo.
        entries: The manifest groups.
        root: The data directory the manifest's file paths are under.
    """
    from quail.server.inputs import PreparedInput

    group = entries.get(item.group) or {"tables": {}}
    uploads = []
    for spec in item.tables:
        table = group["tables"].get(spec.name)
        if table is None:
            continue
        uploads.append(PreparedInput(
            {"kind": "snapshot", "content_id": table["content_id"],
             "id_col": table["id_col"], "columns": table["columns"]},
            root / table["file"]))
    return uploads


def write_manifest(root: Path, entries: dict) -> Path:
    path = root / "manifest.json"
    existing = json.loads(path.read_text("utf-8")) if path.exists() else {}
    groups = {**existing.get("groups", {}), **entries}
    payload = {"built_at": time.time(), "groups": groups}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        return {"groups": {}}
    return json.loads(path.read_text("utf-8"))


def token_length_counter(hf_name: str):
    """Callable(list of texts) -> token counts, with a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    def count(texts):
        return [len(ids) for ids in
                tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    return count


def build(root: Path, groups=GROUPS, *, workdir: Path | None = None,
          compaction_limit: int = COMPACTION_LIMIT,
          compaction_seed: int = COMPACTION_SEED) -> dict:
    """Build the requested groups under ``root`` and update the manifest."""
    from playground.demos import QWEN3_4B
    from quail.specs import MODELS

    workdir = workdir or root / "work"
    entries = {}
    for group in groups:
        started = time.perf_counter()
        if group == "imdb":
            reviews, records = select_reviews(load_imdb())
            entries[group] = write_group(
                root, group, {"reviews": reviews}, {"reviews": records},
                {"dataset": IMDB_DATASET, "revision": IMDB_REVISION,
                 "reviews_per_label": REVIEWS_PER_LABEL})
        elif group == "bio":
            reports, terms = load_bio()
            counter = token_length_counter(MODELS[QWEN3_4B].hf_name)
            entries[group] = write_group(
                root, group, {"reports": reports, "terms": terms},
                bio_records(reports, terms, counter),
                {"benchmark": "quail-bench", "scale_factor": BIO_SCALE_FACTOR,
                 "token_counts": MODELS[QWEN3_4B].hf_name})
        elif group == "compaction":
            built = prepare_compaction(workdir / "compaction", compaction_limit,
                                       compaction_seed)
            entries[group] = write_group(root, group, built["tables"],
                                         built["page"], built["metadata"])
        else:
            raise ValueError(f"unknown data group {group!r}")
        print(f"built {group} in {time.perf_counter() - started:.1f} s: "
              + ", ".join(f"{name} {item['rows']} rows"
                          for name, item in entries[group]["tables"].items()),
              flush=True)
    write_manifest(root, entries)
    return entries
