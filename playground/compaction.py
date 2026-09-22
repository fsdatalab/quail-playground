"""Prepare recorded OpenHands traces for the compaction demo.

Copied from ``demos/agent_trace_compaction.py`` of
https://github.com/fsdatalab/quail at commit
09c40b3cae37f9ea6cfb917361073452ea850ef9, keeping the parts that build
the query inputs: the retention questions, the compaction state, and
the trajectory sampling. The retention rules are the Python adaptation
of fast-jev-compaction; Quail answers them with Boolean decisions.

Reference: https://github.com/tamaratran/fast-jev-compaction (MIT).
Dataset: https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
(CC BY 4.0; retain the source repository and license in saved records).
"""

# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import itertools
import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = "nvidia/SWE-Zero-openhands-trajectories"
DATASET_REVISION = "7b3cd106d00f60918e722d33a1d74bc67072a7ea"
REFERENCE_REVISION = "e3f262a7f4d42bd8dd32ced30d26176f7cb545b0"
PRESERVE_RECENT = 6
MAX_STATE_TOKENS = 25_000
RESULT_HEAD_CHARS = 300

CONTEXT_TEXT = (
    "A coding assistant conversation is being compacted to free context. "
    "`history` is the whole conversation so far, oldest first; tool outputs "
    "are replaced by a short `result` note and long texts may be abridged. "
    "Each question asks whether one tool call, or the full output of that "
    "call, still needs to stay in the history verbatim. Whatever is not kept "
    "is deleted permanently, but the assistant can always re-run a tool "
    "or re-read a file."
)


def json_text(value) -> str:
    """Serialize state fields with the reference library's compact spacing."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


def text_length(text: str) -> int:
    """Count UTF-16 units, as JavaScript does in the original library."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def text_slice(text: str, start: int, end: int | None = None) -> str:
    """Slice using the reference library's UTF-16 offsets."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return raw[2 * start:None if end is None else 2 * end].decode(
        "utf-16-le", errors="surrogatepass")


def truncate(text: str, limit: int) -> str:
    return text if text_length(text) <= limit else text_slice(text, 0, limit - 1) + "…"


def estimate_tokens(text: str) -> int:
    """Apply the original heuristic; this is not a model tokenizer."""
    total = 0.0
    for piece in re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text):
        if piece[0].isascii() and piece[0].isdigit():
            total += len(piece) / 2
        elif piece[0].isascii() and piece[0].isalpha():
            total += 1 + (len(piece) - 1) // 6
        else:
            total += 0.9 * text_length(piece)
    return math.ceil(total)


def is_pinned(index: int, count: int) -> bool:
    return index == 0 or index >= count - PRESERVE_RECENT


@dataclass(frozen=True)
class ToolCall:
    """One paired call and result in a recorded trajectory."""

    id: str
    source_id: str
    tool: str
    arguments: dict
    call_index: int
    result_index: int
    result: str
    is_error: bool
    pinned: bool


def collect_tool_calls(messages: list[dict]) -> list[ToolCall]:
    """Pair OpenHands tool calls with results without guessing parallel matches."""
    pending, seen, paired = {}, set(), []
    for index, message in enumerate(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        if message["role"] == "tool":
            source_id = message.get("tool_call_id")
            if source_id is None and len(pending) == 1:
                source_id = next(iter(pending))
            if source_id not in pending:
                raise ValueError("ambiguous or unmatched tool result")
            call_index, tool, arguments = pending.pop(source_id)
            paired.append(ToolCall(
                "", source_id, tool, arguments, call_index, index, content,
                bool(message.get("is_error", False)),
                is_pinned(call_index, len(messages)) or is_pinned(index, len(messages)),
            ))
        for call in message.get("tool_calls") or []:
            source_id, function = call["id"], call["function"]
            if source_id in seen:
                raise ValueError(f"duplicate tool call ID: {source_id}")
            seen.add(source_id)
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            pending[source_id] = index, function["name"], arguments
    order = {call["id"]: i for i, call in enumerate(
        call for message in messages for call in message.get("tool_calls") or [])}
    return [replace(call, id=f"t{i + 1}") for i, call in enumerate(
        sorted(paired, key=lambda call: order[call.source_id]))]


def retention_questions(call: ToolCall) -> list[dict]:
    """Fill the original templates with a call's ID, tool name, and result size."""
    call_prompt = (
        f"Tool call {call.id} ({call.tool}) should stay in the history: knowing "
        "this call was made, with its input, still matters for what the "
        "assistant does next"
    )
    result_prompt = (
        f"The full output of tool call {call.id} ({call.tool}, "
        f"{text_length(call.result)} chars) should stay in the history verbatim: "
        "the assistant still needs its contents and re-running the tool would not do"
    )
    return [{"key": f"{kind}_{call.id}", "tool_call_id": call.source_id,
             "kind": kind, "statement": statement}
            for kind, statement in (("call", call_prompt), ("result", result_prompt))]


def build_state(messages: list[dict], calls: list[ToolCall]) -> tuple[dict, int, str]:
    """Fit the decision context using the reference library's reduction stages."""
    goal = "\n".join(truncate(message["content"], 500) for message in [
        m for m in messages if m["role"] == "user" and (m.get("content") or "").strip()
    ][-3:])
    by_message = {}
    for call in calls:
        by_message.setdefault(call.call_index, []).append(call)
    state = {"context": CONTEXT_TEXT, "goal": goal, "history": []}
    base_tokens = estimate_tokens(json_text(state))

    def result(stage):
        return state, base_tokens + sum(sizes), stage

    def fits():
        return base_tokens + sum(sizes) <= MAX_STATE_TOKENS

    def resize(index):
        sizes[index] = estimate_tokens(json_text(history[index])) + 1

    for cap in (1000, 200, 60):
        history = []
        for i, message in enumerate(messages):
            if message["role"] == "tool":
                continue
            text = message.get("content") or ""
            own = by_message.get(i, [])
            if not text.strip() and not own:
                continue
            entry = {"i": i, "role": message["role"], "text": text}
            if own:
                entry["tool_calls"] = [{
                    "id": call.id, "tool": call.tool,
                    "input": truncate(json_text(call.arguments), cap),
                    "result": f"{'error' if call.is_error else 'ok'}, "
                              f"{text_length(call.result)} chars (omitted)",
                } for call in own]
            history.append(entry)
        state["history"] = history
        sizes = [estimate_tokens(json_text(entry)) + 1 for entry in history]
        if fits():
            return result("full" if cap == 1000 else f"inputs<={cap}")

    def pinned(entry):
        return is_pinned(entry["i"], len(messages))

    order = sorted(range(len(history)), key=lambda i: pinned(history[i]))
    for i in order:
        text = history[i]["text"]
        length = text_length(text)
        if length > 590:
            history[i]["text"] = (text_slice(text, 0, 400)
                                  + f"\n[… {length - 550} chars omitted …]\n"
                                  + text_slice(text, -150))
            resize(i)
            if fits():
                return result("texts abridged")
    for i in order:
        entry = history[i]
        if not pinned(entry) and entry["text"]:
            length = text_length(messages[entry["i"]].get("content") or "")
            entry["text"] = f"[… {length} chars omitted …]"
            resize(i)
            if fits():
                return result("old messages collapsed")
    for i in order:
        entry = history[i]
        own = by_message.get(entry["i"])
        if not pinned(entry) and own:
            compact = []
            for call in own:
                values = []
                for key, value in call.arguments.items():
                    text = value if isinstance(value, str) else truncate(
                        json_text({key: value}), 200)
                    values.append(f"{key}={re.sub(r'\s+', ' ', text)}")
                compact.append(
                    f"{call.id} {call.tool} {truncate(' '.join(values), 60)} → "
                    f"{'error' if call.is_error else 'ok'} "
                    f"{text_length(call.result)}ch")
            entry["tool_calls"] = compact
            resize(i)
            if fits():
                return result("old calls compacted")
    removed = set()
    for i in order:
        if not pinned(history[i]) and "tool_calls" not in history[i]:
            removed.add(i)
            sizes[i] = 0
            if fits():
                state["history"] = [entry for j, entry in enumerate(history)
                                    if j not in removed]
                return result("old messages left out")
    merged = []
    for i, entry in enumerate(history):
        if i in removed:
            continue
        foldable = (not pinned(entry) and not entry["text"]
                    and isinstance(entry.get("tool_calls", [None])[0], str))
        previous = merged[-1] if merged else None
        if (foldable and previous and not pinned(previous) and not previous["text"]
                and isinstance(previous.get("tool_calls", [None])[0], str)
                and previous["role"] == entry["role"]):
            previous["tool_calls"] += entry["tool_calls"]
        else:
            merged.append(dict(entry))
    state["history"] = merged
    sizes = [estimate_tokens(json_text(entry)) + 1 for entry in merged]
    if fits():
        return result("old calls merged")
    raise ValueError("history exceeds the state budget after all reduction stages")


def truncated_result(call: ToolCall) -> str:
    """The tool result as the truncate decision keeps it."""
    text, length = call.result, text_length(call.result)
    if length > RESULT_HEAD_CHARS + 120:
        text = text_slice(text, 0, RESULT_HEAD_CHARS) + "\n"
        text += (f"[fast-jev-compaction truncated {length - RESULT_HEAD_CHARS} "
                 "chars of this tool result"
                 f"{' (error)' if call.is_error else ''}; "
                 "re-run the tool if needed]")
    return text


def sample_trajectories(index: pa.Table, limit: int, seed: int) -> pa.Table:
    """Choose one attempt per issue, then sample distinct issues reproducibly."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = index.sort_by([("instance_id", "ascending"),
                             ("trajectory_id", "ascending")])
    groups = pc.run_end_encode(ordered["instance_id"].combine_chunks())
    ends = groups.run_ends.to_numpy()
    starts = np.concatenate(([0], ends[:-1]))
    if limit > len(ends):
        raise ValueError(f"requested {limit} issues, but only {len(ends)} exist")
    rng = np.random.default_rng(seed)
    attempts = starts + rng.integers(ends - starts)
    issues = rng.choice(len(ends), size=limit, replace=False)
    return ordered.take(attempts[issues])


def cache_table(table: pa.Table, path: Path) -> None:
    """Publish a complete cached Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def source_index(filesystem, cache: Path) -> pa.Table:
    """Index all attempts using only ID columns, without reading trace bodies."""
    from huggingface_hub import HfApi

    path = cache / "index.parquet"
    if path.exists():
        return pq.read_table(path)
    files = sorted(name for name in HfApi().list_repo_files(
        DATASET, repo_type="dataset", revision=DATASET_REVISION)
        if name.startswith("data/") and name.endswith(".parquet"))
    parts = []
    for name in files:
        remote = f"datasets/{DATASET}@{DATASET_REVISION}/{name}"
        with filesystem.open(remote, block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(
                    group, columns=["instance_id", "trajectory_id"])
                parts.append(ids
                             .append_column("shard", pa.array([name] * len(ids)))
                             .append_column("row_group", pa.array([group] * len(ids)))
                             .append_column("row_index", pa.array(range(len(ids)))))
        print(f"indexed {name}", flush=True)
    index = pa.concat_tables(parts)
    cache_table(index, path)
    return index


def read_source_group(filesystem, cache: Path, entries: pa.Table) -> pa.Table:
    """Read one record group and cache only the sampled trajectories."""
    shard = entries["shard"][0].as_py()
    group = entries["row_group"][0].as_py()
    path = cache / "selected" / Path(shard).stem / f"{group}.parquet"
    cached = pq.read_table(path) if path.exists() else None
    positions = (pc.index_in(entries["trajectory_id"],
                             value_set=cached["trajectory_id"])
                 if cached is not None else None)
    if positions is not None and positions.null_count == 0:
        selected = cached.take(positions)
    else:
        full_group = cache / Path(shard).stem / f"{group}.parquet"
        if full_group.exists():
            table = pq.read_table(full_group)
        else:
            remote = f"datasets/{DATASET}@{DATASET_REVISION}/{shard}"
            with filesystem.open(remote, block_size=1 << 20) as source:
                table = pq.ParquetFile(source).read_row_group(group, columns=[
                    "instance_id", "trajectory_id", "repo", "license", "trajectory"])
        selected = table.take(entries["row_index"])
        if cached is not None:
            cached = cached.filter(pc.invert(pc.is_in(
                cached["trajectory_id"], value_set=selected["trajectory_id"])))
        cache_table(pa.concat_tables([cached, selected]) if cached is not None
                    else selected, path)
    for column in ("instance_id", "trajectory_id"):
        if not selected[column].equals(entries[column]):
            raise ValueError("cached trajectory does not match the source index")
    print(f"loaded {len(selected)} sampled trajectories from {shard}, "
          f"row group {group}", flush=True)
    return selected


def source_rows(limit: int, seed: int):
    """Sample with Arrow and fetch required record groups four at a time."""
    from huggingface_hub import HfFileSystem
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / "quail-agent-compaction" / DATASET_REVISION
    filesystem = HfFileSystem()
    started = time.perf_counter()
    index = source_index(filesystem, cache)
    print(f"source index: {len(index)} attempts, "
          f"{time.perf_counter() - started:.2f} s", flush=True)
    started = time.perf_counter()
    selected = sample_trajectories(index, limit, seed).sort_by([
        ("shard", "ascending"), ("row_group", "ascending"),
        ("row_index", "ascending")])
    print(f"sampled {len(selected)} issues in "
          f"{time.perf_counter() - started:.3f} s", flush=True)
    same_group = pc.and_(
        pc.equal(selected["shard"].slice(1), selected["shard"].slice(0, limit - 1)),
        pc.equal(selected["row_group"].slice(1),
                 selected["row_group"].slice(0, limit - 1)))
    boundaries = np.concatenate(([0], np.flatnonzero(~same_group.to_numpy()) + 1,
                                 [limit]))
    groups = [selected.slice(int(start), int(end - start))
              for start, end in itertools.pairwise(boundaries)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in itertools.batched(groups, 4):
            pending = [pool.submit(read_source_group, filesystem, cache, entries)
                       for entries in batch]
            for future in pending:
                yield from future.result().to_pylist()
