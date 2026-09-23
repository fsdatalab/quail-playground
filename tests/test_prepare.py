"""Building the demo data and its manifest, on tiny inputs."""

import json

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from playground import prepare
from playground.demos import demo


def test_select_reviews_takes_negatives_then_positives(monkeypatch):
    monkeypatch.setattr(prepare, "REVIEWS_PER_LABEL", 2)
    imdb = pa.table({
        "text": ["bad one", "good one", "bad two", "good two", "bad three"],
        "label": [0, 1, 0, 1, 0],
    })
    reviews, records = prepare.select_reviews(imdb)
    assert reviews.column_names == ["review_id", "review"]
    assert reviews.column("review").to_pylist() == [
        "bad one", "bad two", "good one", "good two"]
    assert [record["label"] for record in records] == [0, 0, 1, 1]
    assert records[2] == {"id": "review-2", "label": 1, "head": "good one"}
    with pytest.raises(ValueError, match="label 1"):
        prepare.select_reviews(imdb.slice(0, 3))


def test_head_cuts_long_text_and_joins_whitespace():
    assert prepare.head("a  b\n\nc") == "a b c"
    cut = prepare.head("word " * 100)
    assert cut.endswith("…") and len(cut) <= prepare.HEAD_CHARS + 1


def test_bio_records_carry_token_counts_and_terms(tiny_tables):
    bio = tiny_tables["bio"]
    records = prepare.bio_records(
        bio["reports"], bio["terms"], lambda texts: [len(t.split()) for t in texts])
    assert records["reports"][0]["id"] == "rp0"
    assert records["reports"][0]["tokens"] == len(
        bio["reports"].column("report")[0].as_py().split())
    assert records["terms"][1] == {"id": "tm1", "term": "arrhythmia"}


def _trajectory():
    call = {"id": "x1", "function": {"name": "bash",
                                     "arguments": json.dumps({"cmd": "ls"})}}
    messages = [{"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "x1",
                 "content": "file.py\n" + "line\n" * 200}]
    # seven trailing messages, so the tool call is not pinned as recent
    for _ in range(7):
        messages.append({"role": "assistant", "content": "thinking"})
    return messages


def test_compaction_records_map_questions_to_calls():
    messages = _trajectory()
    conversations = [{"id": "c0", "instance_id": "repo__issue-1",
                      "messages": json.dumps(messages)}]
    questions = pa.table({
        "conversation_id": ["c0", "c0"], "tool_call_id": ["x1", "x1"],
        "kind": ["call", "result"]})
    records = prepare.compaction_records(conversations, questions)
    (conversation,) = records["conversations"]
    assert conversation["name"] == "repo__issue-1"
    (call,) = conversation["calls"]
    assert call["id"] == "t1" and call["tool"] == "bash" and not call["pinned"]
    assert 0 < call["truncated_tokens"] < call["tokens"]
    assert conversation["tokens"] == call["tokens"]
    assert records["questions"] == [[0, "t1", "call"], [0, "t1", "result"]]


def test_write_group_names_files_by_content_and_lists_uploads(tmp_path, tiny_tables):
    root = tmp_path / "data"
    entry = prepare.write_group(root, "bio", tiny_tables["bio"],
                                {"reports": [], "terms": []}, {"scale_factor": 0.1})
    reports = entry["tables"]["reports"]
    assert reports["rows"] == 3 and reports["id_col"] == "id"
    assert reports["columns"] == ["id", "report"]
    with ipc.open_file(str(root / reports["file"])) as reader:
        assert reader.read_all().num_rows == 3
    assert json.loads((root / entry["page"]).read_text()) == {
        "reports": [], "terms": []}
    with pytest.raises(ValueError, match="lacks tables"):
        prepare.write_group(root, "bio", {"reports": tiny_tables["bio"]["reports"]},
                            {})

    prepare.write_manifest(root, {"bio": entry})
    prepare.write_manifest(root, {"imdb": {"tables": {}, "page": "imdb.json"}})
    manifest = prepare.read_manifest(root)
    assert set(manifest["groups"]) == {"bio", "imdb"}, "groups accumulate"
    prepared = prepare.uploads_for(demo("bio"), manifest["groups"], root)
    assert [item.spec["columns"] for item in prepared] == [
        ["id", "report"], ["id", "term"]]
    assert [item.content_id for item in prepared] == [
        reports["content_id"], entry["tables"]["terms"]["content_id"]]
    assert all(item.upload_path.exists() for item in prepared)
    assert prepare.uploads_for(demo("imdb-ending"), manifest["groups"], root) == []
