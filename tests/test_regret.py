"""quail-bench's token counts, reached through the demo descriptions."""

import numpy as np
import pyarrow as pa
import pytest

from playground import demos, regret
from quail_b.minimum import prefix_trie_size
from tests.conftest import batch_tok, compile_demo, fake_tok


def describe(key, tables, anchors=None):
    item = demos.demo(key)
    session, query = compile_demo(item, tables[item.group])
    try:
        return item, regret.describe_query(query, anchors or {}, fake_tok)
    finally:
        session.close()


def test_filter_pieces_and_minimum_for_a_two_stage_filter(tiny_tables):
    item, description = describe("imdb-ending", tiny_tables)
    assert [f.id for f in description.info.filters] == [
        "filter:r:0", "filter:r:1"]
    assert description.info.relations[0].table == "reviews"
    assert description.info.relations[0].text_column == "review"
    pieces = description.pieces
    assert pieces["preamble"] and len(pieces["filters"]) == 2
    tails = [tuple(item["tail"]) for item in pieces["filters"]]
    assert tails[0] != tails[1]

    # every review asked the first question; reviews 0 and 1 passed and
    # were asked the second
    filters = {
        ("r", 0): pa.table({"r": pa.array([0, 1, 2, 3, 4, 5]),
                            "predicate": [0] * 6,
                            "answer": [True, True, False, False, False, False]}),
        ("r", 1): pa.table({"r": pa.array([0, 1]), "predicate": [1, 1],
                            "answer": [True, False]}),
    }
    reviews = tiny_tables["imdb"]["reviews"]
    docs = [np.asarray(fake_tok(text), np.uint32)
            for text in reviews.column("review").to_pylist()]
    pre = np.asarray(pieces["preamble"], np.uint32)
    expected_minimum = prefix_trie_size(
        [np.concatenate((pre, doc)) for doc in docs])
    expected_minimum += 4 * prefix_trie_size([tails[0]])
    expected_minimum += 2 * prefix_trie_size(tails)
    expected_requested = sum(len(pre) + len(doc) + len(tails[0]) for doc in docs)
    expected_requested += sum(len(pre) + len(docs[i]) + len(tails[1])
                              for i in (0, 1))
    report = {"wall_s": 2.0, "fresh_tokens": expected_minimum + 7,
              "cached_tokens": 3}
    tables = tiny_tables["imdb"]
    id_cols = {"reviews": "review_id"}
    output = regret.run_output(description, report, filters, {}, tables, id_cols)
    assert output.filter_answers["filter:r:0"].column("r").to_pylist()[:2] == [
        "review-0", "review-1"]
    numbers = regret.token_numbers(description, output, tables, id_cols,
                                   batch_tok)
    assert numbers["minimum_tokens"] == expected_minimum
    assert numbers["input_tokens"] == expected_requested
    assert numbers["regret_tokens"] == 7
    summary = regret.metrics(report, numbers, gpus=1, usd_per_hour=3.6)
    assert summary["tokens_per_second"] == expected_requested / 2.0
    assert summary["gpu_cost_usd"] == pytest.approx(2.0 / 3600 * 3.6)
    assert summary["regret_tokens"] == 7 and summary["cached_tokens"] == 3
    assert summary["kv_read_tokens"] == expected_requested - report["fresh_tokens"]
    # before the minimum is counted, a 0 cached counter is not a KV count
    partial = regret.metrics({"wall_s": 2.0, "fresh_tokens": 100, "cached_tokens": 0},
                             {}, gpus=1, usd_per_hour=3.6)
    assert partial["input_tokens"] is None and partial["kv_read_tokens"] is None
    partial = regret.metrics({"wall_s": 2.0, "fresh_tokens": 100, "cached_tokens": 30},
                             {}, gpus=1, usd_per_hour=3.6)
    assert partial["input_tokens"] == 130 and partial["kv_read_tokens"] == 30


def test_join_pieces_count_partners_once_per_anchor(tiny_tables):
    item, description = describe("bio", tiny_tables, {0: "r", 1: "r"})
    assert [join.id for join in description.info.joins] == ["join:0", "join:1"]
    assert description.info.joins[0].relations == ("r", "n")
    pieces = description.pieces
    assert [item["anchor"] for item in pieces["joins"]] == ["r", "r"]
    assert pieces["joins"][0]["frame"] and pieces["joins"][0]["label"]

    filters = {
        ("r", 0): pa.table({"r": pa.array([0, 1, 2]), "predicate": [0] * 3,
                            "answer": [True, False, True]}),
        ("n", 0): pa.table({"n": pa.array([0, 1, 2, 3]), "predicate": [0] * 4,
                            "answer": [True, False, False, True]}),
        ("c", 0): pa.table({"c": pa.array([0, 1, 2, 3]), "predicate": [0] * 4,
                            "answer": [False, True, False, False]}),
    }
    joins = {
        0: pa.table({"r": pa.array([0, 0, 2, 2]), "n": pa.array([0, 3, 0, 3]),
                     "answer": [True, False, True, False]}),
        1: pa.table({"r": pa.array([0, 2]), "c": pa.array([1, 1]),
                     "answer": [False, True]}),
    }
    tables = tiny_tables["bio"]
    id_cols = {"reports": "id", "terms": "id"}
    report = {"wall_s": 1.0, "fresh_tokens": 10_000}
    output = regret.run_output(description, report, filters, joins, tables,
                               id_cols)
    assert output.join_answers["join:0"].column("n").to_pylist() == [
        "tm0", "tm3", "tm0", "tm3"]
    numbers = regret.token_numbers(description, output, tables, id_cols,
                                   batch_tok)
    terms = [fake_tok(t) for t in tables["terms"].column("term").to_pylist()]
    join0, join1 = pieces["joins"]
    partner_cost = 0
    # anchors 0 and 2 each saw partners tm0 and tm3 in join 0, tm1 in join 1
    for _anchor in (0, 2):
        partner_cost += sum(len(join0["label"]) + len(join0["tail"]) + len(terms[t])
                            for t in (0, 3))
        partner_cost += len(join1["label"]) + len(join1["tail"]) + len(terms[1])
    assert numbers["minimum_tokens"] > partner_cost
    assert numbers["regret_tokens"] == 10_000 - numbers["minimum_tokens"]
    assert numbers["input_tokens"] > numbers["minimum_tokens"]


def test_reranker_pieces_come_from_the_layout(tiny_tables):
    item, description = describe("imdb-sentiment", tiny_tables)
    pieces = description.pieces
    assert pieces["preamble"], "the reranker system text is the preamble"
    (only,) = pieces["filters"]
    assert only["id"] == "filter:r:0" and only["tail"]
    filters = {("r", 0): pa.table({"r": pa.array([0, 1, 2])})}
    tables = tiny_tables["imdb"]
    output = regret.run_output(description, {"wall_s": 1.0, "fresh_tokens": 5},
                               filters, {}, tables, {"reviews": "review_id"})
    assert output.filter_answers["filter:r:0"].column("answer").to_pylist() == [
        True, True, True]
    numbers = regret.token_numbers(description, output, tables,
                                   {"reviews": "review_id"}, batch_tok)
    # fresh tokens below the minimum: reported as not measured, not an error
    assert numbers["regret_tokens"] is None and "below the minimum" in numbers["note"]
    summary = regret.metrics({"wall_s": 1.0, "fresh_tokens": 5}, numbers,
                             gpus=1, usd_per_hour=1.0)
    assert summary["regret_tokens"] is None and summary["tokens_per_second"] is None


def test_report_counters_are_available_before_regret_is_scored():
    summary = regret.metrics(
        {"wall_s": 2.0, "fresh_tokens": 5, "cached_tokens": 95}, {},
        gpus=1, usd_per_hour=3.6)
    assert summary["input_tokens"] == 100
    assert summary["kv_read_tokens"] == 95
    assert summary["tokens_per_second"] == 50
    assert summary["minimum_tokens"] is None
    assert summary["regret_tokens"] is None
