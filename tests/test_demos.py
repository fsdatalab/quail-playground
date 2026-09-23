"""The four demos compile on a CPU session against tiny tables."""

import pytest

from playground import demos
from tests.conftest import compile_demo


def test_four_demos_on_three_models():
    assert len(demos.DEMOS) == 4
    assert {item.model for item in demos.DEMOS} == set(demos.MODELS)
    assert len(demos.MODELS) == 3
    assert demos.GROUPS == ("compaction", "imdb", "bio")
    assert [item.view for item in demos.DEMOS] == [
        "compaction", "score", "filter", "join"]


def test_public_fields_carry_what_the_page_needs():
    item = demos.demo("bio")
    public = item.public()
    assert public["model"] == demos.QWEN3_4B
    assert [table["name"] for table in public["tables"]] == ["reports", "terms"]
    assert public["hints"]["joins"] == {"n": "neurological", "c": "cardiovascular"}
    sources = {item.key: item.public()["hints"]["source"]["url"]
               for item in demos.DEMOS}
    assert sources == {
        "imdb-sentiment": "https://huggingface.co/datasets/stanfordnlp/imdb",
        "imdb-ending": "https://huggingface.co/datasets/stanfordnlp/imdb",
        "bio": "https://huggingface.co/datasets/BioDEX/BioDEX-Reactions",
        "compaction": ("https://huggingface.co/datasets/nvidia/"
                       "SWE-Zero-openhands-trajectories"),
    }
    with pytest.raises(KeyError, match="unknown demo"):
        demos.demo("nope")


def test_tables_of_a_group_are_shared_across_its_demos():
    assert set(demos.tables_of("imdb")) == {"reviews"}
    assert demos.tables_of("imdb")["reviews"].id_col == "review_id"
    assert set(demos.tables_of("bio")) == {"reports", "terms"}


@pytest.mark.parametrize("key", [item.key for item in demos.DEMOS])
def test_sql_compiles_with_the_expected_operators(key, tiny_tables):
    item = demos.demo(key)
    session, query = compile_demo(item, tiny_tables[item.group])
    try:
        operators = query.logical.operators()
        filters = {alias: len(predicates)
                   for alias, predicates in operators.filters.items()}
        if key == "imdb-sentiment":
            assert filters == {"r": 1} and not operators.joins
        elif key == "imdb-ending":
            assert filters == {"r": 2} and not operators.joins
            assert operators.filters["r"][0].selectivity == 0.25
        elif key == "bio":
            assert filters == {"r": 1, "n": 1, "c": 1}
            assert [join.anchor for join in operators.joins] == ["r", "r"]
            assert operators.joins[0].selectivity == demos.REACTION_SELECTIVITY
        else:
            assert not filters and len(operators.joins) == 1
            assert operators.joins[0].anchor == "c"
    finally:
        session.close()
