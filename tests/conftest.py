"""Shared fakes: a word tokenizer and tiny tables for every demo."""

from __future__ import annotations

import re

import pyarrow as pa
import pytest

from playground import demos

_WORDS: dict = {}


def fake_tok(text: str) -> list[int]:
    """Deterministic word-level token ids, one id per distinct word."""
    ids = []
    for word in re.findall(r"\S+|\n", text):
        if word not in _WORDS:
            _WORDS[word] = len(_WORDS) + 1
        ids.append(_WORDS[word])
    return ids


def batch_tok(texts) -> list[list[int]]:
    return [fake_tok(text) for text in texts]


@pytest.fixture
def tiny_tables() -> dict:
    """Group -> table name -> a small Arrow table with the demo's columns."""
    return {
        "imdb": {"reviews": pa.table({
            "review_id": [f"review-{i}" for i in range(6)],
            "review": [f"review {i} about the ending " + "pad " * (i + 1)
                       for i in range(6)],
        })},
        "bio": {
            "reports": pa.table({
                "id": ["rp0", "rp1", "rp2"],
                "report": ["report zero of a serious event " + "x " * 5,
                           "report one mild " + "y " * 7,
                           "report two seizure and arrhythmia " + "z " * 3],
            }),
            "terms": pa.table({
                "id": ["tm0", "tm1", "tm2", "tm3"],
                "term": ["seizure", "arrhythmia", "rash", "headache"],
            }),
        },
        "compaction": {
            "conversations": pa.table({
                "id": ["c0", "c1"],
                "state": ["state of conversation zero " + "s " * 4,
                          "state of conversation one " + "t " * 6],
            }),
            "tool_questions": pa.table({
                "id": ["c0:call_t1", "c0:result_t1", "c1:call_t1", "c1:result_t1"],
                "conversation_id": ["c0", "c0", "c1", "c1"],
                "key": ["call_t1", "result_t1", "call_t1", "result_t1"],
                "tool_call_id": ["x1", "x1", "y1", "y1"],
                "kind": ["call", "result", "call", "result"],
                "statement": ["keep call one", "keep result one",
                              "keep call one", "keep result one"],
            }),
        },
    }


def compile_demo(item: demos.Demo, tables: dict):
    """Compile a demo's SQL on a CPU session with the fake tokenizer."""
    from quail.catalog import DocumentProvider
    from quail.execution.session import Session
    from quail.planner.plan import EngineConfig

    session = Session(EngineConfig(model=item.model, device=demos.DEVICE),
                      tokenizer=fake_tok)
    for spec in item.tables:
        session.register(spec.name, DocumentProvider.from_table(
            tables[spec.name], id_col=spec.id_col))
    return session, session.sql(item.sql, dialect=item.dialect)
