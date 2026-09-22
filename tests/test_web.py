"""The page's routes: config, data, the proxy, and the metrics."""

import json
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from playground import prepare, regret, web
from playground.demos import QWEN3_4B, demo
from quail.server.records import QueryStatus
from tests.conftest import batch_tok, compile_demo, fake_tok

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def fake_upstream(seen: list) -> Starlette:
    async def capabilities(request: Request):
        seen.append(("GET", request.url.path, request.headers.get("authorization")))
        return JSONResponse({"models": [QWEN3_4B]})

    async def submit(request: Request):
        body = await request.json()
        seen.append(("POST", request.url.path, request.headers.get("authorization")))
        return JSONResponse({"id": "q1", "state": "queued", "sql": body["sql"]},
                            status_code=201)

    async def status(request: Request):
        seen.append(("GET", request.url.path + "?" + request.url.query, None))
        return Response("{}", media_type="application/json",
                        headers={"x-quail-rows": "3", "server": "hidden"})

    return Starlette(routes=[
        Route("/v1/capabilities", capabilities),
        Route("/v1/queries", submit, methods=["POST"]),
        Route("/v1/queries/{query_id}", status),
    ])


@pytest.fixture
def data_dir(tmp_path, tiny_tables):
    root = tmp_path / "data"
    entry = prepare.write_group(root, "imdb", tiny_tables["imdb"],
                                {"reviews": [{"id": "review-0"}]})
    prepare.write_manifest(root, {"imdb": entry})
    return root


def make_app(data_dir, seen, monkeypatch, **overrides):
    settings = web.WebSettings(
        static_dir=WEB_DIR, data_dir=data_dir,
        servers={QWEN3_4B: "http://upstream"}, token="secret",
        usd_per_hour=3.0, tokenizer_factory=lambda model: batch_tok,
        **overrides)
    upstream = fake_upstream(seen)
    original = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.ASGITransport(app=upstream)
        return original(*args, **kwargs)

    monkeypatch.setattr(web.httpx, "AsyncClient", client)
    return web.create_web_app(settings)


def test_page_config_and_data(data_dir, monkeypatch):
    seen = []
    with TestClient(make_app(data_dir, seen, monkeypatch)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Quail playground" in page.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/../pyproject.toml").status_code == 404
        config = client.get("/config").json()
        assert config["servers"] == {QWEN3_4B: "http://upstream",
                                     "qwen3-reranker-0.6b-bf16": None,
                                     "diffusion-gemma-26b-a4b-fp8": None}
        assert [item["key"] for item in config["demos"]] == [
            "imdb-sentiment", "imdb-ending", "bio-4", "compaction"]
        assert config["groups"]["imdb"]["tables"]["reviews"]["id_col"] == "review_id"
        assert config["usd_per_hour"] == 3.0
        assert client.get("/data/imdb").json() == {"reviews": [{"id": "review-0"}]}
        assert client.get("/data/bio").status_code == 404


def test_proxy_adds_the_token_and_keeps_quail_headers(data_dir, monkeypatch):
    seen = []
    with TestClient(make_app(data_dir, seen, monkeypatch)) as client:
        assert client.get(f"/s/{QWEN3_4B}/v1/capabilities").json() == {
            "models": [QWEN3_4B]}
        submitted = client.post(f"/s/{QWEN3_4B}/v1/queries",
                                json={"sql": "SELECT 1"})
        assert submitted.status_code == 201 and submitted.json()["id"] == "q1"
        status = client.get(f"/s/{QWEN3_4B}/v1/queries/q1?after=3&wait=5")
        assert status.headers["x-quail-rows"] == "3"
        assert "server" not in status.headers or status.headers["server"] != "hidden"
        missing = client.get("/s/qwen3-reranker-0.6b-bf16/v1/capabilities")
        assert missing.status_code == 404
    assert seen[0] == ("GET", "/v1/capabilities", "Bearer secret")
    assert seen[1][2] == "Bearer secret"
    assert seen[2][1] == "/v1/queries/q1?after=3&wait=5"


class FakeServerClient:
    """What Metrics reads off a server: the status, files, and answers."""

    def __init__(self, status: QueryStatus = None, files: dict = None,
                 answers=()):
        self._status = status
        self._files = files or {}
        self._answers = list(answers)

    def status(self, query_id):
        return self._status

    def file(self, query_id, name):
        return self._files[name]

    def answers(self, query_id, *, after=0, limit=1000):
        page = self._answers[after:after + limit]
        return {"answers": page, "next": after + len(page), "done": True}


def _ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _status(state="succeeded", files=None, rows=2) -> QueryStatus:
    return QueryStatus(
        id="q1", state=state, revision=9, created_at=0.0, updated_at=1.0,
        timeout_s=10.0, spec={"sql": "..."},
        config={"model": QWEN3_4B, "device": "h100-sxm", "gpus": 1,
                "backend": "quail"},
        inputs={}, result={"rows": rows, "columns": ["review_id"],
                           "files": {"result": "result.arrow",
                                     "report": "report.json",
                                     "answers": files or
                                     {"filters": [], "joins": []}}})


def test_metrics_use_the_saved_answer_tables(data_dir, monkeypatch, tiny_tables):
    seen = []
    item = demo("imdb-ending")

    def describe(model, item, tables, anchors):
        session, query = compile_demo(item, tables)
        try:
            return regret.describe_query(query, anchors, fake_tok)
        finally:
            session.close()

    app = make_app(data_dir, seen, monkeypatch, describe=describe)
    filters = pa.table({"r": pa.array([0, 1, 2]), "predicate": [0, 0, 0],
                        "answer": [True, True, False]})
    second = pa.table({"r": pa.array([0, 1]), "predicate": [1, 1],
                       "answer": [True, False]})
    report = {"wall_s": 4.0, "fresh_tokens": 100_000, "cached_tokens": 20}
    files = {"report.json": json.dumps(report).encode(),
             "answers/filters/r--0.arrow": _ipc_bytes(filters),
             "answers/filters/r--1.arrow": _ipc_bytes(second)}
    status = _status(files={"filters": [
        {"alias": "r", "position": 0, "file": "answers/filters/r--0.arrow"},
        {"alias": "r", "position": 1, "file": "answers/filters/r--1.arrow"}],
        "joins": []})
    fake = FakeServerClient(status, files)
    monkeypatch.setattr(app.state.metrics, "_client", lambda model: fake)
    with TestClient(app) as client:
        # computed on a thread: 202 until the numbers are there
        for _ in range(200):
            response = client.get(f"/metrics/{QWEN3_4B}/q1?demo={item.key}")
            if response.status_code != 202:
                break
            time.sleep(0.05)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["fresh_tokens"] == 100_000
        assert result["minimum_tokens"] < result["input_tokens"] < 100_000
        assert result["regret_tokens"] == 100_000 - result["minimum_tokens"]
        assert result["tokens_per_second"] == result["input_tokens"] / 4.0
        assert result["gpu_cost_usd"] == pytest.approx(4.0 / 3600 * 3.0)
        assert result["output_rows"] == 2 and result["cached_tokens"] == 20
        # computed once; a second read is the remembered result
        fake._files = {}
        assert client.get(f"/metrics/{QWEN3_4B}/q1?demo={item.key}").json() == result
        assert client.get(f"/metrics/{QWEN3_4B}/q1?demo=nope").status_code == 409

    unfinished = FakeServerClient(_status(state="running"), {})
    monkeypatch.setattr(app.state.metrics, "_client", lambda model: unfinished)
    app.state.metrics._results.clear()
    with TestClient(app) as client:
        for _ in range(200):
            response = client.get(f"/metrics/{QWEN3_4B}/q2?demo={item.key}")
            if response.status_code != 202:
                break
            time.sleep(0.05)
        assert response.status_code == 409 and "running" in response.text


def test_join_pairs_are_read_from_the_saved_tables(data_dir, monkeypatch):
    seen = []
    app = make_app(data_dir, seen, monkeypatch)
    table = pa.table({"r": pa.array([0, 0, 1]), "n": pa.array([2, 3, 2]),
                      "answer": [True, False, True]})
    table = table.replace_schema_metadata({b"quail.anchor": b"r",
                                           b"quail.partners": b"n"})
    status = _status(files={"filters": [], "joins": [
        {"position": 0, "file": "answers/joins/0.arrow"}]})
    fake = FakeServerClient(status, {"answers/joins/0.arrow": _ipc_bytes(table)})
    monkeypatch.setattr(app.state.metrics, "_client", lambda model: fake)
    with TestClient(app) as client:
        assert client.get(f"/joins/{QWEN3_4B}/q1").json() == {"joins": [
            {"position": 0, "anchor": "r", "partners": ["n"], "asked": 3,
             "pairs": [[0, 2], [1, 2]]}]}
