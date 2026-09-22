"""The playground page: static files, a proxy to each server, and metrics.

The page never sees the servers' bearer token. Every
``/s/<model>/v1/...`` request is forwarded to that model's Quail Server
with the token added, so the browser submits and polls through one
origin. The servers already hold the demo tables, registered from the
image at start; ``/config`` gives the page their content ids.
``/metrics`` reads a finished query's saved report and answer tables
off its server and computes the token numbers on this CPU, with the
input tables and the model's tokenizer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from playground import regret
from playground.demos import DEMOS, DEVICE, MODELS, demo
from playground.prepare import read_manifest

# a proxied request waits this long for a server; the page retries its
# readiness ping while a container is still restoring
PROXY_TIMEOUT_S = 120.0
HEARTBEAT_S = 30.0
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect width="16" height="16" rx="3" fill="#c31331"/></svg>'
)
# a long poll on a server holds for at most this long
MAX_WAIT_S = 60.0
HOP_HEADERS = frozenset({
    "connection", "keep-alive", "transfer-encoding", "host", "content-length",
    "authorization", "cookie", "accept-encoding",
})


@dataclass
class WebSettings:
    """What the page needs to run.

    Args:
        static_dir: The directory with index.html, app.js, and style.css.
        data_dir: The directory ``playground.prepare.build`` wrote.
        servers: Model name -> base URL of its Quail Server, or None
            when that server is not deployed.
        token: The servers' bearer token.
        usd_per_hour: The H100 price behind every cost number.
        reload: Called before reading the data directory, to refresh a
            Modal Volume; None when the directory is local.
        client_factory: Callable(endpoint, token) -> a quail-server
            client; ``quail.server.client.ServerClient`` when None.
        tokenizer_factory: Callable(model) -> callable(list of texts) ->
            token id lists; the model's Hugging Face tokenizer when None.
        describe: Callable(model, demo, tables, anchors) ->
            ``regret.QueryDescription``; compiles the SQL on a CPU
            session when None. Tests replace both.

    """

    static_dir: Path
    data_dir: Path
    servers: dict = field(default_factory=dict)
    token: str | None = None
    usd_per_hour: float = 0.0
    reload: Callable[[], None] | None = None
    client_factory: Callable | None = None
    tokenizer_factory: Callable | None = None
    describe: Callable | None = None


def hf_tokenizer(model: str):
    """Batch tokenizer of a model, from its Hugging Face checkpoint."""
    from transformers import AutoTokenizer

    from quail.specs import MODELS as SPECS

    tokenizer = AutoTokenizer.from_pretrained(SPECS[model].hf_name)

    def encode(texts):
        return tokenizer(list(texts), add_special_tokens=False)["input_ids"]

    return encode


def describe_demo(model: str, item, tables: dict, anchors: dict):
    """Compile the demo's SQL on a CPU session and describe it to quail-bench."""
    from quail.catalog import DocumentProvider
    from quail.execution.session import Session
    from quail.planner.plan import EngineConfig

    session = Session(EngineConfig(model=model, device=DEVICE))
    try:
        for spec in item.tables:
            session.register(spec.name, DocumentProvider.from_table(
                tables[spec.name], id_col=spec.id_col))
        query = session.sql(item.sql, dialect=item.dialect)
        return regret.describe_query(query, anchors, session.tokenizer)
    finally:
        session.close()


def server_client(settings: WebSettings, model: str):
    """A quail-server client for one model's deployed server."""
    from quail.server.client import ServerClient

    endpoint = settings.servers.get(model)
    if model not in MODELS or not endpoint:
        raise LookupError(f"no server is deployed for {model}")
    factory = settings.client_factory or ServerClient
    return factory(endpoint, settings.token or "")


def _table(payload: bytes) -> pa.Table:
    with ipc.open_file(pa.BufferReader(payload)) as reader:
        return reader.read_all()


class Metrics:
    """Compute and remember the token numbers of finished queries."""

    def __init__(self, settings: WebSettings):
        self.settings = settings
        self._lock = threading.Lock()
        self._tokenizers: dict = {}
        self._descriptions: dict = {}
        self._results: dict = {}

    def _client(self, model: str):
        return server_client(self.settings, model)

    def _tables(self, item) -> dict:
        manifest = read_manifest(self.settings.data_dir)
        group = manifest["groups"].get(item.group)
        if group is None:
            raise LookupError(f"the {item.group} data has not been prepared")
        tables = {}
        for spec in item.tables:
            path = self.settings.data_dir / group["tables"][spec.name]["file"]
            with ipc.open_file(str(path)) as reader:
                tables[spec.name] = reader.read_all()
        return tables

    def _tokenizer(self, model: str):
        if model not in self._tokenizers:
            factory = self.settings.tokenizer_factory or hf_tokenizer
            self._tokenizers[model] = factory(model)
        return self._tokenizers[model]

    def _description(self, model: str, item, tables: dict, anchors: dict):
        key = (item.key, tuple(sorted(anchors.items())))
        if key not in self._descriptions:
            factory = self.settings.describe or describe_demo
            self._descriptions[key] = factory(model, item, tables, anchors)
        return self._descriptions[key]

    def answers(self, client, status) -> tuple[dict, dict]:
        """The saved answer tables of a succeeded query."""
        files = status.result["files"]["answers"]
        filters = {
            (entry["alias"], entry["position"]):
                _table(client.file(status.id, entry["file"]))
            for entry in files["filters"]
        }
        joins = {}
        for entry in files["joins"]:
            table = _table(client.file(status.id, entry["file"]))
            metadata = table.schema.metadata or {}
            anchor = metadata.get(b"quail.anchor", b"").decode("utf-8")
            joins[entry["position"]] = (anchor, table)
        return filters, joins

    def scored_rows(self, client, status, item) -> dict:
        """Filter tables from the streamed score entries, for a reranker.

        A reranker query saves no filter answer table, so the rows it
        scored come from the answer stream instead.
        """
        rows: dict = {}
        after = 0
        while True:
            page = client.answers(status.id, after=after, limit=10_000)
            for entry in page["answers"]:
                if entry.get("kind") != "score":
                    continue
                (alias,) = entry["aliases"]
                rows.setdefault(alias, []).extend(
                    row if isinstance(row, int) else row[0]
                    for row in entry["rows"])
            if not page["answers"]:
                break
            after = page["next"]
        return {(alias, 0): pa.table({alias: pa.array(found, pa.int64())})
                for alias, found in rows.items()}

    def compute(self, model: str, query_id: str, demo_key: str) -> dict:
        item = demo(demo_key)
        key = (model, query_id, demo_key)
        with self._lock:
            if key in self._results:
                return self._results[key]
        client = self._client(model)
        status = client.status(query_id)
        if status.state != "succeeded":
            raise LookupError(f"query {query_id} is {status.state}")
        report = json.loads(client.file(query_id, status.result["files"]["report"]))
        tables = self._tables(item)
        filters, joins = self.answers(client, status)
        if not filters and not joins and item.view == "score":
            filters = self.scored_rows(client, status, item)
        anchors = {position: anchor for position, (anchor, _) in joins.items()}
        description = self._description(model, item, tables, anchors)
        id_cols = {spec.name: spec.id_col for spec in item.tables}
        output = regret.run_output(
            description, report, filters,
            {position: table for position, (_, table) in joins.items()},
            tables, id_cols)
        numbers = regret.token_numbers(description, output, tables, id_cols,
                                       self._tokenizer(model))
        result = regret.metrics(report, numbers, gpus=int(status.config["gpus"]),
                                usd_per_hour=self.settings.usd_per_hour)
        result.update(query_id=query_id, model=model, demo=demo_key,
                      output_rows=status.result["rows"])
        with self._lock:
            self._results[key] = result
        return result

    def true_pairs(self, model: str, query_id: str) -> dict:
        """Every join answer table's true pairs, as row indices."""
        client = self._client(model)
        status = client.status(query_id)
        if status.state != "succeeded":
            raise LookupError(f"query {query_id} is {status.state}")
        _filters, joins = self.answers(client, status)
        out = []
        for position, (anchor, table) in sorted(joins.items()):
            metadata = table.schema.metadata or {}
            partners = metadata.get(b"quail.partners", b"").decode("utf-8")
            partners = partners.split(",") if partners else []
            true_rows = table.filter(table.column("answer"))
            columns = [anchor, *partners]
            out.append({
                "position": position, "anchor": anchor, "partners": partners,
                "asked": table.num_rows,
                "pairs": [list(row) for row in zip(*(
                    true_rows.column(name).to_pylist() for name in columns))],
            })
        return {"joins": out}


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def _heartbeat(started: float) -> None:
    """Print a line every HEARTBEAT_S from a thread, so a frozen container shows."""
    while True:
        time.sleep(HEARTBEAT_S)
        print(f"page alive {time.time() - started:.0f} s", flush=True)


async def _loop_lag(started: float) -> None:
    """Print how late the event loop wakes, so a blocked loop shows."""
    while True:
        before = time.monotonic()
        await asyncio.sleep(HEARTBEAT_S)
        lag = time.monotonic() - before - HEARTBEAT_S
        print(f"page loop lag {lag * 1000:.0f} ms at {time.time() - started:.0f} s",
              flush=True)


def create_web_app(settings: WebSettings) -> Starlette:
    """Build the page's Starlette application."""
    metrics = Metrics(settings)
    started = time.time()
    threading.Thread(target=_heartbeat, args=(started,), daemon=True,
                     name="page-heartbeat").start()
    client = httpx.AsyncClient(timeout=httpx.Timeout(
        PROXY_TIMEOUT_S, read=PROXY_TIMEOUT_S + MAX_WAIT_S))

    def refreshed_manifest() -> dict:
        if settings.reload is not None:
            settings.reload()
        return read_manifest(settings.data_dir)

    async def index(request: Request):
        path = settings.static_dir / "index.html"
        return HTMLResponse(path.read_text("utf-8"))

    async def favicon(request: Request):
        return Response(FAVICON_SVG, media_type="image/svg+xml")

    async def static(request: Request):
        name = request.path_params["name"]
        path = settings.static_dir / name
        if "/" in name or not path.is_file():
            return _error(f"no static file {name!r}", 404)
        return FileResponse(str(path))

    async def config(request: Request):
        manifest = await run_in_threadpool(refreshed_manifest)
        return JSONResponse({
            "device": DEVICE,
            "models": list(MODELS),
            "servers": {model: settings.servers.get(model) for model in MODELS},
            "usd_per_hour": settings.usd_per_hour,
            "demos": [item.public() for item in DEMOS],
            "groups": manifest.get("groups", {}),
            "built_at": manifest.get("built_at"),
        })

    async def data(request: Request):
        group = request.path_params["group"]
        path = settings.data_dir / f"{group}.json"
        if "/" in group or not path.is_file():
            return _error(f"the {group} data has not been prepared", 404)
        return FileResponse(str(path), media_type="application/json")

    async def proxy(request: Request):
        model = request.path_params["model"]
        path = request.path_params["path"]
        endpoint = settings.servers.get(model)
        if model not in MODELS or not endpoint:
            return _error(f"no server is deployed for {model!r}", 404)
        url = f"{endpoint}/v1/{path}"
        headers = {name: value for name, value in request.headers.items()
                   if name.lower() not in HOP_HEADERS}
        if settings.token:
            headers["authorization"] = f"Bearer {settings.token}"
        body = await request.body()
        try:
            upstream = await client.request(
                request.method, url, params=request.query_params,
                headers=headers, content=body)
        except httpx.HTTPError as error:
            return _error(f"cannot reach the {model} server: {error}", 502)
        return Response(
            upstream.content, status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={name: value for name, value in upstream.headers.items()
                     if name.lower().startswith("x-quail-")})

    async def query_metrics(request: Request):
        model = request.path_params["model"]
        query_id = request.path_params["query_id"]
        demo_key = request.query_params.get("demo", "")
        try:
            result = await run_in_threadpool(
                metrics.compute, model, query_id, demo_key)
        except (LookupError, KeyError) as error:
            return _error(str(error), 409)
        except Exception as error:
            return _error(f"{type(error).__name__}: {error}", 500)
        return JSONResponse(result)

    async def join_pairs(request: Request):
        model = request.path_params["model"]
        query_id = request.path_params["query_id"]
        try:
            result = await run_in_threadpool(metrics.true_pairs, model, query_id)
        except LookupError as error:
            return _error(str(error), 409)
        return JSONResponse(result)

    routes = [
        Route("/", index),
        Route("/favicon.ico", favicon),
        Route("/static/{name}", static),
        Route("/config", config),
        Route("/data/{group}", data),
        Route("/s/{model}/v1/{path:path}", proxy,
              methods=["GET", "POST", "HEAD"]),
        Route("/metrics/{model}/{query_id}", query_metrics),
        Route("/joins/{model}/{query_id}", join_pairs),
    ]
    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(_loop_lag(started))
        try:
            yield
        finally:
            task.cancel()

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.metrics = metrics
    return app


def main(argv=None) -> None:
    """Serve the page locally against running Quail Servers.

    python -m playground.web --data-dir ./data --server MODEL=URL
    """
    import argparse
    import os

    import uvicorn

    from quail.specs import H100_USD_PER_HOUR

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--server", action="append", default=[],
                        metavar="MODEL=URL",
                        help="a running Quail Server; repeat per model")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    servers = dict(item.split("=", 1) for item in args.server)
    settings = WebSettings(
        static_dir=Path(__file__).resolve().parent.parent / "web",
        data_dir=args.data_dir, servers=servers,
        token=os.environ.get("QUAIL_SERVER_TOKEN"),
        usd_per_hour=H100_USD_PER_HOUR)
    uvicorn.run(create_web_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
