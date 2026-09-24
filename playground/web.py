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
import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import httpx
import pyarrow as pa
import pyarrow.csv as pa_csv
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
# the Full Stack Data Lab logo, traced from fsdatalab.github.io's PNG icon
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect width="100" height="100" rx="18" fill="#C41230"/>'
    '<rect x="18" y="16" width="64" height="15" rx="3" fill="#fff"/>'
    '<rect x="18" y="33.5" width="32" height="15" rx="3" fill="#fff"/>'
    '<rect x="18" y="51" width="16" height="15" rx="3" fill="#fff"/>'
    '<rect x="18" y="69" width="16" height="15" rx="3" fill="#fff"/>'
    '</svg>'
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
        servers: Model name -> four server URLs. A single URL is also
            accepted for local use.
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
        prewarm: Load every demo's tokenizer and tokenize its documents
            on a thread when the page starts, so the first query after
            a cold start does not wait for them.

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
    prewarm: bool = False


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


def server_endpoint(settings: WebSettings, model: str,
                    query_id: str | None = None) -> str:
    """Return the server slot holding a query."""
    urls = settings.servers.get(model)
    if model not in MODELS or not urls:
        raise LookupError(f"no server is deployed for {model}")
    if isinstance(urls, str):
        urls = (urls,)
    slot = 0
    assigned = re.match(r"q4s(\d+)_", query_id or "")
    if assigned:
        slot = int(assigned.group(1))
    elif query_id and query_id.startswith("q4_"):
        slot = int.from_bytes(
            hashlib.sha256(query_id.encode()).digest()[:8], "big") % len(urls)
    if slot >= len(urls):
        raise LookupError(f"no server is deployed for {model} slot {slot}")
    endpoint = urls[slot]
    if not endpoint:
        raise LookupError(f"no server is deployed for {model} slot {slot}")
    return endpoint


def server_client(settings: WebSettings, model: str,
                  query_id: str | None = None):
    """A quail-server client for the slot holding a query."""
    from quail.server.client import ServerClient

    endpoint = server_endpoint(settings, model, query_id)
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
        self._tokenizer_locks: dict = {}
        self._tables_cache: dict = {}
        self._document_tokens: dict = {}
        self._document_locks: dict = {}
        self._descriptions: dict = {}
        self._results: dict = {}
        self._partials: dict = {}
        self._pending: dict = {}

    def _client(self, model: str, query_id: str):
        return server_client(self.settings, model, query_id)

    def _tables(self, item) -> dict:
        with self._lock:
            cached = self._tables_cache.get(item.group)
        if cached is not None:
            return cached
        manifest = read_manifest(self.settings.data_dir)
        group = manifest["groups"].get(item.group)
        if group is None:
            raise LookupError(f"the {item.group} data has not been prepared")
        tables = {}
        for spec in item.tables:
            path = self.settings.data_dir / group["tables"][spec.name]["file"]
            with ipc.open_file(str(path)) as reader:
                tables[spec.name] = reader.read_all()
        with self._lock:
            return self._tables_cache.setdefault(item.group, tables)

    def _tokenizer(self, model: str):
        with self._lock:
            cached = self._tokenizers.get(model)
            model_lock = self._tokenizer_locks.setdefault(
                model, threading.Lock())
        if cached is not None:
            return cached
        with model_lock:
            with self._lock:
                cached = self._tokenizers.get(model)
            if cached is None:
                factory = self.settings.tokenizer_factory or hf_tokenizer
                cached = factory(model)
                with self._lock:
                    self._tokenizers[model] = cached
            return cached

    def _documents(self, model: str, item, tables: dict):
        """Reusable document tokens and their per-dataset lock."""
        key = (model, item.group)
        with self._lock:
            cached = self._document_tokens.get(key)
            document_lock = self._document_locks.get(key)
        if cached is None:
            id_cols = {spec.name: spec.id_col for spec in item.tables}
            candidate = regret.DocumentTokens(
                regret.corpus_rows(tables, id_cols), self._tokenizer(model))
            with self._lock:
                cached = self._document_tokens.setdefault(key, candidate)
                document_lock = self._document_locks.setdefault(
                    key, threading.Lock())
        return cached, document_lock

    def _description(self, model: str, item, tables: dict, anchors: dict,
                     sql: str):
        key = (item.key, sql, tuple(sorted(anchors.items())))
        if key not in self._descriptions:
            factory = self.settings.describe or describe_demo
            query_item = replace(item, sql=sql)
            self._descriptions[key] = factory(
                model, query_item, tables, anchors)
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

    def poll(self, model: str, query_id: str, demo_key: str) -> dict | None:
        """The numbers when they are ready, else None with the work started.

        The counts take from seconds to minutes, so they are computed on
        a thread and the page asks again until they are there. A
        failure is remembered and raised to every later poll.
        """
        item = demo(demo_key)
        key = (model, query_id, item.key)
        with self._lock:
            if key in self._results:
                result = self._results[key]
                if isinstance(result, BaseException):
                    raise result
                return result
            if key not in self._pending:
                thread = threading.Thread(
                    target=self._compute_and_keep, args=(model, query_id, item.key),
                    name=f"metrics-{query_id[:8]}", daemon=True)
                self._pending[key] = thread
                thread.start()
        return None

    def partial(self, model: str, query_id: str, demo_key: str) -> dict | None:
        """Return the engine-reported metrics available during scoring."""
        with self._lock:
            return self._partials.get((model, query_id, demo_key))

    def _compute_and_keep(self, model: str, query_id: str, demo_key: str) -> None:
        key = (model, query_id, demo_key)
        try:
            self._prewarm(model, demo_key)
            client = self._client(model, query_id)
            status = client.status(query_id)
            while status.state not in {
                    "succeeded", "failed", "interrupted", "cancelled"}:
                time.sleep(0.5)
                status = client.status(query_id)
            if status.state != "succeeded":
                raise LookupError(f"query {query_id} is {status.state}")
            result = self.compute(model, query_id, demo_key)
        except Exception as error:  # noqa: BLE001 - handed to the poller
            result = error
        with self._lock:
            self._results[key] = result
            self._pending.pop(key, None)
            if isinstance(result, BaseException):
                self._partials.pop(key, None)

    def _prewarm(self, model: str, demo_key: str) -> None:
        """Tokenize a demo's documents while its GPU query is running."""
        item = demo(demo_key)
        tables = self._tables(item)
        documents, document_lock = self._documents(model, item, tables)
        keys = [
            (spec.name, spec.text_col, str(row_id))
            for spec in item.tables
            for row_id in tables[spec.name].column(spec.id_col).to_pylist()
        ]
        with document_lock:
            documents.fetch(keys)

    def prewarm_all(self) -> None:
        """Tokenize every demo's documents, one demo at a time."""
        for item in DEMOS:
            started = time.perf_counter()
            try:
                self._prewarm(item.model, item.key)
            except Exception as error:  # noqa: BLE001 - a query retries it
                print(f"prewarm {item.key}: {type(error).__name__}: {error}",
                      flush=True)
                continue
            print(f"prewarm {item.key}: {time.perf_counter() - started:.1f} s",
                  flush=True)

    def compute(self, model: str, query_id: str, demo_key: str) -> dict:
        item = demo(demo_key)
        key = (model, query_id, demo_key)
        with self._lock:
            if key in self._results and not isinstance(
                    self._results[key], BaseException):
                return self._results[key]
        client = self._client(model, query_id)
        status = client.status(query_id)
        if status.state != "succeeded":
            raise LookupError(f"query {query_id} is {status.state}")
        report = json.loads(client.file(query_id, status.result["files"]["report"]))
        partial = regret.metrics(
            report, {}, gpus=int(status.config["gpus"]),
            usd_per_hour=self.settings.usd_per_hour)
        partial.update(query_id=query_id, model=model, demo=demo_key,
                       output_rows=status.result["rows"], complete=False)
        with self._lock:
            self._partials[key] = partial
        tables = self._tables(item)
        filters, joins = self.answers(client, status)
        if not filters and not joins and item.view == "score":
            filters = self.scored_rows(client, status, item)
        anchors = {position: anchor for position, (anchor, _) in joins.items()}
        query_sql = (status.spec or {}).get("sql") or item.sql
        description = self._description(
            model, item, tables, anchors, query_sql)
        id_cols = {spec.name: spec.id_col for spec in item.tables}
        output = regret.run_output(
            description, report, filters,
            {position: table for position, (_, table) in joins.items()},
            tables, id_cols)
        documents, document_lock = self._documents(model, item, tables)
        with document_lock:
            numbers = regret.token_numbers(
                description, output, tables, id_cols, documents=documents)
        result = regret.metrics(report, numbers, gpus=int(status.config["gpus"]),
                                usd_per_hour=self.settings.usd_per_hour)
        # one answer table row per question the model was asked
        model_calls = (sum(table.num_rows for table in filters.values())
                       + sum(table.num_rows for _, table in joins.values()))
        result.update(query_id=query_id, model=model, demo=demo_key,
                      output_rows=status.result["rows"], model_calls=model_calls,
                      complete=True)
        with self._lock:
            self._results[key] = result
            self._partials.pop(key, None)
        return result

    def preview(self, model: str, query_id: str, limit: int = 1000) -> dict:
        """Return the first result rows of a finished query as JSON values."""
        client = self._client(model, query_id)
        status = client.status(query_id)
        if status.state != "succeeded":
            raise LookupError(f"query {query_id} is {status.state}")
        reader = client.result_batches(query_id)
        schema = reader.schema
        batches = []
        left = limit
        try:
            for batch in reader:
                if left <= 0:
                    break
                part = batch.slice(0, min(left, batch.num_rows))
                batches.append(part)
                left -= part.num_rows
        finally:
            reader.close()
        table = pa.Table.from_batches(batches, schema=schema)
        result = {
            "columns": table.column_names,
            "rows": table.to_pylist(),
            "total_rows": int(status.result["rows"]),
            "truncated": int(status.result["rows"]) > table.num_rows,
        }
        return json.loads(json.dumps(result, default=str))

    def csv_result(self, model: str, query_id: str) -> bytes:
        """Return every result row of a finished query as CSV."""
        client = self._client(model, query_id)
        status = client.status(query_id)
        if status.state != "succeeded":
            raise LookupError(f"query {query_id} is {status.state}")
        reader = client.result_batches(query_id)
        sink = pa.BufferOutputStream()
        try:
            with pa_csv.CSVWriter(sink, reader.schema) as writer:
                for batch in reader:
                    writer.write_batch(batch)
        finally:
            reader.close()
        return sink.getvalue().to_pybytes()

    def true_pairs(self, model: str, query_id: str) -> dict:
        """Every join answer table's true pairs, as row indices."""
        client = self._client(model, query_id)
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
    index_path = settings.static_dir / "index.html"
    asset_bytes = b"".join(
        (settings.static_dir / name).read_bytes()
        for name in ("app.js", "style.css"))
    asset_version = hashlib.sha256(asset_bytes).hexdigest()[:12]
    index_html = index_path.read_text("utf-8")
    index_html = index_html.replace(
        "/static/style.css", f"/static/style.css?v={asset_version}")
    index_html = index_html.replace(
        "/static/app.js", f"/static/app.js?v={asset_version}")
    threading.Thread(target=_heartbeat, args=(started,), daemon=True,
                     name="page-heartbeat").start()
    if settings.prewarm:
        threading.Thread(target=metrics.prewarm_all, daemon=True,
                         name="metrics-prewarm").start()
    client = httpx.AsyncClient(timeout=httpx.Timeout(
        PROXY_TIMEOUT_S, read=PROXY_TIMEOUT_S + MAX_WAIT_S))
    active_queries = {model: {} for model in MODELS}
    assignment_lock = asyncio.Lock()

    async def release_finished_queries(model: str) -> None:
        """Check tracked queries before choosing a server for the next one."""
        async with assignment_lock:
            pending = tuple(active_queries[model])
        if not pending:
            return

        async def finished(query_id: str) -> bool:
            try:
                endpoint = server_endpoint(settings, model, query_id)
                headers = ({"authorization": f"Bearer {settings.token}"}
                           if settings.token else {})
                response = await client.get(
                    f"{endpoint}/v1/queries/{query_id}",
                    headers=headers, timeout=5.0)
                if response.status_code == 404:
                    return True
                return response.json().get("state") in {
                    "succeeded", "failed", "interrupted", "cancelled"}
            except (httpx.HTTPError, LookupError, ValueError):
                return False

        states = await asyncio.gather(*(finished(query_id) for query_id in pending))
        async with assignment_lock:
            for query_id, done in zip(pending, states):
                if done:
                    active_queries[model].pop(query_id, None)

    def refreshed_manifest() -> dict:
        if settings.reload is not None:
            settings.reload()
        return read_manifest(settings.data_dir)

    async def index(request: Request):
        return HTMLResponse(index_html, headers={"cache-control": "no-store"})

    async def favicon(request: Request):
        return Response(FAVICON_SVG, media_type="image/svg+xml")

    async def static(request: Request):
        name = request.path_params["name"]
        path = settings.static_dir / name
        if "/" in name or not path.is_file():
            return _error(f"no static file {name!r}", 404)
        return FileResponse(
            str(path), headers={
                "cache-control": "public, max-age=31536000, immutable"})

    async def config(request: Request):
        manifest = await run_in_threadpool(refreshed_manifest)
        public_servers = {}
        for model in MODELS:
            urls = settings.servers.get(model)
            public_servers[model] = (
                urls[0] if isinstance(urls, (tuple, list)) else urls)
        return JSONResponse({
            "device": DEVICE,
            "models": list(MODELS),
            "servers": public_servers,
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
        body = await request.body()
        query_id = None
        reserved = False
        if path == "queries" and request.method == "POST":
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                query_id = payload.get("query_id")
                if query_id is None:
                    urls = settings.servers.get(model)
                    if isinstance(urls, str):
                        urls = (urls,)
                    if model not in MODELS or not urls or not any(urls):
                        return _error(f"no server is deployed for {model!r}", 404)
                    await release_finished_queries(model)
                    async with assignment_lock:
                        counts = {
                            slot: sum(value == slot for value in
                                      active_queries[model].values())
                            for slot, url in enumerate(urls) if url
                        }
                        slot = min(counts, key=lambda index: (counts[index], index))
                        query_id = f"q4s{slot}_{uuid.uuid4().hex}"
                        active_queries[model][query_id] = slot
                        reserved = True
                    payload["query_id"] = query_id
                    body = json.dumps(payload).encode()
        elif path.startswith("queries/"):
            query_id = path.split("/", 2)[1]
        try:
            endpoint = server_endpoint(settings, model, query_id)
        except LookupError as error:
            if reserved:
                async with assignment_lock:
                    active_queries[model].pop(query_id, None)
            return _error(str(error), 404)
        url = f"{endpoint}/v1/{path}"
        headers = {name: value for name, value in request.headers.items()
                   if name.lower() not in HOP_HEADERS}
        if settings.token:
            headers["authorization"] = f"Bearer {settings.token}"
        try:
            upstream = await client.request(
                request.method, url, params=request.query_params,
                headers=headers, content=body)
        except httpx.HTTPError as error:
            if reserved:
                async with assignment_lock:
                    active_queries[model].pop(query_id, None)
            return _error(f"cannot reach the {model} server: {error}", 502)
        upstream_body = {}
        if path == "queries" or (path.startswith("queries/") and
                                  path.count("/") == 1):
            try:
                upstream_body = upstream.json()
            except ValueError:
                pass
        if reserved and (upstream.status_code >= 300 or
                         upstream_body.get("id") != query_id):
            async with assignment_lock:
                active_queries[model].pop(query_id, None)
        elif query_id and path.count("/") == 1 and request.method == "GET":
            state = upstream_body.get("state")
            if state in {"succeeded", "failed", "interrupted", "cancelled"}:
                async with assignment_lock:
                    active_queries[model].pop(query_id, None)
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
                metrics.poll, model, query_id, demo_key)
        except (LookupError, KeyError) as error:
            return _error(str(error), 409)
        except Exception as error:
            return _error(f"{type(error).__name__}: {error}", 500)
        if result is None:
            return JSONResponse({"status": "computing", "metrics": metrics.partial(
                model, query_id, demo_key)}, status_code=202)
        return JSONResponse(result)

    async def join_pairs(request: Request):
        model = request.path_params["model"]
        query_id = request.path_params["query_id"]
        try:
            result = await run_in_threadpool(metrics.true_pairs, model, query_id)
        except LookupError as error:
            return _error(str(error), 409)
        return JSONResponse(result)

    async def query_result(request: Request):
        model = request.path_params["model"]
        query_id = request.path_params["query_id"]
        try:
            result = await run_in_threadpool(
                metrics.preview, model, query_id)
        except LookupError as error:
            return _error(str(error), 409)
        except Exception as error:
            return _error(f"{type(error).__name__}: {error}", 500)
        return JSONResponse(result)

    async def download_result(request: Request):
        model = request.path_params["model"]
        query_id = request.path_params["query_id"]
        try:
            result = await run_in_threadpool(
                metrics.csv_result, model, query_id)
        except LookupError as error:
            return _error(str(error), 409)
        except Exception as error:
            return _error(f"{type(error).__name__}: {error}", 500)
        return Response(
            result,
            media_type="text/csv",
            headers={"content-disposition":
                     'attachment; filename="quail-results.csv"'},
        )

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
        Route("/results/{model}/{query_id}", query_result),
        Route("/downloads/{model}/{query_id}", download_result),
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
