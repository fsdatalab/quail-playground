"""Deploy the playground on Modal: three Quail Servers and the page.

    modal deploy -m playground.modal_app 2>&1 | tee deploy.log
    modal run -m playground.modal_app::prepare 2>&1 | tee prepare.log
    modal run -m playground.modal_app::warm 2>&1 | tee warm.log

Run these from the repository root: the images copy ``pyproject.toml``
and ``uv.lock`` from the working directory.

One class per model, each on one H100. ``@modal.enter(snap=True)`` runs
a tiny query so the executor child has the model loaded and its kernels
compiled, then Modal takes a memory snapshot that includes the GPU. A
restored container serves its first real query without booting the
model. ``@modal.enter(snap=False)`` restores the server's database from
the Volume and starts the checkpoint thread, the parts that must not be
in a snapshot.

Data on the ``quail-results`` Volume:

- ``/results/quail-playground/data``: the prepared input tables and
  page JSON (``playground.prepare``).
- ``/results/quail-playground/servers/<model>``: that server's inputs,
  results, and database checkpoint.

The prepared tables reach each server through quail-server's own
upload route (``PUT /v1/inputs/<content id>``), from ``upload_inputs``.

The bearer token comes from the ``quail-server-token`` secret, the same
one ``quail.server.modal_app`` uses. Add ``HF_TOKEN`` to it for
DiffusionGemma's gated weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from playground.demos import GEMMA, GROUPS, MODELS, QWEN3_4B, RERANKER
from quail.bench.images import CACHE_ENV, CUDA_BASE, UV_VERSION

APP_NAME = "quail-playground"
VOLUME_DIR = Path("/results")
ROOT = VOLUME_DIR / "quail-playground"
DATA_DIR = ROOT / "data"
SERVERS_DIR = ROOT / "servers"
LOCAL_DIR = Path("/tmp/quail-playground")
STATIC_DIR = Path("/root/web")
# an idle server stays up this long after its last request; a restore
# from the snapshot is what a later request pays
SCALEDOWN_S = 15 * 60
SERVER_CLASSES = {QWEN3_4B: "Qwen3Server", RERANKER: "RerankerServer",
                  GEMMA: "GemmaServer"}

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
secret = modal.Secret.from_name("quail-server-token",
                                required_keys=["QUAIL_SERVER_TOKEN"])

gpu_image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .env(CACHE_ENV)
    .uv_sync(uv_version=UV_VERSION)
    .add_local_python_source("playground")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_sync(uv_version=UV_VERSION, extra_options="--no-install-package vllm")
    .add_local_python_source("playground")
    .add_local_dir("web", remote_path=str(STATIC_DIR))
)


class ServerContainer:
    """The lifecycle of one server container, shared by the three classes."""

    def __init__(self, model: str):
        self.model = model
        self.data_dir = SERVERS_DIR / model
        self.db_copy = self.data_dir / "quail.sqlite3"
        self.local_db = LOCAL_DIR / model / "quail.sqlite3"
        self.executor = None
        self.server = None
        self.checkpoint = None

    def warm(self) -> None:
        """Load the model in the executor child; runs before the snapshot."""
        from playground.servers import warm_up
        from quail.server.executor import ChildProcessExecutor

        executor = ChildProcessExecutor()
        manifest = warm_up(executor, self.model, LOCAL_DIR / self.model / "warmup")
        print(f"warmed {self.model}: {manifest['rows']} rows", flush=True)
        self.executor = executor

    def start(self) -> None:
        """Build the server around the warm executor; runs after a restore."""
        from playground.servers import PlaygroundServer
        from quail.server.checkpoint import Checkpoint, restore

        results_volume.reload()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if restore(self.db_copy, self.local_db):
            print(f"restored {self.db_copy} to {self.local_db}", flush=True)
        self.server = PlaygroundServer(
            self.model, self.data_dir, self.local_db,
            os.environ["QUAIL_SERVER_TOKEN"], executor=self.executor)
        inner = self.server.server
        self.checkpoint = Checkpoint(inner.store, self.db_copy,
                                     commit=results_volume.commit)
        self.checkpoint.start()
        inner.add_sync(self.checkpoint.sync)
        inner.add_closer(self.checkpoint.stop)
        if inner.recovered:
            print(f"marked interrupted: {inner.recovered}", flush=True)

    def stop(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint.stop()


def server_class(cls):
    """Apply the shared container options of a server class."""
    return app.cls(
        image=gpu_image,
        gpu="H100!",
        memory=98_304,
        volumes={
            str(VOLUME_DIR): results_volume,
            "/root/.cache/huggingface": hf_cache,
            "/root/.cache/kernels": kernel_cache,
        },
        secrets=[secret],
        timeout=24 * 3600,
        scaledown_window=SCALEDOWN_S,
        max_containers=1,
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
    )(modal.concurrent(max_inputs=64)(cls))


@server_class
class Qwen3Server:
    """Quail Server for qwen3-4b-fp8: the IMDB ending query and BIO-4."""

    @modal.enter(snap=True)
    def warm(self):
        self.container = ServerContainer(QWEN3_4B)
        self.container.warm()

    @modal.enter(snap=False)
    def start(self):
        self.container.start()

    @modal.asgi_app()
    def web(self):
        return self.container.server.asgi

    @modal.method()
    def ping(self) -> str:
        return QWEN3_4B

    @modal.exit()
    def stop(self):
        self.container.stop()


@server_class
class RerankerServer:
    """Quail Server for qwen3-reranker-0.6b-bf16: the IMDB sentiment query."""

    @modal.enter(snap=True)
    def warm(self):
        self.container = ServerContainer(RERANKER)
        self.container.warm()

    @modal.enter(snap=False)
    def start(self):
        self.container.start()

    @modal.asgi_app()
    def web(self):
        return self.container.server.asgi

    @modal.method()
    def ping(self) -> str:
        return RERANKER

    @modal.exit()
    def stop(self):
        self.container.stop()


@server_class
class GemmaServer:
    """Quail Server for diffusion-gemma-26b-a4b-fp8: agent trace compaction."""

    @modal.enter(snap=True)
    def warm(self):
        self.container = ServerContainer(GEMMA)
        self.container.warm()

    @modal.enter(snap=False)
    def start(self):
        self.container.start()

    @modal.asgi_app()
    def web(self):
        return self.container.server.asgi

    @modal.method()
    def ping(self) -> str:
        return GEMMA

    @modal.exit()
    def stop(self):
        self.container.stop()


def deployed_server(model: str):
    """The deployed class of a model's server, looked up by name."""
    return modal.Cls.from_name(APP_NAME, SERVER_CLASSES[model])()


def deployed_server_urls() -> dict:
    """Model name -> the web URL of its deployed server, or None."""
    urls = {}
    for model in MODELS:
        try:
            urls[model] = deployed_server(model).web.get_web_url()
        except Exception as error:  # noqa: BLE001 - not deployed yet
            print(f"{model}: no deployed server ({error})", flush=True)
            urls[model] = None
    return urls


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_DIR): results_volume},
    secrets=[secret],
    timeout=600,
    scaledown_window=SCALEDOWN_S,
    memory=8_192,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def page():
    """The playground page and its proxy to the three servers."""
    from playground.web import WebSettings, create_web_app
    from quail.specs import H100_USD_PER_HOUR

    return create_web_app(WebSettings(
        static_dir=STATIC_DIR, data_dir=DATA_DIR,
        servers=deployed_server_urls(),
        token=os.environ.get("QUAIL_SERVER_TOKEN"),
        usd_per_hour=H100_USD_PER_HOUR,
        reload=results_volume.reload))


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_DIR): results_volume,
             "/root/.cache/huggingface": hf_cache},
    secrets=[secret],
    timeout=4 * 3600,
    memory=16_384,
)
def build_inputs(groups: list, compaction_limit: int,
                 compaction_seed: int) -> dict:
    """Build the data groups on the Volume; returns the manifest entries."""
    from playground.prepare import build

    results_volume.reload()
    try:
        return build(DATA_DIR, groups, compaction_limit=compaction_limit,
                     compaction_seed=compaction_seed)
    finally:
        results_volume.commit()
        hf_cache.commit()


@app.function(image=cpu_image, volumes={str(VOLUME_DIR): results_volume},
              timeout=300)
def saved_manifest() -> dict:
    """The manifest on the Volume, for registering inputs again."""
    from playground.prepare import read_manifest

    results_volume.reload()
    return read_manifest(DATA_DIR)["groups"]


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_DIR): results_volume},
    secrets=[secret],
    timeout=3600,
)
def upload_inputs(entries: dict, servers: dict) -> dict:
    """Upload each group's Arrow files to the servers whose demos read them.

    Uses the Quail Server client, so an upload the server already has
    is skipped by its content id.

    Args:
        entries: The manifest groups.
        servers: Model name -> the server's web URL.

    """
    from playground.prepare import uploads_for
    from quail.server.client import ServerClient

    results_volume.reload()
    token = os.environ["QUAIL_SERVER_TOKEN"]
    uploaded = {}
    for model, prepared in uploads_for(entries, DATA_DIR).items():
        endpoint = servers.get(model)
        if not endpoint:
            print(f"{model}: no deployed server, skipped", flush=True)
            continue
        client = ServerClient(endpoint, token=token)
        for item in prepared:
            print(f"{model}: uploading {item.content_id[:12]} "
                  f"({item.upload_path.name})", flush=True)
            client.upload_input(item)
        uploaded[model] = [item.content_id for item in prepared]
    return uploaded


def register_inputs(entries: dict) -> None:
    """Hand each deployed server the tables its demos read."""
    urls = deployed_server_urls()
    call = upload_inputs.spawn(entries, urls)
    print(f"function call id (upload_inputs): {call.object_id}", flush=True)
    print(call.get(), flush=True)


@app.local_entrypoint()
def prepare(groups: str = ",".join(GROUPS), register: bool = True,
            compaction_limit: int = 100, compaction_seed: int = 42):
    """Build the demo data, then register it with the deployed servers."""
    wanted = [group.strip() for group in groups.split(",") if group.strip()]
    call = build_inputs.spawn(wanted, compaction_limit, compaction_seed)
    print(f"function call id (build_inputs): {call.object_id}", flush=True)
    entries = call.get()
    print(f"data: {DATA_DIR} on the quail-results Volume", flush=True)
    if register:
        register_inputs(entries)


@app.local_entrypoint()
def register():
    """Register the data already on the Volume with the deployed servers."""
    register_inputs(saved_manifest.remote())


@app.local_entrypoint()
def warm():
    """Start every server once, so each takes its snapshot now."""
    for model in MODELS:
        print(f"{model}: starting", flush=True)
        print(f"{model}: {deployed_server(model).ping.remote()} ready", flush=True)
