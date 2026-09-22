"""Deploy the playground on Modal: three Quail Servers and the page.

    modal deploy playground/modal_app.py 2>&1 | tee deploy.log

Run it from the repository root: the images copy ``pyproject.toml`` and
``uv.lock`` from the working directory, and ``uv sync`` inside the
image installs ``quail-engine`` from the GitHub commit the lock pins.

One class per model, each on one H100, at most one container each.
``@modal.enter(snap=True)`` runs a tiny query so the executor child has
the model loaded and its kernels compiled, then Modal takes a memory
snapshot that includes the GPU. A restored container serves its first
real query without booting the model. ``@modal.enter(snap=False)``
restores the server's database from the Volume and starts the
checkpoint thread, the parts that must not be in a snapshot.

The page and the servers share one image. The demo data
(``playground.prepare``) is built into it when the image builds, and
each server registers the tables its demos read when it starts, the
same way an upload lands. Pressing Run on the page submits the query.

Data on the ``quail-results`` Volume:
``/results/quail-playground/servers/<model>`` holds that server's
inputs, results, and database checkpoint.

The bearer token comes from the ``quail-service-token`` secret of the
workspace, as ``QUAIL_SERVER_TOKEN`` or ``QUAIL_SERVICE_TOKEN``. Add
``HF_TOKEN`` to it for DiffusionGemma's gated weights when the
``quail-hf-cache`` Volume does not hold them yet.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from playground.demos import GEMMA, MODELS, QWEN3_4B, RERANKER
from quail.bench.images import CACHE_ENV, CUDA_BASE, UV_VERSION

APP_NAME = "quail-playground"
VOLUME_DIR = Path("/results")
SERVERS_DIR = VOLUME_DIR / "quail-playground" / "servers"
LOCAL_DIR = Path("/tmp/quail-playground")
STATIC_DIR = Path("/root/web")
DEMO_DATA_DIR = Path("/root/demo-data")
HF_CACHE_DIR = "/root/.cache/huggingface"
KERNEL_CACHE_DIR = "/root/.cache/kernels"
# an idle server stays up this long after its last request; a restore
# from the snapshot is what a later request pays
SCALEDOWN_S = 15 * 60
SERVER_CLASSES = {QWEN3_4B: "Qwen3Server", RERANKER: "RerankerServer",
                  GEMMA: "GemmaServer"}

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
SECRET_NAME = "quail-service-token"
secret = modal.Secret.from_name(SECRET_NAME)


def server_token() -> str:
    """The bearer token of the servers, from either variable of the secret."""
    token = os.environ.get("QUAIL_SERVER_TOKEN") or os.environ.get(
        "QUAIL_SERVICE_TOKEN")
    if not token:
        raise RuntimeError(f"the {SECRET_NAME} secret needs QUAIL_SERVER_TOKEN")
    return token


def build_demo_data() -> None:
    """Build every demo's tables and page data; runs while the image builds."""
    import shutil

    from playground.prepare import build

    build(DEMO_DATA_DIR, workdir=LOCAL_DIR / "build")
    # a Volume mounts only on an empty path; importing quail here fills
    # the kernel cache directory the servers mount their Volume on
    shutil.rmtree(KERNEL_CACHE_DIR, ignore_errors=True)


# bump to force a fresh image build; Modal reuses a build whose
# definition is unchanged, including one still in progress
IMAGE_VERSION = "2"

image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .env({**CACHE_ENV, "PLAYGROUND_IMAGE_VERSION": IMAGE_VERSION})
    .uv_sync(uv_version=UV_VERSION)
    .add_local_python_source("playground", copy=True)
    .run_function(build_demo_data, secrets=[secret], timeout=4 * 3600,
                  memory=16_384,
                  volumes={HF_CACHE_DIR: hf_cache})
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
            server_token(), executor=self.executor)
        inner = self.server.server
        self.checkpoint = Checkpoint(inner.store, self.db_copy,
                                     commit=results_volume.commit)
        self.checkpoint.start()
        inner.add_sync(self.checkpoint.sync)
        inner.add_closer(self.checkpoint.stop)
        if inner.recovered:
            print(f"marked interrupted: {inner.recovered}", flush=True)
        self.register_inputs()

    def register_inputs(self) -> None:
        """Register the demo tables this model's queries read, from the image."""
        from playground.demos import DEMOS
        from playground.prepare import read_manifest, uploads_for

        groups = read_manifest(DEMO_DATA_DIR)["groups"]
        added = 0
        for item in DEMOS:
            if item.model != self.model:
                continue
            for prepared in uploads_for(item, groups, DEMO_DATA_DIR):
                added += self.server.add_input(prepared.content_id,
                                               prepared.upload_path)
        if added:
            self.checkpoint.sync()
        print(f"{self.model}: {added} demo tables registered", flush=True)

    def stop(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint.stop()


def server_class(cls):
    """Apply the shared container options of a server class."""
    return app.cls(
        image=image,
        gpu="H100!",
        memory=98_304,
        volumes={
            str(VOLUME_DIR): results_volume,
            HF_CACHE_DIR: hf_cache,
            KERNEL_CACHE_DIR: kernel_cache,
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
    image=image,
    # the tokenizers behind the metrics come from the same cache as the
    # servers' weights
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[secret],
    timeout=600,
    scaledown_window=SCALEDOWN_S,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def page():
    """The playground page and its proxy to the three servers."""
    from playground.web import WebSettings, create_web_app
    from quail.specs import H100_USD_PER_HOUR

    return create_web_app(WebSettings(
        static_dir=STATIC_DIR, data_dir=DEMO_DATA_DIR,
        servers=deployed_server_urls(),
        token=server_token(),
        usd_per_hour=H100_USD_PER_HOUR))


@app.local_entrypoint()
def warm():
    """Start every server once, so each takes its snapshot before a demo.

    modal run playground/modal_app.py::warm
    """
    for model in MODELS:
        print(f"{model}: starting", flush=True)
        print(f"{model}: {deployed_server(model).ping.remote()} ready", flush=True)
