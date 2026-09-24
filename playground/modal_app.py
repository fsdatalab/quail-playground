"""Deploy the playground on Modal: four server slots per model and the page.

    modal deploy playground/modal_app.py 2>&1 | tee deploy.log

Run it from the repository root: the images copy ``pyproject.toml`` and
``uv.lock`` from the working directory, and ``uv sync`` inside the
image installs ``quail-engine`` from the GitHub commit the lock pins.

Each model has four server classes, each on one H100 with one container.
The page routes a query and all later requests for it to the same class.
``@modal.enter`` boots the model with a tiny query, so the first real
query on a container does not pay the boot, then restores the server's
database from the Volume and starts quail-server. There is no memory
snapshot: quail's boot is 12 to 50 seconds from the cached weights and
kernels, and restoring a GPU snapshot of the KV arena took as long or
longer. A container stays up ``SCALEDOWN_S`` after its last request.

The page and the servers share one image. The demo data
(``playground.prepare``) is built into it when the image builds, and
each server registers the tables its demos read when it starts, the
same way an upload lands. Pressing Run on the page submits the query.

Data on the ``quail-results`` Volume:
``/results/quail-playground/servers/<model>`` holds slot zero's inputs,
results, and database checkpoint. The other slots use separate paths.

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
# an idle server stays up this long after its last request; a later
# request pays the model boot again
SCALEDOWN_S = 15 * 60
SERVER_SLOTS = 4
SERVER_CLASSES = {
    QWEN3_4B: tuple(f"Qwen3Server{slot}" for slot in range(SERVER_SLOTS)),
    RERANKER: tuple(f"RerankerServer{slot}" for slot in range(SERVER_SLOTS)),
    GEMMA: tuple(f"GemmaServer{slot}" for slot in range(SERVER_SLOTS)),
}

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
    """The lifecycle of one model server slot."""

    def __init__(self, model: str, slot: int):
        self.model = model
        self.slot = slot
        self.data_dir = (SERVERS_DIR / model if slot == 0 else
                         SERVERS_DIR / f"{model}-replica-{slot}")
        self.db_copy = self.data_dir / "quail.sqlite3"
        self.local_dir = (LOCAL_DIR / model if slot == 0 else
                          LOCAL_DIR / f"{model}-replica-{slot}")
        self.local_db = self.local_dir / "quail.sqlite3"
        self.executor = None
        self.server = None
        self.checkpoint = None

    def warm(self) -> None:
        """Load the model in the executor child with one tiny query."""
        from playground.servers import warm_up
        from quail.server.executor import ChildProcessExecutor

        executor = ChildProcessExecutor()
        manifest = warm_up(executor, self.model, self.local_dir / "warmup")
        print(f"warmed {self.model}: {manifest['rows']} rows", flush=True)
        self.executor = executor

    def start(self) -> None:
        """Restore the database and build the server around the warm executor."""
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
    )(modal.concurrent(max_inputs=64)(cls))


def register_server(model: str, slot: int) -> None:
    """Register one independently routed server slot."""
    class_name = SERVER_CLASSES[model][slot]

    class ModelServer:
        @modal.enter()
        def start(self):
            self.container = ServerContainer(model, slot)
            self.container.warm()
            self.container.start()

        @modal.asgi_app()
        def web(self):
            return self.container.server.asgi

        @modal.method()
        def ping(self) -> str:
            return model

        @modal.exit()
        def stop(self):
            self.container.stop()

    ModelServer.__name__ = class_name
    ModelServer.__qualname__ = class_name
    globals()[class_name] = server_class(ModelServer)


for _model in MODELS:
    for _slot in range(SERVER_SLOTS):
        register_server(_model, _slot)


def deployed_server(model: str, slot: int = 0):
    """The deployed class of a model's server slot, looked up by name."""
    return modal.Cls.from_name(APP_NAME, SERVER_CLASSES[model][slot])()


def deployed_server_urls() -> dict:
    """Model name -> the four server URLs, with None for missing slots."""
    urls = {}
    for model in MODELS:
        slots = []
        for slot in range(SERVER_SLOTS):
            try:
                slots.append(deployed_server(model, slot).web.get_web_url())
            except Exception as error:  # noqa: BLE001 - not deployed yet
                print(f"{model} slot {slot}: no deployed server ({error})",
                      flush=True)
                slots.append(None)
        urls[model] = tuple(slots)
    return urls


@app.function(
    image=image,
    # the tokenizers behind the metrics come from the same cache as the
    # servers' weights
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[secret],
    cpu=8.0,
    memory=8_192,
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
        usd_per_hour=H100_USD_PER_HOUR,
        prewarm=True))


@app.local_entrypoint()
def warm():
    """Start every server once, so each boots its model before a demo.

    Run with ``modal run playground/modal_app.py::warm``.
    """
    from playground.warm import warm_servers

    first_slots = {model: urls[0] for model, urls in
                   deployed_server_urls().items()}
    for line in warm_servers(first_slots):
        print(line, flush=True)
