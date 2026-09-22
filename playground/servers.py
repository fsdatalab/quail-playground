"""One Quail Server per model, with its model booted before it serves.

The server is quail-server as shipped: ``quail.server.app.create_app``
with one model in its settings. ``warm_up`` runs one tiny query through
a ``ChildProcessExecutor`` so the executor child has the model loaded
and its kernels compiled, and ``PlaygroundServer`` builds the server
around that warm executor, so the first real query on a container does
not boot the model.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pyarrow as pa

from playground.demos import DEVICE, MODELS
from quail.server.app import ServerSettings, create_app
from quail.server.artifacts import write_ipc_file
from quail.server.executor import Executor, Job
from quail.server.inputs import file_digest

WARMUP_TIMEOUT_S = 40 * 60.0

# the query kind each model answers: a reranker scores, the others judge
SCORE_MODELS = frozenset(name for name in MODELS if "reranker" in name)


def warmup_table() -> pa.Table:
    return pa.table({
        "id": ["w1", "w2"],
        "body": ["The ending was excellent and the acting was strong.",
                 "The soundtrack was fine but the plot dragged."],
    })


def warmup_sql(model: str) -> str:
    """A query that touches the model once: AI.SCORE for a reranker."""
    if model in SCORE_MODELS:
        return ("SELECT d.id FROM docs AS d WHERE AI.SCORE(PROMPT("
                "'Does this review praise the movie?\\n\\n{0}', d.body)) >= 0.5")
    return ("SELECT d.id FROM docs AS d WHERE AI.IF(PROMPT("
            "'Does this review praise the movie?\\n\\n{0}', d.body))")


def warmup_job(model: str, workdir: str | Path) -> Job:
    """Build the warm-up job; its input and artifacts live under workdir."""
    workdir = Path(workdir)
    inputs_dir = workdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    staging = inputs_dir / "warmup.arrow"
    write_ipc_file(staging, warmup_table())
    content_id = file_digest(staging)
    path = inputs_dir / f"{content_id}.arrow"
    staging.replace(path)
    job_id = f"warmup-{uuid.uuid4().hex}"
    artifact_dir = workdir / "results" / job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return Job(
        id=job_id,
        spec={"sql": warmup_sql(model), "dialect": "bq", "order": None},
        config={"model": model, "device": DEVICE, "gpus": 1,
                "backend": "quail"},
        inputs={"docs": {"kind": "snapshot", "content_id": content_id,
                         "id_col": "id", "columns": ["id", "body"]}},
        snapshot_paths={content_id: str(path)},
        artifact_dir=str(artifact_dir),
    )


def warm_up(executor: Executor, model: str, workdir: str | Path,
            timeout_s: float = WARMUP_TIMEOUT_S) -> dict:
    """Run the warm-up job and return its finished manifest.

    Raises RuntimeError when the job fails or does not finish in time,
    so a deployment whose model cannot load fails loudly.
    """
    events = []
    execution = executor.start(warmup_job(model, workdir),
                               lambda kind, payload: events.append(
                                   (kind, payload)))
    if not execution.wait(timeout_s):
        execution.stop()
        raise RuntimeError(
            f"warm-up of {model} did not finish within {timeout_s:g} s")
    for kind, payload in events:
        if kind == "failed":
            raise RuntimeError(
                f"warm-up of {model} failed: {payload.get('type')}: "
                f"{payload.get('message')}\n{payload.get('traceback', '')}")
        if kind == "finished":
            return payload
    raise RuntimeError(f"warm-up of {model} ended without a result")


class PlaygroundServer:
    """Quail Server for one model.

    Args:
        model: The one model this server runs.
        data_dir: Where inputs and results are kept (a Volume on Modal).
        db_path: The live SQLite file, on local disk.
        token: The bearer token every /v1 request must carry.
        executor: A warm executor to run queries with; the server's own
            child-process executor when omitted.

    """

    def __init__(self, model: str, data_dir: str | Path, db_path: str | Path,
                 token: str | None, executor: Executor | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown playground model {model!r}")
        self.model = model
        self.settings = ServerSettings(
            data_dir=Path(data_dir), db_path=Path(db_path), models=(model,),
            device=DEVICE, gpus=(1,), token=token)
        self.asgi = create_app(self.settings)
        self.server = self.asgi.state.server
        if executor is not None:
            # the scheduler has not started; the executor it built has no
            # child yet, so swapping it in costs nothing
            self.server.scheduler.executor.close()
            self.server.scheduler.executor = executor

    def add_input(self, content_id: str, source: str | Path) -> bool:
        """Register an Arrow IPC file as an uploaded input snapshot.

        The file is copied into the server's inputs directory under its
        content id, where an upload through ``PUT /v1/inputs`` lands.
        Returns False when the server already has this snapshot.
        """
        store = self.server.store
        if store.get_input(content_id) is not None:
            return False
        source = Path(source)
        if file_digest(source) != content_id:
            raise ValueError(f"{source} does not hash to {content_id}")
        target = self.server.inputs_dir / f"{content_id}.arrow"
        temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        store.put_input(content_id, str(target), target.stat().st_size)
        return True
