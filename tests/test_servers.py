"""Warm-up before the snapshot, and the server built around the warm executor."""

import threading

import pytest

from playground import servers
from playground.demos import GEMMA, QWEN3_4B, RERANKER
from quail.server.executor import Job


class FakeExecution:
    def __init__(self, events, emit, fail=False, hang=False):
        self.stopped = False
        self._done = threading.Event()
        if not hang:
            for kind, payload in events:
                emit(kind, payload)
            self._done.set()

    def wait(self, timeout):
        return self._done.wait(timeout)

    def stop(self):
        self.stopped = True


class FakeExecutor:
    def __init__(self, events, hang=False):
        self.events = events
        self.hang = hang
        self.jobs = []
        self.execution = None

    def start(self, job, emit):
        self.jobs.append(job)
        self.execution = FakeExecution(self.events, emit, hang=self.hang)
        return self.execution

    def close(self):
        pass


def test_warmup_job_is_a_snapshot_input_and_the_model_kind_of_query(tmp_path):
    job = servers.warmup_job(QWEN3_4B, tmp_path)
    assert isinstance(job, Job)
    assert job.config == {"model": QWEN3_4B, "device": "h100-sxm", "gpus": 1,
                          "backend": "quail"}
    (content_id,) = job.snapshot_paths
    assert job.inputs["docs"]["content_id"] == content_id
    assert job.snapshot_paths[content_id].endswith(f"{content_id}.arrow")
    assert "AI.IF" in job.spec["sql"]
    assert "AI.SCORE" in servers.warmup_sql(RERANKER)
    assert "AI.IF" in servers.warmup_sql(GEMMA)


def test_warm_up_returns_the_manifest_and_raises_on_failure(tmp_path):
    manifest = {"rows": 1, "columns": ["id"], "files": {}}
    executor = FakeExecutor([("plan", {}), ("model_ready", {"model": QWEN3_4B}),
                             ("finished", manifest)])
    assert servers.warm_up(executor, QWEN3_4B, tmp_path) == manifest
    assert executor.jobs[0].config["model"] == QWEN3_4B

    failing = FakeExecutor([("failed", {"type": "RuntimeError",
                                        "message": "no weights"})])
    with pytest.raises(RuntimeError, match="warm-up of qwen3-4b-fp8 failed: "
                                           "RuntimeError: no weights"):
        servers.warm_up(failing, QWEN3_4B, tmp_path)

    hanging = FakeExecutor([], hang=True)
    with pytest.raises(RuntimeError, match="did not finish within 0.05 s"):
        servers.warm_up(hanging, QWEN3_4B, tmp_path, timeout_s=0.05)
    assert hanging.execution.stopped


def test_server_is_quail_server_with_one_model_and_the_warm_executor(tmp_path):
    executor = FakeExecutor([])
    server = servers.PlaygroundServer(
        RERANKER, tmp_path / "data", tmp_path / "live.sqlite3", "secret",
        executor=executor)
    assert server.settings.models == (RERANKER,)
    assert server.settings.device == "h100-sxm"
    assert server.settings.token == "secret"
    assert server.server.scheduler.executor is executor
    assert server.server.capabilities()["models"] == [RERANKER]
    assert (tmp_path / "live.sqlite3").exists()
    with pytest.raises(ValueError, match="unknown playground model"):
        servers.PlaygroundServer("gpt-9", tmp_path, tmp_path / "x.sqlite3", None)
    server.server.stop()


def test_add_input_registers_a_file_the_way_an_upload_lands(tmp_path):
    from quail.server.artifacts import write_ipc_file
    from quail.server.inputs import file_digest

    server = servers.PlaygroundServer(
        QWEN3_4B, tmp_path / "data", tmp_path / "live.sqlite3", None,
        executor=FakeExecutor([]))
    source = tmp_path / "reviews.arrow"
    write_ipc_file(source, servers.warmup_table())
    content_id = file_digest(source)
    assert server.add_input(content_id, source)
    record = server.server.store.get_input(content_id)
    assert record.path == str(tmp_path / "data" / "inputs" / f"{content_id}.arrow")
    assert record.byte_count == source.stat().st_size
    assert not server.add_input(content_id, source), "already registered"
    with pytest.raises(ValueError, match="does not hash"):
        server.add_input("0" * 64, source)
    server.server.stop()


def test_warm_servers_reports_each_model(monkeypatch):
    from playground import warm

    monkeypatch.setattr(warm, "ping", lambda endpoint, timeout_s=0: 401)
    lines = warm.warm_servers({QWEN3_4B: "http://a", RERANKER: None})
    assert lines[0].startswith("qwen3-4b-fp8: HTTP 401 after")
    assert lines[1] == "qwen3-reranker-0.6b-bf16: no server deployed"
    assert lines[2] == "diffusion-gemma-26b-a4b-fp8: no server deployed"

