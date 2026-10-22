"""CPU process-control seams for dynamic-router vLLM engine isolation."""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import sys
import time
from collections.abc import Sequence

import pytest

from labs.dynamic_router import vllm_runner


class _FakeWorkerWrapper:
    """Pickle-safe engine double instantiated only inside a spawn worker."""

    def __init__(
        self,
        gpu_id: str,
        device_index: int,
        model_id: str,
        *,
        attention_backend: str | None,
    ) -> None:
        self.gpu_id = gpu_id
        self.device_index = device_index
        self.model_id = model_id
        self.attention_backend = attention_backend
        self._inflight: dict[str, vllm_runner._RequestRuntime] = {}
        self._prompts: dict[str, tuple[int, ...]] = {}
        self._completed_output_token_ids: dict[str, tuple[int, ...]] = {}
        self._closed = False

    def add_request(
        self,
        runtime: vllm_runner._RequestRuntime,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> None:
        prompt = tuple(prompt_token_ids or [1] * runtime.req.prompt_tokens)
        self._inflight[runtime.req.req_id] = runtime
        self._prompts[runtime.req.req_id] = prompt

    def step(self):
        finished_ids = list(self._inflight)
        ttft = [(request_id, 1.0) for request_id in finished_ids]
        tokens = 0
        for request_id in finished_ids:
            runtime = self._inflight.pop(request_id)
            prompt = self._prompts.pop(request_id)
            output = tuple(sum(prompt) + index for index in range(runtime.req.expected_new_tokens))
            self._completed_output_token_ids[request_id] = output
            tokens += len(output)
        return finished_ids, ttft, tokens

    def queue_depth(self) -> int:
        return len(self._inflight)

    def reset_request_state(self) -> None:
        if self._inflight:
            raise RuntimeError("fake worker still has unfinished requests")
        self._completed_output_token_ids.clear()

    def snapshot_metrics(
        self,
        ttft_ema: float | None,
        tpot_ema: float,
    ) -> dict[str, float]:
        return {
            "pid": float(os.getpid()),
            "device_index": float(self.device_index),
            "rank_env_count": float(
                sum(key in os.environ for key in vllm_runner._WORKER_RANK_ENV_KEYS)
            ),
            "ttft": float(ttft_ema or 0.0),
            "tpot": tpot_ema,
        }

    def close(self, *, force: bool = False) -> None:
        if self._inflight and not force:
            raise RuntimeError("fake worker still has unfinished requests")
        self._inflight.clear()
        self._closed = True


class _FailingStepWorkerWrapper(_FakeWorkerWrapper):
    def step(self):
        raise ValueError("synthetic worker step failure")


class _FailingStartupWorkerWrapper(_FakeWorkerWrapper):
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise ValueError("synthetic worker startup failure")


class _NonrespondingStepWorkerWrapper(_FakeWorkerWrapper):
    def step(self):
        time.sleep(60)
        raise AssertionError("nonresponding worker should have been terminated")


class _NonrespondingStartupWorkerWrapper(_FakeWorkerWrapper):
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        time.sleep(60)
        raise AssertionError("nonresponding worker should have been terminated")


class _NonrespondingCloseWorkerWrapper(_FakeWorkerWrapper):
    def close(self, *, force: bool = False) -> None:
        del force
        time.sleep(60)
        raise AssertionError("nonresponding worker should have been terminated")


class _NonrespondingMetricsWorkerWrapper(_FakeWorkerWrapper):
    def snapshot_metrics(
        self,
        ttft_ema: float | None,
        tpot_ema: float,
    ) -> dict[str, float]:
        del ttft_ema, tpot_ema
        time.sleep(60)
        raise AssertionError("nonresponding worker should have been terminated")


def _send_fake_startup(connection: socket.socket, gpu_id: str, visible: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    for key in vllm_runner._WORKER_RANK_ENV_KEYS:
        os.environ.pop(key, None)
    vllm_runner._worker_socket_send(
        connection,
        {
            "id": 0,
            "ok": True,
            "result": {
                "pid": os.getpid(),
                "cuda_initialized_before_binding": False,
                "vllm_imported_before_binding": False,
                "python_argv": tuple(sys.argv),
                "os_command_line": (),
                "cuda_visible_devices": visible,
                "vllm_enable_v1_multiprocessing": "0",
                "present_rank_env": (),
                "gpu_id": gpu_id,
            },
        },
    )


def _live_nonreading_control_worker(
    connection: socket.socket,
    wrapper_cls,
    gpu_id: str,
    cuda_visible_device: str,
    model_id: str,
    attention_backend: str | None,
) -> None:
    del wrapper_cls, model_id, attention_backend
    _send_fake_startup(connection, gpu_id, cuda_visible_device)
    time.sleep(60)


def _live_partial_response_frame_worker(
    connection: socket.socket,
    wrapper_cls,
    gpu_id: str,
    cuda_visible_device: str,
    model_id: str,
    attention_backend: str | None,
) -> None:
    del wrapper_cls, model_id, attention_backend
    _send_fake_startup(connection, gpu_id, cuda_visible_device)
    vllm_runner._worker_socket_receive(connection)
    connection.sendall(vllm_runner._WORKER_RPC_HEADER.pack(4096) + b"partial")
    time.sleep(60)


def _runtime(request_id: str, gpu_id: str, prompt_tokens: int = 3):
    return vllm_runner._RequestRuntime(
        req=vllm_runner.Request(
            req_id=request_id,
            prompt_tokens=prompt_tokens,
            expected_new_tokens=2,
        ),
        gpu_id=gpu_id,
        admitted_at=10.0,
        admitted_monotonic=10.0,
    )


def _active_child_pids() -> set[int]:
    return {child.pid for child in multiprocessing.active_children() if child.pid is not None}


def test_spawn_workers_bind_one_visible_gpu_and_batch_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setenv("LOCAL_RANK", "9")
    monkeypatch.setenv("RANK", "9")
    workers = [
        vllm_runner._SpawnedVllmWrapper(
            f"gpu{index}",
            index,
            "/models/fake",
            attention_backend=None,
            wrapper_cls=_FakeWorkerWrapper,
        )
        for index in range(2)
    ]
    worker_pids = {worker.worker_pid for worker in workers}
    try:
        assert len(worker_pids) == 2
        assert os.getpid() not in worker_pids
        assert [worker.worker_cuda_visible_devices for worker in workers] == ["4", "7"]
        assert [worker.worker_v1_multiprocessing for worker in workers] == ["0", "0"]
        assert [worker.worker_rank_environment for worker in workers] == [(), ()]
        assert not any(worker.worker_cuda_initialized_before_binding for worker in workers)
        assert not any(worker.worker_vllm_imported_before_binding for worker in workers)
        assert all(worker.worker_pid == worker._process.pid for worker in workers)
        assert all(worker.worker_python_argv == tuple(sys.argv) for worker in workers)
        if sys.platform.startswith("linux"):
            assert all(
                any(
                    "multiprocessing.spawn" in argument
                    for argument in worker.worker_os_command_line
                )
                for worker in workers
            )
            assert all(
                "--multiprocessing-fork" in worker.worker_os_command_line for worker in workers
            )

        first = workers[0]
        first.add_request(_runtime("request-a", "gpu0"), [2, 3, 4])
        first.add_request(_runtime("request-b", "gpu0"), [5, 6, 7])
        assert first.queue_depth() == 2
        assert first._request_id == 0

        first.flush_requests()
        assert first._request_id == 1
        assert first.queue_depth() == 2
        finished, ttft, token_count = first.step()
        assert finished == ["request-a", "request-b"]
        assert ttft == [("request-a", 1.0), ("request-b", 1.0)]
        assert token_count == 4
        assert first.queue_depth() == 0
        assert first._completed_output_token_ids == {
            "request-a": (9, 10),
            "request-b": (18, 19),
        }
        metrics = first.snapshot_metrics(ttft_ema=3.0, tpot_ema=4.0)
        assert metrics == {
            "pid": float(first.worker_pid),
            "device_index": 0.0,
            "rank_env_count": 0.0,
            "ttft": 3.0,
            "tpot": 4.0,
        }
        first.reset_request_state()
        assert first._completed_output_token_ids == {}
    finally:
        for worker in workers:
            worker.close(force=True)

    assert worker_pids.isdisjoint(_active_child_pids())
    for index, worker in enumerate(workers):
        assert worker.lifecycle_metadata() == {
            "gpu_id": f"gpu{index}",
            "worker_pid": worker.worker_pid,
            "parent_logical_device_index": index,
            "cuda_visible_devices": ("4", "7")[index],
            "child_device": "cuda:0",
            "cuda_initialized_before_binding": False,
            "vllm_imported_before_binding": False,
            "rank_environment": [],
            "vllm_enable_v1_multiprocessing": "0",
            "multiprocessing_start_method": "spawn",
            "fallback_enabled": False,
            "python_argv": list(worker.worker_python_argv),
            "os_command_line": list(worker.worker_os_command_line),
            "exitcode": 0,
            "termination": "clean",
            "residual_alive": False,
            "startup_completed": True,
            "failure_operation": None,
            "failure_reason": None,
            "rpc_deadlines_seconds": {
                "startup": 900.0,
                "operation": 300.0,
                "join": 30.0,
            },
        }


def test_worker_error_is_reported_and_owned_process_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    worker = vllm_runner._SpawnedVllmWrapper(
        "gpu0",
        0,
        "/models/fake",
        attention_backend=None,
        wrapper_cls=_FailingStepWorkerWrapper,
    )
    worker_pid = worker.worker_pid
    worker.add_request(_runtime("request-a", "gpu0"), [1, 2, 3])
    worker.flush_requests()

    with pytest.raises(
        RuntimeError,
        match="ValueError: synthetic worker step failure",
    ):
        worker.step()
    with pytest.raises(RuntimeError, match="vLLM worker gpu0"):
        worker.close(force=True)

    assert worker_pid not in _active_child_pids()
    assert worker.worker_exitcode == 1
    assert worker.worker_termination == "clean"
    assert worker.worker_residual_alive is False


def test_startup_failure_reaps_spawned_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    children_before = _active_child_pids()

    with pytest.raises(
        RuntimeError,
        match="ValueError: synthetic worker startup failure",
    ) as caught:
        vllm_runner._SpawnedVllmWrapper(
            "gpu0",
            0,
            "/models/fake",
            attention_backend=None,
            wrapper_cls=_FailingStartupWorkerWrapper,
        )

    metadata = caught.value.vllm_worker_lifecycle_metadata
    assert metadata["worker_pid"] > 0
    assert metadata["startup_completed"] is False
    assert metadata["failure_operation"] == "startup"
    assert "synthetic worker startup failure" in metadata["failure_reason"]
    assert metadata["exitcode"] == 1
    assert metadata["termination"] == "clean"
    assert metadata["residual_alive"] is False
    assert _active_child_pids() == children_before


def test_live_nonresponding_step_times_out_and_retains_failed_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv(vllm_runner._WORKER_OPERATION_TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(vllm_runner._WORKER_JOIN_TIMEOUT_ENV, "0.5")
    monkeypatch.setattr(vllm_runner, "_requires_spawned_worker", lambda _wrapper: True)
    children_before = _active_child_pids()
    session = vllm_runner.VllmEngineSession(
        workload_kind="dynamic_router",
        mode="optimized",
        handles=(vllm_runner._GPUHandle("gpu0", 0, True, True, 0),),
        model_id="/models/fake",
        attention_backend=None,
        wrapper_cls=_NonrespondingStepWorkerWrapper,
        warmup_runs=0,
    )
    phase, _ = session.begin_run()
    worker = session.engines["gpu0"]
    worker.add_request(_runtime("request-a", "gpu0"), [1, 2, 3])
    worker.flush_requests()

    started = time.monotonic()
    with pytest.raises(
        vllm_runner._VllmWorkerRPCTimeout,
        match=("timed out during step while receiving response header after 0.2 seconds"),
    ) as caught:
        worker.step()
    assert time.monotonic() - started < 10.0
    session.abort_run(caught.value)
    session.close(preserve_primary_error=True)

    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["disposition"] == "failed_run"
    assert lifecycle["primary_failure"]["type"] == "_VllmWorkerRPCTimeout"
    assert lifecycle["engine_worker_residual_count"] == 0
    assert len(lifecycle["engine_workers"]) == 1
    worker_record = lifecycle["engine_workers"][0]
    assert worker_record["startup_completed"] is True
    assert worker_record["failure_operation"] == "step"
    assert "after 0.2 seconds" in worker_record["failure_reason"]
    assert worker_record["termination"] == "terminated"
    assert worker_record["exitcode"] < 0
    assert worker_record["residual_alive"] is False
    assert worker_record["rpc_deadlines_seconds"] == {
        "startup": 900.0,
        "operation": 0.2,
        "join": 0.5,
    }
    assert _active_child_pids() == children_before


def test_live_nonresponding_startup_is_reaped_and_retained_in_receipt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8")
    monkeypatch.setenv(vllm_runner._WORKER_STARTUP_TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(vllm_runner._WORKER_JOIN_TIMEOUT_ENV, "0.5")
    monkeypatch.setattr(vllm_runner, "_requires_spawned_worker", lambda _wrapper: True)
    children_before = _active_child_pids()

    started = time.monotonic()
    with pytest.raises(
        vllm_runner._VllmWorkerRPCTimeout,
        match=("timed out during startup while receiving response header after 0.2 seconds"),
    ):
        vllm_runner.VllmEngineSession(
            workload_kind="dynamic_router",
            mode="optimized",
            handles=(vllm_runner._GPUHandle("gpu0", 0, True, True, 0),),
            model_id="/models/fake",
            attention_backend=None,
            wrapper_cls=_NonrespondingStartupWorkerWrapper,
            warmup_runs=0,
        )
    assert time.monotonic() - started < 10.0

    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["disposition"] == "startup_failed"
    assert lifecycle["primary_failure"]["type"] == "_VllmWorkerRPCTimeout"
    assert lifecycle["engine_worker_residual_count"] == 0
    assert len(lifecycle["engine_workers"]) == 1
    worker_record = lifecycle["engine_workers"][0]
    assert worker_record["gpu_id"] == "gpu0"
    assert worker_record["worker_pid"] > 0
    assert worker_record["cuda_visible_devices"] == "8"
    assert worker_record["startup_completed"] is False
    assert worker_record["failure_operation"] == "startup"
    assert "after 0.2 seconds" in worker_record["failure_reason"]
    assert worker_record["termination"] == "terminated"
    assert worker_record["exitcode"] < 0
    assert worker_record["residual_alive"] is False
    assert worker_record["rpc_deadlines_seconds"] == {
        "startup": 0.2,
        "operation": 300.0,
        "join": 0.5,
    }
    assert _active_child_pids() == children_before


def test_live_nonresponding_close_is_bounded_and_retained_in_receipt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv(vllm_runner._WORKER_OPERATION_TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(vllm_runner._WORKER_JOIN_TIMEOUT_ENV, "0.5")
    monkeypatch.setattr(vllm_runner, "_requires_spawned_worker", lambda _wrapper: True)
    children_before = _active_child_pids()
    session = vllm_runner.VllmEngineSession(
        workload_kind="dynamic_router",
        mode="optimized",
        handles=(vllm_runner._GPUHandle("gpu0", 0, True, True, 0),),
        model_id="/models/fake",
        attention_backend=None,
        wrapper_cls=_NonrespondingCloseWorkerWrapper,
        warmup_runs=0,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="vLLM engine teardown failed"):
        session.close()
    assert time.monotonic() - started < 10.0

    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["disposition"] == "completed_with_teardown_errors"
    assert lifecycle["primary_failure"] is None
    assert len(lifecycle["shutdown_errors"]) == 1
    assert lifecycle["engine_worker_residual_count"] == 0
    worker_record = lifecycle["engine_workers"][0]
    assert worker_record["startup_completed"] is True
    assert worker_record["failure_operation"] == "close"
    assert "after 0.2 seconds" in worker_record["failure_reason"]
    assert worker_record["termination"] == "terminated"
    assert worker_record["exitcode"] < 0
    assert worker_record["residual_alive"] is False
    assert _active_child_pids() == children_before


def test_live_nonresponding_metrics_uses_explicit_deadline_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    children_before = _active_child_pids()
    worker = vllm_runner._SpawnedVllmWrapper(
        "gpu0",
        0,
        "/models/fake",
        attention_backend=None,
        wrapper_cls=_NonrespondingMetricsWorkerWrapper,
        operation_timeout_seconds=0.2,
        join_timeout_seconds=0.5,
    )

    started = time.monotonic()
    with pytest.raises(
        vllm_runner._VllmWorkerRPCTimeout,
        match=(
            "timed out during snapshot_metrics while receiving response header after 0.2 seconds"
        ),
    ):
        worker.snapshot_metrics(ttft_ema=1.0, tpot_ema=2.0)
    assert time.monotonic() - started < 10.0
    metadata = worker.lifecycle_metadata()
    assert metadata["failure_operation"] == "snapshot_metrics"
    assert metadata["termination"] == "terminated"
    assert metadata["exitcode"] < 0
    assert metadata["residual_alive"] is False
    assert metadata["rpc_deadlines_seconds"] == {
        "startup": 900.0,
        "operation": 0.2,
        "join": 0.5,
    }
    assert _active_child_pids() == children_before


def test_large_request_to_live_nonreader_times_out_while_sending_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "10")
    monkeypatch.setattr(
        vllm_runner,
        "_vllm_engine_worker",
        _live_nonreading_control_worker,
    )
    children_before = _active_child_pids()
    worker = vllm_runner._SpawnedVllmWrapper(
        "gpu0",
        0,
        "/models/fake",
        attention_backend=None,
        wrapper_cls=_FakeWorkerWrapper,
        operation_timeout_seconds=0.5,
        join_timeout_seconds=0.5,
    )
    worker._connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

    started = time.monotonic()
    with pytest.raises(
        vllm_runner._VllmWorkerRPCTimeout,
        match="timed out during large_request while sending request after 0.5 seconds",
    ):
        worker._call(
            "large_request",
            arguments={"payload": b"x" * (4 * 1024 * 1024)},
        )
    assert time.monotonic() - started < 10.0
    metadata = worker.lifecycle_metadata()
    assert metadata["failure_operation"] == "large_request"
    assert "while sending request" in metadata["failure_reason"]
    assert metadata["termination"] == "terminated"
    assert metadata["exitcode"] < 0
    assert metadata["residual_alive"] is False
    assert _active_child_pids() == children_before


def test_live_partial_response_frame_times_out_on_body_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "11")
    monkeypatch.setattr(
        vllm_runner,
        "_vllm_engine_worker",
        _live_partial_response_frame_worker,
    )
    children_before = _active_child_pids()
    worker = vllm_runner._SpawnedVllmWrapper(
        "gpu0",
        0,
        "/models/fake",
        attention_backend=None,
        wrapper_cls=_FakeWorkerWrapper,
        operation_timeout_seconds=0.5,
        join_timeout_seconds=0.5,
    )

    started = time.monotonic()
    with pytest.raises(
        vllm_runner._VllmWorkerRPCTimeout,
        match=("timed out during partial_response while receiving response body after 0.5 seconds"),
    ):
        worker._call("partial_response")
    assert time.monotonic() - started < 10.0
    metadata = worker.lifecycle_metadata()
    assert metadata["failure_operation"] == "partial_response"
    assert "while receiving response body" in metadata["failure_reason"]
    assert metadata["termination"] == "terminated"
    assert metadata["exitcode"] < 0
    assert metadata["residual_alive"] is False
    assert _active_child_pids() == children_before


def test_session_uses_spawn_proxies_and_emits_v2_worker_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(vllm_runner, "_requires_spawned_worker", lambda _wrapper: True)
    session = vllm_runner.VllmEngineSession(
        workload_kind="dynamic_router",
        mode="optimized",
        handles=(
            vllm_runner._GPUHandle("gpu0", 0, True, True, 0),
            vllm_runner._GPUHandle("gpu1", 1, True, True, 1),
        ),
        model_id="/models/fake",
        attention_backend=None,
        wrapper_cls=_FakeWorkerWrapper,
        warmup_runs=0,
    )
    assert all(
        isinstance(engine, vllm_runner._SpawnedVllmWrapper) for engine in session.engines.values()
    )

    phase, _ = session.begin_run()
    session.finish_run(phase)
    session.close()

    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["schema"] == "aisp.dynamic-router-vllm-profile-lifecycle.v2"
    assert lifecycle["engine_worker_residual_count"] == 0
    assert len(lifecycle["engine_workers"]) == 2
    assert {worker["parent_logical_device_index"] for worker in lifecycle["engine_workers"]} == {
        0,
        1,
    }
    assert {worker["cuda_visible_devices"] for worker in lifecycle["engine_workers"]} == {
        "4",
        "7",
    }
    assert len({worker["worker_pid"] for worker in lifecycle["engine_workers"]}) == 2
    for worker in lifecycle["engine_workers"]:
        assert worker["child_device"] == "cuda:0"
        assert worker["cuda_initialized_before_binding"] is False
        assert worker["vllm_imported_before_binding"] is False
        assert worker["rank_environment"] == []
        assert worker["vllm_enable_v1_multiprocessing"] == "0"
        assert worker["multiprocessing_start_method"] == "spawn"
        assert worker["fallback_enabled"] is False
        assert worker["exitcode"] == 0
        assert worker["termination"] == "clean"
        assert worker["residual_alive"] is False


def test_production_wrapper_selection_requires_spawn_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RealModuleEngine:
        pass

    _RealModuleEngine.__module__ = "vllm.engine.llm_engine"
    monkeypatch.setattr(vllm_runner, "LLMEngine", _RealModuleEngine)

    assert vllm_runner._requires_spawned_worker(vllm_runner._VllmWrapper)
    assert vllm_runner._requires_spawned_worker(vllm_runner._VllmV1Wrapper)
    assert not vllm_runner._requires_spawned_worker(_FakeWorkerWrapper)

    class _UnexpectedEngine:
        pass

    _UnexpectedEngine.__module__ = "unexpected.engine"
    monkeypatch.setattr(vllm_runner, "LLMEngine", _UnexpectedEngine)
    with pytest.raises(RuntimeError, match="refusing an in-process engine"):
        vllm_runner._requires_spawned_worker(vllm_runner._VllmWrapper)
