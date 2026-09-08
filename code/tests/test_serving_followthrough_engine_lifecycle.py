"""Focused contracts for reusable serving engines and direct 1P1D routing."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    DecodeAffinityMode,
    DistributedRequestPlan,
    _affinity_opportunity_count,
    _direct_1p1d_barriers_avoided_per_request,
    _use_direct_1p1d_sync_fast_path,
)
from labs.dynamic_router import vllm_runner


class _FakeReusableWrapper:
    created: list[_FakeReusableWrapper] = []

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
        self.reset_calls = 0
        self.close_calls = 0
        self.force_close_calls = 0
        self.created.append(self)

    def reset_request_state(self) -> None:
        self.reset_calls += 1

    def close(self, *, force: bool = False) -> None:
        self.close_calls += 1
        self.force_close_calls += int(force)


def test_engine_session_reuses_engines_and_reports_all_lifecycle_phases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _FakeReusableWrapper.created.clear()
    handles = [
        vllm_runner._GPUHandle("gpu0", 0, True, False, 0),
        vllm_runner._GPUHandle("gpu1", 1, False, True, 1),
    ]
    session = vllm_runner.VllmEngineSession(
        workload_kind="dual_pool",
        mode="dual",
        handles=handles,
        model_id="/models/local",
        attention_backend="TRITON_ATTN",
        wrapper_cls=_FakeReusableWrapper,
        warmup_runs=1,
    )

    warmup_phase, warmup_prefix = session.begin_run()
    session.finish_run(warmup_phase)
    steady_phase, steady_prefix = session.begin_run()
    session.finish_run(steady_phase)

    assert warmup_phase == "warmup"
    assert steady_phase == "steady_state"
    assert warmup_prefix != steady_prefix
    assert len(_FakeReusableWrapper.created) == 2
    assert [wrapper.reset_calls for wrapper in _FakeReusableWrapper.created] == [2, 2]
    metrics = session.lifecycle_metrics()
    assert metrics["lifecycle.warmup_runs"] == 1.0
    assert metrics["lifecycle.steady_state_runs"] == 1.0
    assert metrics["lifecycle.engine_reuse_count"] == 1.0
    assert "lifecycle.engine_teardown_ms" not in metrics

    session.close()
    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["event"] == "vllm_engine_lifecycle"
    assert lifecycle["disposition"] == "completed"
    assert lifecycle["request_state_reset_per_run"] is True
    assert len(lifecycle["warmup_request_processing_ms"]) == 1
    assert len(lifecycle["steady_state_request_processing_ms"]) == 1
    assert lifecycle["engine_teardown_ms"] >= 0.0
    assert lifecycle["end_to_end_ms"] >= lifecycle["engine_startup_ms"]
    assert lifecycle["shutdown_errors"] == []
    assert [wrapper.close_calls for wrapper in _FakeReusableWrapper.created] == [1, 1]


def test_partial_engine_startup_closes_created_engine_and_keeps_primary_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailSecondWrapper(_FakeReusableWrapper):
        def __init__(self, gpu_id: str, *args, **kwargs) -> None:
            if gpu_id == "gpu1":
                raise ValueError("second engine startup failed")
            super().__init__(gpu_id, *args, **kwargs)

    _FakeReusableWrapper.created.clear()
    handles = [
        vllm_runner._GPUHandle("gpu0", 0, True, False, 0),
        vllm_runner._GPUHandle("gpu1", 1, False, True, 1),
    ]
    with pytest.raises(ValueError, match="second engine startup failed"):
        vllm_runner.VllmEngineSession(
            workload_kind="dual_pool",
            mode="dual",
            handles=handles,
            model_id="/models/local",
            attention_backend="TRITON_ATTN",
            wrapper_cls=FailSecondWrapper,
            warmup_runs=1,
        )

    assert len(_FakeReusableWrapper.created) == 1
    assert _FakeReusableWrapper.created[0].force_close_calls == 1
    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["disposition"] == "startup_failed"
    assert lifecycle["primary_failure"] == {
        "type": "ValueError",
        "message": "second engine startup failed",
    }
    assert lifecycle["shutdown_errors"] == []


def test_owned_session_failure_keeps_primary_error_and_emits_teardown_disposition(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingCloseWrapper(_FakeReusableWrapper):
        def close(self, *, force: bool = False) -> None:
            super().close(force=force)
            raise RuntimeError("shutdown failed")

    created_sessions: list[vllm_runner.VllmEngineSession] = []

    def factory(mode: str, **kwargs) -> vllm_runner.VllmEngineSession:
        del kwargs
        session = vllm_runner.VllmEngineSession(
            workload_kind="test",
            mode=mode,
            handles=[vllm_runner._GPUHandle("gpu0", 0, True, True, 0)],
            model_id="/models/local",
            attention_backend=None,
            wrapper_cls=FailingCloseWrapper,
            warmup_runs=0,
        )
        created_sessions.append(session)
        return session

    @vllm_runner._manage_engine_session(factory)
    def failing_run(mode: str, *, topology_snapshot, engine_session=None) -> None:
        del mode, topology_snapshot
        engine_session.begin_run()
        raise ValueError("request processing failed")

    with pytest.raises(ValueError, match="request processing failed") as exc_info:
        failing_run("test", topology_snapshot=object())

    assert len(created_sessions) == 1
    assert created_sessions[0]._closed is True
    assert any("shutdown failed" in note for note in (exc_info.value.__notes__ or []))
    lifecycle = json.loads(capsys.readouterr().err)
    assert lifecycle["disposition"] == "failed_run_with_teardown_errors"
    assert lifecycle["primary_failure"] == {
        "type": "ValueError",
        "message": "request processing failed",
    }
    assert lifecycle["failed_runs"][0]["phase"] == "steady_state"
    assert lifecycle["shutdown_errors"] == ["gpu0: shutdown failed"]


def test_wrapper_reset_requires_an_idle_engine_and_clears_completed_outputs() -> None:
    wrapper = vllm_runner._VllmWrapper.__new__(vllm_runner._VllmWrapper)
    wrapper.gpu_id = "gpu0"
    wrapper.engine = SimpleNamespace(get_num_unfinished_requests=lambda: 0)
    wrapper._inflight = {}
    wrapper._completed_output_token_ids = {"old-request": (1, 2)}
    wrapper._closed = False

    wrapper.reset_request_state()
    assert wrapper._completed_output_token_ids == {}

    wrapper.engine = SimpleNamespace(get_num_unfinished_requests=lambda: 1)
    with pytest.raises(RuntimeError, match="unfinished requests"):
        wrapper.reset_request_state()


def test_wrapper_force_close_aborts_vllm_request_list_before_shutdown() -> None:
    aborted: list[list[str]] = []
    shutdown_calls: list[bool] = []
    wrapper = vllm_runner._VllmWrapper.__new__(vllm_runner._VllmWrapper)
    wrapper.gpu_id = "gpu0"
    wrapper.engine = SimpleNamespace(
        get_num_unfinished_requests=lambda: 2,
        abort_request=lambda request_ids: aborted.append(list(request_ids)),
        engine_core=SimpleNamespace(shutdown=lambda: shutdown_calls.append(True)),
    )
    wrapper._inflight = {"request-a": object(), "request-b": object()}
    wrapper._completed_output_token_ids = {}
    wrapper._closed = False

    wrapper.close(force=True)

    assert aborted == [["request-a", "request-b"]]
    assert shutdown_calls == [True]
    assert wrapper._inflight == {}
    assert wrapper._closed is True


def test_one_decode_rank_exposes_no_affinity_opportunity_and_exact_fast_path() -> None:
    plans = [
        DistributedRequestPlan(0, 0, 0, warm_chunks=2, total_chunks=4),
        DistributedRequestPlan(0, 1, 1, warm_chunks=0, total_chunks=4),
    ]

    assert _affinity_opportunity_count(
        plans,
        prefill_ranks=1,
        decode_ranks=1,
    ) == 0
    assert _affinity_opportunity_count(
        plans,
        prefill_ranks=1,
        decode_ranks=2,
    ) > 0
    assert _use_direct_1p1d_sync_fast_path(
        affinity_mode=DecodeAffinityMode.STICKY,
        world_size=2,
        prefill_ranks=1,
        decode_ranks=1,
    )
    assert not _use_direct_1p1d_sync_fast_path(
        affinity_mode=DecodeAffinityMode.ROUND_ROBIN,
        world_size=2,
        prefill_ranks=1,
        decode_ranks=1,
    )
    assert not _use_direct_1p1d_sync_fast_path(
        affinity_mode=DecodeAffinityMode.STICKY,
        world_size=3,
        prefill_ranks=1,
        decode_ranks=2,
    )
    assert _direct_1p1d_barriers_avoided_per_request(plans) == 7.0


@pytest.mark.parametrize(
    "module_name",
    [
        "labs.dynamic_router.baseline_dynamic_router_vllm",
        "labs.dynamic_router.optimized_dynamic_router_vllm",
        "labs.dynamic_router.baseline_dual_pool_vllm",
        "labs.dynamic_router.optimized_dual_pool_vllm",
    ],
)
def test_vllm_benchmarks_separate_five_warmups_from_three_wall_clock_runs(
    module_name: str,
) -> None:
    benchmark = importlib.import_module(module_name).get_benchmark()
    config = benchmark.get_config()

    assert config is not None
    assert config.warmup == vllm_runner.WARMUP_ITERATIONS == 5
    assert config.iterations == vllm_runner.STEADY_STATE_ITERATIONS == 3
    assert config.adaptive_iterations is False
    assert config.timing_method == "wall_clock"
