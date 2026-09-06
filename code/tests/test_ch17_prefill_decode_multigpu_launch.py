from __future__ import annotations

import importlib
import inspect
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ch17 import prefill_decode_disagg_multigpu_common as common
from ch17.prefill_decode_disagg_multigpu_result import (
    PREFILL_DECODE_LAUNCH_WALL_NS_ENV,
    PREFILL_DECODE_RESULT_CALLBACK,
    PrefillDecodeChildResultMixin,
    _input_signature,
    make_prefill_decode_result_contract,
    write_prefill_decode_child_result,
)
from core.harness import benchmark_worker

CODE_ROOT = Path(__file__).resolve().parents[1]

CASES = (
    (
        "ch17.baseline_prefill_decode_disagg_batched_multigpu",
        "BaselinePrefillDecodeDisaggBatchedMultiGPUBenchmark",
        "baseline_prefill_decode_disagg_batched_multigpu",
    ),
    (
        "ch17.optimized_prefill_decode_disagg_batched_multigpu",
        "OptimizedPrefillDecodeDisaggBatchedMultiGPUBenchmark",
        "optimized_prefill_decode_disagg_batched_multigpu",
    ),
    (
        "ch17.baseline_prefill_decode_disagg_overlap_multigpu",
        "BaselinePrefillDecodeDisaggOverlapMultiGPUBenchmark",
        "baseline_prefill_decode_disagg_overlap_multigpu",
    ),
    (
        "ch17.optimized_prefill_decode_disagg_overlap_multigpu",
        "OptimizedPrefillDecodeDisaggOverlapMultiGPUBenchmark",
        "optimized_prefill_decode_disagg_overlap_multigpu",
    ),
    (
        "ch17.baseline_prefill_decode_disagg_tpot_long_multigpu",
        "BaselinePrefillDecodeDisaggTPOTLongMultiGPUBenchmark",
        "baseline_prefill_decode_disagg_tpot_long_multigpu",
    ),
    (
        "ch17.optimized_prefill_decode_disagg_tpot_long_multigpu",
        "OptimizedPrefillDecodeDisaggTPOTLongMultiGPUBenchmark",
        "optimized_prefill_decode_disagg_tpot_long_multigpu",
    ),
    (
        "ch17.baseline_prefill_decode_disagg_ttft_multigpu",
        "BaselinePrefillDecodeDisaggTTFTMultiGPUBenchmark",
        "baseline_prefill_decode_disagg_ttft_multigpu",
    ),
    (
        "ch17.optimized_prefill_decode_disagg_ttft_multigpu",
        "OptimizedPrefillDecodeDisaggTTFTMultiGPUBenchmark",
        "optimized_prefill_decode_disagg_ttft_multigpu",
    ),
)


class _ResultHarness(PrefillDecodeChildResultMixin):
    pass


def _contract():
    return make_prefill_decode_result_contract(
        label="baseline_prefill_decode_disagg_batched_multigpu",
        handoff_mode="serial",
        world_size=2,
        prefill_ranks=1,
        hidden_size=4,
        num_layers=1,
        batch_size=1,
        requests_per_rank=2,
        context_window=2,
        decode_tokens=2,
        transfer_group=1,
        sync_per_request=False,
        barrier_per_request=False,
        dtype=torch.float32,
        iterations=1,
        warmup=0,
    )


def _prepare_result_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_ResultHarness, int, dict[str, str]]:
    harness = _ResultHarness()
    result_env = harness.prepare_prefill_decode_child_result(_contract())
    launch_wall_ns = time.time_ns() - 1_000_000
    result_env[PREFILL_DECODE_LAUNCH_WALL_NS_ENV] = str(launch_wall_ns)
    for key, value in result_env.items():
        monkeypatch.setenv(key, value)
    return harness, launch_wall_ns, result_env


@pytest.mark.parametrize(("module_name", "class_name", "label"), CASES)
def test_ch17_specs_call_explicit_worker_and_bind_profile_and_result(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    label: str,
) -> None:
    monkeypatch.setattr(common, "_resolve_world_size", lambda: 2)
    module = importlib.import_module(module_name)
    benchmark = getattr(module, class_name)()
    config = benchmark.get_config()
    spec = benchmark.get_torchrun_spec(config)
    try:
        assert spec.script_path == Path(benchmark_worker.__file__).resolve()
        assert spec.module_name is None
        assert spec.script_args == [
            "--module",
            module_name,
            "--callable",
            "main",
            "--",
            "--prefill-ranks",
            "1",
        ]
        assert spec.config_arg_map == {
            "iterations": "--iters",
            "warmup": "--warmup",
        }
        assert spec.result_callback == PREFILL_DECODE_RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == config.iterations
        assert config.nsys_nvtx_include == [f"compute_kernel:{label}"]
        assert config.ncu_replay_mode == "app-range"
        assert config.ncu_replay_mode_override is True
        assert spec.name == label
    finally:
        context = benchmark._prefill_decode_result_context
        if context is not None:
            shutil.rmtree(context["result_dir"], ignore_errors=True)


@pytest.mark.parametrize("module_name", [case[0] for case in CASES])
def test_ch17_explicit_worker_reaches_cuda_capability_gate(module_name: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(CODE_ROOT)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(benchmark_worker.__file__).resolve()),
            "--module",
            module_name,
            "--callable",
            "main",
            "--",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--prefill-ranks",
            "1",
        ],
        cwd=CODE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode != 0
    assert "SKIPPED: CUDA required for disaggregated prefill/decode" in completed.stderr


def test_prefill_decode_child_result_requires_full_fresh_rank_quorum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, launch_wall_ns, _ = _prepare_result_harness(monkeypatch)
    reference = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4) / 10
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=0,
        reference_output=reference,
        timed_output=None,
        verification_prompt=torch.ones((1, 2, 4), dtype=torch.float32),
    )
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=1,
        reference_output=None,
        timed_output=reference.clone(),
        verification_prompt=None,
    )
    harness.consume_prefill_decode_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=time.time_ns() + 1_000_000,
        returncode=0,
    )

    torch.testing.assert_close(harness.get_verify_output(), reference)
    torch.testing.assert_close(
        harness.get_verify_inputs()["prompt"],
        torch.ones((1, 2, 4), dtype=torch.float32),
    )
    signature = harness.get_input_signature()
    assert signature.shapes["prompt"] == (1, 2, 4)
    assert signature.shapes["decode_tokens"] == (2,)
    assert signature.shapes["output"] == (2, 1, 4)
    assert signature.world_size == 2
    for name, shape in signature.shapes.items():
        if name != "output":
            assert tuple(harness.get_verify_inputs()[name].shape) == shape
    assert harness.get_output_tolerance() == (1e-5, 1e-8)
    assert harness._prefill_decode_result_bundle is not None


def test_prefill_decode_child_result_rejects_missing_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, launch_wall_ns, result_env = _prepare_result_harness(monkeypatch)
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=0,
        reference_output=torch.zeros((2, 1, 4), dtype=torch.float32),
        timed_output=None,
        verification_prompt=torch.ones((1, 2, 4), dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
        harness.consume_prefill_decode_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=time.time_ns() + 1_000_000,
            returncode=0,
        )
    assert harness._prefill_decode_result_context["retention"] == (
        "retained-incomplete-rank-quorum"
    )
    shutil.rmtree(result_env["AISP_PREFILL_DECODE_RESULT_DIR"])


def test_pair_signatures_match_while_child_contract_binds_handoff_mode() -> None:
    baseline_contract = _contract()
    optimized_contract = replace(
        baseline_contract,
        label="optimized_prefill_decode_disagg_batched_multigpu",
        handoff_mode="batched",
    )

    assert baseline_contract.to_dict() != optimized_contract.to_dict()
    assert _input_signature(baseline_contract, tf32=False).to_dict() == (
        _input_signature(optimized_contract, tf32=False).to_dict()
    )


def test_prefill_decode_child_result_rejects_any_output_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, launch_wall_ns, result_env = _prepare_result_harness(monkeypatch)
    reference = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4) / 10
    corrupted = reference.clone()
    corrupted[1, 0, 2] += 1.0
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=0,
        reference_output=reference,
        timed_output=None,
        verification_prompt=torch.ones((1, 2, 4), dtype=torch.float32),
    )
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=1,
        reference_output=None,
        timed_output=corrupted,
        verification_prompt=None,
    )
    with pytest.raises(RuntimeError, match="full timed output differs"):
        harness.consume_prefill_decode_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=time.time_ns() + 1_000_000,
            returncode=0,
        )
    result_dir = Path(result_env["AISP_PREFILL_DECODE_RESULT_DIR"])
    assert result_dir.is_dir()
    assert harness._prefill_decode_result_context["retention"] == (
        "retained-invalid-child-result"
    )
    shutil.rmtree(result_dir)


def test_prefill_decode_child_result_rejects_stale_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, _, result_env = _prepare_result_harness(monkeypatch)
    future_launch_wall_ns = time.time_ns() + 10_000_000_000
    monkeypatch.setenv(
        PREFILL_DECODE_LAUNCH_WALL_NS_ENV,
        str(future_launch_wall_ns),
    )
    reference = torch.zeros((2, 1, 4), dtype=torch.float32)
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=0,
        reference_output=reference,
        timed_output=None,
        verification_prompt=torch.ones((1, 2, 4), dtype=torch.float32),
    )
    write_prefill_decode_child_result(
        contract=_contract(),
        rank=1,
        reference_output=None,
        timed_output=reference,
        verification_prompt=None,
    )
    with pytest.raises(RuntimeError, match="Stale prefill/decode"):
        harness.consume_prefill_decode_child_results(
            launch_wall_ns=future_launch_wall_ns,
            finish_wall_ns=future_launch_wall_ns + 1_000_000,
            returncode=0,
        )
    shutil.rmtree(result_env["AISP_PREFILL_DECODE_RESULT_DIR"])


@pytest.mark.parametrize(
    "handoff_mode",
    [common.HandoffMode.SERIAL, common.HandoffMode.BATCHED, common.HandoffMode.OVERLAP],
)
def test_prefill_reference_runs_real_model_outside_timing(
    handoff_mode: common.HandoffMode,
) -> None:
    torch.manual_seed(123)
    cfg = common.PrefillDecodeConfig(
        hidden_size=4,
        num_layers=1,
        batch_size=2,
        requests_per_rank=3,
        context_window=2,
        decode_tokens=2,
        transfer_group=2,
        dtype=torch.float32,
    )
    model = common.TinyPrefillDecode(
        cfg.hidden_size,
        cfg.num_layers,
        torch.device("cpu"),
        cfg.dtype,
    ).eval()
    prompts = torch.randn(
        cfg.requests_per_rank,
        cfg.batch_size,
        cfg.context_window,
        cfg.hidden_size,
    )
    output = common._build_prefill_reference_output(
        cfg,
        model,
        prompts,
        handoff_mode,
    )
    assert output.shape == (cfg.requests_per_rank, cfg.batch_size, cfg.hidden_size)
    assert bool(torch.isfinite(output).all())


def test_prefill_decode_profile_range_matches_rank0_timing_boundary() -> None:
    source = inspect.getsource(common._run_torchrun_worker)
    range_start = source.index("with nvtx_range(profile_range, enable=True):")
    timed_loop = source.index("for _ in range(iters):", range_start)
    synchronize = source.index("torch.cuda.synchronize(device)", timed_loop)
    barrier = source.index("_barrier()", synchronize)
    elapsed = source.index("elapsed = time.perf_counter() - start", barrier)
    result_write = source.index("write_prefill_decode_child_result(", elapsed)
    timing_line = source.index('print(f"rank0 time_per_iter_ms:', result_write)

    assert range_start < timed_loop < synchronize < barrier < elapsed
    assert elapsed < result_write < timing_line
    assert "torch.manual_seed(" not in source
    assert "torch.cuda.manual_seed_all(" not in source
