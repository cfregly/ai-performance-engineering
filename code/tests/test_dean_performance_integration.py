from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import torch

from ch04.baseline_nvshmem_vs_nccl_benchmark_multigpu import (
    NVSHMEMVsNCCLBenchmarkMultiGPU,
)
from ch04.nvshmem_child_result import (
    NVSHMEM_CHILD_RESULT_CALLBACK,
    NVSHMEM_CHILD_RESULT_DIR_ENV,
    NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
    NVSHMEMWorkloadResult,
    write_nvshmem_child_result,
)
from ch04.optimized_nvshmem_vs_nccl_benchmark_multigpu import (
    OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU,
)
from core.harness.benchmark_harness import BenchmarkConfig, TorchrunLaunchSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"

_COLLECTIVE_CASES = (
    (NVSHMEMVsNCCLBenchmarkMultiGPU, "baseline", "nccl"),
    (OptimizedNVSHMEMVsNCCLBenchmarkMultiGPU, "optimized", "nvshmem"),
)


def _collective_spec(benchmark) -> TorchrunLaunchSpec:
    return benchmark.get_torchrun_spec(
        BenchmarkConfig(
            nproc_per_node=2,
            nnodes=1,
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=300,
        )
    )


def _cpu_collective_result(
    *,
    rank: int,
    configuration: dict[str, bool | int | str],
    time_per_iter_ms: float,
) -> NVSHMEMWorkloadResult:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4).add_(rank)
    reference = torch.arange(8, dtype=torch.float32).reshape(2, 4).square()
    return NVSHMEMWorkloadResult(
        workload="collective",
        rank=rank,
        world_size=2,
        iterations=500,
        time_per_iter_ms=time_per_iter_ms,
        configuration=configuration,
        verify_inputs={"source_tensor": source},
        verify_output=reference.clone(),
        reference_output=reference,
        batch_size=2,
        parameter_count=0,
        collective_type="broadcast",
        output_tolerance=(1e-5, 1e-6),
    )


def test_repo_performance_skill_is_routed_and_source_bounded() -> None:
    skill = (REPO_ROOT / ".agents/skills/dean-performance-review/SKILL.md").read_text(
        encoding="utf-8"
    )
    agents = (CODE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    readme_generator = (CODE_ROOT / "core/scripts/refresh_readmes.py").read_text(
        encoding="utf-8"
    )

    assert "https://abseil.io/fast/hints.html" in skill
    assert "single-binary performance" in skill
    assert "Static review identifies\n  hypotheses, not measured wins." in skill
    assert ".agents/skills/dean-performance-review/SKILL.md" in agents
    assert ".agents/skills/dean-performance-review/SKILL.md" in readme_generator


def test_performance_intake_requires_cost_model_and_evidence() -> None:
    intake = (CODE_ROOT / "templates/performance_intake.yaml").read_text(encoding="utf-8")
    methodology = (CODE_ROOT / "docs/benchmark_methodology.md").read_text(encoding="utf-8")

    assert "hot_path_model:" in intake
    assert "invocation_frequency:" in intake
    assert "best_case_primary_kpi_improvement_pct:" in intake
    assert "baseline_artifact:" in intake
    assert "## Dean/Ghemawat Optimization Loop" in methodology


def test_nvshmem_wrappers_launch_explicit_measured_collective_workers() -> None:
    for benchmark_type, variant, mode in _COLLECTIVE_CASES:
        benchmark = benchmark_type()
        spec = _collective_spec(benchmark)
        result_dir = Path(spec.env[NVSHMEM_CHILD_RESULT_DIR_ENV])
        try:
            assert isinstance(spec, TorchrunLaunchSpec)
            assert spec.script_path is None
            assert spec.module_name == "core.harness.benchmark_worker"
            assert spec.script_args[:7] == [
                "--module",
                "ch04.nvshmem_worker",
                "--callable",
                "main",
                "--",
                "--workload",
                "collective",
            ]
            worker_args = spec.script_args[7:]
            assert worker_args[worker_args.index("--variant") + 1] == variant
            assert worker_args[worker_args.index("--mode") + 1] == mode
            assert spec.result_callback == NVSHMEM_CHILD_RESULT_CALLBACK
            assert spec.timing_source == "rank0_time_per_iter_ms"
            assert spec.timing_iterations_per_sample == 500
            assert spec.multi_gpu_required is True
            assert result_dir.is_dir() and not any(result_dir.iterdir())
            assert benchmark.get_custom_metrics() is None
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)


def test_nvshmem_wrappers_consume_actual_full_rank_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for benchmark_type, variant, _mode in _COLLECTIVE_CASES:
        benchmark = benchmark_type()
        spec = _collective_spec(benchmark)
        context = benchmark._nvshmem_child_result_context
        assert context is not None
        result_dir = Path(spec.env[NVSHMEM_CHILD_RESULT_DIR_ENV])
        launch_wall_ns = time.time_ns()
        try:
            for key, value in spec.env.items():
                monkeypatch.setenv(key, value)
            monkeypatch.setenv(
                NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
                str(launch_wall_ns),
            )
            configuration = dict(context["configuration"])
            assert write_nvshmem_child_result(
                _cpu_collective_result(
                    rank=0,
                    configuration=configuration,
                    time_per_iter_ms=0.011,
                ),
                variant=variant,
            )
            assert write_nvshmem_child_result(
                _cpu_collective_result(
                    rank=1,
                    configuration=configuration,
                    time_per_iter_ms=0.013,
                ),
                variant=variant,
            )
            benchmark.consume_nvshmem_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )

            benchmark.capture_verification_payload()
            assert benchmark.validate_result() is None
            assert not result_dir.exists()
            assert benchmark._nvshmem_child_result_bundle is not None
            rank_receipts = benchmark._nvshmem_child_result_bundle["ranks"]
            assert [item["time_per_iter_ms"] for item in rank_receipts] == [0.011, 0.013]
            assert all(item["configuration"] == configuration for item in rank_receipts)
            signature = benchmark.get_input_signature()
            assert signature.world_size == 2
            assert signature.ranks == [0, 1]
            assert signature.collective_type == "broadcast"
            assert benchmark.get_verify_output().shape == (16,)
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)


def test_nvshmem_wrappers_fail_closed_without_a_full_rank_receipt_quorum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for benchmark_type, variant, _mode in _COLLECTIVE_CASES:
        benchmark = benchmark_type()
        spec = _collective_spec(benchmark)
        context = benchmark._nvshmem_child_result_context
        assert context is not None
        result_dir = Path(spec.env[NVSHMEM_CHILD_RESULT_DIR_ENV])
        launch_wall_ns = time.time_ns()
        try:
            assert benchmark.validate_result() == "Fresh full-rank NVSHMEM worker output is missing"
            with pytest.raises(RuntimeError, match="fresh full-rank worker result"):
                benchmark.capture_verification_payload()

            for key, value in spec.env.items():
                monkeypatch.setenv(key, value)
            monkeypatch.setenv(
                NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
                str(launch_wall_ns),
            )
            assert write_nvshmem_child_result(
                _cpu_collective_result(
                    rank=0,
                    configuration=dict(context["configuration"]),
                    time_per_iter_ms=0.011,
                ),
                variant=variant,
            )
            with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
                benchmark.consume_nvshmem_child_results(
                    launch_wall_ns=launch_wall_ns,
                    finish_wall_ns=time.time_ns(),
                    returncode=0,
                )
            assert context["retention"] == "retained-incomplete-rank-quorum"
            assert result_dir.is_dir()
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
