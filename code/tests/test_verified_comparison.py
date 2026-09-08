"""Real CPU coverage for fail-closed convenience comparisons."""

from __future__ import annotations

import pytest
import torch

from core.benchmark.runtime_comparison import ExecutedRuntimeComparisonError
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verified_comparison import (
    VerifiedComparisonError,
    calculate_speed_metrics,
)
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    ExecutionMode,
    compare_benchmarks,
)


class _CpuComparisonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self, *, output_offset: float = 0.0, fail: bool = False) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.output_offset = output_offset
        self.fail = fail
        self.benchmark_calls = 0
        self.input_tensor: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    def setup(self) -> None:
        self.input_tensor = torch.arange(32, dtype=torch.float32, device=self.device)

    def benchmark_fn(self) -> torch.Tensor:
        self.benchmark_calls += 1
        if self.fail:
            raise RuntimeError("intentional comparison execution failure")
        if self.input_tensor is None:  # pragma: no cover - lifecycle guard
            raise RuntimeError("setup() must initialize input_tensor")
        self.output = self.input_tensor * 2.0 + self.output_offset
        return self.output

    def capture_verification_payload(self) -> None:
        if self.input_tensor is None or self.output is None:
            raise RuntimeError("timed execution did not produce a verification payload")
        self._set_verification_payload(
            inputs={"input": self.input_tensor},
            output=self.output,
            batch_size=32,
            parameter_count=0,
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self) -> None:
        return None


def _cpu_harness() -> BenchmarkHarness:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        use_subprocess=False,
        execution_mode=ExecutionMode.THREAD,
        enable_profiling=False,
        enable_memory_tracking=False,
        enable_cleanup=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        detect_setup_precomputation=False,
        adaptive_iterations=False,
        clear_compile_cache=False,
        measurement_timeout_seconds=15,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_public_compare_benchmarks_returns_verified_real_cpu_results() -> None:
    baseline = _CpuComparisonBenchmark()
    optimized = _CpuComparisonBenchmark()

    comparison = compare_benchmarks(
        baseline,
        optimized,
        harness=_cpu_harness(),
        name="real CPU comparison",
    )

    assert comparison["name"] == "real CPU comparison"
    assert not comparison["baseline_result"].errors
    assert not comparison["optimized_result"].errors
    assert comparison["runtime_comparison"]["matches"] is True
    assert comparison["input_verification"]["passed"] is True
    assert comparison["verification"]["passed"] is True
    assert comparison["speedup"] == pytest.approx(
        comparison["baseline"]["mean_ms"] / comparison["optimized"]["mean_ms"]
    )
    assert baseline.benchmark_calls > 5
    assert optimized.benchmark_calls > 5


def test_public_compare_benchmarks_rejects_real_execution_failure() -> None:
    baseline = _CpuComparisonBenchmark()
    optimized = _CpuComparisonBenchmark(fail=True)

    with pytest.raises(ExecutedRuntimeComparisonError) as exc_info:
        compare_benchmarks(baseline, optimized, harness=_cpu_harness())

    failures = exc_info.value.comparison.integrity_failures
    assert any(
        failure.code == "result_errors" and failure.run == "candidate"
        for failure in failures
    )
    assert any(
        "intentional comparison execution failure" in failure.detail
        for failure in failures
    )


def test_public_compare_benchmarks_rejects_incorrect_full_output() -> None:
    baseline = _CpuComparisonBenchmark(output_offset=0.0)
    optimized = _CpuComparisonBenchmark(output_offset=1.0)

    with pytest.raises(VerifiedComparisonError, match="output_mismatch") as exc_info:
        compare_benchmarks(baseline, optimized, harness=_cpu_harness())

    assert exc_info.value.code == "output_mismatch"
    assert exc_info.value.evidence["passed"] is False
    assert exc_info.value.evidence["max_diff"] == pytest.approx(1.0)


def test_slowdown_percentage_uses_elapsed_time_ratio() -> None:
    metrics = calculate_speed_metrics(
        100.0,
        150.0,
        regression_threshold_pct=40.0,
    )

    assert metrics == {
        "speedup": pytest.approx(2.0 / 3.0),
        "regression": True,
        "regression_pct": pytest.approx(50.0),
    }
