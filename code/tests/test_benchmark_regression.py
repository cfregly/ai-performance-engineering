"""Pytest tests for benchmark performance regression detection.

Tests that benchmarks maintain performance characteristics and detects regressions.

These tests are marked as slow and require CUDA. They can be skipped in CI
by using pytest's -m flag to exclude slow tests.

Usage:
    pytest tests/test_benchmark_regression.py -m "not slow"  # Skip regression tests
    pytest tests/test_benchmark_regression.py  # Run regression tests (slow)
"""

import math
from pathlib import Path

import pytest

from core.env import apply_env_defaults

apply_env_defaults()

import torch

from core.benchmark.comparison import compare_results
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from core.utils.chapter_compare_template import get_last_load_error, load_benchmark

# Skip tests if CUDA is not available (NVIDIA GPU required)
# Tests are marked as slow and can be skipped with: pytest -m "not slow"
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required - NVIDIA GPU and tools must be available"
    ),
    pytest.mark.slow,
]


REPO_ROOT = Path(__file__).parent.parent
REGRESSION_CHAPTER = "ch02"
BASELINE_BENCHMARK = "baseline_cublas.py"
OPTIMIZED_BENCHMARK = "optimized_cublas.py"


def _load_required_benchmark(filename: str) -> BaseBenchmark:
    benchmark_path = REPO_ROOT / REGRESSION_CHAPTER / filename
    assert benchmark_path.is_file(), f"Required regression benchmark is missing: {benchmark_path}"

    benchmark = load_benchmark(benchmark_path)
    assert benchmark is not None, f"Failed to load {benchmark_path}: {get_last_load_error()}"
    assert isinstance(benchmark, BaseBenchmark)
    return benchmark


def _positive_mean_ms(result) -> float:
    timing = result.timing
    assert timing is not None
    assert timing.iterations > 0
    assert math.isfinite(timing.mean_ms) and timing.mean_ms > 0.0
    return timing.mean_ms


@pytest.fixture(scope="module")
def harness():
    """Create a benchmark harness for regression testing."""
    config = BenchmarkConfig(
        iterations=5,  # Minimal iterations for lightweight regression checks
        warmup=5,
        measurement_timeout_seconds=10,
        timeout_multiplier=1.0,
        adaptive_iterations=False,
        enable_profiling=False,  # Disable profiling for regression tests
        enable_nsys=False,
        enable_ncu=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_quick_baseline_optimized_speedup(harness):
    """Lightweight sanity check: verify one benchmark pair shows expected speedup.

    This is a minimal regression check that only tests a single benchmark pair
    to keep CI times reasonable. Additional slow tests can be run by removing
    the -m "not slow" filter.
    """
    baseline = _load_required_benchmark(BASELINE_BENCHMARK)
    optimized = _load_required_benchmark(OPTIMIZED_BENCHMARK)

    # Run benchmarks with minimal iterations
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)

    # Both should complete successfully
    _positive_mean_ms(baseline_result)
    _positive_mean_ms(optimized_result)

    # Use comparison utility to verify
    comparison = compare_results(
        baseline_result,
        optimized_result,
        regression_threshold_pct=20.0,
    )
    assert math.isfinite(comparison.speedup) and comparison.speedup > 0.0
    # Optimized should not show significant regression (allow variance for small sample)
    assert not comparison.regression, (
        f"Optimized benchmark shows regression: {comparison.regression_pct:.1f}% slower"
    )


@pytest.mark.slow
def test_benchmark_result_consistency(harness):
    """Test that benchmark results are consistent across runs (variance check)."""
    # Fresh instances ensure teardown from the first run cannot affect the second.
    result1 = harness.benchmark(_load_required_benchmark(BASELINE_BENCHMARK))
    result2 = harness.benchmark(_load_required_benchmark(BASELINE_BENCHMARK))

    # Results should be within reasonable variance (50% tolerance for small sample sizes)
    result1_mean = _positive_mean_ms(result1)
    result2_mean = _positive_mean_ms(result2)
    relative_spread = abs(result1_mean - result2_mean) / max(result1_mean, result2_mean)
    assert relative_spread < 0.5, (
        "Benchmark results too inconsistent: "
        f"{result1_mean:.3f}ms vs {result2_mean:.3f}ms "
        f"(relative spread: {relative_spread:.1%})"
    )


@pytest.mark.slow
def test_benchmark_memory_usage():
    """Test that benchmarks report memory usage when enabled."""
    # Create harness with memory tracking enabled
    config = BenchmarkConfig(
        iterations=3,  # Minimal iterations
        warmup=5,
        measurement_timeout_seconds=10,
        timeout_multiplier=1.0,
        adaptive_iterations=False,
        enable_memory_tracking=True,
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
    )
    memory_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    result = memory_harness.benchmark(_load_required_benchmark(BASELINE_BENCHMARK))

    _positive_mean_ms(result)
    assert result.memory is not None
    assert result.memory.peak_mb is not None
    assert math.isfinite(result.memory.peak_mb) and result.memory.peak_mb > 0.0
    assert result.memory.allocated_mb is not None
    assert math.isfinite(result.memory.allocated_mb) and result.memory.allocated_mb > 0.0
    assert result.memory.peak_mb >= result.memory.allocated_mb


@pytest.mark.slow
def test_benchmark_timeout_handling():
    """Test that benchmarks respect timeout limits."""
    # Create harness with very short timeout
    config = BenchmarkConfig(
        iterations=1000,  # Many iterations
        warmup=5,
        measurement_timeout_seconds=1,
        timeout_multiplier=1.0,
        adaptive_iterations=False,
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
        use_subprocess=True,
    )
    timeout_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    # A fast GPU may finish all iterations. Otherwise the harness must return a
    # structured measurement-timeout result rather than an empty success.
    result = timeout_harness.benchmark(_load_required_benchmark(BASELINE_BENCHMARK))
    if result.timeout_stage is None:
        _positive_mean_ms(result)
    else:
        assert result.timeout_stage == "measurement"
        assert result.timing.iterations == 0
        assert result.timeout_limit_seconds == 1
        assert any("timeout" in error.lower() for error in result.errors)


@pytest.mark.slow
def test_benchmark_validation_coverage():
    """Test that benchmarks with validation actually validate correctly."""
    for benchmark_file in (BASELINE_BENCHMARK, OPTIMIZED_BENCHMARK):
        benchmark = _load_required_benchmark(benchmark_file)
        benchmark.setup()
        try:
            benchmark.benchmark_fn()
            validation_error = benchmark.validate_result()
            assert validation_error is None, (
                f"Validation failed for {benchmark_file}: {validation_error}"
            )
        finally:
            benchmark.teardown()
