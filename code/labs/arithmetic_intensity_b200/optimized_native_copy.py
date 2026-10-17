"""Harness candidate for the native B200 copy-kernel optimization."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.arithmetic_intensity_b200.optimization_benchmarks import (
    OptimizedNativeCopyBenchmark,
)


def get_benchmark() -> BaseBenchmark:
    return OptimizedNativeCopyBenchmark()
