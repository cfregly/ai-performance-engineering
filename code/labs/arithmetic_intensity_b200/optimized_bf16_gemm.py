"""Harness candidate for the dense BF16 B200 GEMM optimization."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.arithmetic_intensity_b200.optimization_benchmarks import (
    OptimizedBf16GemmBenchmark,
)


def get_benchmark() -> BaseBenchmark:
    return OptimizedBf16GemmBenchmark()
