"""Fused BF16-to-NVFP4 operation family."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_quantization.benchmarks import NVFP4Benchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(NVFP4Benchmark("add_rmsnorm", optimized=True), __file__)
