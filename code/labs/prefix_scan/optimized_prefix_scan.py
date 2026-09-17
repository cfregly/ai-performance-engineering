"""Inclusive prefix scan benchmark."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.prefix_scan.benchmarks import PrefixScanBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(PrefixScanBenchmark(optimized=True), __file__)
