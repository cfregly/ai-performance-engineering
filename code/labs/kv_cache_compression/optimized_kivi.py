"""KIVI cache with four two-bit codes packed into each storage byte."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.kv_cache_compression.kivi_benchmark import KiviBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(KiviBenchmark(packed=True), __file__)
