"""KIVI two-bit quantization with one byte of storage per quantized code."""

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.kv_cache_compression.kivi_benchmark import KiviBenchmark


def get_benchmark() -> BaseBenchmark:
    return attach_benchmark_metadata(KiviBenchmark(packed=False), __file__)
