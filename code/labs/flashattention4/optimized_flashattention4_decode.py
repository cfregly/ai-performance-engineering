"""Colfax PR #2817 candidate: ping-pong S/P across two TMEM slots."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.colfax_benchmarks import ColfaxBenchmark


class OptimizedFlashAttention4DecodeBenchmark(ColfaxBenchmark):
    def __init__(self):
        super().__init__("decode", optimized=True)


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttention4DecodeBenchmark()
