"""Colfax PR #2804 candidate: separate P/dS TMEM and warp-local signaling."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.colfax_benchmarks import ColfaxBenchmark


class OptimizedFlashAttention4BackwardBenchmark(ColfaxBenchmark):
    def __init__(self):
        super().__init__("backward", optimized=True)


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttention4BackwardBenchmark()
