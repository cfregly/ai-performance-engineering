"""Colfax PR #2804 control: aliased TMEM and compute-wide synchronization."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.colfax_benchmarks import ColfaxBenchmark


class BaselineFlashAttention4BackwardBenchmark(ColfaxBenchmark):
    def __init__(self):
        super().__init__("backward", optimized=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttention4BackwardBenchmark()
