"""Colfax PR #2817 control: single S/P TMEM slot during decode."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.flashattention4.colfax_benchmarks import ColfaxBenchmark


class BaselineFlashAttention4DecodeBenchmark(ColfaxBenchmark):
    def __init__(self):
        super().__init__("decode", optimized=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttention4DecodeBenchmark()
