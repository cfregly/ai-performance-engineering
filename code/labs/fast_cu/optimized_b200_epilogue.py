"""SM100 candidate: FP32-to-FP16 epilogue with one 256-bit store."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.b200_epilogue import B200EpilogueBenchmark


class OptimizedB200EpilogueBenchmark(B200EpilogueBenchmark):
    def __init__(self):
        super().__init__(optimized=True)


def get_benchmark() -> BaseBenchmark:
    return OptimizedB200EpilogueBenchmark()
