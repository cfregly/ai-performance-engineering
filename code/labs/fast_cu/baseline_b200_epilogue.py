"""SM100 control: FP32-to-FP16 epilogue with two 128-bit stores."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.b200_epilogue import B200EpilogueBenchmark


class BaselineB200EpilogueBenchmark(B200EpilogueBenchmark):
    def __init__(self):
        super().__init__(optimized=False)


def get_benchmark() -> BaseBenchmark:
    return BaselineB200EpilogueBenchmark()
