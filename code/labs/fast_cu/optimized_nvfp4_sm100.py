"""B200 K64 GEMM with fast.cu's TMA pipeline and wide FP16 stores."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.nvfp4 import Nvfp4Workload
from labs.fast_cu.nvfp4_sm100 import FastCuNvfp4Sm100Benchmark


class FastCuNvfp4Sm100KernelBenchmark(FastCuNvfp4Sm100Benchmark):
    def __init__(self, workload: Nvfp4Workload | None = None) -> None:
        super().__init__(optimized=True, workload=workload)


def get_benchmark() -> BaseBenchmark:
    return FastCuNvfp4Sm100KernelBenchmark()
