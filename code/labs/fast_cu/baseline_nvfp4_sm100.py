"""cuBLASLt baseline for the B200 NVFP4 GEMM port."""

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.nvfp4 import Nvfp4Workload
from labs.fast_cu.nvfp4_sm100 import FastCuNvfp4Sm100Benchmark


class FastCuNvfp4Sm100CublasLtBenchmark(FastCuNvfp4Sm100Benchmark):
    def __init__(self, workload: Nvfp4Workload | None = None) -> None:
        super().__init__(optimized=False, workload=workload)


def get_benchmark() -> BaseBenchmark:
    return FastCuNvfp4Sm100CublasLtBenchmark()
