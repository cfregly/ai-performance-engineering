"""cuBLASLt control for the pinned fast.cu GB300 NVFP4 workload."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.nvfp4 import FastCuNvfp4Benchmark, Nvfp4Workload


class FastCuNvfp4CublasLtBenchmark(FastCuNvfp4Benchmark):
    """First correctness-gated cuBLASLt NVFP4 algorithm on the shared inputs."""

    def __init__(self, workload: Nvfp4Workload | None = None) -> None:
        super().__init__(optimized=False, workload=workload)


def get_benchmark() -> BaseBenchmark:
    return FastCuNvfp4CublasLtBenchmark()


__all__ = ["FastCuNvfp4CublasLtBenchmark", "get_benchmark"]
