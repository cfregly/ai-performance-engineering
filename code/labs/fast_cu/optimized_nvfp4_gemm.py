"""Selected fast.cu rung for the pinned GB300 NVFP4 workload."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.nvfp4 import DEFAULT_RUNG, FastCuNvfp4Benchmark, Nvfp4Workload


class FastCuNvfp4KernelBenchmark(FastCuNvfp4Benchmark):
    """Run fast.cu r0-r9, with r9 as the default."""

    def __init__(
        self,
        workload: Nvfp4Workload | None = None,
        rung: int = DEFAULT_RUNG,
    ) -> None:
        super().__init__(optimized=True, workload=workload, rung=rung)


def get_benchmark() -> BaseBenchmark:
    return FastCuNvfp4KernelBenchmark(rung=DEFAULT_RUNG)


__all__ = ["FastCuNvfp4KernelBenchmark", "get_benchmark"]
