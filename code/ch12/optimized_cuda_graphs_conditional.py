"""Python harness wrapper for optimized_cuda_graphs_conditional.cu."""

from __future__ import annotations

from ch12.baseline_cuda_graphs_conditional import _CudaGraphsConditionalBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedCudaGraphsConditionalBenchmark(_CudaGraphsConditionalBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        super().__init__(
            binary_name="optimized_cuda_graphs_conditional",
            friendly_name="Optimized Cuda Graphs Conditional",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedCudaGraphsConditionalBenchmark()
