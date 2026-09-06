"""NVSHMEM put-and-signal pipeline optimization via the strict multi-GPU wrapper."""

from __future__ import annotations

from ch04.optimized_nvshmem_pipeline_parallel_multigpu import (
    OptimizedNVSHMEMPipelineParallelMultiGPU,
)
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVSHMEMPipelineParallelMultiGPU()
