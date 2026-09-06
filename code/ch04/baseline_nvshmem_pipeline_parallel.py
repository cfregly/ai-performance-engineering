"""NCCL P2P baseline for the NVSHMEM pipeline comparison."""

from __future__ import annotations

from ch04.baseline_nvshmem_pipeline_parallel_multigpu import (
    NVSHMEMPipelineParallelMultiGPU,
)
from core.harness.benchmark_harness import BaseBenchmark


def get_benchmark() -> BaseBenchmark:
    return NVSHMEMPipelineParallelMultiGPU()
