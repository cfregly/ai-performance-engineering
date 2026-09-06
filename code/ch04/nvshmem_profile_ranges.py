"""NVTX range contract shared by direct NVSHMEM torchrun workloads."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from core.profiling.nvtx_helper import nvtx_range

PROFILE_NVTX_RANGE_ENV = "AISP_NVSHMEM_PROFILE_NVTX_RANGE"

PIPELINE_BASELINE_NVTX_RANGE = "compute_kernel:nvshmem_pipeline_synchronous"
PIPELINE_OPTIMIZED_NVTX_RANGE = "compute_kernel:nvshmem_pipeline_async"
TRAINING_EXAMPLE_BASELINE_NVTX_RANGE = (
    "compute_kernel:nvshmem_training_pipeline_allocate_each_step"
)
TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE = "compute_kernel:nvshmem_training_pipeline_reuse_buffers"
TRAINING_PATTERNS_BASELINE_NVTX_RANGE = "reduce:nvshmem_gradient_naive_allreduce"
TRAINING_PATTERNS_OPTIMIZED_NVTX_RANGE = "reduce:nvshmem_gradient_fused_sync"
COLLECTIVE_BASELINE_NVTX_RANGE = "transfer_sync:nccl_broadcast"
COLLECTIVE_OPTIMIZED_NVTX_RANGE = "transfer_async:nvshmem_broadcast_overlap"

KNOWN_NVTX_RANGES = frozenset(
    {
        PIPELINE_BASELINE_NVTX_RANGE,
        PIPELINE_OPTIMIZED_NVTX_RANGE,
        TRAINING_EXAMPLE_BASELINE_NVTX_RANGE,
        TRAINING_EXAMPLE_OPTIMIZED_NVTX_RANGE,
        TRAINING_PATTERNS_BASELINE_NVTX_RANGE,
        TRAINING_PATTERNS_OPTIMIZED_NVTX_RANGE,
        COLLECTIVE_BASELINE_NVTX_RANGE,
        COLLECTIVE_OPTIMIZED_NVTX_RANGE,
    }
)


@contextmanager
def selected_nvtx_range() -> Iterator[None]:
    """Emit the exact range selected by an owning benchmark wrapper, if any."""
    label = os.environ.get(PROFILE_NVTX_RANGE_ENV)
    if label is None:
        yield
        return
    if label not in KNOWN_NVTX_RANGES:
        raise RuntimeError(f"Unknown NVSHMEM profile NVTX range: {label!r}")
    with nvtx_range(label, enable=True):
        yield
