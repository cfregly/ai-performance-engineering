"""Optimized cache-aware disaggregated inference benchmark (multi-GPU torchrun)."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark
from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import (
    CACHE_AWARE_OPTIMIZED_NVTX_RANGE,
    CacheAwareDisaggMultiGPUBenchmark,
)

PROFILE_NVTX_RANGE = CACHE_AWARE_OPTIMIZED_NVTX_RANGE


def get_benchmark() -> BaseBenchmark:
    bench = CacheAwareDisaggMultiGPUBenchmark(
        optimized=True,
        label="optimized_cache_aware_disagg_multigpu",
        script_path=__file__,
    )
    return attach_benchmark_metadata(bench, __file__)
