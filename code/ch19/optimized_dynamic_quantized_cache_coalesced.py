"""Optimized dynamic quantized cache refresh with repeated-width coalescing."""

from __future__ import annotations

from ch19.baseline_dynamic_quantized_cache import _DynamicQuantizedCacheBenchmark


class OptimizedDynamicQuantizedCacheCoalescedBenchmark(_DynamicQuantizedCacheBenchmark):
    """Optimized: collapse consecutive same-width refreshes into one physical copy."""

    def __init__(self) -> None:
        schedule = [8] * 12 + [6] * 8 + [4] * 12
        super().__init__(
            schedule_bits=schedule,
            use_fp32_baseline=False,
            coalesce_repeated_bits=True,
        )


def get_benchmark():
    return OptimizedDynamicQuantizedCacheCoalescedBenchmark()
