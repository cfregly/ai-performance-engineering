"""Baseline for coalesced dynamic quantized cache refresh.

This keeps the adaptive-bitwidth schedule but performs every logical refresh.
"""

from __future__ import annotations

from ch19.baseline_dynamic_quantized_cache import _DynamicQuantizedCacheBenchmark


class BaselineDynamicQuantizedCacheCoalescedBenchmark(_DynamicQuantizedCacheBenchmark):
    """Baseline: adaptive quantized cache refresh without coalescing."""

    def __init__(self) -> None:
        schedule = [8] * 12 + [6] * 8 + [4] * 12
        super().__init__(
            schedule_bits=schedule,
            use_fp32_baseline=False,
            coalesce_repeated_bits=False,
        )


def get_benchmark():
    return BaselineDynamicQuantizedCacheCoalescedBenchmark()
