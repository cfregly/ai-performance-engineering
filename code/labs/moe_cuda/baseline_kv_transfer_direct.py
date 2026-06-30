"""Baseline wrapper for the direct-destination KV-transfer variant.

This keeps the same sequential compute-then-copy path as ``kv_transfer`` so the
new direct-placement example can compare against the existing transfer baseline
without changing the original overlap-only target.
"""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_kv_transfer import BaselineKVTransferBenchmark


def get_benchmark() -> BaseBenchmark:
    return BaselineKVTransferBenchmark()

