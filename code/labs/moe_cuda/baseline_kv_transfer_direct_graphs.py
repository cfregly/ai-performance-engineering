"""Baseline wrapper for the direct-destination graphed KV-transfer variant.

The optimized target combines direct destination placement with CUDA Graph
replay. The baseline intentionally stays on the original sequential
compute-then-copy path so the pair measures the full serving-pipeline change.
"""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_kv_transfer import BaselineKVTransferBenchmark


def get_benchmark() -> BaseBenchmark:
    return BaselineKVTransferBenchmark()

