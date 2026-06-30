"""Baseline for trusted speculative decode: verify every draft block."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.speculative_decode.optimized_speculative_decode import (
    OptimizedSpeculativeDecodeBenchmark,
)


class BaselineSpeculativeDecodeTrustedBenchmark(OptimizedSpeculativeDecodeBenchmark):
    """Baseline: standard speculative decode with target verification enabled."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics.update({
            "speculative.target_verify_calls": 0.0,
            "speculative.trusted_draft": 0.0,
        })

    def get_custom_metrics(self) -> dict[str, float]:
        metrics = dict(super().get_custom_metrics())
        metrics["speculative.target_verify_calls"] = metrics.get("speculative.rounds", 0.0)
        metrics["speculative.trusted_draft"] = 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodeTrustedBenchmark()
