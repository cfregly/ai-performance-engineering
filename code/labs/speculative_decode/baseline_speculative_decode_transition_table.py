"""Baseline for transition-table speculative decode: trusted draft MLP decode."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.speculative_decode.optimized_speculative_decode_trusted import (
    OptimizedSpeculativeDecodeTrustedBenchmark,
)


class BaselineSpeculativeDecodeTransitionTableBenchmark(OptimizedSpeculativeDecodeTrustedBenchmark):
    """Baseline: trusted draft path still runs the draft MLP for every token."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics.update({
            "speculative.transition_table": 0.0,
            "speculative.draft_model_calls": 0.0,
        })

    def get_custom_metrics(self) -> dict[str, float]:
        metrics = dict(super().get_custom_metrics())
        metrics["speculative.transition_table"] = 0.0
        metrics["speculative.draft_model_calls"] = metrics.get("speculative.draft_tokens", 0.0)
        return metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodeTransitionTableBenchmark()
