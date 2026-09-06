"""Shared utilities for TMA-gated threshold benchmarks."""

from __future__ import annotations

import torch

from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported

from ch08.threshold_benchmark_base import ThresholdBenchmarkBase


class ThresholdBenchmarkBaseTMA(ThresholdBenchmarkBase):
    """Blackwell threshold comparison with per-thread async-copy staging.

    The chapter-native warp-divergence story lives in `threshold`. This pair keeps
    the same threshold workload shape but swaps in a Blackwell-gated, per-thread
    async-copy staging path. The legacy `thresholdtma` name does not denote bulk TMA.
    """

    requirement_label = "threshold_tma"
    rows: int = ThresholdBenchmarkBase.rows
    threshold: float = ThresholdBenchmarkBase.threshold
    inner_iterations: int = ThresholdBenchmarkBase.inner_iterations

    def setup(self) -> None:
        ensure_blackwell_tma_supported("Chapter 8 threshold TMA pipeline")
        super().setup()

    def _generate_inputs(self) -> torch.Tensor:  # type: ignore[override]
        return super()._generate_inputs()

    def get_custom_metrics(self) -> dict | None:
        metrics = super().get_custom_metrics() or {}
        metrics.update(
            {
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
                "story.bridge_to_ch10": 1.0,
            }
        )
        return metrics

    def get_optimization_goal(self) -> str:
        """Keep the TMA bridge pair runnable without headline speed gating."""
        return "comparison"
