"""Baseline Ch15 MoE route placement with remote overlap and scatter."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_overlap_local_route_common import MoeOverlapLocalRouteBenchmark


class BaselineMoeOverlapLocalRouteBenchmark(MoeOverlapLocalRouteBenchmark):
    """No-arg class wrapper for subprocess benchmark isolation."""

    def __init__(self) -> None:
        super().__init__(
            variant="remote_overlap",
            label="baseline_moe_overlap_local_route",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineMoeOverlapLocalRouteBenchmark()
