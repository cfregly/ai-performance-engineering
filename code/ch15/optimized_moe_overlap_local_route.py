"""Optimized Ch15 MoE route placement with local direct routed expert compute."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch15.moe_overlap_local_route_common import MoeOverlapLocalRouteBenchmark


class OptimizedMoeOverlapLocalRouteBenchmark(MoeOverlapLocalRouteBenchmark):
    """No-arg class wrapper for subprocess benchmark isolation."""

    def __init__(self) -> None:
        super().__init__(
            variant="local_direct",
            label="optimized_moe_overlap_local_route",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoeOverlapLocalRouteBenchmark()
