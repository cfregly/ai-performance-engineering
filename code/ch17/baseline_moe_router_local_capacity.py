"""Baseline for local-capacity MoE routing: topology-aware routing with overflow spill."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch17.optimized_moe_router_uniform_topology import OptimizedMoERouterTopologyBenchmark


class BaselineMoERouterLocalCapacityBenchmark(OptimizedMoERouterTopologyBenchmark):
    """Baseline: topology-aware routing still spills every 8th token remote."""

    def __init__(self) -> None:
        super().__init__(spill_period=8)


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERouterLocalCapacityBenchmark()
