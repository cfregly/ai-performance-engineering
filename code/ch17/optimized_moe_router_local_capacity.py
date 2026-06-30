"""Optimized local-capacity MoE routing: keep all tokens on local experts."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from ch17.optimized_moe_router_uniform_topology import OptimizedMoERouterTopologyBenchmark


class OptimizedMoERouterLocalCapacityBenchmark(OptimizedMoERouterTopologyBenchmark):
    """Optimized: reserve enough local expert capacity to remove overflow spill."""

    def __init__(self) -> None:
        super().__init__(spill_period=0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoERouterLocalCapacityBenchmark()
