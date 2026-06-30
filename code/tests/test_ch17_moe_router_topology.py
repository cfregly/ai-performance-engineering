from __future__ import annotations

import inspect

import pytest
import torch

from ch17.optimized_moe_router_uniform_topology import OptimizedMoERouterTopologyBenchmark
from ch17.moe_router_topology_demo import (
    _build_islands,
    _build_spillover_order,
    _route_one,
)


def test_topology_router_precomputes_spillover_order() -> None:
    route_source = inspect.getsource(_route_one)
    assert "for isl in spillover_order[local_island]:" in route_source
    assert "sorted(islands.keys()" not in route_source

    islands = _build_islands(num_islands=4, experts_per_island=1)
    spillover_order = _build_spillover_order(islands)
    assert spillover_order == {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [1, 3, 0],
        3: [2, 1, 0],
    }

    loads = {expert: 0 for experts in islands.values() for expert in experts}
    assert _route_one(
        token_id=0,
        local_island=0,
        islands=islands,
        spillover_order=spillover_order,
        loads=loads,
        capacity_per_expert=1,
    ) == 0
    assert _route_one(
        token_id=1,
        local_island=0,
        islands=islands,
        spillover_order=spillover_order,
        loads=loads,
        capacity_per_expert=1,
    ) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for MoE local-capacity parity")
def test_local_capacity_spill_policy_preserves_shared_expert_output() -> None:
    baseline = OptimizedMoERouterTopologyBenchmark(spill_period=8)
    optimized = OptimizedMoERouterTopologyBenchmark(spill_period=0)
    for bench in (baseline, optimized):
        bench.hidden_size = 32
        bench.ffn_size = 16
        bench.num_islands = 4
        bench.experts_per_island = 2
        bench.num_experts = 8
        bench.batch = 4
        bench.seq = 8
        bench.remote_round_trips = 2
    try:
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.cuda.synchronize()

        torch.testing.assert_close(baseline.output, optimized.output, rtol=0.0, atol=0.0)
        baseline_metrics = baseline.get_custom_metrics()
        optimized_metrics = optimized.get_custom_metrics()
        assert baseline_metrics["moe.remote_tokens"] == 4.0
        assert baseline_metrics["moe.remote_fraction_pct"] == 12.5
        assert optimized_metrics["moe.remote_tokens"] == 0.0
        assert optimized_metrics["moe.remote_fraction_pct"] == 0.0
    finally:
        baseline.teardown()
        optimized.teardown()
