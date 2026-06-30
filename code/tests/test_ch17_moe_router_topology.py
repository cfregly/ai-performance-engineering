from __future__ import annotations

import inspect

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
