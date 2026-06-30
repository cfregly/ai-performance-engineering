"""moe_router_topology_demo.py - Chapter 17 MoE topology-aware routing demo (non-benchmark).

Topology-aware MoE router: groups experts by NVSwitch island and routes tokens
to local experts first, falling back to nearby islands only when local capacity
is exhausted. Uses simple round-robin within each island.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List


def _build_islands(*, num_islands: int, experts_per_island: int) -> Dict[int, List[int]]:
    islands: Dict[int, List[int]] = {}
    expert_id = 0
    for island in range(num_islands):
        islands[island] = []
        for _ in range(experts_per_island):
            islands[island].append(expert_id)
            expert_id += 1
    return islands


def _build_spillover_order(islands: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {
        local_island: [
            island
            for island in sorted(islands.keys(), key=lambda i: (abs(i - local_island), i))
            if island != local_island
        ]
        for local_island in islands
    }


def _route_one(
    *,
    token_id: int,
    local_island: int,
    islands: Dict[int, List[int]],
    spillover_order: Dict[int, List[int]],
    loads: Dict[int, int],
    capacity_per_expert: int,
) -> int:
    for exp in islands[local_island]:
        if loads[exp] < capacity_per_expert:
            loads[exp] += 1
            return exp

    for isl in spillover_order[local_island]:
        for exp in islands[isl]:
            if loads[exp] < capacity_per_expert:
                loads[exp] += 1
                return exp

    fallback = islands[local_island][0]
    loads[fallback] += 1
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 17 MoE topology-aware router demo")
    parser.add_argument("--tokens", type=int, default=4096, help="Number of tokens to route.")
    parser.add_argument("--num-islands", type=int, default=4, help="Number of NVSwitch islands.")
    parser.add_argument("--experts-per-island", type=int, default=4, help="Experts per island.")
    parser.add_argument("--capacity-per-expert", type=int, default=256, help="Max tokens per expert before spilling.")
    args = parser.parse_args()

    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.num_islands <= 0:
        raise ValueError("--num-islands must be positive")
    if args.experts_per_island <= 0:
        raise ValueError("--experts-per-island must be positive")
    if args.capacity_per_expert <= 0:
        raise ValueError("--capacity-per-expert must be positive")

    islands = _build_islands(num_islands=int(args.num_islands), experts_per_island=int(args.experts_per_island))
    spillover_order = _build_spillover_order(islands)
    loads = {exp: 0 for experts in islands.values() for exp in experts}
    assignments: list[int] = []
    remote_tokens = 0

    for token in range(int(args.tokens)):
        local_island = token % len(islands)
        expert = _route_one(
            token_id=token,
            local_island=local_island,
            islands=islands,
            spillover_order=spillover_order,
            loads=loads,
            capacity_per_expert=int(args.capacity_per_expert),
        )
        assignments.append(expert)
        if expert not in islands[local_island]:
            remote_tokens += 1

    counts = Counter(assignments)
    total_experts = int(args.num_islands) * int(args.experts_per_island)
    remote_frac = remote_tokens / max(int(args.tokens), 1)

    print("Topology-aware router demo")
    print(f"tokens:             {args.tokens}")
    print(f"num_islands:        {args.num_islands}")
    print(f"experts_per_island: {args.experts_per_island}")
    print(f"capacity_per_expert:{args.capacity_per_expert}")
    print(f"remote_fraction:    {remote_frac:.3f}")
    print("expert_load:")
    for expert_id in range(total_experts):
        print(f"  {expert_id:>3}: {counts.get(expert_id, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
