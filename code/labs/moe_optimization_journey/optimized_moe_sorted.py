#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 2: Fused SiLU and multiplication.

Requests fused SiLU/multiply in the shared BF16 model; Triton execution depends on runtime availability.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level2_sorted import Level2Sorted


def get_benchmark() -> Level2Sorted:
    return Level2Sorted()


__all__ = ["Level2Sorted", "get_benchmark"]
