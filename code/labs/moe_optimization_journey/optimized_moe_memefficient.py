#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 3: Intermediate-buffer reuse.

Adds buffer reuse to the shared BF16 model.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level3_memefficient import Level3MemEfficient


def get_benchmark() -> Level3MemEfficient:
    return Level3MemEfficient()


__all__ = ["Level3MemEfficient", "get_benchmark"]


