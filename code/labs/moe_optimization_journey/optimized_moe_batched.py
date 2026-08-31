#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 1: Batched expert computation.

Uses the shared batched BF16 model.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level1_batched import Level1Batched


def get_benchmark() -> Level1Batched:
    return Level1Batched()


__all__ = ["Level1Batched", "get_benchmark"]
