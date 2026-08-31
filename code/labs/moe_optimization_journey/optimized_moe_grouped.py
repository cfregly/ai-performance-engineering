#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 4: Sorted per-expert GEMMs.

Adds sorting and per-expert GEMMs to the shared BF16 model; no multi-stream dispatch is enabled.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level4_grouped import Level4Grouped


def get_benchmark() -> Level4Grouped:
    return Level4Grouped()


__all__ = ["Level4Grouped", "get_benchmark"]
