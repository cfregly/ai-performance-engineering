#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 7: torch.compile on the graph-friendly model.

Requests torch.compile for the shared BF16 graph-friendly model; inspect compilation and graph metrics to establish actual execution.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level7_compiled import Level7Compiled, run_level


def get_benchmark():
    return Level7Compiled()


