#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 6: CUDA graph replay of the expert path.

Requests CUDA graph capture/replay for the shared BF16 expert path; inspect capture metrics to establish actual execution.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""
from labs.moe_optimization_journey.level5_cudagraphs import Level6CUDAGraphs


def get_benchmark() -> Level6CUDAGraphs:
    return Level6CUDAGraphs()


__all__ = ["Level6CUDAGraphs", "get_benchmark"]
