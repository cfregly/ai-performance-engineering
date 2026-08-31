#!/usr/bin/env python3
"""Compatibility entrypoint for shared MoE level 5: Padded batched GEMMs.

Adds padded batched-matmul dispatch to the shared BF16 model; no distributed or multi-stream expert parallelism is enabled.
The target name is retained; legacy FP8/streams/parallel labels are not
evidence that those techniques execute. Shared-model precision is BF16.
Fresh correctness and profiling evidence is required for a speedup claim."""

from labs.moe_optimization_journey.level5_bmm_fusion import Level5BMMFusion

def get_benchmark():
    return Level5BMMFusion()

