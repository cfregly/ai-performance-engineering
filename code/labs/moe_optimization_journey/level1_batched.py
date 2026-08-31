#!/usr/bin/env python3
"""Level 1: Batched expert computation.

Uses the shared batched BF16 model.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level1Batched(MoEJourneyBenchmark):
    """Shared level 1: Batched expert computation."""
    LEVEL = 1

def get_benchmark() -> Level1Batched:
    return Level1Batched()


