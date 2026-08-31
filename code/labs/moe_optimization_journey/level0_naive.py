#!/usr/bin/env python3
"""Level 0: Naive expert loops.

Runs the naive shared BF16 model.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level0Naive(MoEJourneyBenchmark):
    """Shared level 0: Naive expert loops."""
    LEVEL = 0

def get_benchmark() -> Level0Naive:
    return Level0Naive()


