#!/usr/bin/env python3
"""Level 3: Intermediate-buffer reuse.

Adds buffer reuse to the shared BF16 model.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3Sorted(MoEJourneyBenchmark):
    """Shared level 3: Intermediate-buffer reuse."""
    LEVEL = 3

def get_benchmark() -> Level3Sorted:
    return Level3Sorted()


