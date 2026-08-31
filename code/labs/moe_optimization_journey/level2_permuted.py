#!/usr/bin/env python3
"""Level 2: Fused SiLU and multiplication.

Requests fused SiLU/multiply in the shared BF16 model; Triton execution depends on runtime availability.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Permuted(MoEJourneyBenchmark):
    """Shared level 2: Fused SiLU and multiplication."""
    LEVEL = 2

def get_benchmark() -> Level2Permuted:
    return Level2Permuted()


