#!/usr/bin/env python3
"""Level 4: Sorted per-expert GEMMs.

Adds sorting and per-expert GEMMs to the shared BF16 model; no multi-stream dispatch is enabled.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Grouped(MoEJourneyBenchmark):
    """Shared level 4: Sorted per-expert GEMMs."""
    LEVEL = 4

def get_benchmark() -> Level4Grouped:
    return Level4Grouped()


