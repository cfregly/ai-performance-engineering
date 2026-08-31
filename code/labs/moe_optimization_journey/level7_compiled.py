#!/usr/bin/env python3
"""Level 7: torch.compile on the graph-friendly model.

Requests torch.compile for the shared BF16 graph-friendly model; inspect compilation and graph metrics to establish actual execution.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""

import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level7Compiled(MoEJourneyBenchmark):
    """Shared level 7: torch.compile on the graph-friendly model."""

    LEVEL = 7

def get_benchmark() -> MoEJourneyBenchmark:
    return Level7Compiled()


