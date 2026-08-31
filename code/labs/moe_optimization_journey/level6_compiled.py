#!/usr/bin/env python3
"""Level 6: CUDA graph replay of the expert path.

Requests CUDA graph capture/replay for the shared BF16 expert path; inspect capture metrics to establish actual execution.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level6Compiled(MoEJourneyBenchmark):
    """Shared level 6: CUDA graph replay of the expert path."""

    LEVEL = 6

def get_benchmark() -> Level6Compiled:
    return Level6Compiled()


