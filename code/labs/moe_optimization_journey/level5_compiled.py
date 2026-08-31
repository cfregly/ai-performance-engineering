#!/usr/bin/env python3
"""Level 5: Padded batched GEMMs.

Adds padded batched-matmul dispatch to the shared BF16 model; no distributed or multi-stream expert parallelism is enabled.

The filename and class are retained for compatibility. The LEVEL mapping,
not a legacy filename, determines execution. These shared levels do not use
FP8 quantization. No speedup is established by selecting a level."""

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5Compiled(MoEJourneyBenchmark):
    """Shared level 5: Padded batched GEMMs."""

    LEVEL = 5

def get_benchmark() -> Level5Compiled:
    return Level5Compiled()


