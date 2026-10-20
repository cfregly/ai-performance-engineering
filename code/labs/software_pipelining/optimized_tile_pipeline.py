"""Optimized two-stage software-pipelined tile loop for the lab."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.software_pipelining.software_pipelining_common import TilePipelineBenchmark


class OptimizedTilePipelineBenchmark(TilePipelineBenchmark):
    def __init__(self) -> None:
        super().__init__(
            op_name="optimized_tile_pipeline",
            label="software_pipelining_optimized",
            pipeline_stage_count=2,
            notes=(
                "Two-stage tile loop on a block-scoped cuda::pipeline (sm_80+): all eight "
                "warps issue 16-byte cp.async copies (cuda::memcpy_async) for the next "
                "tile while the block computes the current tile with float4 vectors. Tile "
                "size, block size, vector width and math match the baseline, so only the "
                "serialized versus two-stage schedule differs."
            ),
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedTilePipelineBenchmark()
