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
                "Two-stage warp-specialized TMA (cp.async.bulk) tile pipeline on sm_90+: "
                "producer warp issues bulk copies against tx-count mbarriers, consumer "
                "warps compute with float4 vectors and release stages early. Falls back "
                "to a block-scoped cuda::pipeline on sm_80."
            ),
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedTilePipelineBenchmark()
