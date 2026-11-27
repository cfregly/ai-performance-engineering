"""Optimized async input pipeline benchmark (pinned + non-blocking + copy stream)."""

from __future__ import annotations

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.common.async_input_pipeline import AsyncInputPipelineBenchmark, PipelineConfig


def get_benchmark() -> AsyncInputPipelineBenchmark:
    cfg = PipelineConfig(
        batch_size=256,
        feature_shape=(3, 256, 256),
        dataset_size=2048,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        non_blocking=True,
        use_copy_stream=True,
    )
    benchmark = AsyncInputPipelineBenchmark(cfg, label="optimized_async_input_pipeline")
    return attach_benchmark_metadata(benchmark, __file__)


