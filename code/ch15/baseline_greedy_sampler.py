"""Baseline greedy sampling: materialize probabilities before argmax."""

from __future__ import annotations

from ch15.greedy_sampler_common import GreedySamplerBenchmark, GreedySamplerConfig
from core.benchmark.wrapper_utils import attach_benchmark_metadata


def get_benchmark() -> GreedySamplerBenchmark:
    cfg = GreedySamplerConfig(
        materialize_probabilities=True,
        label="baseline_greedy_sampler",
    )
    return attach_benchmark_metadata(GreedySamplerBenchmark(cfg), __file__)  # type: ignore[return-value]
