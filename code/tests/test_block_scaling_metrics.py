from __future__ import annotations

import torch

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from labs.block_scaling.baseline_block_scaling import BaselineBlockScalingBenchmark
from labs.block_scaling.optimized_block_scaling import OptimizedBlockScalingBenchmark


def test_block_scaling_path_indicator_survives_numeric_metric_resolution() -> None:
    baseline = BaselineBlockScalingBenchmark()
    optimized = OptimizedBlockScalingBenchmark()
    harness = BenchmarkHarness(
        config=BenchmarkConfig(device=torch.device("cpu"), validity_profile="portable")
    )

    baseline_metrics = baseline.get_custom_metrics()
    optimized_metrics = optimized.get_custom_metrics()

    assert baseline_metrics == {
        "block_scaling.hardware_blockscaled": 0.0,
        "block_scaling.sf_vec_size": float(baseline.config.sf_vec_size),
    }
    assert optimized_metrics == {
        "block_scaling.hardware_blockscaled": 1.0,
        "block_scaling.sf_vec_size": float(optimized.config.sf_vec_size),
        "block_scaling.graph_replay": 0.0,
    }
    assert all(type(value) is float for value in baseline_metrics.values())
    assert all(type(value) is float for value in optimized_metrics.values())

    assert harness._resolve_custom_metrics(baseline) == baseline_metrics
    assert harness._resolve_custom_metrics(optimized) == optimized_metrics

    baseline_story = harness._resolve_story_metadata(baseline)
    optimized_story = harness._resolve_story_metadata(optimized)
    assert baseline_story is not None
    assert optimized_story is not None
    assert baseline_story["block_scaling.path"] == "software_dequant_bf16"
    assert optimized_story["block_scaling.path"] == "hardware_blockscaled_cutlass"
