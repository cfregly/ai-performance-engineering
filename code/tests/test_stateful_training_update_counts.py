"""Training pairs must reach the same optimizer step before output comparison."""

import importlib

import pytest
import torch


@pytest.mark.parametrize(
    "baseline_name,optimized_name,iterations,warmup",
    [
        ("ch01.baseline_performance", "ch01.optimized_performance", 5, 10),
        ("ch01.baseline_performance", "ch01.optimized_performance_fusion", 5, 10),
        ("ch01.baseline_performance_fp16", "ch01.optimized_performance_fp16", 5, 10),
        ("ch13.baseline_autograd_standard", "ch13.optimized_autograd_standard", 50, 10),
        ("ch13.baseline_precisionfp8", "ch13.optimized_precisionfp8", 50, 10),
        ("ch13.baseline_precisionfp8", "ch13.optimized_precisionfp8_rowwise", 50, 10),
        ("ch13.baseline_precisionfp8", "ch13.optimized_precisionfp8_rowwise_gw_hp", 50, 10),
        ("ch13.baseline_precisionfp8_te", "ch13.optimized_precisionfp8_te", 50, 10),
        ("ch13.baseline_precisionmixed", "ch13.optimized_precisionmixed", 50, 10),
        ("ch13.baseline_training_speed", "ch13.optimized_training_speed", 30, 10),
        ("ch13.baseline_training_standard", "ch13.optimized_training_standard", 10, 5),
        pytest.param(
            "ch19.baseline_nvfp4_training", "ch19.optimized_nvfp4_training", 8, 5,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 factory requires CUDA"),
        ),
        ("ch20.baseline_training_single", "ch20.optimized_training_single", 50, 10),
    ],
)
def test_actual_factory_configs_preserve_equal_training_updates(
    baseline_name, optimized_name, iterations, warmup
):
    baseline = importlib.import_module(baseline_name).get_benchmark().get_config()
    optimized = importlib.import_module(optimized_name).get_benchmark().get_config()
    assert baseline.iterations == optimized.iterations == iterations
    assert baseline.warmup == optimized.warmup == warmup
    assert not baseline.adaptive_iterations
    assert not optimized.adaptive_iterations
