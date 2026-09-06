"""Controls for Chapter 13 failures found by the direct B200 sweep."""
import pytest
import torch

from ch13.baseline_memory_profiling import BaselineMemoryProfilingBenchmark, SimpleModel
from ch13.optimized_memory_profiling import OptimizedMemoryProfilingBenchmark, OptimizedModel
from ch13.optimized_kv_cache_naive_pool import SimpleAttentionLayer
from core.harness.run_benchmarks import _resolve_optimization_goal


def test_declared_memory_goal_takes_precedence_over_historical_speed_metadata():
    benchmark = object.__new__(OptimizedMemoryProfilingBenchmark)
    assert _resolve_optimization_goal(benchmark.get_optimization_goal(), "speed") == "memory"
    assert _resolve_optimization_goal(None, "memory") == "memory"
    assert _resolve_optimization_goal(None, None) == "speed"


def test_memory_pair_matches_all_gradients_on_repeated_iterations():
    class CpuBaseline(BaselineMemoryProfilingBenchmark):
        allow_cpu = True

    class CpuOptimized(OptimizedMemoryProfilingBenchmark):
        allow_cpu = True

    torch.manual_seed(42)
    baseline = CpuBaseline()
    optimized = CpuOptimized()
    baseline.model = SimpleModel(hidden_dim=16)
    optimized.model = OptimizedModel(hidden_dim=16)
    optimized.model.load_state_dict(baseline.model.state_dict())
    for benchmark in (baseline, optimized):
        benchmark.inputs = torch.randn(4, 16)
        benchmark.targets = torch.randn(4, 16)
        benchmark.criterion = torch.nn.MSELoss()
        benchmark.output = None
    optimized.inputs = baseline.inputs.clone()
    optimized.targets = baseline.targets.clone()
    first = None
    for _ in range(3):
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.testing.assert_close(baseline.output, optimized.output)
        gradients = torch.cat([p.grad.flatten() for p in baseline.model.parameters()])
        candidate = torch.cat([p.grad.flatten() for p in optimized.model.parameters()])
        torch.testing.assert_close(gradients, candidate)
        if first is None:
            first = gradients.clone()
        else:
            torch.testing.assert_close(gradients, first, rtol=0, atol=0)


@pytest.mark.parametrize("with_bias", [True, False])
def test_pool_projection_fuses_bias_and_reuses_full_output_storage(with_bias):
    torch.manual_seed(13)
    layer = SimpleAttentionLayer(16, 4, 4, dtype=torch.float32).eval()
    if not with_bias:
        layer.qkv.bias = None
    layer.prepare_inference()
    inputs = torch.randn(2, 1, 16)
    with torch.inference_mode():
        expected = layer.qkv(inputs)
        actual = layer._project_qkv(inputs)
        torch.testing.assert_close(actual, expected)
        pointer = actual.data_ptr()
        next_input = inputs * 2
        next_expected = layer.qkv(next_input)
        next_actual = layer._project_qkv(next_input)
        torch.testing.assert_close(next_actual, next_expected)
        assert next_actual.data_ptr() == pointer
