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


def test_regional_pair_verifies_latest_bucket_and_every_output_token():
    from ch13.baseline_regional_compile import (
        BaselineFullGraphCompileBenchmark,
        TinyTransformerBlock as FullBlock,
    )
    from ch13.optimized_regional_compile import (
        OptimizedRegionalCompileBenchmark,
        TinyTransformerBlock as RegionalBlock,
    )

    class CpuBaseline(BaselineFullGraphCompileBenchmark):
        allow_cpu = True

    class CpuOptimized(OptimizedRegionalCompileBenchmark):
        allow_cpu = True

    torch.manual_seed(42)
    baseline, optimized = CpuBaseline(), CpuOptimized()
    baseline.model = FullBlock(hidden=16, num_heads=4, mlp_hidden=32).eval()
    baseline.compiled_model = baseline.model
    optimized.model = RegionalBlock(hidden=16, num_heads=4, mlp_hidden=32).eval()
    optimized.model.load_state_dict(baseline.model.state_dict())
    inputs = {seq: torch.randn(2, seq, 16) for seq in (4, 129)}
    for benchmark in (baseline, optimized):
        benchmark.device = torch.device("cpu")
        benchmark.batch_size = 2
        benchmark.sequence_schedule = [4, 129]
        benchmark.inputs = {seq: x.clone() for seq, x in inputs.items()}
        benchmark._verify_output_buffer = torch.empty(2, 129, 16)
        assert benchmark.get_config().adaptive_iterations is False

    # Exercise the real model and benchmark/payload lifecycle at two bucket
    # sizes. CUDA compilation and speed remain separate GPU validation gates.
    for seq in (4, 129, 4, 129):
        for benchmark in (baseline, optimized):
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            torch.testing.assert_close(benchmark.get_verify_inputs()["input"], inputs[seq])
            actual = benchmark.get_verify_output()
            assert actual.shape == (2, seq, 16)
            torch.testing.assert_close(actual, benchmark.output)
        torch.testing.assert_close(baseline.get_verify_output(), optimized.get_verify_output())


def test_te_pair_declares_only_precision_signature_equivalence():
    from ch13.baseline_precisionfp8_te import BaselineTEFP8Benchmark
    from ch13.optimized_precisionfp8_te import OptimizedTEFP8Benchmark
    from core.benchmark.verification import get_signature_equivalence_spec

    baseline = get_signature_equivalence_spec(BaselineTEFP8Benchmark)
    optimized = get_signature_equivalence_spec(OptimizedTEFP8Benchmark)
    assert baseline == optimized
    assert baseline.group == "ch13_precisionfp8_te_precision"
    assert baseline.ignore_fields == ("precision_flags",)
    # Real output capture comes from VerificationPayloadMixin, never a legacy
    # accessor that returned the input unchanged.
    assert "get_output_for_verification" not in BaselineTEFP8Benchmark.__dict__
    assert "get_output_for_verification" not in OptimizedTEFP8Benchmark.__dict__


def test_pool_reuses_views_and_matches_every_prefix_across_requests():
    from ch13.optimized_kv_cache_naive_pool import OptimizedKVCache

    cache = OptimizedKVCache(23, 2, 2, 3, 4, torch.float32, torch.device("cpu"))
    torch.manual_seed(13)
    for request_number, length in enumerate((23, 7, 19)):
        request = str(request_number)
        cache.allocate(request)
        keys, values = [[], []], [[], []]
        for pos in range(length):
            for layer in range(2):
                k, v = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
                keys[layer].append(k)
                values[layer].append(v)
                cache.append(request, layer, k, v, pos)
                actual = cache.get(request, layer, 0, pos + 1)
                expected_k = torch.stack(keys[layer], dim=2)
                expected_v = torch.stack(values[layer], dim=2)
                torch.testing.assert_close(actual[0], expected_k, rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected_v, rtol=0, atol=0)
                assert cache.get(request, layer, 0, pos + 1) is actual
                tail_k, tail_v = cache.get(request, layer, 1, pos + 1)
                torch.testing.assert_close(tail_k, expected_k[:, :, 1:, :], rtol=0, atol=0)
                torch.testing.assert_close(tail_v, expected_v[:, :, 1:, :], rtol=0, atol=0)
        cache.free(request)
    assert len(cache.free_indices) == 2
