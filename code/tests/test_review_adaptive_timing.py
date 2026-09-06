"""Real CUDA work must satisfy sampling duration or the explicit safety cap."""

import pytest
import torch

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness


@pytest.mark.parametrize("kwargs", [
    {"max_adaptive_iterations": 0},
    {"max_adaptive_iterations": True},
    {"min_total_duration_ms": -1},
    {"min_total_duration_ms": float("nan")},
    {"min_total_duration_ms": float("inf")},
])
def test_adaptive_configuration_rejects_impossible_sampling_contract(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(adaptive_iterations=True, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA timing")
def test_cuda_adaptive_cap_rejects_an_impossible_sample_minimum():
    config = BenchmarkConfig(
        device=torch.device("cuda"), iterations=10, max_adaptive_iterations=3,
        use_subprocess=False, enable_profiling=False,
    )
    harness = BenchmarkHarness(config=config)
    value = torch.zeros(1, device="cuda")
    with pytest.raises(ValueError, match="must be >= iterations"):
        harness._benchmark_custom(lambda: value.add_(1), config)
    assert value.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA timing")
@pytest.mark.parametrize("iterations,duration,cap", [(10, 0.001, 20), (1, 1000000.0, 3)])
def test_adaptive_sampling_respects_requested_minimum_and_safety_cap(iterations, duration, cap):
    value = torch.zeros(1024, device="cuda")
    config = BenchmarkConfig(
        device=torch.device("cuda"), iterations=iterations, warmup=5,
        use_subprocess=False, enable_profiling=False,
        adaptive_iterations=True, min_total_duration_ms=duration,
        max_adaptive_iterations=cap,
    )
    harness = BenchmarkHarness(config=config)
    samples, _ = harness._benchmark_custom(lambda: value.add_(1), config)
    assert len(samples) <= cap
    assert len(samples) >= iterations
    assert sum(samples) >= duration or len(samples) == cap
    torch.testing.assert_close(value, torch.full_like(value, len(samples)), rtol=0, atol=0)
