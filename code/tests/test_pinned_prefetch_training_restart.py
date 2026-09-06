"""Real training lifecycle checks; CUDA prefetch coverage requires actual CUDA."""

import pytest
import torch

from ch03.baseline_pinned_prefetch_mlp import BaselinePinnedPrefetchMLPBenchmark
from ch03.optimized_pinned_prefetch_mlp import OptimizedPinnedPrefetchMLPBenchmark


def _small(benchmark, device):
    benchmark.device = torch.device(device)
    benchmark.input_dim = 7
    benchmark.hidden_dim = 5
    benchmark.output_dim = 3
    benchmark.batch_size = 4
    benchmark.num_batches = 3
    return benchmark


def test_baseline_restarts_training_from_first_batch(monkeypatch):
    # Only suppress CUDA lifecycle hooks for this actual CPU tensor execution.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    benchmark = _small(BaselinePinnedPrefetchMLPBenchmark(), "cpu")
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        original_input = benchmark._payload_x.clone()
        original_output = benchmark.output.clone()
        benchmark.benchmark_fn()
        benchmark.teardown()
        benchmark.setup()
        benchmark.benchmark_fn()
        torch.testing.assert_close(benchmark._payload_x, original_input, rtol=0, atol=0)
        torch.testing.assert_close(benchmark.output, original_output, rtol=0, atol=0)
        assert len(benchmark.host_batches) == benchmark.num_batches
    finally:
        benchmark.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA prefetch required")
def test_real_cuda_prefetch_matches_blocking_training_across_restarts():
    baseline = _small(BaselinePinnedPrefetchMLPBenchmark(), "cuda")
    optimized = _small(OptimizedPinnedPrefetchMLPBenchmark(), "cuda")
    try:
        for steps in (2, 5):
            baseline.setup()
            optimized.setup()
            for _ in range(steps):
                baseline.benchmark_fn()
                optimized.benchmark_fn()
                torch.cuda.synchronize()
                torch.testing.assert_close(baseline._payload_x, optimized._payload_inputs, rtol=0, atol=0)
                torch.testing.assert_close(baseline._payload_y, optimized._payload_targets, rtol=0, atol=0)
                torch.testing.assert_close(baseline.output, optimized.output, rtol=1e-5, atol=1e-5)
                for left, right in zip(baseline.model.parameters(), optimized.model.parameters(), strict=True):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
            baseline.teardown()
            optimized.teardown()
    finally:
        baseline.teardown()
        optimized.teardown()
