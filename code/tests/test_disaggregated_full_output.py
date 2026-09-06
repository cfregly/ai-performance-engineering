"""Execute both inference phases; verification must retain every phase output."""

import torch

from ch04.baseline_disaggregated import BaselineDisaggregatedBenchmark
from ch04.optimized_disaggregated import OptimizedDisaggregatedBenchmark


def test_full_prefill_and_decode_outputs_match_and_refresh(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    benchmarks = [BaselineDisaggregatedBenchmark(), OptimizedDisaggregatedBenchmark()]
    try:
        for benchmark in benchmarks:
            benchmark.device = torch.device("cpu")
            benchmark.batch_size = 1
            benchmark.prefill_len = 3
            benchmark.setup()
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
        left, right = [benchmark.get_verify_output() for benchmark in benchmarks]
        assert left.shape == right.shape == (1, 4, 256)
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
        original = left.clone()
        for benchmark in benchmarks:
            benchmark.prefill_input.add_(1)
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            actual = benchmark.get_verify_output()
            assert not torch.equal(actual[:, :3], original[:, :3])
            torch.testing.assert_close(actual[:, 3:], original[:, 3:], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            benchmarks[0].get_verify_output(), benchmarks[1].get_verify_output(),
            rtol=1e-5, atol=1e-5,
        )
    finally:
        for benchmark in benchmarks:
            benchmark.teardown()
