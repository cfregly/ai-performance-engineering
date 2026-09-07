from __future__ import annotations

import pytest
import torch

from labs.nanochat_fullstack.baseline_nanochat_inference import (
    BaselineNanochatInferenceBenchmark,
)
from labs.nanochat_fullstack.optimized_nanochat_inference import (
    OptimizedNanochatInferenceBenchmark,
)


def test_nanochat_pair_rejects_degenerate_logits() -> None:
    for benchmark_type in (
        BaselineNanochatInferenceBenchmark,
        OptimizedNanochatInferenceBenchmark,
    ):
        benchmark = benchmark_type()
        benchmark.output = torch.zeros(2, 1, 8)
        assert benchmark.validate_result() == "benchmark_fn() produced degenerate all-zero logits"
        benchmark.output[0, 0, 0] = 1
        assert benchmark.validate_result() is None


def test_nanochat_requires_setup_before_replay() -> None:
    benchmark = OptimizedNanochatInferenceBenchmark()
    with pytest.raises(RuntimeError, match=r"setup\(\) must run"):
        benchmark.benchmark_fn()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph execution requires CUDA")
@pytest.mark.parametrize("seed", [42, 1042])
def test_nanochat_graph_replays_complete_requests_with_changed_inputs(seed: int) -> None:
    baseline = BaselineNanochatInferenceBenchmark()
    optimized = OptimizedNanochatInferenceBenchmark()
    try:
        for benchmark in (baseline, optimized):
            benchmark.batch_size = 2
            benchmark.prompt_len = 16
            benchmark.decode_len = 4
            benchmark.vocab_size = 64
            benchmark.n_layer = 2
            benchmark.n_head = benchmark.n_kv_head = 2
            benchmark.n_embd = 64
            torch.manual_seed(seed)
            benchmark.setup()
            assert torch.initial_seed() == seed
        assert torch.equal(baseline.prompt, optimized.prompt)
        assert torch.equal(baseline.decode_tokens, optimized.decode_tokens)
        for original, captured in zip(
            baseline.model.parameters(), optimized.model.parameters(), strict=True
        ):
            assert torch.equal(original, captured)

        prompt = baseline.prompt.clone()
        decode = baseline.decode_tokens.clone()
        original_output = None
        output_pointer = None
        for case in ("original", "changed_prompt", "early_decode", "original_again"):
            for benchmark in (baseline, optimized):
                benchmark.prompt.copy_(prompt)
                benchmark.decode_tokens.copy_(decode)
                if case == "changed_prompt":
                    benchmark.prompt.add_(1).remainder_(benchmark.vocab_size)
                elif case == "early_decode":
                    benchmark.decode_tokens[:, 0].add_(1).remainder_(benchmark.vocab_size)
                benchmark.benchmark_fn()
                benchmark.capture_verification_payload()
                assert benchmark.validate_result() is None
                assert torch.equal(benchmark.output, benchmark._verify_output_buffer)
            assert torch.equal(baseline.output, optimized.output)
            assert optimized.output.numel() == 128
            if original_output is None:
                original_output = optimized.output.clone()
                output_pointer = optimized.output.data_ptr()
            elif case == "original_again":
                assert torch.equal(original_output, optimized.output)
            else:
                assert not torch.equal(original_output, optimized.output)
            assert optimized.output.data_ptr() == output_pointer
        assert optimized.capture_ms > 0
    finally:
        optimized.teardown()
        baseline.teardown()
