import torch
import pytest

from ch15.speculative_decoding_common import TokenMLP as ChapterTokenMLP
from labs.speculative_decode.baseline_speculative_decode_trusted import (
    BaselineSpeculativeDecodeTrustedBenchmark,
)
from labs.speculative_decode.baseline_speculative_decode_transition_table import (
    BaselineSpeculativeDecodeTransitionTableBenchmark,
)
from labs.speculative_decode.optimized_speculative_decode_trusted import (
    OptimizedSpeculativeDecodeTrustedBenchmark,
)
from labs.speculative_decode.optimized_speculative_decode_transition_table import (
    OptimizedSpeculativeDecodeTransitionTableBenchmark,
)
from labs.speculative_decode.speculative_decode_common import SpecDecodeWorkload
from labs.speculative_decode.speculative_decode_common import TokenMLP as LabTokenMLP


def _assert_forward_into_matches_forward(model_cls) -> None:
    torch.manual_seed(1234)
    model = model_cls(
        vocab_size=17,
        hidden_size=8,
        num_layers=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    logits_out = torch.empty(
        (token_ids.size(0), token_ids.size(1), model.vocab_size),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = model(token_ids)
        actual = model.forward_into(token_ids, logits_out)

    assert actual.data_ptr() == logits_out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_lab_token_mlp_forward_into_matches_forward() -> None:
    _assert_forward_into_matches_forward(LabTokenMLP)


def test_ch15_token_mlp_forward_into_matches_forward() -> None:
    _assert_forward_into_matches_forward(ChapterTokenMLP)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for trusted speculative decode")
def test_lab_trusted_speculative_decode_matches_verified_decode() -> None:
    workload = SpecDecodeWorkload(
        vocab_size=512,
        target_hidden=128,
        target_layers=1,
        draft_hidden=32,
        speculative_k=8,
        total_tokens=16,
        tail_scale=1e-8,
        dtype=torch.float32,
    )
    baseline = BaselineSpeculativeDecodeTrustedBenchmark()
    optimized = OptimizedSpeculativeDecodeTrustedBenchmark()
    baseline.workload = workload
    optimized.workload = workload
    try:
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.cuda.synchronize()

        assert torch.equal(baseline.output, optimized.output)
        baseline_metrics = baseline.get_custom_metrics()
        optimized_metrics = optimized.get_custom_metrics()
        assert baseline_metrics["speculative.target_verify_calls"] == 2.0
        assert baseline_metrics["speculative.trusted_draft"] == 0.0
        assert optimized_metrics["speculative.target_verify_calls"] == 0.0
        assert optimized_metrics["speculative.trusted_draft"] == 1.0
        assert optimized_metrics["speculative.acceptance_rate_pct"] == 100.0
    finally:
        baseline.teardown()
        optimized.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for transition-table speculative decode")
def test_lab_transition_table_speculative_decode_matches_trusted_draft() -> None:
    workload = SpecDecodeWorkload(
        vocab_size=512,
        target_hidden=128,
        target_layers=1,
        draft_hidden=32,
        speculative_k=8,
        total_tokens=16,
        tail_scale=1e-8,
        dtype=torch.float32,
    )
    baseline = BaselineSpeculativeDecodeTransitionTableBenchmark()
    optimized = OptimizedSpeculativeDecodeTransitionTableBenchmark()
    baseline.workload = workload
    optimized.workload = workload
    try:
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.cuda.synchronize()

        assert torch.equal(baseline.output, optimized.output)
        baseline_metrics = baseline.get_custom_metrics()
        optimized_metrics = optimized.get_custom_metrics()
        assert baseline_metrics["speculative.draft_model_calls"] == 16.0
        assert baseline_metrics["speculative.transition_table"] == 0.0
        assert optimized_metrics["speculative.draft_model_calls"] == 0.0
        assert optimized_metrics["speculative.transition_table"] == 1.0
        assert optimized_metrics["speculative.target_verify_calls"] == 0.0
    finally:
        baseline.teardown()
        optimized.teardown()
