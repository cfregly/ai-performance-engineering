from dataclasses import replace

import torch
import pytest

from ch15.speculative_decoding_benchmarks import SpeculativeDecodingBenchmark
from ch15.speculative_decoding_common import TRITON_AVAILABLE as CHAPTER_TRITON_AVAILABLE
from ch15.speculative_decoding_common import TokenMLP as ChapterTokenMLP
from ch15.speculative_decoding_common import accept_prefix_length as chapter_accept_prefix_length
from labs.speculative_decode.baseline_speculative_decode import (
    BaselineSpeculativeDecodeBenchmark,
)
from labs.speculative_decode.baseline_speculative_decode_trusted import (
    BaselineSpeculativeDecodeTrustedBenchmark,
)
from labs.speculative_decode.baseline_speculative_decode_transition_table import (
    BaselineSpeculativeDecodeTransitionTableBenchmark,
)
from labs.speculative_decode.optimized_speculative_decode_trusted import (
    OptimizedSpeculativeDecodeTrustedBenchmark,
)
from labs.speculative_decode.optimized_speculative_decode import (
    OptimizedSpeculativeDecodeBenchmark,
)
from labs.speculative_decode.optimized_speculative_decode_transition_table import (
    OptimizedSpeculativeDecodeTransitionTableBenchmark,
)
from labs.speculative_decode.speculative_decode_common import SpecDecodeWorkload
from labs.speculative_decode.speculative_decode_common import TRITON_AVAILABLE as LAB_TRITON_AVAILABLE
from labs.speculative_decode.speculative_decode_common import TokenMLP as LabTokenMLP
from labs.speculative_decode.speculative_decode_common import accept_prefix_length as lab_accept_prefix_length


def _assert_accept_prefix_lengths(helper, device: torch.device) -> None:
    cases = [
        ([], 0),
        ([True, True, True, True], 4),
        ([True, True, False, True], 2),
        ([False, True, True, True], 0),
    ]
    for values, expected in cases:
        matches = torch.tensor([values], device=device, dtype=torch.bool)
        out = torch.empty((), device=device, dtype=torch.int64)
        helper(matches, out)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        assert int(out.detach().cpu()) == expected


def test_accept_prefix_length_helpers_match_expected_on_cpu() -> None:
    _assert_accept_prefix_lengths(chapter_accept_prefix_length, torch.device("cpu"))
    _assert_accept_prefix_lengths(lab_accept_prefix_length, torch.device("cpu"))


@pytest.mark.skipif(
    not torch.cuda.is_available() or not (CHAPTER_TRITON_AVAILABLE and LAB_TRITON_AVAILABLE),
    reason="requires CUDA and Triton",
)
def test_accept_prefix_length_helpers_match_expected_on_cuda() -> None:
    _assert_accept_prefix_lengths(chapter_accept_prefix_length, torch.device("cuda"))
    _assert_accept_prefix_lengths(lab_accept_prefix_length, torch.device("cuda"))


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


def test_lab_token_mlp_forward_into_prepared_matches_forward() -> None:
    torch.manual_seed(1234)
    model = LabTokenMLP(
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
    buffers = model.prepare_forward_buffers(
        token_ids.numel(),
        device=token_ids.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = model(token_ids)
        actual = model.forward_into_prepared(token_ids, logits_out, buffers)

    assert actual.data_ptr() == logits_out.data_ptr()
    torch.testing.assert_close(actual, expected)

    unchecked_out = torch.empty_like(logits_out)
    with torch.inference_mode():
        unchecked = model.forward_into_prepared_unchecked(token_ids, unchecked_out, buffers)

    assert unchecked.data_ptr() == unchecked_out.data_ptr()
    torch.testing.assert_close(unchecked, expected)


def test_ch15_token_mlp_forward_into_matches_forward() -> None:
    _assert_forward_into_matches_forward(ChapterTokenMLP)


def test_ch15_token_mlp_forward_into_prepared_unchecked_matches_forward() -> None:
    torch.manual_seed(1234)
    model = ChapterTokenMLP(
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
    buffers = model.prepare_forward_buffers(
        token_ids.numel(),
        device=token_ids.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected = model(token_ids)
        actual = model.forward_into_prepared_unchecked(token_ids, logits_out, buffers)

    assert actual.data_ptr() == logits_out.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_ch15_speculative_decode_fallback_matches_greedy_when_draft_rejects() -> None:
    workload_overrides = {
        "vocab_size": 64,
        "target_hidden": 32,
        "target_layers": 1,
        "draft_hidden": 8,
        "speculative_k": 4,
        "total_tokens": 12,
        "tail_scale": 1.0,
        "dtype": torch.float32,
    }
    outputs = []
    metrics = []
    for use_speculative in (False, True):
        bench = SpeculativeDecodingBenchmark(use_speculative=use_speculative, label="fallback_check")
        bench.workload = replace(bench.workload, **workload_overrides)
        try:
            bench.setup()
            bench.benchmark_fn()
            assert bench.output is not None
            outputs.append(bench.output.detach().cpu().clone())
            metrics.append(bench.get_custom_metrics())
        finally:
            bench.teardown()

    assert torch.equal(outputs[0], outputs[1])
    assert metrics[1] is not None
    assert metrics[1]["speculative.acceptance_rate_pct"] < 100.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for lab speculative decode")
def test_lab_speculative_decode_matches_greedy_when_draft_rejects() -> None:
    workload = SpecDecodeWorkload(
        vocab_size=128,
        target_hidden=64,
        target_layers=1,
        draft_hidden=16,
        speculative_k=4,
        total_tokens=12,
        tail_scale=1.0,
        dtype=torch.float32,
    )
    baseline = BaselineSpeculativeDecodeBenchmark()
    optimized = OptimizedSpeculativeDecodeBenchmark()
    baseline.workload = workload
    optimized.workload = workload
    try:
        baseline.setup()
        optimized.setup()
        baseline.benchmark_fn()
        optimized.benchmark_fn()
        torch.cuda.synchronize()

        assert torch.equal(baseline.output, optimized.output)
        optimized_metrics = optimized.get_custom_metrics()
        assert optimized_metrics["speculative.acceptance_rate_pct"] < 100.0
    finally:
        baseline.teardown()
        optimized.teardown()


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
