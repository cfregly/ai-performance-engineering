"""W2-055: the Chapter 14 sparse FlexAttention pair must time the same window.

The optimized side attended 128 keys per query while the dense baseline
attended 129. Verification sampled only the first 128 query rows, where both
masks agree, and used a (0.1, 1.0) tolerance on outputs with a standard
deviation near 0.03, so it could not see the difference.

The mask tests use the benchmark's full shapes. The CPU payload tests run the
real module forwards and payload capture with the benchmark's heads, head size
and window, but a 1024-token sequence so the dense CPU baseline stays fast.
Inductor cannot compile FlexAttention for CPU on this platform, so the CPU
tests run the optimized module with eager FlexAttention. The CUDA test runs
both benchmarks end to end at their full shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from ch14.baseline_flex_attention_sparse import (
    SPARSE_ATTENTION_OUTPUT_TOLERANCE,
    BaselineFlexAttentionSparseBenchmark,
    DenseMaskedSlidingWindowAttention,
    dense_sliding_window_causal_mask,
    verification_row_slice,
)
from ch14.optimized_flex_attention_sparse import (
    FlexAttentionSparseBenchmark,
    SlidingWindowCausalAttention,
    _causal_window_key_pairs,
    block_sparse_mask,
    sliding_window_causal_mask,
)
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verify_runner import VerifyRunner

PRE_FIX_TOLERANCE = (0.1, 1.0)
CPU = torch.device("cpu")
CPU_SEQ_LEN = 1024
DTYPE = torch.bfloat16


def _pre_fix_baseline_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """The baseline mask before this fix, which admits window_size + 1 keys."""
    idx = torch.arange(seq_len)
    q_idx, kv_idx = idx[:, None], idx[None, :]
    return ((q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size))[None, None]


def _captured_baseline(seed: int) -> BaselineFlexAttentionSparseBenchmark:
    torch.manual_seed(seed)
    bench = BaselineFlexAttentionSparseBenchmark()
    bench.device = CPU
    bench.dtype = DTYPE
    bench.seq_len = CPU_SEQ_LEN
    # Same state that setup() builds, without its CUDA requirement.
    bench.model = DenseMaskedSlidingWindowAttention(
        embed_dim=bench.embed_dim,
        num_heads=bench.num_heads,
        window_size=bench.window_size,
    ).to(bench.device, dtype=bench.dtype)
    bench.parameter_count = sum(p.numel() for p in bench.model.parameters())
    bench.allowed_mask = dense_sliding_window_causal_mask(bench.seq_len, bench.window_size, bench.device)
    bench.x = torch.randn(bench.batch_size, bench.seq_len, bench.embed_dim, device=bench.device, dtype=bench.dtype)
    bench._verify_output_buffer = torch.empty(
        bench.batch_size,
        min(128, bench.seq_len),
        min(256, bench.embed_dim),
        device=bench.device,
        dtype=torch.float32,
    )
    bench._verify_rows = verification_row_slice(bench.seq_len, bench._verify_output_buffer.shape[1])
    bench.benchmark_fn()
    bench.capture_verification_payload()
    return bench


def _captured_optimized(seed: int) -> FlexAttentionSparseBenchmark:
    torch.manual_seed(seed)
    bench = FlexAttentionSparseBenchmark()
    bench.device = CPU
    bench.seq_len = CPU_SEQ_LEN
    embed_dim = bench.num_heads * bench.head_dim
    # Same state that setup() builds, without its CUDA requirement.
    bench.attn = SlidingWindowCausalAttention(
        embed_dim=embed_dim,
        num_heads=bench.num_heads,
        window_size=bench.window_size,
    ).to(bench.device, dtype=DTYPE)
    bench.attn._compiled_flex = flex_attention
    bench.attn.cache_weight_views()
    bench.model = bench.attn
    bench.parameter_count = sum(p.numel() for p in bench.attn.parameters())
    bench.x = torch.randn(bench.batch_size, bench.seq_len, embed_dim, device=bench.device, dtype=DTYPE)
    bench._verify_output_buffer = torch.empty(
        bench.batch_size,
        min(128, bench.seq_len),
        min(256, bench.hidden_dim),
        device=bench.device,
        dtype=torch.float32,
    )
    bench._verify_rows = verification_row_slice(bench.seq_len, bench._verify_output_buffer.shape[1])
    bench.benchmark_fn()
    bench.capture_verification_payload()
    return bench


@pytest.fixture
def runner(tmp_path: Path) -> VerifyRunner:
    return VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path),
    )


def test_pair_shares_shape_and_window() -> None:
    baseline = BaselineFlexAttentionSparseBenchmark()
    optimized = FlexAttentionSparseBenchmark()

    for name in ("batch_size", "num_heads", "head_dim", "seq_len", "window_size"):
        assert getattr(baseline, name) == getattr(optimized, name), name


def test_optimized_block_mask_admits_exactly_the_baseline_keys() -> None:
    baseline = BaselineFlexAttentionSparseBenchmark()
    seq_len, window = baseline.seq_len, baseline.window_size
    dense = dense_sliding_window_causal_mask(seq_len, window, CPU)[0, 0]

    attn = SlidingWindowCausalAttention(
        embed_dim=baseline.embed_dim,
        num_heads=baseline.num_heads,
        window_size=window,
    )
    block_mask = attn._get_block_mask(baseline.batch_size, seq_len, CPU)
    idx = torch.arange(seq_len)
    zero = torch.zeros((), dtype=torch.int64)
    flex_allowed = block_mask.mask_mod(zero, zero, idx[:, None], idx[None, :])

    assert torch.equal(flex_allowed, dense)
    keys_per_query = dense.sum(dim=-1)
    assert int(keys_per_query[window:].min()) == window
    assert int(keys_per_query[window:].max()) == window
    # get_custom_metrics() counts FLOPs from this helper.
    assert int(dense.sum()) == _causal_window_key_pairs(seq_len, window)


def test_demo_mask_factories_build_block_masks() -> None:
    seq_len = 512
    sliding = create_block_mask(
        sliding_window_causal_mask(128), B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )
    sparse = create_block_mask(
        block_sparse_mask(128, 0.5), B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=CPU
    )
    idx = torch.arange(seq_len)
    zero = torch.zeros((), dtype=torch.int64)

    assert torch.equal(
        sliding.mask_mod(zero, zero, idx[:, None], idx[None, :]),
        dense_sliding_window_causal_mask(seq_len, 128, CPU)[0, 0],
    )
    sparse_allowed = sparse.mask_mod(zero, zero, idx[:, None], idx[None, :])
    q_block, kv_block = idx[:, None] // 128, idx[None, :] // 128
    assert torch.equal(sparse_allowed, (q_block == kv_block) | (kv_block % 2 == 0))


def test_verification_rows_reach_steady_state_queries() -> None:
    baseline = BaselineFlexAttentionSparseBenchmark()
    rows = torch.arange(baseline.seq_len)[verification_row_slice(baseline.seq_len, 128)]

    assert rows.numel() == 128
    assert int(rows[-1]) == baseline.seq_len - 1
    assert int((rows >= baseline.window_size).sum()) > 100
    assert torch.arange(10)[verification_row_slice(10, 10)].tolist() == list(range(10))
    with pytest.raises(ValueError):
        verification_row_slice(64, 128)


@pytest.mark.parametrize("seed", [0, 42])
def test_payloads_match_for_the_shared_window(runner: VerifyRunner, seed: int) -> None:
    baseline = _captured_baseline(seed)
    optimized = _captured_optimized(seed)

    assert torch.equal(baseline.model.qkv_proj.weight, optimized.attn.qkv_proj.weight)
    assert torch.equal(baseline.model.out_proj.weight, optimized.attn.out_proj.weight)
    assert torch.equal(baseline.get_verify_inputs()["input"], optimized.get_verify_inputs()["input"])
    assert baseline.get_output_tolerance() == SPARSE_ATTENTION_OUTPUT_TOLERANCE
    assert optimized.get_output_tolerance() == SPARSE_ATTENTION_OUTPUT_TOLERANCE
    expected = baseline.get_verify_output()
    actual = optimized.get_verify_output()
    torch.testing.assert_close(actual, optimized.output[:, verification_row_slice(optimized.seq_len, 128), :256].float())
    comparison = runner.compare_perf_outputs(expected, actual, SPARSE_ATTENTION_OUTPUT_TOLERANCE)
    assert comparison.passed, comparison


@pytest.mark.parametrize("seed", [0, 42])
def test_payload_rejects_off_by_one_window(runner: VerifyRunner, seed: int) -> None:
    baseline = _captured_baseline(seed)
    optimized = _captured_optimized(seed)
    baseline.allowed_mask = _pre_fix_baseline_mask(baseline.seq_len, baseline.window_size)
    baseline.benchmark_fn()
    baseline.capture_verification_payload()

    # The pre-fix payload sampled rows [0, 128), where both masks agree, so
    # the pre-fix check accepted the mismatched pair.
    head_rows = slice(0, 128)
    assert runner.compare_perf_outputs(
        baseline.output[:, head_rows, :256].float(),
        optimized.output[:, head_rows, :256].float(),
        PRE_FIX_TOLERANCE,
    ).passed
    assert runner.compare_perf_outputs(
        baseline.get_verify_output(), optimized.get_verify_output(), PRE_FIX_TOLERANCE
    ).passed
    assert not runner.compare_perf_outputs(
        baseline.get_verify_output(),
        optimized.get_verify_output(),
        SPARSE_ATTENTION_OUTPUT_TOLERANCE,
    ).passed


def test_payload_tolerance_rejects_zero_output(runner: VerifyRunner) -> None:
    baseline = _captured_baseline(0)
    expected = baseline.get_verify_output()

    assert runner.compare_perf_outputs(expected, torch.zeros_like(expected), PRE_FIX_TOLERANCE).passed
    assert not runner.compare_perf_outputs(
        expected, torch.zeros_like(expected), SPARSE_ATTENTION_OUTPUT_TOLERANCE
    ).passed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SKIPPED: CUDA required for compiled FlexAttention")
def test_cuda_pair_payloads_match(runner: VerifyRunner) -> None:
    outputs: dict[str, torch.Tensor] = {}
    tolerances: dict[str, tuple[float, float]] = {}
    for bench_cls in (BaselineFlexAttentionSparseBenchmark, FlexAttentionSparseBenchmark):
        torch.manual_seed(42)
        bench = bench_cls()
        bench.setup()
        try:
            bench.benchmark_fn()
            torch.cuda.synchronize(bench.device)
            bench.capture_verification_payload()
            outputs[bench_cls.__name__] = bench.get_verify_output()
            tolerances[bench_cls.__name__] = bench.get_output_tolerance()
        finally:
            bench.teardown()

    assert set(tolerances.values()) == {SPARSE_ATTENTION_OUTPUT_TOLERANCE}
    comparison = runner.compare_perf_outputs(
        outputs[BaselineFlexAttentionSparseBenchmark.__name__],
        outputs[FlexAttentionSparseBenchmark.__name__],
        SPARSE_ATTENTION_OUTPUT_TOLERANCE,
    )
    assert comparison.passed, comparison
