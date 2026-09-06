"""Execute attention tensor paths on CPU; full CUDA setup is tested on B200."""

import pytest
import torch

from ch16.baseline_dense_attention_flash import BaselineDenseAttentionFlashBenchmark
from ch16.optimized_dense_attention_flash import OptimizedDenseAttentionFlashBenchmark
from ch16.optimized_dense_attention_flash_blackwell_variant import DenseAttentionFlashBlackwellVariantBenchmark


@pytest.mark.parametrize("benchmark_class", [
    OptimizedDenseAttentionFlashBenchmark,
    DenseAttentionFlashBlackwellVariantBenchmark,
])
def test_flash_head_merge_matches_full_naive_attention_and_reuses_storage(benchmark_class):
    torch.manual_seed(16)
    # Exercise the production tensor method without its CUDA-only constructor.
    baseline = object.__new__(BaselineDenseAttentionFlashBenchmark)
    optimized = object.__new__(benchmark_class)
    batch, seq, hidden, heads = 2, 17, 16, 4
    qkv = torch.nn.Linear(hidden, hidden * 3, bias=False)
    out = torch.nn.Linear(hidden, hidden, bias=False)
    for benchmark in (baseline, optimized):
        benchmark.hidden_dim = hidden
        benchmark.num_heads = heads
        benchmark.head_dim = hidden // heads
        benchmark.qkv_proj = qkv
        benchmark.out_proj = out
    baseline._causal_mask = torch.ones(seq, seq, dtype=torch.bool).triu(1)
    optimized._qkv_weight_t = qkv.weight.t()
    optimized._out_proj_weight_t = out.weight.t()
    optimized._qkv_buffer = torch.empty(batch, seq, hidden * 3)
    optimized._attn_merge_buffer = torch.empty(batch, seq, hidden)
    optimized._output_buffer = torch.empty(batch, seq, hidden)
    output_pointer = optimized._output_buffer.data_ptr()
    with torch.inference_mode():
        for _ in range(3):
            baseline.inputs = torch.randn(batch, seq, hidden)
            optimized.inputs = baseline.inputs.clone()
            expected = baseline._forward_naive()
            actual = optimized._forward_flash()
            assert actual.shape == (batch, seq, hidden)
            assert actual.data_ptr() == output_pointer
            torch.testing.assert_close(actual, expected)
