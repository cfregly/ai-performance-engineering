from __future__ import annotations

import pytest
import torch

from labs.moe_cuda.baseline_decode_attention import BaselineDecodeAttentionBenchmark
from labs.moe_cuda.optimized_decode_attention import OptimizedDecodeAttentionBenchmark

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _shrink_decode_workload(bench) -> None:
    bench.batch = 2
    bench.num_heads = 2
    bench.kv_seq = 16
    bench.head_dim = 16


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("benchmark_cls", "output_dtype"),
    [
        (BaselineDecodeAttentionBenchmark, torch.float32),
        (OptimizedDecodeAttentionBenchmark, torch.bfloat16),
    ],
)
def test_moe_cuda_decode_attention_reuses_timing_and_meta(
    benchmark_cls: type,
    output_dtype: torch.dtype,
) -> None:
    bench = benchmark_cls()
    _shrink_decode_workload(bench)
    bench.setup()
    try:
        meta = bench._payload_meta
        assert meta is not None

        bench.benchmark_fn()
        torch.cuda.synchronize(bench.device)

        timing_pair = bench._timing_pair
        assert timing_pair is not None
        assert bench._pending_timing_pair is timing_pair
        assert bench._payload_meta is meta
        assert bench.output is not None
        assert bench.output.dtype == output_dtype

        bench.capture_verification_payload()
        payload = bench._verification_payload
        assert payload.inputs["meta"] is meta
        assert payload.output.dtype == torch.float32
        assert payload.output.data_ptr() != bench.output.data_ptr()

        bench.benchmark_fn()
        torch.cuda.synchronize(bench.device)
        assert bench._timing_pair is timing_pair
        assert bench._payload_meta is meta
    finally:
        bench.teardown()
