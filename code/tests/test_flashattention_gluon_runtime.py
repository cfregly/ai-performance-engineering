from __future__ import annotations

import importlib.util

import pytest
import torch

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _require_triton() -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton is required by the FlashAttention lab runtime")


def _shrink_workload(bench) -> None:
    bench.batch = 1
    bench.batch_size = 1
    bench.seq_len = 32
    bench.heads = 2
    bench.head_dim = 16
    bench.hidden_dim = bench.heads * bench.head_dim


def _assert_deferred_verification_clone(bench) -> None:
    bench.benchmark_fn()
    assert bench.output is not None
    assert bench.output.dtype == torch.float16
    output_ptr = bench.output.data_ptr()

    bench.capture_verification_payload()
    payload = bench._verification_payload
    assert payload.output.dtype == torch.float32
    assert payload.output.data_ptr() != output_ptr
    assert payload.output.data_ptr() == bench._verify_output_buffer.data_ptr()
    inputs = bench.inputs
    scores = torch.matmul(inputs.q.float(), inputs.k.float().transpose(-1, -2))
    scores *= inputs.q.shape[-1] ** -0.5
    reference = torch.matmul(torch.softmax(scores, dim=-1), inputs.v.float())
    torch.testing.assert_close(payload.output, reference, rtol=2e-2, atol=1e-2)


@CUDA_REQUIRED
def test_baseline_flashattention_gluon_defers_output_clone_to_capture() -> None:
    _require_triton()
    from labs.flashattention_gluon.baseline_flashattention_gluon import (
        BaselineFlashAttentionGluonBenchmark,
    )

    bench = BaselineFlashAttentionGluonBenchmark()
    _shrink_workload(bench)
    bench.setup()
    try:
        _assert_deferred_verification_clone(bench)
    finally:
        bench.teardown()


@CUDA_REQUIRED
def test_optimized_flashattention_gluon_defers_output_clone_to_capture() -> None:
    _require_triton()
    from labs.flashattention_gluon.optimized_flashattention_gluon import (
        OptimizedFlashAttentionGluonBenchmark,
    )

    bench = OptimizedFlashAttentionGluonBenchmark()
    _shrink_workload(bench)
    bench.setup()
    try:
        assert bench.kernel.provider == "triton_tiled_attention"
        assert bench._output_buffer is not None
        output_buffer_ptr = bench._output_buffer.data_ptr()
        _assert_deferred_verification_clone(bench)
        assert bench.output is not None
        assert bench.output.data_ptr() == output_buffer_ptr
    finally:
        bench.teardown()
