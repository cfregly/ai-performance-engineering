from __future__ import annotations

import pytest
import torch

import labs.flashattention_gluon.optimized_flashattention_gluon as optimized_module
from labs.flashattention_gluon.baseline_flashattention_gluon import (
    BaselineFlashAttentionGluonBenchmark,
)
from labs.flashattention_gluon.flashattention_gluon_common import FlashAttentionKernel
from labs.flashattention_gluon.optimized_flashattention_gluon import (
    OptimizedFlashAttentionGluonBenchmark,
)

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


@CUDA_REQUIRED
def test_baseline_flashattention_gluon_defers_output_clone_to_capture() -> None:
    bench = BaselineFlashAttentionGluonBenchmark()
    _shrink_workload(bench)
    bench.setup()
    try:
        _assert_deferred_verification_clone(bench)
    finally:
        bench.teardown()


@CUDA_REQUIRED
def test_optimized_flashattention_gluon_defers_output_clone_to_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_kernel(
        q: torch.Tensor,
        _k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert out is not None
        torch.add(q, v, out=out)
        return out

    monkeypatch.setattr(
        optimized_module,
        "resolve_gluon_flash_attention",
        lambda: FlashAttentionKernel(fn=fake_kernel, provider="fake"),
    )

    bench = OptimizedFlashAttentionGluonBenchmark()
    _shrink_workload(bench)
    bench.setup()
    try:
        assert bench._output_buffer is not None
        output_buffer_ptr = bench._output_buffer.data_ptr()
        _assert_deferred_verification_clone(bench)
        assert bench.output is not None
        assert bench.output.data_ptr() == output_buffer_ptr
    finally:
        bench.teardown()
