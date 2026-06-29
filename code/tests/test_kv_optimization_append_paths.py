from __future__ import annotations

import inspect

import pytest
import torch

from labs.kv_optimization.baseline_kv_standard import BaselineKVStandard
from labs.kv_optimization.optimized_kv_standard import OptimizedKVFP8Compressed

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for KV runtime checks")
FP8_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"),
    reason="CUDA with torch.float8_e4m3fn required for FP8 KV runtime checks",
)


def test_kv_standard_uses_host_seq_lengths_and_single_device_fill() -> None:
    for benchmark_cls in (BaselineKVStandard, OptimizedKVFP8Compressed):
        init_source = inspect.getsource(benchmark_cls.__init__)
        setup_source = inspect.getsource(benchmark_cls.setup)
        get_kv_source = inspect.getsource(benchmark_cls.get_kv)
        benchmark_source = inspect.getsource(benchmark_cls.benchmark_fn)
        teardown_source = inspect.getsource(benchmark_cls.teardown)

        assert "self._generated_step_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []" in init_source
        assert "self._output_view: Optional[torch.Tensor] = None" in init_source
        assert "self._active_layer_slice = slice(0, active_layers)" in init_source
        assert "self._generated_step_pairs = list(" in setup_source
        assert "zip(self._generated_k_steps, self._generated_v_steps, strict=True)" in setup_source
        if benchmark_cls is BaselineKVStandard:
            assert "self._generated_step_layer_view_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []" in init_source
            assert "self._generated_step_layer_view_pairs = [" in setup_source
            assert "(k_step.unsqueeze(1), v_step.unsqueeze(1))" in setup_source
        assert "self._output_view = self.kv_cache[:1, :1, :, :, :1, : min(8, self.head_dim)]" in setup_source
        assert "seq_len = self._seq_lengths_host[batch_idx]" in get_kv_source
        assert ".item()" not in get_kv_source
        assert "self.seq_lengths += 1" not in benchmark_source
        assert "self.seq_lengths.zero_()" not in benchmark_source
        if benchmark_cls is BaselineKVStandard:
            assert "for pos, (new_k_layer, new_v_layer) in enumerate(" in benchmark_source
            assert "self._generated_step_layer_view_pairs" in benchmark_source
            assert "self.append_active_layer_views(new_k_layer, new_v_layer, pos=pos)" in benchmark_source
        else:
            assert "for pos, (new_k, new_v) in enumerate(self._generated_step_pairs):" in benchmark_source
        assert "self.seq_lengths.fill_(num_decode_steps)" in benchmark_source
        assert "self._set_host_seq_lengths(0)" in benchmark_source
        assert "self._set_host_seq_lengths(num_decode_steps)" in benchmark_source
        assert "self.output = self._output_view" in benchmark_source
        assert "self._output_view.detach()" not in benchmark_source
        assert "new_k = self._generated_k_steps[pos]" not in benchmark_source
        assert "new_v = self._generated_v_steps[pos]" not in benchmark_source
        assert "self.kv_cache[:1, :1" not in benchmark_source
        assert "self._seq_lengths_host = [0] * self.batch_size" not in benchmark_source
        assert "self._seq_lengths_host = [num_decode_steps] * self.batch_size" not in benchmark_source
        assert "self._generated_step_pairs = []" in teardown_source
        if benchmark_cls is BaselineKVStandard:
            assert "self._generated_step_layer_view_pairs = []" in teardown_source
        assert "self._output_view = None" in teardown_source


def test_kv_standard_cache_allocation_avoids_zero_fill() -> None:
    for benchmark_cls in (BaselineKVStandard, OptimizedKVFP8Compressed):
        setup_source = inspect.getsource(benchmark_cls.setup)
        cache_allocation = setup_source.split("# Current sequence lengths", maxsplit=1)[0]

        assert "self.kv_cache = torch.empty(" in cache_allocation
        assert "self.kv_cache = torch.zeros(" not in cache_allocation


def test_baseline_append_kv_vectorizes_full_batch_writes() -> None:
    append_source = inspect.getsource(BaselineKVStandard.append_kv)

    assert "torch.arange(self.batch_size" not in append_source
    assert "for i, batch_idx in enumerate(batch_indices)" not in append_source
    assert "self.kv_cache[:, layer_idx, 0, :, pos, :].copy_(k)" in append_source
    assert "self.kv_cache[:, layer_idx, 1, :, pos, :].copy_(v)" in append_source
    assert "self.kv_cache[batch_indices, layer_idx, 0, :, pos, :].copy_(k)" in append_source
    assert "self.kv_cache[batch_indices, layer_idx, 1, :, pos, :].copy_(v)" in append_source


def test_fp8_append_paths_reuse_quantization_buffers() -> None:
    setup_source = inspect.getsource(OptimizedKVFP8Compressed.setup)
    quantize_source = inspect.getsource(OptimizedKVFP8Compressed._quantize_step_into)
    append_source = inspect.getsource(OptimizedKVFP8Compressed.append_kv)
    append_active_source = inspect.getsource(OptimizedKVFP8Compressed.append_active_layers)

    assert "self._k_quantized_step = torch.empty(" in setup_source
    assert "self._v_quantized_step = torch.empty_like(self._k_quantized_step)" in setup_source
    assert "self._k_quantized_layer_view = self._k_quantized_step.unsqueeze(1)" in setup_source
    assert "self._v_quantized_layer_view = self._v_quantized_step.unsqueeze(1)" in setup_source
    assert "torch.mul(x, scale, out=out)" in quantize_source
    assert "k_quantized = self._quantize_step_into(k, k_scale, self._k_quantized_step)" in append_source
    assert "v_quantized = self._quantize_step_into(v, v_scale, self._v_quantized_step)" in append_source
    assert "k_quantized = self._quantize_step_into(k, k_scale, self._k_quantized_step)" in append_active_source
    assert "v_quantized = self._quantize_step_into(v, v_scale, self._v_quantized_step)" in append_active_source
    assert "k_layer = (" in append_active_source
    assert "self._k_quantized_layer_view" in append_active_source
    assert "self.kv_cache[:, active, 0, :, pos, :].copy_(k_layer)" in append_active_source
    assert "self.kv_cache[:, active, 1, :, pos, :].copy_(v_layer)" in append_active_source
    assert "(k * k_scale).to(self.cache_dtype)" not in append_source
    assert "(v * v_scale).to(self.cache_dtype)" not in append_source
    assert "(k * k_scale).to(self.cache_dtype)" not in append_active_source
    assert "(v * v_scale).to(self.cache_dtype)" not in append_active_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for KV append parity")
def test_baseline_active_layer_append_matches_per_layer_append() -> None:
    bench = BaselineKVStandard(
        batch_size=2,
        num_layers=5,
        num_heads=4,
        head_dim=16,
        max_seq_length=8,
        active_layers=3,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        k = bench._generated_k_steps[0]
        v = bench._generated_v_steps[0]
        assert k is not None
        assert v is not None

        for layer_idx in range(bench.active_layers):
            bench.append_kv(layer_idx, k, v, pos=0)
        bench.append_active_layers(k, v, pos=1)

        per_layer = bench.kv_cache[:, : bench.active_layers, :, :, 0, :]
        batched = bench.kv_cache[:, : bench.active_layers, :, :, 1, :]
        assert torch.equal(per_layer, batched)
    finally:
        bench.teardown()


@CUDA_REQUIRED
def test_baseline_benchmark_reuses_timing_pair_and_defers_output_clone() -> None:
    bench = BaselineKVStandard(
        batch_size=2,
        num_layers=3,
        num_heads=2,
        head_dim=16,
        max_seq_length=8,
        active_layers=2,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        bench.benchmark_fn()
        torch.cuda.synchronize()
        timing_pair = bench._timing_pair
        assert timing_pair is not None
        assert bench._pending_timing_pair is timing_pair
        assert bench.output is not None
        assert bench.output.dtype == torch.bfloat16
        assert bench.output.data_ptr() == bench.kv_cache.data_ptr()

        bench.capture_verification_payload()
        payload = bench._verification_payload
        assert payload.output.dtype == torch.float32
        assert payload.output.data_ptr() != bench.output.data_ptr()

        bench.benchmark_fn()
        torch.cuda.synchronize()
        assert bench._timing_pair is timing_pair
        assert bench._pending_timing_pair is timing_pair
    finally:
        bench.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for KV append parity")
def test_fp8_active_layer_append_matches_per_layer_append() -> None:
    bench = OptimizedKVFP8Compressed(
        batch_size=2,
        num_layers=5,
        num_heads=4,
        head_dim=16,
        max_seq_length=8,
        active_layers=3,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        k = bench._generated_k_steps[0]
        v = bench._generated_v_steps[0]
        assert k is not None
        assert v is not None

        for layer_idx in range(bench.active_layers):
            bench.append_kv(layer_idx, k, v, pos=0)
        bench.append_active_layers(k, v, pos=1)

        per_layer = bench.kv_cache[:, : bench.active_layers, :, :, 0, :]
        batched = bench.kv_cache[:, : bench.active_layers, :, :, 1, :]
        assert torch.equal(per_layer, batched)
        assert torch.equal(
            bench.k_scales[: bench.active_layers, 0],
            bench.k_scales[: bench.active_layers, 1],
        )
        assert torch.equal(
            bench.v_scales[: bench.active_layers, 0],
            bench.v_scales[: bench.active_layers, 1],
        )
    finally:
        bench.teardown()


@FP8_REQUIRED
def test_fp8_benchmark_reuses_timing_pair_and_defers_dequantized_output() -> None:
    bench = OptimizedKVFP8Compressed(
        batch_size=2,
        num_layers=3,
        num_heads=2,
        head_dim=16,
        max_seq_length=8,
        active_layers=2,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        bench.benchmark_fn()
        torch.cuda.synchronize()
        timing_pair = bench._timing_pair
        assert timing_pair is not None
        assert bench._pending_timing_pair is timing_pair
        assert bench.output is not None
        assert bench.output.dtype == bench.cache_dtype
        assert bench.output.data_ptr() == bench.kv_cache.data_ptr()

        bench.capture_verification_payload()
        payload = bench._verification_payload
        assert payload.output.dtype == torch.float32
        assert payload.output.data_ptr() != bench.output.data_ptr()

        bench.benchmark_fn()
        torch.cuda.synchronize()
        assert bench._timing_pair is timing_pair
        assert bench._pending_timing_pair is timing_pair
    finally:
        bench.teardown()
