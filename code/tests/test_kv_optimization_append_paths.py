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
        get_kv_source = inspect.getsource(benchmark_cls.get_kv)
        benchmark_source = inspect.getsource(benchmark_cls.benchmark_fn)

        assert "seq_len = self._seq_lengths_host[batch_idx]" in get_kv_source
        assert ".item()" not in get_kv_source
        assert "self.seq_lengths += 1" not in benchmark_source
        assert "self.seq_lengths.zero_()" not in benchmark_source
        assert "self.seq_lengths.fill_(num_decode_steps)" in benchmark_source
        assert "self._set_host_seq_lengths(0)" in benchmark_source
        assert "self._set_host_seq_lengths(num_decode_steps)" in benchmark_source
        assert "self._seq_lengths_host = [0] * self.batch_size" not in benchmark_source
        assert "self._seq_lengths_host = [num_decode_steps] * self.batch_size" not in benchmark_source


def test_kv_standard_cache_allocation_avoids_zero_fill() -> None:
    for benchmark_cls in (BaselineKVStandard, OptimizedKVFP8Compressed):
        setup_source = inspect.getsource(benchmark_cls.setup)
        cache_allocation = setup_source.split("# Current sequence lengths", maxsplit=1)[0]

        assert "self.kv_cache = torch.empty(" in cache_allocation
        assert "self.kv_cache = torch.zeros(" not in cache_allocation


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
