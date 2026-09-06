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


def test_kv_standard_uses_host_seq_lengths_and_full_written_verification_region() -> None:
    for benchmark_cls in (BaselineKVStandard, OptimizedKVFP8Compressed):
        init_source = inspect.getsource(benchmark_cls.__init__)
        setup_source = inspect.getsource(benchmark_cls.setup)
        written_view_source = inspect.getsource(benchmark_cls._written_cache_view)
        build_source = inspect.getsource(benchmark_cls._build_verification_output)
        get_kv_source = inspect.getsource(benchmark_cls.get_kv)
        benchmark_source = inspect.getsource(benchmark_cls.benchmark_fn)
        capture_source = inspect.getsource(benchmark_cls.capture_verification_payload)
        finalize_source = inspect.getsource(benchmark_cls.finalize_iteration_metrics)
        teardown_source = inspect.getsource(benchmark_cls.teardown)

        assert "self._generated_step_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []" in init_source
        assert "self._output_view: Optional[torch.Tensor] = None" in init_source
        assert "self._batch_size_tensor: Optional[torch.Tensor] = None" in init_source
        assert "self._seq_lengths_payload: Optional[torch.Tensor] = None" in init_source
        assert "self._active_layer_slice = slice(0, active_layers)" in init_source
        assert "self._generated_step_pairs = list(" in setup_source
        assert "zip(self._generated_k_steps, self._generated_v_steps, strict=True)" in setup_source
        if benchmark_cls is BaselineKVStandard:
            assert "self._generated_step_layer_view_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []" in init_source
            assert "self._generated_step_layer_position_pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []" in init_source
            assert "self._generated_step_layer_position_count = 0" in init_source
            assert "self._generated_step_layer_view_pairs = [" in setup_source
            assert "(k_step.unsqueeze(1), v_step.unsqueeze(1))" in setup_source
            assert "self._generated_step_layer_position_pairs = [" in setup_source
            assert "self._generated_step_layer_position_count = len(self._generated_step_layer_position_pairs)" in setup_source
            assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in init_source
            assert "self._verify_output_buffer.copy_(self.output)" in build_source
            assert "output=self._build_verification_output()" in capture_source
            assert "self.output.float().clone()" not in capture_source
        else:
            assert "self._generated_step_position_pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []" in init_source
            assert "self._generated_step_position_count = 0" in init_source
            assert "self._generated_step_position_pairs = [" in setup_source
            assert "self._generated_step_position_count = len(self._generated_step_position_pairs)" in setup_source
            assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in init_source
            assert "self._verify_output_buffer.copy_(self.output)" in build_source
            assert "self.k_scales[" in build_source
            assert "self.v_scales[" in build_source
            assert "self._verify_output_buffer[:, :, 0]" in build_source
            assert "self._verify_output_buffer[:, :, 1]" in build_source
            assert "return self._verify_output_buffer" in build_source
            assert "torch.stack(" not in build_source
            assert ".detach().clone()" not in build_source
        assert "self._output_view = self._written_cache_view()" in setup_source
        assert "self._verify_output_buffer = None" in setup_source
        assert ":, self._active_layer_slice, :, :, : self.num_decode_steps, :" in written_view_source
        assert "torch.empty_like(self.output, dtype=torch.float32)" in build_source
        assert 'self._batch_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")' in setup_source
        assert "self._batch_size_tensor[0] = self.batch_size" in setup_source
        assert "self._seq_lengths_payload = torch.empty_like(self.seq_lengths)" in setup_source
        assert '"batch_size": self._batch_size_tensor' in capture_source
        assert "self._seq_lengths_payload.copy_(self.seq_lengths)" in capture_source
        assert '"seq_lengths": self._seq_lengths_payload' in capture_source
        assert "torch.tensor([self.batch_size]" not in capture_source
        assert "self.seq_lengths.detach().clone()" not in capture_source
        assert "seq_len = self._seq_lengths_host[batch_idx]" in get_kv_source
        assert ".item()" not in get_kv_source
        assert "self.seq_lengths += 1" not in benchmark_source
        assert "self.seq_lengths.zero_()" not in benchmark_source
        assert "with torch.inference_mode():" in benchmark_source
        if benchmark_cls is BaselineKVStandard:
            assert "for pos, new_k_layer, new_v_layer in self._generated_step_layer_position_pairs:" in benchmark_source
            assert "self._generated_step_layer_position_pairs" in benchmark_source
            assert "self._generated_step_layer_position_count != self.num_decode_steps" in benchmark_source
            assert "len(self._generated_step_layer_position_pairs)" not in benchmark_source
            assert "self.append_active_layer_views(new_k_layer, new_v_layer, pos=pos)" in benchmark_source
        else:
            assert "for pos, new_k, new_v in self._generated_step_position_pairs:" in benchmark_source
            assert "self._generated_step_position_pairs" in benchmark_source
            assert "self._generated_step_position_count != self.num_decode_steps" in benchmark_source
            assert "len(self._generated_step_position_pairs)" not in benchmark_source
        assert "enumerate(self._generated_step" not in benchmark_source
        assert "self.seq_lengths.fill_(num_decode_steps)" in benchmark_source
        assert "self._set_host_seq_lengths(0)" in benchmark_source
        assert "self._set_host_seq_lengths(num_decode_steps)" in benchmark_source
        assert "current_stream = torch.cuda.current_stream(self.device)" in benchmark_source
        assert "start_event.record(current_stream)" in benchmark_source
        assert "end_event.record(current_stream)" in benchmark_source
        assert "start_event.record()" not in benchmark_source
        assert "end_event.record()" not in benchmark_source
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
            assert "self._generated_step_layer_position_pairs = []" in teardown_source
            assert "self._generated_step_layer_position_count = 0" in teardown_source
        else:
            assert "self._generated_step_position_pairs = []" in teardown_source
            assert "self._generated_step_position_count = 0" in teardown_source
            assert "self._verify_output_buffer = None" in teardown_source
        assert "self._output_view = None" in teardown_source
        assert "self._batch_size_tensor = None" in teardown_source
        assert "self._seq_lengths_payload = None" in teardown_source
        assert "metrics = self._last_metrics" in finalize_source
        assert 'metrics["latency_ms"] = elapsed_ms_value' in finalize_source
        assert 'metrics["tokens_per_sec"] = tokens_per_sec' in finalize_source
        assert 'metrics["memory_gb"] = memory_gb' in finalize_source
        assert "self._last_metrics = {" not in finalize_source
        assert "logger.debug(f" not in finalize_source
        if benchmark_cls is OptimizedKVFP8Compressed:
            assert 'metrics["compression_ratio"] = 2.0 / self.bytes_per_element' in finalize_source


def _prepare_cpu_verification_metadata(
    bench: BaselineKVStandard | OptimizedKVFP8Compressed,
) -> None:
    bench._batch_size_tensor = torch.tensor([bench.batch_size], dtype=torch.int64)
    bench.seq_lengths = torch.full(
        (bench.batch_size,), bench.num_decode_steps, dtype=torch.int64
    )
    bench._seq_lengths_payload = torch.empty_like(bench.seq_lengths)


def test_baseline_verification_payload_covers_only_the_full_written_region_on_cpu() -> None:
    bench = BaselineKVStandard(
        batch_size=2,
        num_layers=4,
        num_heads=3,
        head_dim=5,
        max_seq_length=7,
        active_layers=3,
        num_decode_steps=4,
    )
    bench.device = torch.device("cpu")
    bench.kv_cache = torch.full(
        (2, 4, 2, 3, 7, 5), -1000, dtype=torch.bfloat16
    )
    expected_shape = (2, 3, 2, 3, 4, 5)
    expected = torch.arange(2 * 3 * 2 * 3 * 4 * 5, dtype=torch.float32)
    expected = expected.reshape(expected_shape).to(torch.bfloat16)
    bench.kv_cache[:, :3, :, :, :4, :].copy_(expected)
    bench.output = bench._written_cache_view()
    _prepare_cpu_verification_metadata(bench)

    assert bench._verify_output_buffer is None
    bench.capture_verification_payload()
    output = bench.get_verify_output()

    assert output.shape == expected_shape
    assert output.dtype == torch.float32
    assert torch.equal(output, expected.float())
    assert output[-1, -1, -1, -1, -1, -1] == expected[-1, -1, -1, -1, -1, -1]
    assert not torch.any(output == -1000)


def test_optimized_verification_dequantizes_the_full_written_region_on_cpu() -> None:
    bench = OptimizedKVFP8Compressed(
        batch_size=2,
        num_layers=4,
        num_heads=3,
        head_dim=5,
        max_seq_length=7,
        active_layers=3,
        num_decode_steps=4,
    )
    bench.device = torch.device("cpu")
    bench.cache_dtype = torch.float32
    bench.kv_cache = torch.full((2, 4, 2, 3, 7, 5), -1000, dtype=torch.float32)
    expected_shape = (2, 3, 2, 3, 4, 5)
    quantized = torch.arange(1, 2 * 3 * 2 * 3 * 4 * 5 + 1, dtype=torch.float32)
    quantized = quantized.reshape(expected_shape)
    bench.kv_cache[:, :3, :, :, :4, :].copy_(quantized)
    bench.k_scales = torch.ones(4, 7, dtype=torch.float32)
    bench.v_scales = torch.ones(4, 7, dtype=torch.float32)
    bench.k_scales[:3, :4] = torch.tensor(
        [[2, 4, 5, 8], [4, 5, 8, 10], [5, 8, 10, 16]], dtype=torch.float32
    )
    bench.v_scales[:3, :4] = torch.tensor(
        [[4, 5, 8, 10], [5, 8, 10, 16], [8, 10, 16, 20]], dtype=torch.float32
    )
    bench.output = bench._written_cache_view()
    _prepare_cpu_verification_metadata(bench)

    expected = quantized.clone()
    expected[:, :, 0].div_(bench.k_scales[:3, :4].view(1, 3, 1, 4, 1))
    expected[:, :, 1].div_(bench.v_scales[:3, :4].view(1, 3, 1, 4, 1))
    assert bench._verify_output_buffer is None
    bench.capture_verification_payload()
    output = bench.get_verify_output()

    assert output.shape == expected_shape
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)
    assert output[-1, -1, -1, -1, -1, -1] == expected[-1, -1, -1, -1, -1, -1]
    assert not torch.any(output == -1000)


def test_kv_standard_prefers_full_application_range_replay_for_ncu() -> None:
    for benchmark_cls in (BaselineKVStandard, OptimizedKVFP8Compressed):
        bench = benchmark_cls(
            batch_size=1,
            num_layers=1,
            num_heads=1,
            head_dim=1,
            max_seq_length=1,
            active_layers=1,
            num_decode_steps=1,
        )
        assert bench.preferred_ncu_replay_mode == "app-range"
        assert bench.get_config().ncu_replay_mode == "app-range"


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
    init_source = inspect.getsource(OptimizedKVFP8Compressed.__init__)
    setup_source = inspect.getsource(OptimizedKVFP8Compressed.setup)
    compute_scale_source = inspect.getsource(OptimizedKVFP8Compressed._compute_scale)
    quantize_source = inspect.getsource(OptimizedKVFP8Compressed._quantize_step_into)
    append_source = inspect.getsource(OptimizedKVFP8Compressed.append_kv)
    append_active_source = inspect.getsource(OptimizedKVFP8Compressed.append_active_layers)
    teardown_source = inspect.getsource(OptimizedKVFP8Compressed.teardown)

    assert "self._scale_abs_buffer: Optional[torch.Tensor] = None" in init_source
    assert "self._k_quantized_step = torch.empty(" in setup_source
    assert "self._v_quantized_step = torch.empty_like(self._k_quantized_step)" in setup_source
    assert "self._scale_abs_buffer = torch.empty_like(self._generated_k_steps[0])" in setup_source
    assert "self._k_quantized_layer_view = self._k_quantized_step.unsqueeze(1)" in setup_source
    assert "self._v_quantized_layer_view = self._v_quantized_step.unsqueeze(1)" in setup_source
    assert "or self._scale_abs_buffer.numel() < numel" in compute_scale_source
    assert "abs_buffer = self._scale_abs_buffer[:numel].view(shape)" in compute_scale_source
    assert "torch.abs(x, out=abs_buffer)" in compute_scale_source
    assert "absmax = abs_buffer.amax().float()" in compute_scale_source
    assert "x.abs().amax().float()" not in compute_scale_source
    assert "scale_for_promotion = scale.reshape(1)" in quantize_source
    assert "torch.mul(x, scale_for_promotion, out=out)" in quantize_source
    assert "(x * scale_for_promotion).to(self.cache_dtype)" in quantize_source
    assert "torch.mul(x, scale, out=out)" not in quantize_source
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
    assert "self._scale_abs_buffer = None" in teardown_source

    bench = OptimizedKVFP8Compressed(
        batch_size=2,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        max_seq_length=4,
        active_layers=1,
        num_decode_steps=1,
    )
    bench.device = torch.device("cpu")
    bench.cache_dtype = torch.float32
    bench.use_fp8 = True
    bench.use_fp4 = False
    large = torch.randn(2, 2, 4)
    small = torch.randn(1, 2, 4)

    large_scale = bench._compute_scale(large)
    buffer_ptr = bench._scale_abs_buffer.data_ptr()
    small_scale = bench._compute_scale(small)

    assert torch.isfinite(large_scale)
    assert torch.isfinite(small_scale)
    assert bench._scale_abs_buffer.data_ptr() == buffer_ptr
    assert bench._scale_abs_buffer.numel() >= large.numel()


@pytest.mark.parametrize("use_fp8", [True, False], ids=["preallocated_out", "conversion_return"])
def test_quantize_step_promotes_bf16_product_before_low_precision_cast(use_fp8: bool) -> None:
    bench = OptimizedKVFP8Compressed(
        batch_size=1,
        num_layers=1,
        num_heads=1,
        head_dim=1,
        max_seq_length=1,
        active_layers=1,
        num_decode_steps=1,
    )
    # E4M3 also exercises the conversion-return branch on CPU installations
    # that do not expose the production FP4 dtype.
    bench.use_fp8 = use_fp8
    bench.cache_dtype = torch.float8_e4m3fn
    x = torch.tensor([[[-2.015625]]], dtype=torch.bfloat16)
    scale = torch.tensor(135.2452850341797, dtype=torch.float32)
    out = torch.zeros_like(x, dtype=torch.float8_e4m3fn)

    double_rounded = (x * scale).to(torch.float8_e4m3fn)
    expected = (x.float() * scale).to(torch.float8_e4m3fn)
    result = bench._quantize_step_into(x, scale, out)

    assert scale.ndim == 0
    assert double_rounded.item() == -256.0
    assert expected.item() == -288.0
    assert torch.equal(result, expected)
    if use_fp8:
        assert result is out
        assert result.data_ptr() == out.data_ptr()
    else:
        assert result is not out
        assert torch.count_nonzero(out.float()).item() == 0


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
        metrics = bench.get_custom_metrics()

        bench.benchmark_fn()
        torch.cuda.synchronize()
        assert bench._timing_pair is timing_pair
        assert bench._pending_timing_pair is timing_pair
        next_metrics = bench.get_custom_metrics()
        assert next_metrics is metrics
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
        metrics = bench.get_custom_metrics()

        bench.benchmark_fn()
        torch.cuda.synchronize()
        assert bench._timing_pair is timing_pair
        assert bench._pending_timing_pair is timing_pair
        next_metrics = bench.get_custom_metrics()
        assert next_metrics is metrics
    finally:
        bench.teardown()
