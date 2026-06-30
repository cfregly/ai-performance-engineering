from __future__ import annotations

import inspect
import sys
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn

from core.harness.benchmark_harness import ExecutionMode
from labs.kv_cache_compression import kv_cache_common
from labs.kv_cache_compression.baseline_kv_cache import BaselineKVCacheBenchmark
from labs.kv_cache_compression.kv_cache_common import (
    DirectWriteKVCacheAttention,
    KVCacheAttention,
    allocate_kv_cache,
    allocate_token_major_kv_cache,
    build_token_batches,
)
from labs.kv_cache_compression.optimized_kv_cache_nvfp4 import OptimizedKVCacheNVFP4Benchmark
from labs.nvfp4_group_gemm import custom_cuda_submission
from labs.persistent_decode import paged_kv_offload_common as paged_kv
from labs.persistent_decode.optimized_paged_kv_offload import (
    get_benchmark as get_optimized_paged_kv_offload,
)
from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark
from labs.trtllm_phi_3_5_moe import trtllm_common
from labs.trtllm_phi_3_5_moe.baseline_trtllm_phi_3_5_moe import (
    BaselineTrtLlmPhi35MoeBenchmark,
)
from labs.trtllm_phi_3_5_moe.optimized_trtllm_phi_3_5_moe import (
    OptimizedTrtLlmPhi35MoeBenchmark,
)


def test_kv_cache_decode_batches_reuse_single_storage_on_cpu() -> None:
    _, decode = build_token_batches(
        batch_size=2,
        prefill_seq=8,
        decode_seq=4,
        decode_steps=6,
        hidden_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(decode) == 6
    assert len({tensor.data_ptr() for tensor in decode}) == 1


def test_kv_cache_benchmark_defaults_keep_single_gpu_shape_bounded() -> None:
    bench = BaselineKVCacheBenchmark()

    assert bench.batch_size == 8
    assert bench.prefill_seq == 4096
    assert bench.decode_seq == 128
    assert bench.decode_steps == 128


def test_kv_cache_compression_benchmarks_overwrite_without_full_cache_reset() -> None:
    for benchmark_cls in (BaselineKVCacheBenchmark, OptimizedKVCacheNVFP4Benchmark):
        benchmark_source = inspect.getsource(benchmark_cls.benchmark_fn)
        assert "reset_cache(self.cache)" not in benchmark_source
        assert "torch.inference_mode()" in benchmark_source
        assert "torch.no_grad()" not in benchmark_source
        assert "for prefill, offset in self._prefill_groups:" in benchmark_source
        assert "for decode, offset in self._decode_groups:" in benchmark_source
        assert "offset += prefill.shape[1]" not in benchmark_source
        assert "offset += decode.shape[1]" not in benchmark_source

    for method in (
        BaselineKVCacheBenchmark._calibrate_fp8,
        BaselineKVCacheBenchmark._warmup_runtime,
    ):
        source = inspect.getsource(method)
        assert "reset_cache(self.cache)" not in source
        assert "torch.inference_mode()" in source
        assert "torch.no_grad()" not in source
        assert "for prefill, offset in self._prefill_groups:" in source
        assert "for decode, offset in self._decode_groups:" in source
        assert "offset += prefill.shape[1]" not in source
        assert "offset += decode.shape[1]" not in source

    setup_source = inspect.getsource(BaselineKVCacheBenchmark._setup_with_recipe)
    teardown_source = inspect.getsource(BaselineKVCacheBenchmark.teardown)
    assert "self._prefill_groups = []" in setup_source
    assert "self._decode_groups = []" in setup_source
    assert "self._prefill_groups.append((prefill, offset))" in setup_source
    assert "self._decode_groups.append((decode, offset))" in setup_source
    assert "self._prefill_groups = []" in teardown_source
    assert "self._decode_groups = []" in teardown_source


@pytest.mark.parametrize(
    "benchmark_cls",
    (BaselineKVCacheBenchmark, OptimizedKVCacheNVFP4Benchmark),
)
def test_kv_cache_verification_output_is_built_after_benchmark(
    benchmark_cls, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = benchmark_cls()
    bench.device = torch.device("cpu")
    bench.batch_size = 2
    bench.prefill_seq = 4
    bench.decode_seq = 2
    bench.decode_steps = 1
    bench.cache = allocate_kv_cache(
        batch_size=2,
        total_tokens=6,
        num_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    bench.cache.cache_k.copy_(torch.arange(bench.cache.cache_k.numel()).view_as(bench.cache.cache_k))
    bench.cache.cache_v.copy_(torch.arange(bench.cache.cache_v.numel()).view_as(bench.cache.cache_v).mul_(2))
    bench._batch_size_tensor = torch.tensor([bench.batch_size], dtype=torch.int64)
    bench._seq_meta_tensor = torch.tensor([bench.prefill_seq, bench.decode_seq, bench.decode_steps], dtype=torch.int64)
    bench._verify_output_buffer = torch.empty(2, 2, 1, 1, 4, dtype=torch.float32)

    def fail_stack(*args, **kwargs):
        raise AssertionError("benchmark_fn() should not stack verification output")

    monkeypatch.setattr(torch, "stack", fail_stack)
    bench._mark_cache_output_ready()
    assert bench.output is None
    monkeypatch.undo()

    bench.capture_verification_payload()

    verify_output = bench.get_verify_output()
    assert verify_output.shape == (2, 2, 1, 1, 4)
    assert verify_output.dtype == torch.float32


class _DummyLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, *, params_dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer = nn.LayerNorm(hidden_dim, device=device, dtype=params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _DummyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        params_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_kv_cache_attention_routes_through_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cpu")
    bench_attn = KVCacheAttention(
        hidden_dim=16,
        num_heads=4,
        linear_cls=_DummyLinear,
        layernorm_cls=_DummyLayerNorm,
        params_dtype=torch.float32,
        device=device,
    )
    cache = allocate_kv_cache(
        batch_size=2,
        total_tokens=8,
        num_heads=4,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    tokens = torch.randn(2, 3, 16, device=device, dtype=torch.float32)
    captured: dict[str, object] = {}

    def _fake_sdpa(query, key, value, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        captured["key_shape"] = tuple(key.shape)
        captured["value_shape"] = tuple(value.shape)
        captured["kwargs"] = kwargs
        return torch.zeros_like(query)

    monkeypatch.setattr(kv_cache_common, "prefer_sdpa_backends", lambda: nullcontext())
    monkeypatch.setattr(kv_cache_common.F, "scaled_dot_product_attention", _fake_sdpa)

    out = bench_attn(tokens, cache, start_offset=2)

    assert out.shape == (2, 3, 16)
    assert captured["query_shape"] == (2, 4, 3, 4)
    assert captured["key_shape"] == (2, 4, 5, 4)
    assert captured["value_shape"] == (2, 4, 5, 4)
    assert captured["kwargs"] == {
        "dropout_p": 0.0,
        "is_causal": False,
        "scale": bench_attn.scale,
    }


def _dummy_linear(module: nn.Module) -> nn.Linear:
    return module.layer  # type: ignore[return-value]


def test_direct_write_kv_cache_attention_writes_projection_outputs_into_cache() -> None:
    source = inspect.getsource(DirectWriteKVCacheAttention.forward)

    assert "qkv = self.qkv(x)" not in source
    assert ".copy_(" not in source
    assert "k_dest = cache.cache_k[token_slice].flatten(2)" in source
    assert "v_dest = cache.cache_v[token_slice].flatten(2)" in source
    assert "self._project_into(x_t, self.k_proj, k_dest)" in source
    assert "self._project_into(x_t, self.v_proj, v_dest)" in source


def test_direct_write_kv_cache_attention_matches_fused_qkv_reference_on_cpu() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1234)
    fused = KVCacheAttention(
        hidden_dim=16,
        num_heads=4,
        linear_cls=_DummyLinear,
        layernorm_cls=_DummyLayerNorm,
        params_dtype=torch.float32,
        device=device,
    )
    direct = DirectWriteKVCacheAttention(
        hidden_dim=16,
        num_heads=4,
        linear_cls=_DummyLinear,
        layernorm_cls=_DummyLayerNorm,
        params_dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        direct.ln.layer.load_state_dict(fused.ln.layer.state_dict())
        qkv = _dummy_linear(fused.qkv)
        q = _dummy_linear(direct.q_proj)
        k = _dummy_linear(direct.k_proj)
        v = _dummy_linear(direct.v_proj)
        q.weight.copy_(qkv.weight[:16])
        q.bias.copy_(qkv.bias[:16])
        k.weight.copy_(qkv.weight[16:32])
        k.bias.copy_(qkv.bias[16:32])
        v.weight.copy_(qkv.weight[32:48])
        v.bias.copy_(qkv.bias[32:48])
        _dummy_linear(direct.proj).load_state_dict(_dummy_linear(fused.proj).state_dict())

    batch_major_cache = allocate_kv_cache(
        batch_size=2,
        total_tokens=8,
        num_heads=4,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    token_major_cache = allocate_token_major_kv_cache(
        batch_size=2,
        total_tokens=8,
        num_heads=4,
        head_dim=4,
        device=device,
        dtype=torch.float32,
    )
    batch_major_cache.cache_k.zero_()
    batch_major_cache.cache_v.zero_()
    token_major_cache.cache_k.zero_()
    token_major_cache.cache_v.zero_()
    tokens = torch.randn(2, 3, 16, device=device, dtype=torch.float32)

    with torch.inference_mode():
        fused_out = fused(tokens, batch_major_cache, start_offset=2)
        direct_out = direct(tokens, token_major_cache, start_offset=2)

    torch.testing.assert_close(direct_out, fused_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        token_major_cache.cache_k[2:5].permute(1, 0, 2, 3),
        batch_major_cache.cache_k[:, 2:5],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        token_major_cache.cache_v[2:5].permute(1, 0, 2, 3),
        batch_major_cache.cache_v[:, 2:5],
        rtol=1e-6,
        atol=1e-6,
    )


def test_paged_kv_offload_skips_when_fused_fp8_is_required_but_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paged_kv, "_supports_fp8_kv", lambda: True)
    monkeypatch.setattr(paged_kv, "_supports_fused_fp8_attention", lambda: False)

    bench = PagedKVOffloadBenchmark(
        PagedKVConfig(prefer_fp8=True, require_fused_fp8=True, fallback_dtype=torch.float16)
    )

    with pytest.raises(RuntimeError, match="SKIPPED: FP8 KV requested"):
        bench._select_runtime_dtype()


def test_optimized_paged_kv_offload_falls_back_instead_of_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paged_kv, "_supports_fp8_kv", lambda: True)
    monkeypatch.setattr(paged_kv, "_supports_fused_fp8_attention", lambda: False)

    bench = get_optimized_paged_kv_offload()

    assert bench.cfg.require_fused_fp8 is False
    assert bench.cfg.use_pinned_stage is True
    assert bench.cfg.use_direct_h2d is True
    assert bench._select_runtime_dtype() == torch.float16


def test_nvfp4_group_gemm_custom_cuda_module_keeps_sys_import_available() -> None:
    assert custom_cuda_submission.sys is sys


def test_nvfp4_group_gemm_scale_packer_reuses_scratch_tensor() -> None:
    source = inspect.getsource(custom_cuda_submission._pack_scale_tiles_for_tcgen05)
    assert "src = torch.zeros((4, 32, 4, 4)" not in source
    assert "src = torch.empty((4, 32, 4, 4)" in source
    assert "src.zero_()" not in source
    assert "src[seg_avail:].zero_()" in source

    raw = torch.arange(4 * 32 * 4 * 4, dtype=torch.int64)
    sfa = raw.remainder(251).to(torch.uint8).view(1, 4, 32, 4, 4)
    sfb = raw.add(3).remainder(251).to(torch.uint8).view(1, 4, 32, 4, 4)

    sfa_tiles, sfb_tiles = custom_cuda_submission._pack_scale_tiles_for_tcgen05(
        sfa,
        sfb,
        m=128,
        n=128,
        k_scales=16,
    )

    torch.testing.assert_close(sfa_tiles[0, 0], sfa[0, :4].contiguous().reshape(128, 16))
    torch.testing.assert_close(sfb_tiles[0, 0], sfb[0, :4].contiguous().reshape(128, 16))

    partial_raw = torch.arange(3 * 32 * 4 * 4, dtype=torch.int64)
    partial_sfa = partial_raw.remainder(251).to(torch.uint8).view(1, 3, 32, 4, 4)
    partial_sfb = partial_raw.add(3).remainder(251).to(torch.uint8).view(1, 3, 32, 4, 4)
    partial_sfa_tiles, partial_sfb_tiles = custom_cuda_submission._pack_scale_tiles_for_tcgen05(
        partial_sfa,
        partial_sfb,
        m=128,
        n=128,
        k_scales=12,
    )
    expected_sfa = torch.zeros(4, 32, 4, 4, dtype=torch.uint8)
    expected_sfb = torch.zeros_like(expected_sfa)
    expected_sfa[:3].copy_(partial_sfa[0])
    expected_sfb[:3].copy_(partial_sfb[0])

    torch.testing.assert_close(partial_sfa_tiles[0, 0], expected_sfa.reshape(128, 16))
    torch.testing.assert_close(partial_sfb_tiles[0, 0], expected_sfb.reshape(128, 16))


def test_nvfp4_group_gemm_prepare_zeroes_only_padding_tails() -> None:
    source = inspect.getsource(custom_cuda_submission.prepare_custom_cuda)

    assert "a_pad = torch.zeros((m_padded, k_bytes)" not in source
    assert "b_pad = torch.zeros((n_padded_ab, k_bytes)" not in source
    assert "a_pad = torch.empty((m_padded, k_bytes)" in source
    assert "b_pad = torch.empty((n_padded_ab, k_bytes)" in source
    assert "a_pad[m:, :].zero_()" in source
    assert "b_pad[n:, :].zero_()" in source
    assert "padded = torch.zeros((m_tiles_tma,)" not in source
    assert "padded = torch.zeros((n_tiles_tma,)" not in source
    assert "padded[m_tiles_actual:].zero_()" in source
    assert "padded[n_tiles_actual:].zero_()" in source


def test_nvfp4_group_gemm_fused_context_preallocates_metadata() -> None:
    ctxs = [
        {
            "a_ptrs": torch.tensor([10, 11], dtype=torch.int64),
            "m_sizes": torch.tensor([128, 256], dtype=torch.int32),
            "cta_group_idx_map": torch.tensor([0, 1, 0], dtype=torch.int32),
        },
        {
            "a_ptrs": torch.tensor([12, 13, 14], dtype=torch.int64),
            "m_sizes": torch.tensor([384, 512, 640], dtype=torch.int32),
            "cta_group_idx_map": torch.tensor([0, 2], dtype=torch.int32),
        },
    ]

    fused_sizes = custom_cuda_submission._fuse_grouped_ctx_tensor(ctxs, "m_sizes")
    fused_cta_map = custom_cuda_submission._fuse_cta_group_idx_map(ctxs)

    torch.testing.assert_close(
        fused_sizes,
        torch.tensor([128, 256, 384, 512, 640], dtype=torch.int32),
    )
    torch.testing.assert_close(
        fused_cta_map,
        torch.tensor([0, 1, 0, 2, 4], dtype=torch.int32),
    )
    assert fused_sizes.is_contiguous()
    assert fused_cta_map.is_contiguous()

    prepare_source = inspect.getsource(custom_cuda_submission.prepare_custom_cuda)
    assert "_fuse_grouped_ctx_tensor(grouped_ctxs, key)" in prepare_source
    assert "_fuse_cta_group_idx_map(grouped_ctxs)" in prepare_source
    assert "torch.cat([ctx[key] for ctx in grouped_ctxs]" not in prepare_source
    assert "cta_group_idx_parts" not in prepare_source


def test_trtllm_capture_verification_payload_uses_small_cpu_slice() -> None:
    source = inspect.getsource(OptimizedTrtLlmPhi35MoeBenchmark.benchmark_fn)
    assert "self._generated_output_ids = output_ids" in source
    assert "output_ids.detach()" not in source

    bench = OptimizedTrtLlmPhi35MoeBenchmark()
    bench.input_ids = torch.arange(256, dtype=torch.long).view(1, 256)
    bench.prompt_lengths = [0]
    bench._generated_output_ids = torch.arange(256, dtype=torch.long).view(2, 128)
    bench._verify_output_buffer = torch.empty(1, trtllm_common.VERIFICATION_TOKEN_PREFIX, dtype=torch.long)

    bench.capture_verification_payload()

    verify_output = bench.get_verify_output()
    assert verify_output.shape == (1, trtllm_common.VERIFICATION_TOKEN_PREFIX)
    assert verify_output.device.type == "cpu"


def test_trtllm_prompt_builder_expands_single_encoded_prompt() -> None:
    class _Tokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = 0

        def encode(self, _text: str, add_special_tokens: bool = True) -> list[int]:
            return [11, 12, 13, 14]

    source = inspect.getsource(trtllm_common.build_prompt_tokens)
    input_ids, attention_mask = trtllm_common.build_prompt_tokens(
        _Tokenizer(),
        prompt_len=3,
        batch_size=4,
    )

    expected = torch.tensor(
        [
            [11, 12, 13],
            [11, 12, 13],
            [11, 12, 13],
            [11, 12, 13],
        ],
        dtype=torch.long,
    )
    assert torch.equal(input_ids, expected)
    assert torch.equal(attention_mask, torch.ones_like(expected))
    assert "encoded_ids.unsqueeze(0).expand(batch_size, -1).contiguous()" in source
    assert "[encoded] * batch_size" not in source


def test_optimized_trtllm_reuses_static_batch_inputs() -> None:
    baseline_setup_source = inspect.getsource(BaselineTrtLlmPhi35MoeBenchmark.setup)
    setup_source = inspect.getsource(OptimizedTrtLlmPhi35MoeBenchmark.setup)
    benchmark_source = inspect.getsource(OptimizedTrtLlmPhi35MoeBenchmark.benchmark_fn)

    for source in (baseline_setup_source, setup_source):
        assert "self.prompt_lengths = [input_ids.size(1)] * self.batch_size" in source
        assert "attention_mask.sum" not in source

    assert "self._batch_inputs = [" in setup_source
    assert "self.input_ids[i, :valid_len].contiguous()" in setup_source
    assert "batch_inputs = []" not in benchmark_source
    assert "self.input_ids[i, :valid_len].contiguous()" not in benchmark_source
    assert "self.runner.generate(self._batch_inputs" in benchmark_source


def test_trtllm_benchmarks_use_wall_clock_timing() -> None:
    baseline_config = BaselineTrtLlmPhi35MoeBenchmark().get_config()
    optimized_config = OptimizedTrtLlmPhi35MoeBenchmark().get_config()

    assert baseline_config.timing_method == "wall_clock"
    assert baseline_config.full_device_sync is True
    assert optimized_config.timing_method == "wall_clock"
    assert optimized_config.full_device_sync is True


def test_optimized_trtllm_uses_subprocess_execution_after_local_descendant_cleanup() -> None:
    config = OptimizedTrtLlmPhi35MoeBenchmark().get_config()

    assert config.use_subprocess is True
    assert config.execution_mode == ExecutionMode.SUBPROCESS


def test_optimized_trtllm_profiler_path_uses_hard_exit_cleanup() -> None:
    bench = OptimizedTrtLlmPhi35MoeBenchmark()

    assert getattr(bench, "profile_require_teardown", False) is False


def test_optimized_trtllm_teardown_calls_runner_release_hooks_without_local_descendant_reap() -> None:
    calls: list[str] = []

    class _FakeRunner:
        def shutdown(self) -> None:
            calls.append("shutdown")

        def close(self) -> None:
            calls.append("close")

    bench = OptimizedTrtLlmPhi35MoeBenchmark()
    bench.runner = _FakeRunner()
    bench.teardown()

    assert calls == ["shutdown", "close"]
    assert bench.runner is None


def test_trtllm_generated_token_slice_normalizes_beams_and_padding() -> None:
    module_source = inspect.getsource(trtllm_common)
    source = inspect.getsource(trtllm_common.slice_generated_token_ids)
    trtllm_common._TOKEN_OFFSET_CACHE.clear()
    output_ids = torch.tensor(
        [
            [[11, 12, 13, 21, 22], [91, 92, 93, 94, 95]],
            [[31, 32, 41, 42, 43], [81, 82, 83, 84, 85]],
        ],
        dtype=torch.long,
    )

    normalized = trtllm_common.slice_generated_token_ids(
        output_ids,
        prompt_lengths=[3, 2],
        max_new_tokens=4,
        pad_token_id=0,
    )

    assert torch.equal(
        normalized,
        torch.tensor(
            [
                [21, 22, 0, 0],
                [41, 42, 43, 0],
            ],
            dtype=torch.long,
        ),
    )
    first_offsets = trtllm_common._token_offsets_for(4, output_ids.device)
    _ = trtllm_common.slice_generated_token_ids(
        output_ids,
        prompt_lengths=[3, 2],
        max_new_tokens=4,
        pad_token_id=0,
    )
    second_offsets = trtllm_common._token_offsets_for(4, output_ids.device)
    assert second_offsets.data_ptr() == first_offsets.data_ptr()
    assert "_TOKEN_OFFSET_CACHE" in module_source
    assert "def _token_offsets_for" in module_source
    assert "token_offsets = _token_offsets_for(max_new_tokens, output_ids.device)" in source
    assert "torch.arange(max_new_tokens, device=output_ids.device" not in source
    assert "gather_positions = prompt_offsets.unsqueeze(1) + token_offsets.unsqueeze(0)" in source
    assert "torch.where(valid_positions, gathered" not in source
    assert "torch.full_like(gathered" not in source
    assert "gathered.masked_fill_(valid_positions.logical_not_(), pad_value)" in source
    assert "generated = torch.cat" not in source
    assert "rows.append" not in source


def test_trtllm_generated_token_slice_normalizes_output_dtype_to_int64() -> None:
    output_ids = torch.tensor(
        [[[11, 12, 13, 21, 22]]],
        dtype=torch.int32,
    )

    normalized = trtllm_common.slice_generated_token_ids(
        output_ids,
        prompt_lengths=[3],
        max_new_tokens=4,
        pad_token_id=0,
    )

    assert normalized.dtype == torch.int64
    assert torch.equal(normalized, torch.tensor([[21, 22, 0, 0]], dtype=torch.int64))


def test_trtllm_verification_prefix_length_uses_stable_decode_prefix() -> None:
    assert trtllm_common.verification_token_prefix_length(128) == trtllm_common.VERIFICATION_TOKEN_PREFIX
    assert trtllm_common.verification_token_prefix_length(4) == 4
