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
    KVCacheAttention,
    allocate_kv_cache,
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
    assert "src.zero_()" in source

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


def test_trtllm_capture_verification_payload_uses_small_cpu_slice() -> None:
    bench = OptimizedTrtLlmPhi35MoeBenchmark()
    bench.input_ids = torch.arange(256, dtype=torch.long).view(1, 256)
    bench.prompt_lengths = [0]
    bench._generated_output_ids = torch.arange(256, dtype=torch.long).view(2, 128)

    bench.capture_verification_payload()

    verify_output = bench.get_verify_output()
    assert verify_output.shape == (1, trtllm_common.VERIFICATION_TOKEN_PREFIX)
    assert verify_output.device.type == "cpu"


def test_optimized_trtllm_reuses_static_batch_inputs() -> None:
    setup_source = inspect.getsource(OptimizedTrtLlmPhi35MoeBenchmark.setup)
    benchmark_source = inspect.getsource(OptimizedTrtLlmPhi35MoeBenchmark.benchmark_fn)

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
    source = inspect.getsource(trtllm_common.slice_generated_token_ids)
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
    assert "gather_positions = prompt_offsets.unsqueeze(1) + token_offsets.unsqueeze(0)" in source
    assert "torch.where(valid_positions, gathered" in source
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
