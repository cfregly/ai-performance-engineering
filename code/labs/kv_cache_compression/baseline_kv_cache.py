"""Baseline KV-cache benchmark using MXFP8 block scaling."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.env import apply_env_defaults
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.kv_cache_compression.kv_cache_common import (
    KVCache,
    KVCacheAttention,
    allocate_kv_cache,
    build_token_batches,
    cache_is_finite,
    resolve_device,
)

apply_env_defaults()


def _preload_torch_cuda_symbols() -> None:
    """Ensure torch CUDA shared objects are globally visible before TE import."""
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    libs = [
        "libtorch_cuda.so",
        "libtorch_cuda_linalg.so",
        "libtorch_nvshmem.so",
        "libc10_cuda.so",
    ]
    for name in libs:
        candidate = torch_lib_dir / name
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


_preload_torch_cuda_symbols()

try:  # Transformer Engine is optional; fail fast in setup when missing.
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch import LayerNorm as TELayerNorm
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import autocast as te_autocast
    from transformer_engine.pytorch import quantized_model_init

    TE_AVAILABLE = True
    TE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc
    TELinear = TELayerNorm = te_autocast = quantized_model_init = te_recipe = None  # type: ignore


class BaselineKVCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """MXFP8 KV-cache benchmark (prefill + decode)."""

    def __init__(self) -> None:
        super().__init__()
        self.device = None
        self.tensor_dtype = torch.bfloat16
        # Keep the lab large enough to exercise KV pressure without relying on
        # near-capacity allocator behavior on single-GPU runs.
        self.batch_size = 8
        self.hidden_dim = 16384
        self.num_heads = 64
        self.prefill_seq = 4096
        self.decode_seq = 128
        self.decode_steps = 128
        self.prefill_inputs: List[torch.Tensor] = []
        self.decode_inputs: List[torch.Tensor] = []
        self.cache: Optional[KVCache] = None
        self.model: Optional[nn.Module] = None
        self.fp8_recipe = (
            te_recipe.DelayedScaling(amax_history_len=16, amax_compute_algo="max") if TE_AVAILABLE else None
        )
        self.runtime_recipe = self.fp8_recipe
        self.output: Optional[torch.Tensor] = None
        self._cache_output_ready = False
        self._payload_parameter_count = 0
        self._prefill_groups: list[tuple[torch.Tensor, int]] = []
        self._decode_groups: list[tuple[torch.Tensor, int]] = []
        self._batch_size_tensor: Optional[torch.Tensor] = None
        self._seq_meta_tensor: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def _resolve_device(self) -> torch.device:
        return resolve_device()

    def setup(self) -> None:
        self._setup_with_recipe(self.fp8_recipe)

    def _setup_with_recipe(self, recipe) -> None:
        if not TE_AVAILABLE or recipe is None:
            raise RuntimeError(f"SKIPPED: Transformer Engine not available: {TE_IMPORT_ERROR}")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with quantized_model_init(enabled=True, recipe=recipe):
            self.model = KVCacheAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                linear_cls=TELinear,
                layernorm_cls=TELayerNorm,
                params_dtype=self.tensor_dtype,
                device=self.device,
            )

        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.prefill_inputs, self.decode_inputs = build_token_batches(
            batch_size=self.batch_size,
            prefill_seq=self.prefill_seq // 2,  # two prefill windows
            decode_seq=self.decode_seq,
            decode_steps=self.decode_steps,
            hidden_dim=self.hidden_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        offset = 0
        self._prefill_groups = []
        for prefill in self.prefill_inputs:
            self._prefill_groups.append((prefill, offset))
            offset += prefill.shape[1]
        self._decode_groups = []
        for decode in self.decode_inputs:
            self._decode_groups.append((decode, offset))
            offset += decode.shape[1]
        total_tokens = self.prefill_seq + self.decode_seq * self.decode_steps
        tokens_per_iteration = self.batch_size * total_tokens
        self.cache = allocate_kv_cache(
            batch_size=self.batch_size,
            total_tokens=total_tokens,
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            device=self.device,
            dtype=self.tensor_dtype,
        )
        self.runtime_recipe = recipe
        self.register_workload_metadata(tokens_per_iteration=float(tokens_per_iteration))
        self._batch_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        self._seq_meta_tensor = torch.empty(3, dtype=torch.int64, device="cpu")
        self._batch_size_tensor[0] = self.batch_size
        self._seq_meta_tensor[0] = self.prefill_seq
        self._seq_meta_tensor[1] = self.decode_seq
        self._seq_meta_tensor[2] = self.decode_steps
        self._verify_output_buffer = torch.empty(
            2,
            self.batch_size,
            1,
            1,
            min(8, self.hidden_dim // self.num_heads),
            device=self.device,
            dtype=torch.float32,
        )
        self._calibrate_fp8(recipe)
        self._warmup_runtime(recipe)
        self._cache_output_ready = False
        torch.cuda.synchronize()

    def _calibrate_fp8(self, recipe) -> None:
        if self.model is None or self.cache is None or recipe is None:
            return
        with torch.inference_mode(), te_autocast(enabled=True, recipe=recipe, calibrating=True):
            for prefill, offset in self._prefill_groups:
                _ = self.model(prefill, self.cache, offset)
            for decode, offset in self._decode_groups:
                _ = self.model(decode, self.cache, offset)

    def _warmup_runtime(self, recipe) -> None:
        if self.model is None or self.cache is None or recipe is None:
            return
        with torch.inference_mode(), te_autocast(enabled=True, recipe=recipe):
            for prefill, offset in self._prefill_groups:
                _ = self.model(prefill, self.cache, offset)
            for decode, offset in self._decode_groups:
                _ = self.model(decode, self.cache, offset)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.cache is None
            or self.runtime_recipe is None
            or not self._prefill_groups
            or not self._decode_groups
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode(), te_autocast(enabled=True, recipe=self.runtime_recipe):
            for prefill, offset in self._prefill_groups:
                _ = self.model(prefill, self.cache, offset)
            for decode, offset in self._decode_groups:
                _ = self.model(decode, self.cache, offset)
        self._mark_cache_output_ready()

    def _mark_cache_output_ready(self) -> None:
        if self.cache is None:
            raise RuntimeError("Benchmark cache not initialized")
        self._cache_output_ready = True

    def _build_verification_output(self) -> torch.Tensor:
        if self.cache is None or not self._cache_output_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        k_slice = self.cache.cache_k[
            :,
            : min(1, self.cache.cache_k.shape[1]),
            :1,
            : min(8, self.cache.cache_k.shape[-1]),
        ]
        v_slice = self.cache.cache_v[
            :,
            : min(1, self.cache.cache_v.shape[1]),
            :1,
            : min(8, self.cache.cache_v.shape[-1]),
        ]
        self._verify_output_buffer[0].copy_(k_slice)
        self._verify_output_buffer[1].copy_(v_slice)
        return self._verify_output_buffer

    def capture_verification_payload(self) -> None:
        self.output = self._build_verification_output()
        if self._batch_size_tensor is None or self._seq_meta_tensor is None:
            raise RuntimeError("setup() must initialize verification metadata tensors")
        self._set_verification_payload(
            inputs={
                "batch_size": self._batch_size_tensor,
                "seq_meta": self._seq_meta_tensor,
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.prefill_inputs = []
        self.decode_inputs = []
        self._prefill_groups = []
        self._decode_groups = []
        self.cache = None
        self.model = None
        self.output = None
        self._batch_size_tensor = None
        self._seq_meta_tensor = None
        self._verify_output_buffer = None
        self._cache_output_ready = False
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.cache is None:
            return "Cache not initialized"
        if not cache_is_finite(self.cache):
            return "Non-finite entries detected in KV cache"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5, deterministic=False, enable_memory_tracking=True)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        total_tokens = self.prefill_seq + self.decode_seq * self.decode_steps
        return {
            "kv_cache.batch_size": float(self.batch_size),
            "kv_cache.seq_len": float(total_tokens),
            "kv_cache.hidden_dim": float(self.hidden_dim),
        }

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheBenchmark()
