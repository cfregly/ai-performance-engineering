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

        self.prefill_inputs, self.decode_inputs = build_token_batches(
            batch_size=self.batch_size,
            prefill_seq=self.prefill_seq // 2,  # two prefill windows
            decode_seq=self.decode_seq,
            decode_steps=self.decode_steps,
            hidden_dim=self.hidden_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        )
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
        self._calibrate_fp8(recipe)
        self._warmup_runtime(recipe)
        self._cache_output_ready = False
        torch.cuda.synchronize()

    def _calibrate_fp8(self, recipe) -> None:
        if self.model is None or self.cache is None or recipe is None:
            return
        with te_autocast(enabled=True, recipe=recipe, calibrating=True):
            offset = 0
            for prefill in self.prefill_inputs:
                _ = self.model(prefill, self.cache, offset)
                offset += prefill.shape[1]
            for decode in self.decode_inputs:
                _ = self.model(decode, self.cache, offset)
                offset += decode.shape[1]

    def _warmup_runtime(self, recipe) -> None:
        if self.model is None or self.cache is None or recipe is None:
            return
        with te_autocast(enabled=True, recipe=recipe):
            offset = 0
            for prefill in self.prefill_inputs:
                _ = self.model(prefill, self.cache, offset)
                offset += prefill.shape[1]
            for decode in self.decode_inputs:
                _ = self.model(decode, self.cache, offset)
                offset += decode.shape[1]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.model is None or self.cache is None or self.runtime_recipe is None:
            raise RuntimeError("Benchmark not initialized")
        offset = 0
        with te_autocast(enabled=True, recipe=self.runtime_recipe):
            for prefill in self.prefill_inputs:
                _ = self.model(prefill, self.cache, offset)
                offset += prefill.shape[1]
            for decode in self.decode_inputs:
                _ = self.model(decode, self.cache, offset)
                offset += decode.shape[1]
        self._mark_cache_output_ready()

    def _mark_cache_output_ready(self) -> None:
        if self.cache is None:
            raise RuntimeError("Benchmark cache not initialized")
        self._cache_output_ready = True

    def _build_verification_output(self) -> torch.Tensor:
        if self.cache is None or not self._cache_output_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
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
        return torch.stack([k_slice, v_slice], dim=0).detach().float().clone()

    def capture_verification_payload(self) -> None:
        self.output = self._build_verification_output()
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor([self.batch_size], dtype=torch.int64, device="cpu"),
                "seq_meta": torch.tensor(
                    [self.prefill_seq, self.decode_seq, self.decode_steps], dtype=torch.int64, device="cpu"
                ),
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
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
        self.cache = None
        self.model = None
        self.output = None
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
