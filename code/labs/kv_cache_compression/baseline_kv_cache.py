"""Per-tensor delayed-scaling FP8 compute with BF16 KV-cache storage."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.env import apply_env_defaults
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.kv_cache_compression.accuracy import (
    assert_cache_accuracy, cache_accuracy, load_accuracy_limits, reference_cache,
)
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
    """FP8 compute benchmark (prefill + decode); the cache itself remains BF16."""

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
        self._accuracy_variant = "fp8"
        self._accuracy_limits = None
        self._accuracy_metrics: dict[str, float] = {}

    def _resolve_device(self) -> torch.device:
        return resolve_device()

    def setup(self) -> None:
        self._setup_with_recipe(self.fp8_recipe)

    def _setup_with_recipe(self, recipe, *, require_accuracy_policy: bool = True) -> None:
        if not TE_AVAILABLE or recipe is None:
            raise RuntimeError(f"SKIPPED: Transformer Engine not available: {TE_IMPORT_ERROR}")

        self._accuracy_limits = (load_accuracy_limits(self._accuracy_variant)
                                 if require_accuracy_policy else None)
        self.device = self._resolve_device()
        # Preserve common BF16 weights for an independent unquantized reference.
        # TE autocast still selects FP8/NVFP4 GEMMs during the benchmark.
        # The harness owns RNG seeding; do not reset it inside setup().
        with quantized_model_init(enabled=False):
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
        self._verify_output_buffer = None  # Full snapshot is allocated outside timing.
        if recipe.delayed():
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

    def measure_accuracy(self) -> dict[str, float]:
        if self.cache is None or self.model is None or not self._cache_output_ready:
            raise RuntimeError("benchmark_fn() must run before accuracy measurement")
        reference = reference_cache(self.model, self._prefill_groups + self._decode_groups, self.cache)
        if self._accuracy_limits is None:
            return cache_accuracy(self.cache, reference)
        return assert_cache_accuracy(self.cache, reference, self._accuracy_limits)

    def _build_verification_output(self) -> torch.Tensor:
        if self.cache is None or not self._cache_output_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._accuracy_limits is None:
            raise RuntimeError("Uncalibrated KV run cannot produce an accepted verification payload")
        self._accuracy_metrics = self.measure_accuracy()
        # Snapshot all tokens, heads and channels; no one-token/eight-value probe.
        return torch.cat((self.cache.cache_k.reshape(-1), self.cache.cache_v.reshape(-1)))

    def capture_verification_payload(self) -> None:
        self.output = self._build_verification_output()
        if self._batch_size_tensor is None or self._seq_meta_tensor is None:
            raise RuntimeError("setup() must initialize verification metadata tensors")
        self._set_verification_payload(
            inputs={
                "batch_size": self._batch_size_tensor,
                "seq_meta": self._seq_meta_tensor,
                **{f"prefill_{i}": value for i, value in enumerate(self.prefill_inputs)},
                "decode": self.decode_inputs[0],
                "qkv_weight": self.model.qkv.weight.detach(),
                "qkv_bias": self.model.qkv.bias.detach(),
                "ln_weight": self.model.ln.weight.detach(),
                "ln_bias": self.model.ln.bias.detach(),
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(self._accuracy_limits.pairwise_rtol, self._accuracy_limits.pairwise_atol),
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
        self._accuracy_metrics = {}
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.cache is None:
            return "Cache not initialized"
        if not cache_is_finite(self.cache):
            return "Non-finite entries detected in KV cache"
        if self._accuracy_limits is None:
            return "Uncalibrated KV accuracy policy; runtime acceptance is pending"
        try:
            self._accuracy_metrics = self.measure_accuracy()
        except (AssertionError, RuntimeError) as exc:
            return str(exc)
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5, deterministic=False, enable_memory_tracking=True)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        total_tokens = self.prefill_seq + self.decode_seq * self.decode_steps
        if self.cache is None:
            raise RuntimeError("Cache storage metrics require setup()")
        tensors = (self.cache.cache_k, self.cache.cache_v)
        storage_bytes = sum(t.numel() * t.element_size() for t in tensors)
        bf16_bytes = sum(t.numel() * 2 for t in tensors)
        return {
            "kv_cache.storage_bytes": float(storage_bytes),
            "kv_cache.storage_bits_per_element": float(8 * storage_bytes / sum(t.numel() for t in tensors)),
            "kv_cache.compression_ratio": float(bf16_bytes / storage_bytes),
            **{f"kv_cache.accuracy.{k}": v for k, v in self._accuracy_metrics.items()},
            "kv_cache.batch_size": float(self.batch_size),
            "kv_cache.seq_len": float(total_tokens),
            "kv_cache.hidden_dim": float(self.hidden_dim),
        }

    def get_optimization_goal(self) -> str:
        """Compare projection-compute latency; both caches occupy the same bytes."""
        return "speed"


def get_benchmark() -> BaseBenchmark:
    return BaselineKVCacheBenchmark()
