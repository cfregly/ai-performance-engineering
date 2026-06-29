"""Optimized KV-cache benchmark using NVFP4 block scaling."""

from __future__ import annotations

from typing import Optional

import torch

from labs.kv_cache_compression.baseline_kv_cache import (
    TE_AVAILABLE,
    TE_IMPORT_ERROR,
    BaselineKVCacheBenchmark,
)

if TE_AVAILABLE:
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.pytorch import autocast as te_autocast
    from transformer_engine.pytorch import is_nvfp4_available
else:  # pragma: no cover
    te_autocast = is_nvfp4_available = te_recipe = None  # type: ignore


class OptimizedKVCacheNVFP4Benchmark(BaselineKVCacheBenchmark):
    """Calibrate in FP8 and then run NVFP4 for KV-cache heavy attention."""

    def __init__(self) -> None:
        super().__init__()
        self.nvfp4_recipe = (
            te_recipe.NVFP4BlockScaling(calibration_steps=20, amax_history_len=16, fp4_tensor_block=16)
            if TE_AVAILABLE
            else None
        )
        self.nvfp4_active = False
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not TE_AVAILABLE:
            raise RuntimeError(f"Transformer Engine not available: {TE_IMPORT_ERROR}")
        if self.nvfp4_recipe is None:
            raise RuntimeError("NVFP4 recipe not available in this Transformer Engine version")
        if not is_nvfp4_available():
            raise RuntimeError("NVFP4 kernels unavailable on this hardware/driver.")
        self._setup_with_recipe(self.nvfp4_recipe)
        self.nvfp4_active = True

    def validate_result(self) -> Optional[str]:
        return super().validate_result()

    def benchmark_fn(self) -> None:
        if self.model is None or self.cache is None or not self._prefill_groups or not self._decode_groups:
            raise RuntimeError("Benchmark not initialized")
        recipe = self.runtime_recipe
        if recipe is None:
            raise RuntimeError("No NVFP4 recipe available for benchmark")
        with torch.inference_mode(), te_autocast(enabled=True, recipe=recipe):
            for prefill, offset in self._prefill_groups:
                _ = self.model(prefill, self.cache, offset)
            for decode, offset in self._decode_groups:
                _ = self.model(decode, self.cache, offset)
        self._mark_cache_output_ready()

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

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVFP4-specific metrics."""
        metrics = super().get_custom_metrics()
        if metrics is None:
            raise RuntimeError("Base KV-cache metrics missing for NVFP4 benchmark")
        metrics = dict(metrics)
        metrics.update({
            "kv_cache.nvfp4_active": 1.0 if self.nvfp4_active else 0.0,
            "kv_cache.compression_ratio": 4.0 if self.nvfp4_active else 2.0,  # NVFP4=4x, FP8=2x
        })
        return metrics

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"


def get_benchmark() -> BaselineKVCacheBenchmark:
    return OptimizedKVCacheNVFP4Benchmark()
