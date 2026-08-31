"""NVFP4 projection compute with unchanged BF16 KV-cache storage."""

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
    """Run NVFP4 GEMMs for KV-cache heavy attention; stored K/V remain BF16."""

    def __init__(self) -> None:
        super().__init__()
        self.nvfp4_recipe = (
            te_recipe.NVFP4BlockScaling()
            if TE_AVAILABLE
            else None
        )
        self._accuracy_variant = "nvfp4"
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

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVFP4-specific metrics."""
        metrics = super().get_custom_metrics()
        if metrics is None:
            raise RuntimeError("Base KV-cache metrics missing for NVFP4 benchmark")
        metrics = dict(metrics)
        metrics.update({
            "kv_cache.nvfp4_active": 1.0 if self.nvfp4_active else 0.0,
        })
        return metrics

    def get_optimization_goal(self) -> str:
        """Compare compute latency with unchanged BF16 cache storage."""
        return "speed"


def get_benchmark() -> BaselineKVCacheBenchmark:
    return OptimizedKVCacheNVFP4Benchmark()
