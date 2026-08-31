"""Full-cache accuracy checks and independent unquantized BF16 reference.

No quantization acceptance threshold has been calibrated for this workload. A
missing policy is an error, not permission to use a permissive default.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from labs.kv_cache_compression.kv_cache_common import KVCache


@dataclass(frozen=True)
class AccuracyLimits:
    relative_l2: float
    normalized_max_abs: float
    pairwise_rtol: float
    pairwise_atol: float

    def __post_init__(self):
        for name in ("relative_l2", "normalized_max_abs", "pairwise_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not math.isfinite(self.pairwise_atol) or self.pairwise_atol < 0:
            raise ValueError("pairwise_atol must be finite and nonnegative")


def load_accuracy_limits(variant: str) -> AccuracyLimits:
    path = os.environ.get("AISP_KV_CACHE_ACCURACY_POLICY")
    if not path:
        raise RuntimeError(
            "KV compute accuracy is uncalibrated: AISP_KV_CACHE_ACCURACY_POLICY is required. "
            "Use python -m labs.kv_cache_compression.calibrate_accuracy to collect errors; "
            "a configured policy alone is not measured accuracy evidence."
        )
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("KV accuracy policy requires schema_version=1")
    return AccuracyLimits(**policy[variant])


def reference_cache(model, groups, cache: KVCache) -> KVCache:
    """Compute every K/V from original BF16 weights, bypassing TE and its packing."""
    end = 0
    for tokens, offset in groups:
        if offset != end or tokens.shape[0] != cache.cache_k.shape[0]:
            raise ValueError("Reference groups must cover the cache in order without gaps")
        end += tokens.shape[1]
    if end != cache.cache_k.shape[1] or cache.cache_k.shape != cache.cache_v.shape:
        raise ValueError("Reference groups must cover the entire K/V cache")
    reference = KVCache(torch.empty_like(cache.cache_k), torch.empty_like(cache.cache_v))
    weight, bias = model.qkv.weight.detach(), model.qkv.bias.detach()
    if weight.dtype != torch.bfloat16 or type(weight) is not torch.Tensor:
        raise RuntimeError("Independent reference requires unquantized BF16 model weights")
    with torch.inference_mode():
        for tokens, offset in groups:
            # Bounded temporary storage; every token, head and channel is covered.
            for start in range(0, tokens.shape[1], 128):
                part = tokens[:, start:start + 128]
                x = F.layer_norm(part, (model.hidden_dim,), model.ln.weight,
                                 model.ln.bias, model.ln.eps)
                qkv = F.linear(x, weight, bias).reshape(
                    part.shape[0], part.shape[1], 3, model.num_heads, model.head_dim)
                destination = slice(offset + start, offset + start + part.shape[1])
                reference.cache_k[:, destination].copy_(qkv[:, :, 1])
                reference.cache_v[:, destination].copy_(qkv[:, :, 2])
    return reference


def cache_accuracy(actual: KVCache, expected: KVCache) -> dict[str, float]:
    """Measure full K/V tensors without checksum cancellation or a giant FP64 copy."""
    result = {}
    for name in ("cache_k", "cache_v"):
        got, ref = getattr(actual, name), getattr(expected, name)
        if got.shape != ref.shape or got.dtype != ref.dtype or not got.numel():
            raise AssertionError(f"{name}: empty or different output shape/dtype")
        if got.untyped_storage().data_ptr() == ref.untyped_storage().data_ptr():
            raise AssertionError(f"{name}: reference aliases candidate storage")
        error_squared = reference_squared = max_error = max_reference = 0.0
        flat_got, flat_ref = got.reshape(-1), ref.reshape(-1)
        for start in range(0, got.numel(), 1 << 20):
            g, r = flat_got[start:start + (1 << 20)].double(), flat_ref[start:start + (1 << 20)].double()
            if not torch.isfinite(g).all() or not torch.isfinite(r).all():
                raise AssertionError(f"{name}: non-finite output/reference")
            error = g - r
            error_squared += float(torch.sum(error * error))
            reference_squared += float(torch.sum(r * r))
            max_error = max(max_error, float(error.abs().max()))
            max_reference = max(max_reference, float(r.abs().max()))
        result[f"{name}.relative_l2"] = (math.sqrt(error_squared / reference_squared)
            if reference_squared else (0.0 if error_squared == 0 else math.inf))
        result[f"{name}.normalized_max_abs"] = (max_error / max_reference
            if max_reference else (0.0 if max_error == 0 else math.inf))
    return result


def assert_cache_accuracy(actual: KVCache, expected: KVCache, limits: AccuracyLimits) -> dict[str, float]:
    metrics = cache_accuracy(actual, expected)
    failures = [f"{name}={value:.8g} > {getattr(limits, name.split('.')[-1]):.8g}"
                for name, value in metrics.items()
                if not math.isfinite(value) or value > getattr(limits, name.split('.')[-1])]
    if failures:
        raise AssertionError("KV cache accuracy failed: " + "; ".join(failures))
    return metrics
