"""Full-cache accuracy checks and independent unquantized BF16 reference.

The checked-in policy defines conservative arithmetic requirements from the
operand formats.  A caller must still select that policy explicitly: merely
collecting candidate errors never changes an acceptance bound.
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


REFERENCE_ID = "pytorch-unquantized-bf16-full-cache-v1"
POLICY_ID = "kv-cache-projection-format-ceilings-v1"
DEFAULT_POLICY_PATH = Path(__file__).with_name("accuracy_policy.json")
WORKLOAD = {
    "batch_size": 8,
    "hidden_dim": 16384,
    "num_heads": 64,
    "prefill_seq": 4096,
    "decode_seq": 128,
    "decode_steps": 128,
    "storage_dtype": "bfloat16",
}
QUALIFICATION_VARIANTS = ["fp8", "nvfp4"]
QUALIFICATION_RECEIPTS = [
    {"cohort": "nominal", "seeds": [2026]},
    {"cohort": "holdout", "seeds": [2027, 2029]},
    {"cohort": "alternating", "seeds": [2039]},
    {"cohort": "sparse_outlier", "seeds": [2053]},
]


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


# These ceilings are set from the quantized operand representations, before
# candidate execution. E4M3 has three stored fraction bits, so half an ULP at
# a normal binade is 2^-4. E2M1 has one stored fraction bit, so the analogous
# bound is 2^-2. The full-cache aggregate and global-maximum requirements use
# those format-scale bounds. The pairwise check compares FP8 and NVFP4 caches
# only after each arm passes its independent reference check, so it uses the
# coarser E2M1 ceiling plus one E4M3-scale absolute allowance near zero.
ENGINEERING_CEILINGS = {
    "fp8": AccuracyLimits(
        relative_l2=2.0**-4,
        normalized_max_abs=2.0**-4,
        pairwise_rtol=2.0**-2,
        pairwise_atol=2.0**-4,
    ),
    "nvfp4": AccuracyLimits(
        relative_l2=2.0**-2,
        normalized_max_abs=2.0**-2,
        pairwise_rtol=2.0**-2,
        pairwise_atol=2.0**-4,
    ),
}


def _limits_from_item(item: dict) -> AccuracyLimits:
    try:
        return AccuracyLimits(**{
            name: float(item[name])
            for name in ("relative_l2", "normalized_max_abs", "pairwise_rtol", "pairwise_atol")
        })
    except KeyError as exc:
        raise ValueError(f"KV accuracy policy missing {exc.args[0]}") from exc


def load_accuracy_policy(path: Path) -> dict:
    """Load and validate the reviewed policy contract, without running a candidate."""
    policy = json.loads(path.read_text())
    schema_version = policy.get("schema_version")
    if schema_version == 1:
        # Preserve exact synthetic fixture policies. A nonzero legacy policy has
        # no reference/workload identity and cannot accept a quantized run.
        for variant in ("fp8", "nvfp4"):
            if variant not in policy:
                continue
            limits = _limits_from_item(policy[variant])
            if any(getattr(limits, name) != 0 for name in (
                "relative_l2", "normalized_max_abs", "pairwise_rtol", "pairwise_atol"
            )):
                raise ValueError("schema_version=1 is permitted only for exact-zero test policies")
        return policy
    if schema_version != 2:
        raise ValueError("KV accuracy policy requires schema_version=2")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError(f"KV accuracy policy requires policy_id={POLICY_ID}")
    if policy.get("reference", {}).get("id") != REFERENCE_ID:
        raise ValueError(f"KV accuracy policy requires reference.id={REFERENCE_ID}")
    if policy.get("workload") != WORKLOAD:
        raise ValueError("KV accuracy policy workload does not match the benchmark contract")
    qualification = policy.get("qualification", {})
    if (qualification.get("variants") != QUALIFICATION_VARIANTS or
            qualification.get("required_receipts") != QUALIFICATION_RECEIPTS):
        raise ValueError("KV accuracy policy qualification matrix does not match the source contract")
    variants = policy.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("KV accuracy policy requires a variants object")
    for variant, ceiling in ENGINEERING_CEILINGS.items():
        if variant not in variants:
            raise ValueError(f"KV accuracy policy missing variant {variant}")
        limits = _limits_from_item(variants[variant])
        for name in ("relative_l2", "normalized_max_abs", "pairwise_rtol", "pairwise_atol"):
            if getattr(limits, name) > getattr(ceiling, name):
                raise ValueError(
                    f"{variant}.{name} exceeds the source-defined engineering ceiling "
                    f"{getattr(ceiling, name):.8g}"
                )
    return policy


def load_accuracy_limits(variant: str) -> AccuracyLimits:
    path = os.environ.get("AISP_KV_CACHE_ACCURACY_POLICY")
    if not path:
        raise RuntimeError(
            "KV compute accuracy is uncalibrated because no policy is selected: "
            "AISP_KV_CACHE_ACCURACY_POLICY is required. "
            f"The reviewed repository policy is {DEFAULT_POLICY_PATH}. A configured policy alone "
            "is not target-hardware accuracy evidence."
        )
    policy = load_accuracy_policy(Path(path))
    try:
        item = policy[variant] if policy["schema_version"] == 1 else policy["variants"][variant]
    except KeyError as exc:
        raise ValueError(f"KV accuracy policy missing variant {variant}") from exc
    return _limits_from_item(item)


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
