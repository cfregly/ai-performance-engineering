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
PAIRWISE_ENVELOPE_METHOD = "shared-reference-max-triangle-envelope-v1"
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
    # Retained only for schema-v1 exact-zero fixtures. Schema-v2 policies use
    # PairwiseEnvelope because raw torch.allclose rtol/atol have different units.
    pairwise_rtol: float = 0.0
    pairwise_atol: float = 0.0

    def __post_init__(self):
        for name in (
            "relative_l2",
            "normalized_max_abs",
            "pairwise_rtol",
            "pairwise_atol",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not boolean")
        for name in ("relative_l2", "normalized_max_abs", "pairwise_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not math.isfinite(self.pairwise_atol) or self.pairwise_atol < 0:
            raise ValueError("pairwise_atol must be finite and nonnegative")


@dataclass(frozen=True)
class PairwiseEnvelope:
    """Reference-normalized full-output envelope shared by both benchmark arms."""

    normalized_max_abs: float
    output_rtol: float = 0.0
    method: str = PAIRWISE_ENVELOPE_METHOD
    reference_id: str = REFERENCE_ID

    def __post_init__(self) -> None:
        for name in ("normalized_max_abs", "output_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"pairwise envelope {name} must be numeric, not boolean")
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(
                    f"pairwise envelope {name} must be finite and in [0, 1)"
                )
        if self.output_rtol != 0:
            raise ValueError("pairwise envelope output_rtol must be zero")
        if self.method != PAIRWISE_ENVELOPE_METHOD:
            raise ValueError(f"pairwise envelope method must be {PAIRWISE_ENVELOPE_METHOD}")
        if self.reference_id != REFERENCE_ID:
            raise ValueError(f"pairwise envelope reference_id must be {REFERENCE_ID}")


@dataclass(frozen=True)
class CacheAccuracyEvidence:
    metrics: dict[str, float]
    reference_max_abs: float


# These ceilings are set from the quantized operand representations, before
# candidate execution. E4M3 has three stored fraction bits, so half an ULP at
# a normal binade is 2^-4. E2M1 has one stored fraction bit, so the analogous
# bound is 2^-2. The full-cache aggregate and global-maximum requirements use
# those format-scale bounds. The pairwise envelope is derived separately from
# the sum of both selected normalized-maximum limits and the shared reference
# magnitude. It is never fitted to a candidate output.
ENGINEERING_CEILINGS = {
    "fp8": AccuracyLimits(
        relative_l2=2.0**-4,
        normalized_max_abs=2.0**-4,
    ),
    "nvfp4": AccuracyLimits(
        relative_l2=2.0**-2,
        normalized_max_abs=2.0**-2,
    ),
}


def _limits_from_item(item: dict) -> AccuracyLimits:
    def policy_float(name: str, *, default: float | None = None) -> float:
        if name not in item:
            if default is not None:
                return default
            raise KeyError(name)
        value = item[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"KV accuracy policy {name} must be numeric, not boolean")
        return float(value)

    try:
        return AccuracyLimits(
            relative_l2=policy_float("relative_l2"),
            normalized_max_abs=policy_float("normalized_max_abs"),
            pairwise_rtol=policy_float("pairwise_rtol", default=0.0),
            pairwise_atol=policy_float("pairwise_atol", default=0.0),
        )
    except KeyError as exc:
        raise ValueError(f"KV accuracy policy missing {exc.args[0]}") from exc


def _pairwise_envelope_from_policy(policy: dict) -> PairwiseEnvelope:
    if policy.get("schema_version") == 1:
        return PairwiseEnvelope(normalized_max_abs=0.0)
    item = policy.get("pairwise_envelope")
    if not isinstance(item, dict):
        raise ValueError("KV accuracy policy requires a pairwise_envelope object")
    if item.get("method") != PAIRWISE_ENVELOPE_METHOD:
        raise ValueError(
            f"KV pairwise envelope requires method={PAIRWISE_ENVELOPE_METHOD}"
        )
    if item.get("reference_id") != REFERENCE_ID:
        raise ValueError(f"KV pairwise envelope requires reference_id={REFERENCE_ID}")
    if item.get("variants") != QUALIFICATION_VARIANTS:
        raise ValueError("KV pairwise envelope variants do not match the source contract")
    if item.get("coefficient") != "sum_variant_normalized_max_abs":
        raise ValueError(
            "KV pairwise envelope coefficient must be sum_variant_normalized_max_abs"
        )
    output_rtol = item.get("output_rtol")
    if isinstance(output_rtol, bool) or not isinstance(output_rtol, int | float):
        raise ValueError("KV pairwise envelope output_rtol must be numeric")
    variants = policy.get("variants", {})
    coefficient = sum(
        _limits_from_item(variants[variant]).normalized_max_abs
        for variant in QUALIFICATION_VARIANTS
    )
    return PairwiseEnvelope(
        normalized_max_abs=coefficient,
        output_rtol=float(output_rtol),
        method=str(item["method"]),
        reference_id=str(item["reference_id"]),
    )


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
            if any(
                getattr(limits, name) != 0
                for name in (
                    "relative_l2",
                    "normalized_max_abs",
                    "pairwise_rtol",
                    "pairwise_atol",
                )
            ):
                raise ValueError(
                    "schema_version=1 is permitted only for exact-zero test policies"
                )
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
    if (
        qualification.get("variants") != QUALIFICATION_VARIANTS
        or qualification.get("required_receipts") != QUALIFICATION_RECEIPTS
    ):
        raise ValueError(
            "KV accuracy policy qualification matrix does not match the source contract"
        )
    variants = policy.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("KV accuracy policy requires a variants object")
    for variant, ceiling in ENGINEERING_CEILINGS.items():
        if variant not in variants:
            raise ValueError(f"KV accuracy policy missing variant {variant}")
        limits = _limits_from_item(variants[variant])
        legacy_pairwise = {
            name
            for name in ("pairwise_rtol", "pairwise_atol")
            if name in variants[variant]
        }
        if legacy_pairwise:
            raise ValueError(
                f"{variant} uses obsolete raw allclose fields: {sorted(legacy_pairwise)}"
            )
        for name in ("relative_l2", "normalized_max_abs"):
            if getattr(limits, name) > getattr(ceiling, name):
                raise ValueError(
                    f"{variant}.{name} exceeds the source-defined engineering ceiling "
                    f"{getattr(ceiling, name):.8g}"
                )
    envelope = _pairwise_envelope_from_policy(policy)
    source_envelope_ceiling = sum(
        limits.normalized_max_abs for limits in ENGINEERING_CEILINGS.values()
    )
    if envelope.normalized_max_abs > source_envelope_ceiling:
        raise ValueError(
            "pairwise envelope exceeds the source-derived normalized-maximum ceiling "
            f"{source_envelope_ceiling:.8g}"
        )
    return policy


def load_accuracy_contract(variant: str) -> tuple[AccuracyLimits, PairwiseEnvelope]:
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
    return _limits_from_item(item), _pairwise_envelope_from_policy(policy)


def load_accuracy_limits(variant: str) -> AccuracyLimits:
    return load_accuracy_contract(variant)[0]


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


def cache_accuracy_evidence(actual: KVCache, expected: KVCache) -> CacheAccuracyEvidence:
    """Measure full K/V tensors without checksum cancellation or a giant FP64 copy."""
    result = {}
    reference_max_abs = 0.0
    for name in ("cache_k", "cache_v"):
        got, ref = getattr(actual, name), getattr(expected, name)
        if got.shape != ref.shape or got.dtype != ref.dtype or not got.numel():
            raise AssertionError(f"{name}: empty or different output shape/dtype")
        if got.untyped_storage().data_ptr() == ref.untyped_storage().data_ptr():
            raise AssertionError(f"{name}: reference aliases candidate storage")
        error_squared = reference_squared = max_error = max_reference = 0.0
        flat_got, flat_ref = got.reshape(-1), ref.reshape(-1)
        for start in range(0, got.numel(), 1 << 20):
            g = flat_got[start : start + (1 << 20)].double()
            r = flat_ref[start : start + (1 << 20)].double()
            if not torch.isfinite(g).all() or not torch.isfinite(r).all():
                raise AssertionError(f"{name}: non-finite output/reference")
            error = g - r
            error_squared += float(torch.sum(error * error))
            reference_squared += float(torch.sum(r * r))
            max_error = max(max_error, float(error.abs().max()))
            max_reference = max(max_reference, float(r.abs().max()))
        reference_max_abs = max(reference_max_abs, max_reference)
        result[f"{name}.relative_l2"] = (
            math.sqrt(error_squared / reference_squared)
            if reference_squared
            else (0.0 if error_squared == 0 else math.inf)
        )
        result[f"{name}.normalized_max_abs"] = (
            max_error / max_reference
            if max_reference
            else (0.0 if max_error == 0 else math.inf)
        )
    return CacheAccuracyEvidence(metrics=result, reference_max_abs=reference_max_abs)


def cache_accuracy(actual: KVCache, expected: KVCache) -> dict[str, float]:
    return cache_accuracy_evidence(actual, expected).metrics


def _assert_accuracy_evidence(
    evidence: CacheAccuracyEvidence,
    limits: AccuracyLimits,
) -> CacheAccuracyEvidence:
    failures = [
        f"{name}={value:.8g} > {getattr(limits, name.split('.')[-1]):.8g}"
        for name, value in evidence.metrics.items()
        if not math.isfinite(value) or value > getattr(limits, name.split(".")[-1])
    ]
    if failures:
        raise AssertionError("KV cache accuracy failed: " + "; ".join(failures))
    return evidence


def assert_cache_accuracy_evidence(
    actual: KVCache,
    expected: KVCache,
    limits: AccuracyLimits,
) -> CacheAccuracyEvidence:
    return _assert_accuracy_evidence(cache_accuracy_evidence(actual, expected), limits)


def assert_cache_accuracy(
    actual: KVCache,
    expected: KVCache,
    limits: AccuracyLimits,
) -> dict[str, float]:
    return assert_cache_accuracy_evidence(actual, expected, limits).metrics


def pairwise_absolute_tolerance(
    envelope: PairwiseEnvelope,
    reference_max_abs: float,
) -> float:
    if not math.isfinite(reference_max_abs) or reference_max_abs < 0:
        raise ValueError("pairwise reference_max_abs must be finite and nonnegative")
    return envelope.normalized_max_abs * reference_max_abs
