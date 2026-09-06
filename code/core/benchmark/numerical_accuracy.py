"""Scale-invariant numerical checks for low-precision benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


def _require_finite_nonnegative_real(value: object, *, name: str) -> float:
    """Reject booleans and malformed values before comparing error budgets."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


@dataclass(frozen=True)
class ScaleInvariantAccuracyLimits:
    """Full-output relative L2 and normalized maximum-error limits."""

    relative_l2: float
    normalized_max_abs: float

    def __post_init__(self) -> None:
        for name in ("relative_l2", "normalized_max_abs"):
            value = _require_finite_nonnegative_real(getattr(self, name), name=name)
            if value >= 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")

    def check(self, errors: dict[str, float], *, label: str) -> None:
        if set(errors) != {"relative_l2", "normalized_max_abs"}:
            raise ValueError("Both full-output scale-invariant errors are required")
        validated = {
            name: _require_finite_nonnegative_real(value, name=name)
            for name, value in errors.items()
        }
        failures = [
            f"{name}={value:.8g} > {getattr(self, name):.8g}"
            for name, value in validated.items()
            if value > getattr(self, name)
        ]
        if failures:
            raise AssertionError(f"{label} accuracy failed: " + "; ".join(failures))


def measure_scale_invariant_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> dict[str, float]:
    """Compare every element without unstable elementwise relative errors."""

    if actual.shape != expected.shape or actual.ndim == 0 or not actual.numel():
        raise AssertionError(f"{label}: output/reference shape differs or is empty")
    if actual.device != expected.device:
        raise AssertionError(f"{label}: output/reference devices differ")
    if not actual.is_floating_point() or not expected.is_floating_point():
        raise AssertionError(f"{label}: output/reference must be floating point")
    if actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr():
        raise AssertionError(f"{label}: reference aliases candidate storage")

    error_squared = reference_squared = max_error = max_reference = 0.0
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    chunk_size = 1 << 20
    for start in range(0, actual.numel(), chunk_size):
        got = flat_actual[start:start + chunk_size].double()
        ref = flat_expected[start:start + chunk_size].double()
        if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(ref).all()):
            raise AssertionError(f"{label}: nonfinite output/reference")
        delta = got - ref
        error_squared += float(delta.square().sum())
        reference_squared += float(ref.square().sum())
        max_error = max(max_error, float(delta.abs().max()))
        max_reference = max(max_reference, float(ref.abs().max()))

    return {
        "relative_l2": math.sqrt(error_squared / reference_squared)
        if reference_squared
        else (0.0 if error_squared == 0 else math.inf),
        "normalized_max_abs": max_error / max_reference
        if max_reference
        else (0.0 if max_error == 0 else math.inf),
    }


def low_precision_attention_limits(dtype: torch.dtype) -> ScaleInvariantAccuracyLimits:
    """Return the reviewed one-representation-epsilon attention policy."""

    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Attention accuracy policy does not cover {dtype}")
    epsilon = float(torch.finfo(dtype).eps)
    return ScaleInvariantAccuracyLimits(
        relative_l2=epsilon,
        normalized_max_abs=epsilon,
    )


def assert_low_precision_attention_accuracy(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    """Require a complete FP16/BF16 result within one dtype epsilon of a reference."""

    limits = low_precision_attention_limits(actual.dtype)
    errors = measure_scale_invariant_errors(actual, expected, label="attention")
    limits.check(errors, label="attention")
    return errors
