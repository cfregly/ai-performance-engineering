"""Fail-closed comparison of already executed benchmark pairs."""

from __future__ import annotations

import math
from typing import Any

import torch

from core.benchmark.models import BenchmarkRun
from core.benchmark.runtime_comparison import (
    ExecutedRuntimeComparisonError,
    compare_executed_runtime_provenance,
)
from core.benchmark.verification import (
    InputSignature,
    ToleranceSpec,
    coerce_input_signature,
    get_output_tolerance,
    get_signature_equivalence_spec,
    signature_workload_dict,
)
from core.benchmark.verify_runner import VerifyRunner


class VerifiedComparisonError(RuntimeError):
    """Raised when captured verification evidence cannot admit a comparison."""

    def __init__(self, code: str, detail: str, *, evidence: Any = None) -> None:
        self.code = code
        self.detail = detail
        self.evidence = evidence
        super().__init__(f"VERIFIED COMPARISON FAILED [{code}]: {detail}")


def _positive_finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return numeric


def _nonnegative_finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return numeric


def calculate_speed_metrics(
    baseline_mean_ms: float,
    optimized_mean_ms: float,
    *,
    regression_threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Calculate speedup and mathematically accurate slowdown percentage."""

    baseline_mean = _positive_finite(
        baseline_mean_ms,
        field_name="baseline_mean_ms",
    )
    optimized_mean = _positive_finite(
        optimized_mean_ms,
        field_name="optimized_mean_ms",
    )
    threshold = _nonnegative_finite(
        regression_threshold_pct,
        field_name="regression_threshold_pct",
    )

    speedup = baseline_mean / optimized_mean
    slowdown_pct = (optimized_mean / baseline_mean - 1.0) * 100.0
    is_slower = optimized_mean > baseline_mean
    return {
        "speedup": speedup,
        "regression": is_slower and slowdown_pct >= threshold,
        "regression_pct": slowdown_pct if is_slower else None,
    }


def _require_signature(benchmark: Any, *, side: str) -> InputSignature:
    getter = getattr(benchmark, "get_input_signature", None)
    if not callable(getter):
        raise VerifiedComparisonError(
            f"{side}_input_signature_missing",
            f"{side.title()} benchmark must expose get_input_signature().",
        )
    try:
        return coerce_input_signature(getter())
    except Exception as exc:
        raise VerifiedComparisonError(
            f"{side}_input_signature_invalid",
            f"{side.title()} captured input signature is unavailable or invalid: {exc}",
        ) from exc


def _compare_signatures(baseline: Any, optimized: Any) -> dict[str, Any]:
    baseline_signature = _require_signature(baseline, side="baseline")
    optimized_signature = _require_signature(optimized, side="optimized")
    try:
        baseline_equivalence = get_signature_equivalence_spec(baseline)
        optimized_equivalence = get_signature_equivalence_spec(optimized)
    except Exception as exc:
        raise VerifiedComparisonError(
            "signature_equivalence_invalid",
            f"Signature equivalence declaration is invalid: {exc}",
        ) from exc

    if baseline_equivalence != optimized_equivalence:
        raise VerifiedComparisonError(
            "signature_equivalence_mismatch",
            "Baseline and optimized signature-equivalence declarations differ.",
            evidence={
                "baseline": baseline_equivalence,
                "optimized": optimized_equivalence,
            },
        )

    baseline_workload = signature_workload_dict(
        baseline_signature,
        equivalence=baseline_equivalence,
    )
    optimized_workload = signature_workload_dict(
        optimized_signature,
        equivalence=baseline_equivalence,
    )
    if baseline_workload != optimized_workload:
        differing_fields = sorted(
            key
            for key in set(baseline_workload) | set(optimized_workload)
            if baseline_workload.get(key) != optimized_workload.get(key)
        )
        raise VerifiedComparisonError(
            "input_signature_mismatch",
            "Captured input signatures differ for fields: "
            + ", ".join(differing_fields),
            evidence={
                "baseline": baseline_workload,
                "optimized": optimized_workload,
            },
        )

    equivalence_receipt = None
    if baseline_equivalence is not None:
        equivalence_receipt = {
            "group": baseline_equivalence.group,
            "ignore_fields": list(baseline_equivalence.ignore_fields),
        }
    return {
        "passed": True,
        "baseline_signature": baseline_signature.to_dict(),
        "optimized_signature": optimized_signature.to_dict(),
        "equivalence": equivalence_receipt,
    }


def _require_tolerance(benchmark: Any, *, side: str) -> ToleranceSpec:
    try:
        tolerance = get_output_tolerance(benchmark)
    except Exception as exc:
        raise VerifiedComparisonError(
            f"{side}_output_tolerance_invalid",
            f"{side.title()} output tolerance is unavailable or invalid: {exc}",
        ) from exc
    if tolerance is None:
        raise VerifiedComparisonError(
            f"{side}_output_tolerance_missing",
            f"{side.title()} benchmark did not provide an output tolerance.",
        )
    if tolerance.comparator_fn is not None:
        raise VerifiedComparisonError(
            f"{side}_output_tolerance_unsupported",
            "Post-timing comparison requires explicit numeric rtol/atol values.",
        )
    return tolerance


def _require_captured_output(benchmark: Any, *, side: str) -> torch.Tensor | dict[str, torch.Tensor]:
    getter = getattr(benchmark, "get_verify_output", None)
    if not callable(getter):
        raise VerifiedComparisonError(
            f"{side}_verify_output_missing",
            f"{side.title()} benchmark must expose get_verify_output().",
        )
    try:
        output = getter()
    except Exception as exc:
        raise VerifiedComparisonError(
            f"{side}_verify_output_invalid",
            f"{side.title()} captured verify output is unavailable: {exc}",
        ) from exc

    tensors: list[tuple[str, torch.Tensor]]
    if isinstance(output, torch.Tensor):
        tensors = [("output", output)]
    elif isinstance(output, dict):
        if not output:
            raise VerifiedComparisonError(
                f"{side}_verify_output_empty",
                f"{side.title()} captured verify-output dictionary is empty.",
            )
        invalid = {
            name: type(value).__name__
            for name, value in output.items()
            if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor)
        }
        if invalid:
            raise VerifiedComparisonError(
                f"{side}_verify_output_invalid",
                f"{side.title()} verify-output dictionary contains invalid entries: {invalid!r}.",
            )
        tensors = list(output.items())
    else:
        raise VerifiedComparisonError(
            f"{side}_verify_output_invalid",
            f"{side.title()} get_verify_output() returned {type(output).__name__}; "
            "expected a tensor or nonempty tensor dictionary.",
        )

    empty_names = [name for name, tensor in tensors if tensor.numel() == 0]
    if empty_names:
        raise VerifiedComparisonError(
            f"{side}_verify_output_empty",
            f"{side.title()} captured verify output contains empty tensors: {empty_names!r}.",
        )
    return output


def _compare_outputs(baseline: Any, optimized: Any) -> dict[str, Any]:
    baseline_tolerance = _require_tolerance(baseline, side="baseline")
    optimized_tolerance = _require_tolerance(optimized, side="optimized")
    if (
        optimized_tolerance.rtol > baseline_tolerance.rtol
        or optimized_tolerance.atol > baseline_tolerance.atol
    ):
        raise VerifiedComparisonError(
            "optimized_output_tolerance_looser",
            "Optimized output tolerance cannot be looser than the baseline tolerance: "
            f"baseline=({baseline_tolerance.rtol}, {baseline_tolerance.atol}), "
            f"optimized=({optimized_tolerance.rtol}, {optimized_tolerance.atol}).",
        )

    effective_tolerance = ToleranceSpec(
        rtol=min(baseline_tolerance.rtol, optimized_tolerance.rtol),
        atol=min(baseline_tolerance.atol, optimized_tolerance.atol),
        justification=optimized_tolerance.justification or baseline_tolerance.justification,
    )
    baseline_output = _require_captured_output(baseline, side="baseline")
    optimized_output = _require_captured_output(optimized, side="optimized")
    comparison = VerifyRunner().compare_perf_outputs(
        baseline_output,
        optimized_output,
        (effective_tolerance.rtol, effective_tolerance.atol),
    )
    comparison_receipt = comparison.to_dict()
    comparison_receipt.update(
        {
            "rtol": effective_tolerance.rtol,
            "atol": effective_tolerance.atol,
            "baseline_tolerance": baseline_tolerance.to_dict(),
            "optimized_tolerance": optimized_tolerance.to_dict(),
        }
    )
    if not comparison.passed:
        detail = "Captured full timed outputs do not match"
        if comparison.max_diff is not None:
            detail += f" (max_diff={comparison.max_diff})"
        raise VerifiedComparisonError(
            "output_mismatch",
            detail + ".",
            evidence=comparison_receipt,
        )
    return comparison_receipt


def _timing_summary(run: BenchmarkRun) -> dict[str, float]:
    timing = run.result.timing
    return {
        "mean_ms": timing.mean_ms,
        "median_ms": timing.median_ms,
        "std_ms": timing.std_ms,
        "min_ms": timing.min_ms,
        "max_ms": timing.max_ms,
    }


def compare_verified_benchmark_runs(
    baseline: Any,
    optimized: Any,
    baseline_run: BenchmarkRun,
    optimized_run: BenchmarkRun,
    *,
    name: str = "Comparison",
    regression_threshold_pct: float = 5.0,
) -> dict[str, Any]:
    """Compare two completed runs without executing either benchmark again."""

    if not isinstance(baseline_run, BenchmarkRun):
        raise TypeError("baseline_run must be a BenchmarkRun")
    if not isinstance(optimized_run, BenchmarkRun):
        raise TypeError("optimized_run must be a BenchmarkRun")

    runtime_comparison = compare_executed_runtime_provenance(
        baseline_run,
        optimized_run,
    )
    if not runtime_comparison.matches:
        raise ExecutedRuntimeComparisonError(runtime_comparison)

    input_verification = _compare_signatures(baseline, optimized)
    output_verification = _compare_outputs(baseline, optimized)
    speed_metrics = calculate_speed_metrics(
        baseline_run.result.timing.mean_ms,
        optimized_run.result.timing.mean_ms,
        regression_threshold_pct=regression_threshold_pct,
    )

    return {
        "name": name,
        "baseline": _timing_summary(baseline_run),
        "optimized": _timing_summary(optimized_run),
        **speed_metrics,
        "baseline_result": baseline_run.result,
        "optimized_result": optimized_run.result,
        "runtime_comparison": runtime_comparison.model_dump(mode="json"),
        "input_verification": input_verification,
        "verification": output_verification,
    }


__all__ = [
    "VerifiedComparisonError",
    "calculate_speed_metrics",
    "compare_verified_benchmark_runs",
]
