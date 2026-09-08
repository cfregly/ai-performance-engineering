"""Qualify retained Ozaki measurement logs against the predeclared policy.

The CUDA executables intentionally return 2 for measurement-only runs. Capture
their stdout as individual logs, then use this CPU-only command to require the
complete nominal, holdout, and independent long-double edge matrix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from labs.ozaki_scheme.accuracy_policy import (
    DEFAULT_POLICY_PATH,
    WORKLOAD,
    _limits_from_item,
    load_accuracy_policy,
)
from labs.ozaki_scheme.lab_utils import parse_metrics


def _required_cases(policy: dict) -> dict[tuple[str, str, int], dict]:
    cases = {}
    for variant in policy["qualification"]["variants"]:
        for group in policy["qualification"]["required_receipts"]:
            for seed in group["seeds"]:
                key = (variant, group["cohort"], int(seed))
                cases[key] = {**group, "variant": variant, "seed": int(seed)}
    return cases


def _matching_case(required: dict[tuple[str, str, int], dict], metrics: dict) -> tuple[str, str, int] | None:
    variant = str(metrics.get("variant", "")).removeprefix("ozaki_")
    for key, case in required.items():
        if key[0] != variant or key[2] != metrics.get("seed"):
            continue
        if all(metrics.get(name) == case[name] for name in (
            "m", "n", "k", "input_scale", "input_pattern", "reference_mode"
        )):
            return key
    return None


def assess_logs(policy: dict, log_texts: list[str]) -> dict:
    required = _required_cases(policy)
    seen: set[tuple[str, str, int]] = set()
    failures: list[str] = []
    results: list[dict] = []
    expected_provenance: tuple | None = None

    for index, text in enumerate(log_texts):
        metrics = parse_metrics(text)
        key = _matching_case(required, metrics)
        reasons: list[str] = []
        if key is None:
            reasons.append("log does not match a required policy case")
        elif key in seen:
            reasons.append("duplicate policy case")
        if metrics.get("accuracy_status") != "MEASUREMENT_ONLY_NOT_ACCEPTED":
            reasons.append("log is not retained measurement-only evidence")
        if metrics.get("emulation_used") != 1 or int(metrics.get("retained_bits", -1)) < 0:
            reasons.append("cuBLAS did not report active fixed-point emulation")
        provenance = tuple(metrics.get(name) for name in (
            "gpu_name", "compute_capability", "cuda_runtime_version", "cublas_version"
        ))
        if any(value in (None, "") for value in provenance):
            reasons.append("GPU/CUDA/cuBLAS provenance is incomplete")
        elif expected_provenance is None:
            expected_provenance = provenance
        elif provenance != expected_provenance:
            reasons.append("GPU/CUDA/cuBLAS provenance differs across logs")
        variant = str(metrics.get("variant", "")).removeprefix("ozaki_")
        if variant in policy["variants"]:
            limits = _limits_from_item(policy["variants"][variant])
            for metric_name, limit_name in (
                ("relative_l2_error", "relative_l2"),
                ("normalized_max_abs_error", "normalized_max_abs"),
            ):
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    reasons.append(f"{metric_name} is missing or non-finite")
                elif float(value) > limits[limit_name]:
                    reasons.append(f"{metric_name}={float(value):.8g} exceeds {limits[limit_name]:.8g}")
            if variant == "dynamic" and (
                metrics.get("dynamic_max_bits") != WORKLOAD["dynamic_max_bits"] or
                metrics.get("dynamic_offset") != WORKLOAD["dynamic_offset"]
            ):
                reasons.append("dynamic mantissa configuration mismatch")
            if variant == "fixed" and metrics.get("fixed_bits") != WORKLOAD["fixed_bits"]:
                reasons.append("fixed mantissa configuration mismatch")
            if metrics.get("emulation_strategy") != WORKLOAD["emulation_strategy"]:
                reasons.append("emulation strategy mismatch")
        else:
            reasons.append("unknown Ozaki variant")
        if key is not None and key not in seen:
            seen.add(key)
        if reasons:
            failures.extend(f"log[{index}] {key}: {reason}" for reason in reasons)
        results.append({"case": list(key) if key else None, "passed": not reasons,
                        "reasons": reasons, "metrics": metrics})

    for key in sorted(set(required) - seen):
        failures.append(f"missing required log: {key}")
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "status": "qualified_arithmetic_gate" if not failures else "failed_arithmetic_gate",
        "required_case_count": len(required),
        "passing_case_count": sum(item["passed"] for item in results),
        "failures": failures,
        "logs": results,
        "claim_boundary": policy["qualification"]["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_accuracy_policy(args.policy)
    if policy.get("schema_version") != 2:
        raise ValueError("Log qualification requires the reviewed schema_version=2 policy")
    summary = assess_logs(policy, [path.read_text() for path in args.logs])
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "qualified_arithmetic_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
