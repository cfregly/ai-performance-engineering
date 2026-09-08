"""Qualify KV accuracy receipts against the predeclared policy and cohorts.

This command never runs a benchmark. It consumes retained measurement-only
receipts, writes every rejection reason, and exits nonzero unless the complete
nominal, holdout, and edge matrix passes the source-defined ceilings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from labs.kv_cache_compression.accuracy import (
    DEFAULT_POLICY_PATH,
    REFERENCE_ID,
    WORKLOAD,
    _limits_from_item,
    load_accuracy_policy,
)

PROVENANCE_FIELDS = (
    "git_commit",
    "torch",
    "cuda",
    "transformer_engine",
    "gpu",
    "compute_capability",
)
RECEIPT_BINDING = (
    "receipt_consistency_only; execution/source identity requires companion receipts"
)


def required_cases(policy: dict) -> set[tuple[str, str, int]]:
    cases: set[tuple[str, str, int]] = set()
    qualification = policy["qualification"]
    for variant in qualification["variants"]:
        for group in qualification["required_receipts"]:
            for seed in group["seeds"]:
                cases.add((variant, group["cohort"], int(seed)))
    return cases


def assess_receipts(policy: dict, receipts: list[dict]) -> dict:
    """Return a durable pass/fail summary without discarding malformed receipts."""
    required = required_cases(policy)
    seen: set[tuple[str, str, int]] = set()
    failures: list[str] = []
    receipt_results: list[dict] = []
    expected_provenance: tuple | None = None
    provenance_consistent = True
    variants = policy["variants"]
    expected_workload = {key: value for key, value in WORKLOAD.items() if key != "storage_dtype"}

    for index, receipt in enumerate(receipts):
        try:
            seed = int(receipt.get("seed", -1))
        except (TypeError, ValueError):
            seed = -1
        key = (str(receipt.get("variant")), str(receipt.get("cohort")), seed)
        reasons: list[str] = []
        if key not in required:
            reasons.append("receipt is not a required policy case")
        if key in seen:
            reasons.append("duplicate policy case")
        if receipt.get("schema_version") != 2:
            reasons.append("receipt schema_version is not 2")
        if receipt.get("status") != "measurement_only_not_accepted":
            reasons.append("receipt is not measurement-only source evidence")
        if receipt.get("reference_id") != REFERENCE_ID:
            reasons.append("reference identity mismatch")
        if receipt.get("workload") != expected_workload:
            reasons.append("workload mismatch")
        provenance = tuple(receipt.get(name) for name in PROVENANCE_FIELDS)
        if any(value in (None, "", []) for value in provenance):
            reasons.append("hardware/software provenance is incomplete")
            provenance_consistent = False
        elif expected_provenance is None:
            expected_provenance = provenance
        elif provenance != expected_provenance:
            reasons.append("hardware/software provenance differs across receipts")
            provenance_consistent = False
        metrics = receipt.get("metrics")
        if key[0] in variants and isinstance(metrics, dict):
            limits = _limits_from_item(variants[key[0]])
            for tensor in ("cache_k", "cache_v"):
                for metric_name in ("relative_l2", "normalized_max_abs"):
                    name = f"{tensor}.{metric_name}"
                    value = metrics.get(name)
                    limit = getattr(limits, metric_name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(float(value))
                        or float(value) < 0
                    ):
                        reasons.append(f"{name} is missing, boolean, negative, or non-finite")
                    elif float(value) > limit:
                        reasons.append(f"{name}={float(value):.8g} exceeds {limit:.8g}")
        else:
            reasons.append("variant or metrics are invalid")
        if key in required and key not in seen:
            seen.add(key)
        if reasons:
            failures.extend(f"receipt[{index}] {key}: {reason}" for reason in reasons)
        receipt_results.append({"case": list(key), "passed": not reasons, "reasons": reasons})

    for key in sorted(required - seen):
        failures.append(f"missing required receipt: {key}")
    declared_provenance = (
        dict(zip(PROVENANCE_FIELDS, expected_provenance, strict=True))
        if expected_provenance is not None and provenance_consistent
        else None
    )
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "status": "qualified_arithmetic_gate" if not failures else "failed_arithmetic_gate",
        "required_case_count": len(required),
        "passing_case_count": sum(item["passed"] for item in receipt_results),
        "failures": failures,
        "receipts": receipt_results,
        "declared_provenance": declared_provenance,
        "binding": RECEIPT_BINDING,
        "claim_boundary": policy["qualification"]["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_accuracy_policy(args.policy)
    if policy.get("schema_version") != 2:
        raise ValueError("Receipt qualification requires the reviewed schema_version=2 policy")
    receipts = [json.loads(path.read_text()) for path in args.receipts]
    summary = assess_receipts(policy, receipts)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "qualified_arithmetic_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
