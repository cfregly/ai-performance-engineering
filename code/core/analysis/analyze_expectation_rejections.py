#!/usr/bin/env python3
"""Render a provenance-aware ledger of rejected expectation updates for a run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_INFERENCE_RELATIVE_TOLERANCE = 0.25


def _load_jsonl(path: Path, *, label: str) -> Iterable[Dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError(f"Failed to read {label} {path}: {exc}") from exc
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise ValueError(f"Failed to parse {label} {path} line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected JSON object in {label} {path} line {line_number}, got {type(payload).__name__}"
            )
        yield payload


def _load_json_object(path: Path, *, label: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to read {label} {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"Expected JSON object in {label} {path}, got {type(payload).__name__}"
    return payload, None


def _fallback_stored_entry(
    repo_root: Path,
    chapter: str,
    example: str,
    hardware_key: str,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[str]]:
    expectation_path = repo_root / chapter / f"expectations_{hardware_key}.json"
    if not expectation_path.exists():
        return None, None, None
    data, warning = _load_json_object(expectation_path, label="stored expectations file")
    if warning:
        return expectation_path, None, warning
    entry = (data.get("examples") or {}).get(example)
    return expectation_path, entry, None


def _stability_bucket(delta_pct: Optional[float]) -> str:
    if delta_pct is None:
        return "unknown"
    magnitude = abs(delta_pct)
    if magnitude <= 10.0:
        return "stable"
    if magnitude <= 25.0:
        return "moderate"
    return "noisy"


def _rejection_reason(event: Dict[str, Any], validation_issue_types: List[str]) -> str:
    if event.get("update_message"):
        return str(event["update_message"])
    if validation_issue_types:
        return ",".join(validation_issue_types)
    return "rejected"


def _refresh_recommendation(row: Dict[str, Any], threshold_pct: float) -> str:
    delta_pct = row.get("delta_pct")
    if delta_pct is None:
        return "review_missing_delta"
    if abs(float(delta_pct)) < threshold_pct:
        return "ignore_small_delta"
    comparison_stability = row.get("comparison_stability")
    if comparison_stability == "stable":
        return "update_now"
    if comparison_stability == "moderate":
        return "rerun_once_more"
    if comparison_stability == "noisy":
        return "hold_noisy"
    return "pending_rerun"


def _classify_rejection(
    *,
    validation_issue_types: List[str],
    old_score: Optional[float],
    new_score: Optional[float],
) -> str:
    if "provenance_mismatch" in validation_issue_types:
        return "provenance_mismatch"
    if validation_issue_types:
        return ",".join(validation_issue_types)
    if old_score is None or new_score is None:
        return "rejected_unknown"
    if new_score > old_score * (1.0 + PROVENANCE_INFERENCE_RELATIVE_TOLERANCE):
        return "inferred_provenance_mismatch_improvement_rejected"
    if new_score < old_score * (1.0 - PROVENANCE_INFERENCE_RELATIVE_TOLERANCE):
        return "possible_regression_or_provenance_mismatch"
    return "mixed_or_unchanged_rejected"


def _load_expectation_updates(run_dir: Path, repo_root: Path) -> List[Dict[str, Any]]:
    events_path = run_dir / "logs" / "benchmark_events.jsonl"
    rows: List[Dict[str, Any]] = []

    for event in _load_jsonl(events_path, label="expectation update event log"):
        if event.get("event_type") != "expectation_update":
            continue
        if event.get("status") != "rejected":
            continue

        chapter = str(event["chapter"])
        example = str(event["example"])
        target = f"{chapter}:{example}"
        validation_issue_types = list(event.get("validation_issue_types") or [])
        old_provenance = event.get("old_provenance") or {}
        new_provenance = event.get("new_provenance") or {}
        hardware_key = str(
            old_provenance.get("hardware_key")
            or new_provenance.get("hardware_key")
            or "b200"
        )
        expectation_path, stored_entry, stored_warning = _fallback_stored_entry(
            repo_root,
            chapter,
            example,
            hardware_key,
        )
        stored_provenance = old_provenance or ((stored_entry or {}).get("provenance") or {})
        provenance_mismatch_fields = list(event.get("provenance_mismatch_fields") or [])

        if not provenance_mismatch_fields and stored_provenance and new_provenance:
            for field_name in sorted(set(stored_provenance) | set(new_provenance)):
                if stored_provenance.get(field_name) != new_provenance.get(field_name):
                    provenance_mismatch_fields.append(field_name)

        rows.append(
            {
                "target": target,
                "chapter": chapter,
                "example": example,
                "goal": event.get("goal"),
                "status": event.get("status"),
                "rejection_reason": _rejection_reason(event, validation_issue_types),
                "rejection_classification": _classify_rejection(
                    validation_issue_types=validation_issue_types,
                    old_score=event.get("old_score"),
                    new_score=event.get("new_score"),
                ),
                "validation_issue_types": validation_issue_types,
                "provenance_mismatch_fields": provenance_mismatch_fields,
                "old_score": event.get("old_score"),
                "new_score": event.get("new_score"),
                "delta": event.get("delta"),
                "delta_pct": event.get("delta_pct"),
                "baseline_time_ms": event.get("baseline_time_ms"),
                "optimized_time_ms": event.get("optimized_time_ms"),
                "expectations_file": str(expectation_path.relative_to(repo_root)) if expectation_path else None,
                "stored_git_commit": stored_provenance.get("git_commit"),
                "stored_hardware_key": stored_provenance.get("hardware_key"),
                "stored_profile_name": stored_provenance.get("profile_name"),
                "stored_execution_environment": stored_provenance.get("execution_environment"),
                "stored_validity_profile": stored_provenance.get("validity_profile"),
                "stored_dmi_product_name": stored_provenance.get("dmi_product_name"),
                "stored_iterations": stored_provenance.get("iterations"),
                "stored_warmup_iterations": stored_provenance.get("warmup_iterations"),
                "new_git_commit": new_provenance.get("git_commit"),
                "new_hardware_key": new_provenance.get("hardware_key"),
                "new_profile_name": new_provenance.get("profile_name"),
                "new_execution_environment": new_provenance.get("execution_environment"),
                "new_validity_profile": new_provenance.get("validity_profile"),
                "new_dmi_product_name": new_provenance.get("dmi_product_name"),
                "new_iterations": new_provenance.get("iterations"),
                "new_warmup_iterations": new_provenance.get("warmup_iterations"),
                "stored_expectation_warning": stored_warning,
            }
        )

    rows.sort(key=lambda row: (row["chapter"], row["example"]))
    return rows


def _index_by_target(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["target"]: row for row in rows}


def _merge_comparison_indexes(comparison_row_sets: Iterable[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for rows in comparison_row_sets:
        merged.update(_index_by_target(rows))
    return merged


def enrich_with_comparison(
    base_rows: List[Dict[str, Any]],
    comparison_row_sets: Iterable[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    comparison_index = _merge_comparison_indexes(comparison_row_sets)
    enriched: List[Dict[str, Any]] = []

    for row in base_rows:
        result = dict(row)
        comparison = comparison_index.get(row["target"])
        if comparison and row.get("new_score") not in (None, 0) and comparison.get("new_score") is not None:
            rerun_vs_original_delta_pct = (
                (float(comparison["new_score"]) - float(row["new_score"])) / float(row["new_score"])
            ) * 100.0
            result["comparison_score"] = comparison["new_score"]
            result["comparison_delta_pct"] = comparison["delta_pct"]
            result["comparison_vs_original_delta_pct"] = rerun_vs_original_delta_pct
            result["comparison_stability"] = _stability_bucket(rerun_vs_original_delta_pct)
            result["comparison_rejection_reason"] = comparison.get("rejection_reason")
            result["comparison_new_git_commit"] = comparison.get("new_git_commit")
            result["comparison_new_profile_name"] = comparison.get("new_profile_name")
            result["comparison_new_execution_environment"] = comparison.get("new_execution_environment")
            result["comparison_new_validity_profile"] = comparison.get("new_validity_profile")
            result["comparison_new_dmi_product_name"] = comparison.get("new_dmi_product_name")
        else:
            result["comparison_score"] = None
            result["comparison_delta_pct"] = None
            result["comparison_vs_original_delta_pct"] = None
            result["comparison_stability"] = "pending"
            result["comparison_rejection_reason"] = None
            result["comparison_new_git_commit"] = None
            result["comparison_new_profile_name"] = None
            result["comparison_new_execution_environment"] = None
            result["comparison_new_validity_profile"] = None
            result["comparison_new_dmi_product_name"] = None
        enriched.append(result)

    return enriched


def _write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, Any]], path: Path, threshold_pct: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    material = [
        row for row in rows if row.get("delta_pct") is not None and abs(float(row["delta_pct"])) >= threshold_pct
    ]
    warnings = sorted({str(row["stored_expectation_warning"]) for row in rows if row.get("stored_expectation_warning")})
    lines = [
        "# Expectation Rejection Ledger",
        "",
        f"- Total rejected expectation updates: `{len(rows)}`",
        f"- Material mismatches (`abs(delta_pct) >= {threshold_pct:g}`): `{len(material)}`",
        f"- Provenance-linked rejections: `{sum(1 for row in rows if 'provenance' in row['rejection_classification'])}`",
        f"- Comparison rows available: `{sum(1 for row in rows if row['comparison_stability'] != 'pending')}`",
        f"- Stable update-now candidates: `{sum(1 for row in rows if row.get('refresh_recommendation') == 'update_now')}`",
        f"- Stored expectation warnings: `{len(warnings)}`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend([
        "| Target | Delta % | Stored -> Observed | Classification | Comparison Stability | Refresh Recommendation |",
        "|---|---:|---:|---|---|---|",
    ])
    for row in sorted(material, key=lambda item: abs(float(item["delta_pct"])), reverse=True)[:60]:
        lines.append(
            f"| `{row['target']}` | {float(row['delta_pct']):+.1f}% | "
            f"{float(row['old_score']):.3f} -> {float(row['new_score']):.3f} | "
            f"`{row['rejection_classification']}` | "
            f"`{row['comparison_stability']}` | "
            f"`{row.get('refresh_recommendation')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_expectation_rejection_ledger(
    *,
    repo_root: Path,
    run_dir: Path,
    comparison_run_dirs: Optional[List[Path]] = None,
    output_dir: Optional[Path] = None,
    threshold_pct: float = 25.0,
) -> Dict[str, Path]:
    base_rows = _load_expectation_updates(run_dir, repo_root)
    comparison_row_sets: List[List[Dict[str, Any]]] = []
    for comparison_run_dir in comparison_run_dirs or []:
        comparison_row_sets.append(_load_expectation_updates(comparison_run_dir, repo_root))
    rows = enrich_with_comparison(base_rows, comparison_row_sets)
    for row in rows:
        delta_pct = row.get("delta_pct")
        row["material_mismatch"] = delta_pct is not None and abs(float(delta_pct)) >= threshold_pct
        row["refresh_recommendation"] = _refresh_recommendation(row, threshold_pct)

    output_root = output_dir or (run_dir / "analysis")
    prefix = run_dir.name
    json_path = output_root / f"{prefix}_expectation_rejections.json"
    csv_path = output_root / f"{prefix}_expectation_rejections.csv"
    md_path = output_root / f"{prefix}_expectation_rejections.md"

    _write_json(rows, json_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path, threshold_pct)
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rejected expectation updates for a benchmark run.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing chapter expectation files (default: current repo root)",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing logs/benchmark_events.jsonl")
    parser.add_argument(
        "--comparison-run-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional comparison run directory for stability comparison; repeat as needed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (defaults to <run-dir>/analysis)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=25.0,
        help="Material mismatch threshold in percent (default: 25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = render_expectation_rejection_ledger(
            repo_root=args.repo_root.resolve(),
            run_dir=args.run_dir,
            comparison_run_dirs=args.comparison_run_dir,
            output_dir=args.output_dir,
            threshold_pct=args.threshold_pct,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
