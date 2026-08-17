"""Merge one immutable Tier-1 evidence row into the latest canonical history."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.analysis.history_index import (
    SAFE_RUN_ID,
    load_history_index_with_warnings,
    resolve_history_entry_path,
)
from core.analysis.trends import build_trend_snapshot, sort_history_runs
from core.scripts.ci.restore_tier1_history import (
    _comparison_has_valid_shape,
    _summary_has_valid_metric_types,
)

REQUIRED_RUN_PATHS = (
    "summary_path",
    "regression_summary_path",
    "regression_json_path",
    "trend_snapshot_path",
)
ACCEPTANCE_FIELDS = (
    "baseline_acceptance",
    "baseline_acceptance_actor",
    "baseline_acceptance_actor_role",
    "baseline_acceptance_note",
    "baseline_acceptance_workflow_run",
    "baseline_evidence_digest",
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is unreadable ({type(exc).__name__})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _latest_anchor_run_id(runs: list[dict[str, Any]]) -> str | None:
    anchors = [
        entry.get("run_id")
        for entry in sort_history_runs(runs)
        if entry.get("baseline_eligible") is True
    ]
    if not anchors:
        return None
    run_id = anchors[-1]
    if not isinstance(run_id, str) or SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Tier-1 canonical history has an unsafe anchor run id")
    return run_id


def merge_tier1_history_evidence(
    *,
    candidate_history_root: Path,
    canonical_history_root: Path,
    output_history_root: Path,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("Tier-1 merge run id is unsafe")
    candidate_root = Path(candidate_history_root).resolve()
    canonical_root = Path(canonical_history_root).resolve()
    output_root = Path(output_history_root).resolve()
    if output_root.exists():
        raise ValueError("Tier-1 merge output history already exists")

    candidate_index, candidate_warnings = load_history_index_with_warnings(
        candidate_root / "index.json"
    )
    canonical_index, canonical_warnings = load_history_index_with_warnings(
        canonical_root / "index.json"
    )
    if candidate_warnings:
        raise ValueError("Tier-1 candidate history is invalid: " + " | ".join(candidate_warnings))
    if canonical_warnings:
        raise ValueError("Tier-1 canonical history is invalid: " + " | ".join(canonical_warnings))
    candidate_runs = candidate_index.get("runs")
    canonical_runs = canonical_index.get("runs")
    if not isinstance(candidate_runs, list) or not candidate_runs:
        raise ValueError("Tier-1 candidate history has no runs")
    if not isinstance(canonical_runs, list) or not canonical_runs:
        raise ValueError("Tier-1 canonical history has no runs")
    canonical_runs = sort_history_runs(canonical_runs)
    candidate_entry = candidate_runs[-1]
    if candidate_entry.get("run_id") != run_id:
        raise ValueError("Tier-1 merge requires the newest immutable candidate")
    if candidate_entry.get("baseline_eligible") is True or any(
        candidate_entry.get(field) is not None for field in ACCEPTANCE_FIELDS
    ):
        raise ValueError("Tier-1 evidence merge cannot publish a canonical anchor")
    if any(entry.get("run_id") == run_id for entry in canonical_runs):
        raise ValueError("Tier-1 merge run id already exists in canonical history")
    if candidate_index.get("suite_name") != canonical_index.get(
        "suite_name"
    ) or candidate_index.get("suite_version") != canonical_index.get("suite_version"):
        raise ValueError("Tier-1 candidate and canonical suite identities do not match")

    resolved_paths: dict[str, Path] = {}
    for key in REQUIRED_RUN_PATHS:
        raw_path = candidate_entry.get(key)
        if not isinstance(raw_path, str):
            raise ValueError("Tier-1 candidate has an invalid history path")
        relative = Path(raw_path)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != run_id:
            raise ValueError("Tier-1 candidate history path is not bound to its run id")
        resolved = resolve_history_entry_path(
            candidate_root,
            raw_path,
            run_id=run_id,
        )
        if resolved is None or not resolved.is_file():
            raise ValueError("Tier-1 candidate history package is incomplete")
        resolved_paths[key] = resolved

    summary = _load_json_object(
        resolved_paths["summary_path"],
        label="Tier-1 candidate summary",
    )
    comparison = _load_json_object(
        resolved_paths["regression_json_path"],
        label="Tier-1 candidate comparison",
    )
    if summary.get("run_id") != run_id or not _summary_has_valid_metric_types(summary):
        raise ValueError("Tier-1 candidate summary is invalid")
    if comparison.get("current_run_id") != run_id or not _comparison_has_valid_shape(comparison):
        raise ValueError("Tier-1 candidate comparison is invalid")

    latest_anchor = _latest_anchor_run_id(canonical_runs)
    if latest_anchor is None:
        raise ValueError("Tier-1 canonical history has no eligible anchor")
    merged_entry = dict(candidate_entry)
    stale_baseline = comparison.get("baseline_run_id") != latest_anchor
    if stale_baseline:
        merged_entry["run_accepted"] = False
        merged_entry["baseline_eligible"] = False
        for field in ACCEPTANCE_FIELDS:
            merged_entry.pop(field, None)

    updated_runs = sort_history_runs([*canonical_runs, merged_entry])
    updated_index = {
        **canonical_index,
        "history_root": ".",
        "runs": updated_runs,
    }
    updated_trend = build_trend_snapshot(updated_index)
    candidate_run_dir = candidate_root / run_id
    if not candidate_run_dir.is_dir():
        raise ValueError("Tier-1 candidate run directory is missing")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tier1-merge-",
        dir=output_root.parent,
    ) as temp_dir:
        staging_root = Path(temp_dir) / "history"
        shutil.copytree(canonical_root, staging_root)
        staging_run_dir = staging_root / run_id
        if staging_run_dir.exists():
            raise ValueError("Tier-1 candidate run directory collides with canonical history")
        shutil.copytree(candidate_run_dir, staging_run_dir)
        staging_trend_path = resolve_history_entry_path(
            staging_root,
            merged_entry.get("trend_snapshot_path"),
            run_id=run_id,
        )
        if staging_trend_path is None:
            raise ValueError("Tier-1 candidate trend path is invalid")
        _write_json(staging_trend_path, updated_trend)
        latest_entry = updated_runs[-1]
        if latest_entry.get("run_id") != run_id:
            try:
                latest_trend_path = resolve_history_entry_path(
                    staging_root,
                    latest_entry.get("trend_snapshot_path"),
                    run_id=str(latest_entry.get("run_id") or "").strip() or None,
                )
            except ValueError:
                latest_trend_path = None
            if latest_trend_path is not None and latest_trend_path.is_file():
                _write_json(latest_trend_path, updated_trend)
        _write_json(staging_root / "index.json", updated_index)
        os.replace(staging_root, output_root)

    return {
        "success": True,
        "run_id": run_id,
        "stale_baseline": stale_baseline,
        "history_root": ".",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-history-root", type=Path, required=True)
    parser.add_argument("--canonical-history-root", type=Path, required=True)
    parser.add_argument("--output-history-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = merge_tier1_history_evidence(
            candidate_history_root=args.candidate_history_root,
            canonical_history_root=args.canonical_history_root,
            output_history_root=args.output_history_root,
            run_id=args.run_id,
        )
    except Exception as exc:
        print(f"Tier-1 history merge failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
