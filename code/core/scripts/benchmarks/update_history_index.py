#!/usr/bin/env python3
"""Update the tier-1 history index for an existing summary."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from core.analysis.history_index import (
    build_updated_history_index,
    load_history_index_with_warnings,
    resolve_history_entry_path,
    update_history_index,
)
from core.analysis.regressions import compare_suite_summaries, render_regression_summary
from core.analysis.trends import build_trend_snapshot
from core.benchmark.suites.tier1 import _summary_has_baseline_metrics, load_tier1_suite

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _load_summary(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read tier-1 summary JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in tier-1 summary JSON {path}, got {type(payload).__name__}"
        )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"Missing run_id in tier-1 summary JSON {path}")
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"Unsafe run_id in tier-1 summary JSON {path}: {run_id!r}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            f"Expected summary object in tier-1 summary JSON {path}, got {type(summary).__name__}"
        )
    targets = payload.get("targets")
    if not isinstance(targets, list) or not all(isinstance(target, dict) for target in targets):
        raise ValueError(f"Expected targets list in tier-1 summary JSON {path}")
    required_counts = ("target_count", "succeeded", "failed", "skipped", "missing")
    parsed_counts: dict[str, int] = {}
    for key in required_counts:
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid {key} count in tier-1 summary JSON {path}")
        if value < 0:
            raise ValueError(f"Invalid {key} count in tier-1 summary JSON {path}")
        parsed_counts[key] = value
    if parsed_counts["target_count"] != len(targets):
        raise ValueError(f"Tier-1 target_count does not match targets list in summary JSON {path}")
    derived_counts = {
        "succeeded": sum(target.get("status") == "succeeded" for target in targets),
        "failed": sum(str(target.get("status") or "").startswith("failed") for target in targets),
        "skipped": sum(str(target.get("status") or "").startswith("skipped") for target in targets),
        "missing": sum(target.get("status") == "missing" for target in targets),
    }
    for key, derived in derived_counts.items():
        if parsed_counts[key] != derived:
            raise ValueError(
                f"Tier-1 {key} count does not match target statuses in summary JSON {path}"
            )
    if sum(parsed_counts[key] for key in ("succeeded", "failed", "skipped", "missing")) != parsed_counts[
        "target_count"
    ]:
        raise ValueError(f"Tier-1 target statuses are not fully classified in summary JSON {path}")
    for key in (
        "avg_speedup",
        "geomean_speedup",
        "max_speedup",
        "median_speedup",
        "representative_speedup",
    ):
        value = summary.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Invalid {key} metric in tier-1 summary JSON {path}")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Invalid {key} metric in tier-1 summary JSON {path}")
    for target in targets:
        for key in (
            "baseline_memory_mb",
            "baseline_time_ms",
            "best_memory_savings_pct",
            "best_optimized_memory_mb",
            "best_optimized_time_ms",
            "best_speedup",
        ):
            value = target.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"Invalid target {key} metric in tier-1 summary JSON {path}")
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"Invalid target {key} metric in tier-1 summary JSON {path}")
    return payload


def _load_regression_summary(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read tier-1 regression JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in tier-1 regression JSON {path}, "
            f"got {type(payload).__name__}"
        )
    regressions = payload.get("regressions")
    if not isinstance(regressions, list):
        raise ValueError(f"Missing regressions list in tier-1 regression JSON {path}")
    if not isinstance(payload.get("missing_targets"), list):
        raise ValueError(f"Missing missing_targets list in tier-1 regression JSON {path}")
    return payload


def _load_json_object(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read {label} {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {label} {path}")
    return payload


def _require_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path.resolve()


def _copy_atomic(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination.resolve(strict=False):
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _write_text_atomic(destination: Path, content: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _load_latest_eligible_summary(history_root: Path, index: dict) -> dict | None:
    for entry in reversed(index.get("runs", [])):
        if not isinstance(entry, dict) or entry.get("baseline_eligible") is not True:
            continue
        run_id = str(entry.get("run_id") or "")
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("Existing Tier-1 history has an unsafe eligible run id")
        try:
            summary_path = resolve_history_entry_path(
                history_root,
                entry.get("summary_path"),
                run_id=run_id,
            )
        except ValueError as exc:
            raise ValueError("Existing Tier-1 eligible summary escapes the history root") from exc
        if summary_path is None or not summary_path.is_file():
            raise ValueError("Existing Tier-1 eligible summary is missing")
        summary = _load_summary(summary_path)
        if not _summary_has_baseline_metrics(summary):
            raise ValueError("Existing Tier-1 eligible summary has invalid metrics")
        return summary
    return None


def _canonicalize_history_inputs(
    *,
    history_root: Path,
    summary: dict,
    summary_source: Path,
    regression_summary_source: Path,
    regression_json_source: Path,
    trend_snapshot_source: Path,
) -> dict[str, Path]:
    run_id = str(summary["run_id"])
    root = history_root.resolve()
    run_path = root / run_id
    if run_path.is_symlink() or run_path.exists():
        raise ValueError(f"Refusing to overwrite existing Tier-1 history run {run_id!r}")
    run_dir = run_path.resolve(strict=False)
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Tier-1 run directory escapes history root: {run_dir}") from exc
    if run_dir == root:
        raise ValueError(f"Tier-1 run directory must be below history root: {run_dir}")

    validated_summary_source = _require_regular_file(summary_source, label="Tier-1 summary")
    indexed_summary = _load_summary(validated_summary_source)
    if indexed_summary != summary:
        raise ValueError(
            "Tier-1 summary path does not contain the same payload as --summary-json: "
            f"{summary_source}"
        )

    sources: dict[str, Path] = {
        "summary_source": validated_summary_source,
        "regression_summary_source": _require_regular_file(
            regression_summary_source,
            label="Tier-1 regression summary",
        ),
        "regression_json_source": _require_regular_file(
            regression_json_source,
            label="Tier-1 regression JSON",
        ),
        "trend_snapshot_source": _require_regular_file(
            trend_snapshot_source,
            label="Tier-1 trend snapshot",
        ),
    }
    destinations: dict[str, Path] = {
        "summary_path": run_dir / "summary.json",
        "regression_summary_path": run_dir / "regression_summary.md",
        "regression_json_path": run_dir / "regression_summary.json",
        "trend_snapshot_path": run_dir / "trend_snapshot.json",
    }
    return {**sources, **destinations}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update tier-1 history index from an existing summary.json."
    )
    parser.add_argument(
        "--summary-json", type=Path, required=True, help="Path to tier-1 summary.json"
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional summary artifact source. It must match --summary-json.",
    )
    parser.add_argument(
        "--regression-summary", type=Path, required=True, help="Path to regression_summary.md"
    )
    parser.add_argument(
        "--regression-json",
        type=Path,
        required=True,
        help="Path to regression_summary.json",
    )
    parser.add_argument(
        "--trend-snapshot", type=Path, required=True, help="Path to trend_snapshot.json"
    )
    parser.add_argument("--history-root", type=Path, required=True, help="History root directory.")
    parser.add_argument("--config", type=Path, default=None, help="Tier-1 YAML config path.")
    args = parser.parse_args()

    suite = load_tier1_suite(args.config)
    try:
        previous_index, index_warnings = load_history_index_with_warnings(
            args.history_root / "index.json"
        )
        if index_warnings:
            raise ValueError(
                "Refusing to update invalid Tier-1 history index: " + " | ".join(index_warnings)
            )
        summary = _load_summary(args.summary_json)
        if any(
            isinstance(entry, dict) and entry.get("run_id") == summary["run_id"]
            for entry in previous_index.get("runs", [])
        ):
            raise ValueError(
                f"Refusing to overwrite existing Tier-1 history run {summary['run_id']!r}"
            )
        _load_regression_summary(args.regression_json)
        _load_json_object(args.trend_snapshot, label="tier-1 trend snapshot JSON")
        canonical_paths = _canonicalize_history_inputs(
            history_root=args.history_root,
            summary=summary,
            summary_source=args.summary_path or args.summary_json,
            regression_summary_source=args.regression_summary,
            regression_json_source=args.regression_json,
            trend_snapshot_source=args.trend_snapshot,
        )
        previous_summary = _load_latest_eligible_summary(
            args.history_root,
            previous_index,
        )
        derived_comparison = compare_suite_summaries(summary, previous_summary)
        derived_report = render_regression_summary(summary, previous_summary, derived_comparison)
        derived_comparison_json = json.dumps(derived_comparison, indent=2, allow_nan=False)
        provisional_index = build_updated_history_index(
            history_root=args.history_root,
            suite=suite,
            summary=summary,
            summary_path=canonical_paths["summary_path"],
            regression_summary_path=canonical_paths["regression_summary_path"],
            regression_json_path=canonical_paths["regression_json_path"],
            trend_snapshot_path=canonical_paths["trend_snapshot_path"],
            run_accepted=False,
            baseline_eligible=False,
        )
        derived_trend_json = json.dumps(
            build_trend_snapshot(provisional_index),
            indent=2,
            allow_nan=False,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    history_root_preexisting = args.history_root.exists()
    try:
        args.history_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".tier1-import-",
            dir=args.history_root,
        ) as temp_dir:
            staged_run_dir = Path(temp_dir) / str(summary["run_id"])
            staged_run_dir.mkdir()
            _copy_atomic(
                canonical_paths["summary_source"],
                staged_run_dir / "summary.json",
            )
            _write_text_atomic(staged_run_dir / "regression_summary.md", derived_report)
            _write_text_atomic(
                staged_run_dir / "regression_summary.json",
                derived_comparison_json,
            )
            _write_text_atomic(staged_run_dir / "trend_snapshot.json", derived_trend_json)

            final_run_dir = canonical_paths["summary_path"].parent
            os.replace(staged_run_dir, final_run_dir)
            try:
                updated = update_history_index(
                    history_root=args.history_root,
                    suite=suite,
                    summary=summary,
                    summary_path=canonical_paths["summary_path"],
                    regression_summary_path=canonical_paths["regression_summary_path"],
                    regression_json_path=canonical_paths["regression_json_path"],
                    trend_snapshot_path=canonical_paths["trend_snapshot_path"],
                    run_accepted=False,
                    baseline_eligible=False,
                )
            except Exception:
                os.replace(final_run_dir, staged_run_dir)
                raise
    except Exception as exc:
        if (
            not history_root_preexisting
            and args.history_root.is_dir()
            and not any(args.history_root.iterdir())
        ):
            args.history_root.rmdir()
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(updated, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
