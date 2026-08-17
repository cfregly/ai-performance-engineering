"""History index helpers for canonical benchmark suites."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional

from core.benchmark.suites.tier1 import Tier1SuiteDefinition

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BASELINE_ACCEPTANCE_VALUES = {
    "accept_history_anchor",
    "accept_regressions",
    "clean",
    "update_expectations",
}
EXPLICIT_BASELINE_ACCEPTANCE_VALUES = BASELINE_ACCEPTANCE_VALUES - {"clean"}


def _default_history_index() -> Dict[str, Any]:
    return {"suite_name": "tier1", "runs": []}


def load_history_index_with_warnings(index_path: Path) -> tuple[Dict[str, Any], List[str]]:
    if not index_path.exists():
        return _default_history_index(), []

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _default_history_index(), [
            f"Failed to read tier-1 history index ({type(exc).__name__})"
        ]

    if not isinstance(payload, dict):
        return _default_history_index(), [
            f"Expected JSON object in tier-1 history index, got {type(payload).__name__}"
        ]

    normalized = dict(payload)
    warnings: List[str] = []
    runs = normalized.get("runs", [])
    if not isinstance(runs, list):
        warnings.append(f"Expected 'runs' list in tier-1 history index, got {type(runs).__name__}")
        normalized["runs"] = []
    elif any(not isinstance(entry, dict) for entry in runs):
        warnings.append("Tier-1 history index contains a non-object run entry")
        normalized["runs"] = []
    else:
        run_ids = [entry.get("run_id") for entry in runs]
        if any(not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id) for run_id in run_ids):
            warnings.append("Tier-1 history index contains an unsafe run id")
            normalized["runs"] = []
        elif len(run_ids) != len(set(run_ids)):
            warnings.append("Tier-1 history index contains a duplicate run id")
            normalized["runs"] = []
        elif any(
            "baseline_eligible" in entry
            and not isinstance(entry.get("baseline_eligible"), bool)
            for entry in runs
        ):
            warnings.append("Tier-1 history index contains a non-boolean baseline eligibility")
            normalized["runs"] = []
        elif any(
            "run_accepted" in entry and not isinstance(entry.get("run_accepted"), bool)
            for entry in runs
        ):
            warnings.append("Tier-1 history index contains a non-boolean run acceptance")
            normalized["runs"] = []
        elif any(
            entry.get("baseline_eligible") is True and entry.get("run_accepted") is False
            for entry in runs
        ):
            warnings.append("Tier-1 history index has an unaccepted eligible baseline")
            normalized["runs"] = []
        elif any(
            "baseline_acceptance" in entry
            and entry.get("baseline_acceptance") not in BASELINE_ACCEPTANCE_VALUES
            for entry in runs
        ):
            warnings.append("Tier-1 history index contains an invalid baseline acceptance")
            normalized["runs"] = []
        elif any(
            entry.get("baseline_acceptance") is not None
            and entry.get("baseline_eligible") is not True
            for entry in runs
        ):
            warnings.append(
                "Tier-1 history index attaches baseline acceptance to an ineligible run"
            )
            normalized["runs"] = []
        elif any(
            entry.get("baseline_acceptance") in EXPLICIT_BASELINE_ACCEPTANCE_VALUES
            and any(
                not isinstance(entry.get(key), str) or not entry.get(key, "").strip()
                for key in (
                    "baseline_acceptance_actor",
                    "baseline_acceptance_note",
                    "baseline_acceptance_workflow_run",
                )
            )
            for entry in runs
        ):
            warnings.append("Tier-1 history index has incomplete baseline acceptance evidence")
            normalized["runs"] = []
    normalized.setdefault("suite_name", "tier1")
    return normalized, warnings


def load_history_index(index_path: Path) -> Dict[str, Any]:
    index, _ = load_history_index_with_warnings(index_path)
    return index


def _path_within_history_root(history_root: Path, path: Path) -> Path:
    root = history_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Tier-1 history path escapes the configured history root") from exc
    return resolved


def _portable_history_path(history_root: Path, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    resolved = _path_within_history_root(history_root, Path(path))
    return resolved.relative_to(history_root.resolve()).as_posix()


def resolve_history_entry_path(
    history_root: Path,
    raw_path: Any,
    *,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """Resolve portable or legacy Tier-1 index paths under the active history root."""
    if raw_path is None or not str(raw_path).strip():
        return None

    root = history_root.resolve()
    path = Path(str(raw_path))
    if not path.is_absolute():
        return _path_within_history_root(root, root / path)

    try:
        return _path_within_history_root(root, path)
    except ValueError:
        if not run_id:
            raise
        relocated = root / str(run_id) / path.name
        return _path_within_history_root(root, relocated)


def build_updated_history_index(
    *,
    history_root: Path,
    suite: Tier1SuiteDefinition,
    summary: Dict[str, Any],
    summary_path: Path,
    regression_summary_path: Path,
    regression_json_path: Optional[Path] = None,
    trend_snapshot_path: Optional[Path] = None,
    run_accepted: Optional[bool] = None,
    baseline_eligible: Optional[bool] = None,
    baseline_acceptance: Optional[str] = None,
    baseline_acceptance_actor: Optional[str] = None,
    baseline_acceptance_note: Optional[str] = None,
    baseline_acceptance_workflow_run: Optional[str] = None,
) -> Dict[str, Any]:
    index_path = history_root / "index.json"
    index, warnings = load_history_index_with_warnings(index_path)
    if warnings:
        raise ValueError(
            "Refusing to overwrite invalid Tier-1 history index: " + " | ".join(warnings)
        )

    entry = {
        "run_id": summary["run_id"],
        "generated_at": summary.get("generated_at"),
        "summary_path": _portable_history_path(history_root, summary_path),
        "regression_summary_path": _portable_history_path(history_root, regression_summary_path),
        "regression_json_path": _portable_history_path(history_root, regression_json_path),
        "trend_snapshot_path": _portable_history_path(history_root, trend_snapshot_path),
        "avg_speedup": summary.get("summary", {}).get("avg_speedup", 0.0),
        "median_speedup": summary.get("summary", {}).get("median_speedup", 0.0),
        "geomean_speedup": summary.get("summary", {}).get("geomean_speedup", 0.0),
        "representative_speedup": summary.get("summary", {}).get("representative_speedup", 0.0),
        "max_speedup": summary.get("summary", {}).get("max_speedup", 0.0),
        "succeeded": summary.get("summary", {}).get("succeeded", 0),
        "failed": summary.get("summary", {}).get("failed", 0),
        "skipped": summary.get("summary", {}).get("skipped", 0),
        "missing": summary.get("summary", {}).get("missing", 0),
    }
    if run_accepted is not None:
        if not isinstance(run_accepted, bool):
            raise ValueError("Tier-1 run acceptance must be a boolean")
        entry["run_accepted"] = run_accepted
    if baseline_eligible is not None:
        if not isinstance(baseline_eligible, bool):
            raise ValueError("Tier-1 baseline eligibility must be a boolean")
        if baseline_eligible and run_accepted is not True:
            raise ValueError("Tier-1 baseline eligibility requires an accepted run")
        entry["baseline_eligible"] = baseline_eligible
    if baseline_acceptance is not None:
        if baseline_eligible is not True:
            raise ValueError("Tier-1 baseline acceptance requires an eligible run")
        if baseline_acceptance not in BASELINE_ACCEPTANCE_VALUES:
            raise ValueError("Tier-1 baseline acceptance is invalid")
        entry["baseline_acceptance"] = baseline_acceptance
        if baseline_acceptance in EXPLICIT_BASELINE_ACCEPTANCE_VALUES:
            acceptance_evidence = {
                "baseline_acceptance_actor": baseline_acceptance_actor,
                "baseline_acceptance_note": baseline_acceptance_note,
                "baseline_acceptance_workflow_run": baseline_acceptance_workflow_run,
            }
            if any(not isinstance(value, str) or not value.strip() for value in acceptance_evidence.values()):
                raise ValueError("Tier-1 explicit baseline acceptance requires audit evidence")
            entry.update({key: str(value).strip() for key, value in acceptance_evidence.items()})
    elif baseline_eligible is True:
        raise ValueError("Tier-1 eligible history entries require a baseline acceptance")

    runs = [
        existing for existing in index.get("runs", []) if existing.get("run_id") != entry["run_id"]
    ]
    runs.append(entry)
    runs.sort(key=lambda item: str(item.get("generated_at") or item.get("run_id") or ""))

    return {
        "suite_name": suite.name,
        "suite_version": suite.version,
        "history_root": ".",
        "runs": runs,
    }


def update_history_index(
    *,
    history_root: Path,
    suite: Tier1SuiteDefinition,
    summary: Dict[str, Any],
    summary_path: Path,
    regression_summary_path: Path,
    regression_json_path: Optional[Path] = None,
    trend_snapshot_path: Optional[Path] = None,
    run_accepted: Optional[bool] = None,
    baseline_eligible: Optional[bool] = None,
    baseline_acceptance: Optional[str] = None,
    baseline_acceptance_actor: Optional[str] = None,
    baseline_acceptance_note: Optional[str] = None,
    baseline_acceptance_workflow_run: Optional[str] = None,
) -> Dict[str, Any]:
    history_root.mkdir(parents=True, exist_ok=True)
    index_path = history_root / "index.json"
    updated = build_updated_history_index(
        history_root=history_root,
        suite=suite,
        summary=summary,
        summary_path=summary_path,
        regression_summary_path=regression_summary_path,
        regression_json_path=regression_json_path,
        trend_snapshot_path=trend_snapshot_path,
        run_accepted=run_accepted,
        baseline_eligible=baseline_eligible,
        baseline_acceptance=baseline_acceptance,
        baseline_acceptance_actor=baseline_acceptance_actor,
        baseline_acceptance_note=baseline_acceptance_note,
        baseline_acceptance_workflow_run=baseline_acceptance_workflow_run,
    )
    serialized = json.dumps(updated, indent=2, allow_nan=False)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=history_root,
            prefix=".index.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, index_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return updated
