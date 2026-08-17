"""Trend snapshots for canonical benchmark suite history."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

_EARLIEST_GENERATED_AT = datetime.min.replace(tzinfo=timezone.utc)


def _parse_generated_at(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return _EARLIEST_GENERATED_AT
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return _EARLIEST_GENERATED_AT


def sort_history_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return history rows in event-time order with a stable safe tie breaker."""

    def sort_key(run: dict[str, Any]) -> tuple[datetime, str]:
        run_id = run.get("run_id")
        safe_run_id = run_id if isinstance(run_id, str) else ""
        return _parse_generated_at(run.get("generated_at")), safe_run_id

    return sorted(runs, key=sort_key)


def build_trend_snapshot(index: dict[str, Any]) -> dict[str, Any]:
    runs = sort_history_runs(index.get("runs", []))
    evidence_by_date = [
        {
            "run_id": run.get("run_id"),
            "generated_at": run.get("generated_at"),
            "avg_speedup": run.get("avg_speedup", 0.0),
            "median_speedup": run.get("median_speedup", 0.0),
            "geomean_speedup": run.get("geomean_speedup", 0.0),
            "representative_speedup": run.get(
                "representative_speedup", run.get("geomean_speedup", 0.0)
            ),
            "max_speedup": run.get("max_speedup", 0.0),
            "succeeded": run.get("succeeded", 0),
            "failed": run.get("failed", 0),
            "skipped": run.get("skipped", 0),
            "missing": run.get("missing", 0),
        }
        for run in runs
    ]
    by_date = [
        row
        for run, row in zip(runs, evidence_by_date, strict=True)
        if run.get("run_accepted") is True or run.get("baseline_eligible") is True
    ]
    if by_date:
        avg_speedup = sum(item["avg_speedup"] for item in by_date) / len(by_date)
        median_speedup = sum(item["median_speedup"] for item in by_date) / len(by_date)
        geomean_speedup = sum(item["geomean_speedup"] for item in by_date) / len(by_date)
        representative_speedup = sum(item["representative_speedup"] for item in by_date) / len(
            by_date
        )
        max_speedup = max(item["max_speedup"] for item in by_date)
    else:
        avg_speedup = 0.0
        median_speedup = 0.0
        geomean_speedup = 0.0
        representative_speedup = 0.0
        max_speedup = 0.0

    return {
        "suite_name": index.get("suite_name", "tier1"),
        "run_count": len(by_date),
        "evidence_run_count": len(evidence_by_date),
        "history": by_date,
        "evidence_history": evidence_by_date,
        "avg_speedup": avg_speedup,
        "avg_median_speedup": median_speedup,
        "avg_geomean_speedup": geomean_speedup,
        "representative_speedup": representative_speedup,
        "best_speedup_seen": max_speedup,
        "latest_run_id": by_date[-1]["run_id"] if by_date else None,
        "latest_evidence_run_id": evidence_by_date[-1]["run_id"] if evidence_by_date else None,
    }
