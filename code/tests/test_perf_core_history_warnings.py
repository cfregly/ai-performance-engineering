from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.perf_core_base import PerformanceCoreBase


class _HistoryRootCore(PerformanceCoreBase):
    def __init__(
        self, *, history_root: Path, data_file: Path | None = None, bench_root: Path | None = None
    ) -> None:
        super().__init__(data_file=data_file, bench_root=bench_root)
        self._history_root = history_root

    def _tier1_history_root(self) -> Path:
        return self._history_root


def _build_history_root(tmp_path: Path, run_entry: dict) -> Path:
    history_root = tmp_path / "history"
    history_root.mkdir()
    (history_root / "index.json").write_text(
        json.dumps({"suite_name": "tier1", "suite_version": 1, "runs": [run_entry]}),
        encoding="utf-8",
    )
    return history_root


def test_tier1_history_runs_surface_summary_read_warnings(tmp_path: Path) -> None:
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "summary_path": "run_bad/bad_summary.json"},
    )
    summary_path = history_root / "run_bad" / "bad_summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text("{not-json", encoding="utf-8")
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_history_runs()

    assert result["total_runs"] == 0
    assert result["warnings"]
    assert any("JSONDecodeError" in warning for warning in result["warnings"])
    assert all(str(summary_path) not in warning for warning in result["warnings"])


def test_tier1_trends_surface_trend_snapshot_read_warnings(tmp_path: Path) -> None:
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "trend_snapshot_path": "run_bad/bad_trend.json"},
    )
    trend_path = history_root / "run_bad" / "bad_trend.json"
    trend_path.parent.mkdir()
    trend_path.write_text("[]", encoding="utf-8")
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_trends()

    assert result["warnings"]
    assert any("Expected Tier-1 JSON object" in warning for warning in result["warnings"])
    assert all(str(trend_path) not in warning for warning in result["warnings"])


def test_tier1_target_history_surfaces_summary_read_warnings(tmp_path: Path) -> None:
    history_root = _build_history_root(
        tmp_path,
        {"run_id": "run_bad", "summary_path": "run_bad/bad_target_summary.json"},
    )
    summary_path = history_root / "run_bad" / "bad_target_summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text("{not-json", encoding="utf-8")
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_target_history(key="ch01:demo")

    assert result["run_count"] == 0
    assert result["warnings"]
    assert any("JSONDecodeError" in warning for warning in result["warnings"])
    assert all(str(summary_path) not in warning for warning in result["warnings"])


def test_tier1_history_runs_surface_index_read_warnings(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    bad_index = history_root / "index.json"
    bad_index.write_text("{not-json", encoding="utf-8")

    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_history_runs()

    assert result["warnings"]
    assert any("Failed to read tier-1 history index" in warning for warning in result["warnings"])
    assert all(str(bad_index) not in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("index_value", "expected"),
    [(True, True), (False, False), (None, False)],
)
def test_tier1_history_runs_expose_baseline_eligibility(
    tmp_path: Path, index_value: bool | None, expected: bool
) -> None:
    run_entry = {
        "run_id": "run_a",
        "summary_path": "run_a/summary.json",
    }
    if index_value is not None:
        run_entry["baseline_eligible"] = index_value
    history_root = _build_history_root(
        tmp_path,
        run_entry,
    )
    summary_path = history_root / "run_a" / "summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "summary": {
                    "target_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                },
                "targets": [{"key": "demo", "status": "succeeded"}],
            }
        ),
        encoding="utf-8",
    )
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_history_runs()

    assert result["runs"][0]["baseline_eligible"] is expected
    assert result["latest"]["run"]["baseline_eligible"] is expected


def test_tier1_history_consumers_resolve_relocated_index_and_evidence_locator(
    tmp_path: Path,
) -> None:
    original_root = tmp_path / "original" / "history"
    run_dir = original_root / "tier1_run_a"
    run_dir.mkdir(parents=True)
    summary = {
        "suite_name": "tier1",
        "suite_version": 1,
        "run_id": "tier1_run_a",
        "generated_at": "2026-08-16T00:00:00Z",
        "evidence_artifact_name": "tier1-evidence-123-1",
        "source_result_json": "tier1_run_a/results/benchmark_test_results.json",
        "targets": [
            {
                "key": "demo",
                "target": "ch01:demo",
                "status": "succeeded",
                "best_speedup": 2.0,
                "artifacts": {"nsys_rep": "tier1_run_a/profiles/demo.nsys-rep"},
            }
        ],
        "summary": {
            "target_count": 1,
            "succeeded": 1,
            "failed": 0,
            "skipped": 0,
            "missing": 0,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (original_root / "index.json").write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "history_root": ".",
                "runs": [
                    {
                        "run_id": "tier1_run_a",
                        "run_accepted": True,
                        "summary_path": "tier1_run_a/summary.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    relocated_root = tmp_path / "relocated" / "history"
    relocated_root.parent.mkdir()
    original_root.rename(relocated_root)
    core = _HistoryRootCore(history_root=relocated_root, bench_root=tmp_path)

    history = core.get_tier1_history_runs()
    target_history = core.get_tier1_target_history(key="demo")

    assert history["total_runs"] == 1
    assert history["latest"]["run"]["source_result_json"] == (
        "artifact://tier1-evidence-123-1/tier1_run_a/results/benchmark_test_results.json"
    )
    assert target_history["history"][0]["artifacts"]["nsys_rep"] == (
        "artifact://tier1-evidence-123-1/tier1_run_a/profiles/demo.nsys-rep"
    )


def test_tier1_target_history_keeps_unaccepted_outliers_out_of_canonical_headlines(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    for run_id, speedup in (("run_accepted", 2.0), ("run_imported", 1000.0)):
        run_dir = history_root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "targets": [
                        {
                            "key": "demo",
                            "target": "ch01:demo",
                            "status": "succeeded",
                            "best_speedup": speedup,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    (history_root / "index.json").write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "runs": [
                    {
                        "run_id": "run_accepted",
                        "run_accepted": True,
                        "baseline_eligible": True,
                        "summary_path": "run_accepted/summary.json",
                    },
                    {
                        "run_id": "run_imported",
                        "run_accepted": False,
                        "baseline_eligible": False,
                        "summary_path": "run_imported/summary.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_target_history(key="demo")

    assert result["run_count"] == 1
    assert result["evidence_run_count"] == 2
    assert result["best_speedup_seen"] == 2.0
    assert result["latest"]["run_id"] == "run_accepted"
    assert result["latest_evidence"]["run_id"] == "run_imported"
    assert [point["best_speedup"] for point in result["history"]] == [2.0]
    assert [point["best_speedup"] for point in result["evidence_history"]] == [2.0, 1000.0]


def test_tier1_dashboard_consumers_order_delayed_evidence_by_generated_time(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    entries = []
    for run_id, generated_at, accepted in (
        ("newer_anchor", "2026-08-17T00:00:00Z", True),
        ("older_producer", "2026-08-16T23:00:00Z", False),
    ):
        run_dir = history_root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "generated_at": generated_at,
                    "summary": {"target_count": 1, "succeeded": 1},
                    "targets": [
                        {
                            "key": "demo",
                            "target": "ch01:demo",
                            "status": "succeeded",
                            "best_speedup": 2.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "summary_path": f"{run_id}/summary.json",
                "run_accepted": accepted,
                "baseline_eligible": accepted,
            }
        )
    (history_root / "index.json").write_text(
        json.dumps({"suite_name": "tier1", "suite_version": 1, "runs": entries}),
        encoding="utf-8",
    )
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    history = core.get_tier1_history_runs()
    target_history = core.get_tier1_target_history(key="demo")

    assert [run["run_id"] for run in history["runs"]] == [
        "older_producer",
        "newer_anchor",
    ]
    assert history["latest_evidence_run_id"] == "newer_anchor"
    assert history["latest_run_id"] == "newer_anchor"
    assert target_history["latest_evidence"]["run_id"] == "newer_anchor"
    assert target_history["latest"]["run_id"] == "newer_anchor"


def test_tier1_target_history_uses_accepted_identity_not_rejected_metadata(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    fixtures = (
        ("run_rejected", False, "ch99:poison", "rejected"),
        ("run_accepted", True, "ch01:demo", "canonical"),
    )
    entries = []
    for position, (run_id, accepted, target, category) in enumerate(fixtures):
        generated_at = f"2026-08-16T0{position}:00:00Z"
        run_dir = history_root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "generated_at": generated_at,
                    "targets": [
                        {
                            "key": "demo",
                            "target": target,
                            "category": category,
                            "status": "succeeded",
                            "best_speedup": 2.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "run_accepted": accepted,
                "baseline_eligible": accepted,
                "summary_path": f"{run_id}/summary.json",
            }
        )
    (history_root / "index.json").write_text(
        json.dumps({"suite_name": "tier1", "suite_version": 1, "runs": entries}),
        encoding="utf-8",
    )
    core = _HistoryRootCore(history_root=history_root, bench_root=tmp_path)

    result = core.get_tier1_target_history(key="demo")

    assert result["selected_target"] == "ch01:demo"
    assert result["category"] == "canonical"
    assert [point["target"] for point in result["history"]] == ["ch01:demo"]
    assert [point["target"] for point in result["evidence_history"]] == [
        "ch99:poison",
        "ch01:demo",
    ]
