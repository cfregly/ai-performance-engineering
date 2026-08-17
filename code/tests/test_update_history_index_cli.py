from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_summary_payload(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "generated_at": "2026-08-16T00:00:00Z",
        "source_git_commit": "a" * 40,
        "targets": [
            {
                "target": "ch01:demo",
                "status": "succeeded",
                "optimization_goal": "performance",
                "baseline_time_ms": 1.0,
                "best_speedup": 2.0,
                "best_optimized_time_ms": 0.5,
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


def _write_trend_snapshot(tmp_path: Path) -> Path:
    trend_snapshot = tmp_path / "trend_snapshot.json"
    trend_snapshot.write_text('{"run_count": 1}', encoding="utf-8")
    return trend_snapshot


def _history_bytes(history_root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(history_root).as_posix(): path.read_bytes()
        for path in history_root.rglob("*")
        if path.is_file()
    }


def test_update_history_index_cli_rejects_malformed_summary(tmp_path: Path) -> None:
    summary_json = tmp_path / "summary.json"
    summary_json.write_text("{not-json", encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# placeholder\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    history_root = tmp_path / "history"
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Failed to read tier-1 summary JSON" in completed.stderr
    assert str(summary_json) in completed.stderr
    assert "Traceback" not in completed.stderr


def test_update_history_index_cli_copies_external_inputs_into_portable_run_dir(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "external"
    inputs.mkdir()
    summary_json = inputs / "source_summary.json"
    summary_payload = _valid_summary_payload("tier1_run_a")
    summary_json.write_text(json.dumps(summary_payload), encoding="utf-8")
    regression_summary = inputs / "report.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = inputs / "comparison.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = inputs / "trend.json"
    trend_snapshot.write_text('{"run_count": 1}', encoding="utf-8")
    history_root = tmp_path / "history"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    index = json.loads((history_root / "index.json").read_text(encoding="utf-8"))
    entry = index["runs"][0]
    assert entry["summary_path"] == "tier1_run_a/summary.json"
    assert entry["regression_summary_path"] == "tier1_run_a/regression_summary.md"
    assert entry["regression_json_path"] == "tier1_run_a/regression_summary.json"
    assert entry["trend_snapshot_path"] == "tier1_run_a/trend_snapshot.json"
    assert entry["baseline_eligible"] is False
    assert (
        json.loads((history_root / "tier1_run_a" / "summary.json").read_text(encoding="utf-8"))
        == summary_payload
    )
    derived_report = (history_root / "tier1_run_a" / "regression_summary.md").read_text(
        encoding="utf-8"
    )
    assert derived_report.startswith("# Tier-1 Regression Summary\n")
    assert "passes baseline eligibility" in derived_report
    assert "becomes the initial history anchor" not in derived_report
    derived_comparison = json.loads(
        (history_root / "tier1_run_a" / "regression_summary.json").read_text(encoding="utf-8")
    )
    assert derived_comparison["baseline_run_id"] is None
    assert derived_comparison["current_run_id"] == "tier1_run_a"
    derived_trend = json.loads(
        (history_root / "tier1_run_a" / "trend_snapshot.json").read_text(encoding="utf-8")
    )
    assert derived_trend["run_count"] == 0
    assert derived_trend["evidence_run_count"] == 1
    assert derived_trend["latest_run_id"] is None
    assert derived_trend["latest_evidence_run_id"] == "tier1_run_a"


@pytest.mark.parametrize("run_id", ["../escape", "..", "."])
def test_update_history_index_cli_rejects_unsafe_run_id(
    tmp_path: Path,
    run_id: str,
) -> None:
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps({"run_id": run_id, "summary": {"target_count": 0}}),
        encoding="utf-8",
    )
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(tmp_path / "history"),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Unsafe run_id" in completed.stderr
    assert not (tmp_path / "history").exists()


def test_update_history_index_cli_keeps_regressed_run_ineligible(tmp_path: Path) -> None:
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(_valid_summary_payload("tier1_regressed")),
        encoding="utf-8",
    )
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# regression\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        json.dumps(
            {
                "regressions": [{"target": "ch01:demo"}],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    history_root = tmp_path / "history"
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    index = json.loads((history_root / "index.json").read_text(encoding="utf-8"))
    assert index["runs"][0]["baseline_eligible"] is False


def test_update_history_index_cli_preserves_malformed_existing_index(tmp_path: Path) -> None:
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(_valid_summary_payload("tier1_run_a")),
        encoding="utf-8",
    )
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    history_root = tmp_path / "history"
    history_root.mkdir()
    index_path = history_root / "index.json"
    index_path.write_text("{not-json", encoding="utf-8")
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Refusing to update invalid Tier-1 history index" in completed.stderr
    assert index_path.read_text(encoding="utf-8") == "{not-json"
    assert not (history_root / "tier1_run_a").exists()


def test_update_history_index_cli_rejects_mismatched_target_count(tmp_path: Path) -> None:
    summary_payload = _valid_summary_payload("tier1_bad_shape")
    summary_payload["summary"]["target_count"] = 2
    summary_payload["summary"]["succeeded"] = 2
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(json.dumps(summary_payload), encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)
    history_root = tmp_path / "history"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "target_count does not match targets list" in completed.stderr
    assert not history_root.exists()


def test_update_history_index_cli_derives_regression_against_eligible_anchor(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    anchor_dir = history_root / "tier1_anchor"
    anchor_dir.mkdir(parents=True)
    anchor_summary = _valid_summary_payload("tier1_anchor")
    (anchor_dir / "summary.json").write_text(json.dumps(anchor_summary), encoding="utf-8")
    (anchor_dir / "regression_summary.md").write_text("# anchor\n", encoding="utf-8")
    (anchor_dir / "regression_summary.json").write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    (anchor_dir / "trend_snapshot.json").write_text('{"run_count": 1}', encoding="utf-8")
    (history_root / "index.json").write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "history_root": ".",
                "runs": [
                    {
                        "run_id": "tier1_anchor",
                        "generated_at": "2026-08-15T00:00:00Z",
                        "summary_path": "tier1_anchor/summary.json",
                        "regression_summary_path": "tier1_anchor/regression_summary.md",
                        "regression_json_path": "tier1_anchor/regression_summary.json",
                        "trend_snapshot_path": "tier1_anchor/trend_snapshot.json",
                        "baseline_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    current_summary = _valid_summary_payload("tier1_slower")
    current_target = current_summary["targets"][0]
    current_target["best_speedup"] = 1.0
    current_target["best_optimized_time_ms"] = 1.0
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(json.dumps(current_summary), encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# forged clean report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    derived = json.loads(
        (history_root / "tier1_slower" / "regression_summary.json").read_text(encoding="utf-8")
    )
    assert derived["baseline_run_id"] == "tier1_anchor"
    assert [row["target"] for row in derived["regressions"]] == ["ch01:demo"]
    index = json.loads((history_root / "index.json").read_text(encoding="utf-8"))
    current_entry = next(row for row in index["runs"] if row["run_id"] == "tier1_slower")
    assert current_entry["baseline_eligible"] is False


def test_update_history_index_cli_rejects_existing_run_id_without_writes(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    anchor_dir = history_root / "tier1_anchor"
    anchor_dir.mkdir(parents=True)
    anchor_summary = _valid_summary_payload("tier1_anchor")
    (anchor_dir / "summary.json").write_text(json.dumps(anchor_summary), encoding="utf-8")
    (anchor_dir / "regression_summary.md").write_text("# anchor\n", encoding="utf-8")
    (anchor_dir / "regression_summary.json").write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    (anchor_dir / "trend_snapshot.json").write_text('{"run_count": 1}', encoding="utf-8")
    index_path = history_root / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "history_root": ".",
                "runs": [
                    {
                        "run_id": "tier1_anchor",
                        "summary_path": "tier1_anchor/summary.json",
                        "regression_summary_path": "tier1_anchor/regression_summary.md",
                        "regression_json_path": "tier1_anchor/regression_summary.json",
                        "trend_snapshot_path": "tier1_anchor/trend_snapshot.json",
                        "baseline_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = _history_bytes(history_root)

    replacement = _valid_summary_payload("tier1_anchor")
    replacement["targets"][0]["best_speedup"] = 0.5
    summary_json = tmp_path / "replacement.json"
    summary_json.write_text(json.dumps(replacement), encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# replacement\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    after = _history_bytes(history_root)
    assert completed.returncode == 1
    assert "Refusing to overwrite existing Tier-1 history run" in completed.stderr
    assert after == before


def test_update_history_index_cli_rejects_unindexed_run_directory_without_writes(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    orphan_dir = history_root / "tier1_orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "keep.txt").write_text("preserve", encoding="utf-8")
    before = _history_bytes(history_root)

    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(_valid_summary_payload("tier1_orphan")),
        encoding="utf-8",
    )
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Refusing to overwrite existing Tier-1 history run" in completed.stderr
    assert _history_bytes(history_root) == before


@pytest.mark.parametrize("invalid_metric", ["not-a-number", "2.0"])
def test_update_history_index_cli_derivation_failure_leaves_history_unchanged(
    tmp_path: Path,
    invalid_metric: str,
) -> None:
    history_root = tmp_path / "history"
    anchor_dir = history_root / "tier1_anchor"
    anchor_dir.mkdir(parents=True)
    anchor_summary = _valid_summary_payload("tier1_anchor")
    (anchor_dir / "summary.json").write_text(json.dumps(anchor_summary), encoding="utf-8")
    (anchor_dir / "regression_summary.md").write_text("# anchor\n", encoding="utf-8")
    (anchor_dir / "regression_summary.json").write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    (anchor_dir / "trend_snapshot.json").write_text('{"run_count": 1}', encoding="utf-8")
    (history_root / "index.json").write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "history_root": ".",
                "runs": [
                    {
                        "run_id": "tier1_anchor",
                        "summary_path": "tier1_anchor/summary.json",
                        "regression_summary_path": "tier1_anchor/regression_summary.md",
                        "regression_json_path": "tier1_anchor/regression_summary.json",
                        "trend_snapshot_path": "tier1_anchor/trend_snapshot.json",
                        "baseline_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = _history_bytes(history_root)

    invalid_summary = _valid_summary_payload("tier1_invalid_metric")
    invalid_summary["targets"][0]["best_speedup"] = invalid_metric
    summary_json = tmp_path / "invalid_summary.json"
    summary_json.write_text(json.dumps(invalid_summary), encoding="utf-8")
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Invalid target best_speedup metric" in completed.stderr
    assert _history_bytes(history_root) == before


def test_update_history_index_cli_rejects_non_object_existing_entry_without_writes(
    tmp_path: Path,
) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    index_path = history_root / "index.json"
    index_path.write_text('{"suite_name": "tier1", "runs": [42]}', encoding="utf-8")
    before = _history_bytes(history_root)
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(_valid_summary_payload("tier1_new")),
        encoding="utf-8",
    )
    regression_summary = tmp_path / "regression_summary.md"
    regression_summary.write_text("# report\n", encoding="utf-8")
    regression_json = tmp_path / "regression_summary.json"
    regression_json.write_text(
        '{"regressions": [], "missing_targets": []}',
        encoding="utf-8",
    )
    trend_snapshot = _write_trend_snapshot(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.benchmarks.update_history_index",
            "--summary-json",
            str(summary_json),
            "--regression-summary",
            str(regression_summary),
            "--regression-json",
            str(regression_json),
            "--trend-snapshot",
            str(trend_snapshot),
            "--history-root",
            str(history_root),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "non-object run entry" in completed.stderr
    assert _history_bytes(history_root) == before
