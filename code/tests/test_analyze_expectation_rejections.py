import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.analysis.analyze_expectation_rejections import render_expectation_rejection_ledger


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_render_expectation_rejection_ledger_falls_back_to_stored_provenance(tmp_path: Path) -> None:
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "expectations_b200.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "hardware_key": "b200",
                "examples": {
                    "gemm": {
                        "example": "gemm",
                        "type": "python",
                        "metrics": {
                            "baseline_time_ms": 10.0,
                            "best_optimized_time_ms": 5.0,
                            "best_speedup": 2.0,
                            "best_optimized_speedup": 2.0,
                            "is_regression": False,
                        },
                        "metadata": {
                            "optimization_goal": "speed",
                            "best_optimization_speedup": 2.0,
                        },
                        "provenance": {
                            "git_commit": "storedsha",
                            "hardware_key": "b200",
                            "profile_name": "none",
                            "timestamp": "2026-03-10T00:00:00Z",
                            "iterations": 20,
                            "warmup_iterations": 5,
                            "execution_environment": "bare_metal",
                            "validity_profile": "strict",
                            "dmi_product_name": "StoredHost",
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_dir = repo_root / "artifacts" / "runs" / "current_run"
    comparison_run_dir = repo_root / "artifacts" / "runs" / "rerun"
    _write_jsonl(
        run_dir / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.2,
                "delta": -0.8,
                "delta_pct": -40.0,
                "baseline_time_ms": 10.0,
                "optimized_time_ms": 8.333333,
                "new_provenance": {"hardware_key": "b200"},
            }
        ],
    )
    _write_jsonl(
        comparison_run_dir / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.18,
                "delta": -0.82,
                "delta_pct": -41.0,
                "baseline_time_ms": 10.0,
                "optimized_time_ms": 8.474576,
                "update_message": "Provenance mismatch: git_commit differs",
                "validation_issue_types": ["provenance_mismatch"],
                "old_provenance": {
                    "git_commit": "storedsha",
                    "hardware_key": "b200",
                    "profile_name": "none",
                    "timestamp": "2026-03-10T00:00:00Z",
                    "iterations": 20,
                    "warmup_iterations": 5,
                    "execution_environment": "bare_metal",
                    "validity_profile": "strict",
                    "dmi_product_name": "StoredHost",
                },
                "new_provenance": {
                    "git_commit": "newsha",
                    "hardware_key": "b200",
                    "profile_name": "minimal",
                    "timestamp": "2026-03-11T00:00:00Z",
                    "iterations": 20,
                    "warmup_iterations": 5,
                    "execution_environment": "virtualized",
                    "validity_profile": "strict",
                    "dmi_product_name": "Standard PC (Q35 + ICH9, 2009)",
                },
                "provenance_mismatch_fields": ["execution_environment", "git_commit", "profile_name"],
            }
        ],
    )

    outputs = render_expectation_rejection_ledger(
        repo_root=repo_root,
        run_dir=run_dir,
        comparison_run_dirs=[comparison_run_dir],
        threshold_pct=25.0,
    )

    rows = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["expectations_file"] == "ch01/expectations_b200.json"
    assert row["stored_git_commit"] == "storedsha"
    assert row["stored_profile_name"] == "none"
    assert row["stored_execution_environment"] == "bare_metal"
    assert row["new_execution_environment"] is None
    assert row["stored_validity_profile"] == "strict"
    assert row["new_validity_profile"] is None
    assert row["rejection_reason"] == "rejected"
    assert row["comparison_score"] == 1.18
    assert row["comparison_vs_original_delta_pct"] == pytest.approx(-1.6666666666666685)
    assert row["comparison_stability"] == "stable"
    assert row["comparison_new_execution_environment"] == "virtualized"
    assert row["comparison_new_validity_profile"] == "strict"
    assert row["material_mismatch"] is True
    assert row["refresh_recommendation"] == "update_now"

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "Expectation Rejection Ledger" in markdown
    assert "`ch01:gemm`" in markdown


def test_render_expectation_rejection_ledger_surfaces_stored_expectation_warning(tmp_path: Path) -> None:
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "expectations_b200.json").write_text("{not-json", encoding="utf-8")

    run_dir = repo_root / "artifacts" / "runs" / "current_run"
    _write_jsonl(
        run_dir / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.2,
                "delta": -0.8,
                "delta_pct": -40.0,
                "new_provenance": {"hardware_key": "b200"},
            }
        ],
    )

    outputs = render_expectation_rejection_ledger(
        repo_root=repo_root,
        run_dir=run_dir,
        comparison_run_dirs=[],
        threshold_pct=25.0,
    )

    rows = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert rows[0]["stored_expectation_warning"]
    assert str(chapter_dir / "expectations_b200.json") in rows[0]["stored_expectation_warning"]

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "## Warnings" in markdown
    assert str(chapter_dir / "expectations_b200.json") in markdown


def test_render_expectation_rejection_ledger_raises_clear_error_for_malformed_event_log(tmp_path: Path) -> None:
    repo_root = tmp_path
    run_dir = repo_root / "artifacts" / "runs" / "current_run"
    events_path = run_dir / "logs" / "benchmark_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("{not-json\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        render_expectation_rejection_ledger(
            repo_root=repo_root,
            run_dir=run_dir,
            comparison_run_dirs=[],
            threshold_pct=25.0,
        )

    message = str(excinfo.value)
    assert "expectation update event log" in message
    assert str(events_path) in message


def test_analyze_expectation_rejections_cli_supports_repo_root_override(tmp_path: Path) -> None:
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "expectations_b200.json").write_text("{not-json", encoding="utf-8")

    run_dir = repo_root / "artifacts" / "runs" / "current_run"
    _write_jsonl(
        run_dir / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.2,
                "delta": -0.8,
                "delta_pct": -40.0,
                "new_provenance": {"hardware_key": "b200"},
            }
        ],
    )

    output_dir = run_dir / "analysis"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.analysis.analyze_expectation_rejections",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    rows = json.loads((output_dir / "current_run_expectation_rejections.json").read_text(encoding="utf-8"))
    assert rows[0]["stored_expectation_warning"]
    assert str(chapter_dir / "expectations_b200.json") in rows[0]["stored_expectation_warning"]


def test_render_expectation_rejection_ledger_merges_multiple_comparison_runs(tmp_path: Path) -> None:
    repo_root = tmp_path
    for chapter in ("ch01", "ch04"):
        chapter_dir = repo_root / chapter
        chapter_dir.mkdir(parents=True)
        (chapter_dir / "expectations_b200.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "hardware_key": "b200",
                    "examples": {
                        "gemm": {
                            "example": "gemm",
                            "type": "python",
                            "metrics": {
                                "baseline_time_ms": 10.0,
                                "best_optimized_time_ms": 5.0,
                                "best_speedup": 2.0,
                                "best_optimized_speedup": 2.0,
                                "is_regression": False,
                            },
                            "metadata": {"optimization_goal": "speed", "best_optimization_speedup": 2.0},
                            "provenance": {"git_commit": "stored", "hardware_key": "b200", "profile_name": "none"},
                        },
                        "dataparallel": {
                            "example": "dataparallel",
                            "type": "python",
                            "metrics": {
                                "baseline_time_ms": 10.0,
                                "best_optimized_time_ms": 5.0,
                                "best_speedup": 2.0,
                                "best_optimized_speedup": 2.0,
                                "is_regression": False,
                            },
                            "metadata": {"optimization_goal": "speed", "best_optimization_speedup": 2.0},
                            "provenance": {"git_commit": "stored", "hardware_key": "b200", "profile_name": "none"},
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    run_dir = repo_root / "artifacts" / "runs" / "current_run"
    comparison_run_dir_a = repo_root / "artifacts" / "runs" / "rerun_a"
    comparison_run_dir_b = repo_root / "artifacts" / "runs" / "rerun_b"

    _write_jsonl(
        run_dir / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.2,
                "delta": -0.8,
                "delta_pct": -40.0,
                "new_provenance": {"hardware_key": "b200"},
            },
            {
                "event_type": "expectation_update",
                "chapter": "ch04",
                "example": "dataparallel",
                "goal": "speed",
                "status": "rejected",
                "old_score": 4.5,
                "new_score": 5.8,
                "delta": 1.3,
                "delta_pct": 28.9,
                "new_provenance": {"hardware_key": "b200"},
            },
        ],
    )
    _write_jsonl(
        comparison_run_dir_a / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch01",
                "example": "gemm",
                "goal": "speed",
                "status": "rejected",
                "old_score": 2.0,
                "new_score": 1.18,
                "delta": -0.82,
                "delta_pct": -41.0,
                "new_provenance": {"hardware_key": "b200"},
            }
        ],
    )
    _write_jsonl(
        comparison_run_dir_b / "logs" / "benchmark_events.jsonl",
        [
            {
                "event_type": "expectation_update",
                "chapter": "ch04",
                "example": "dataparallel",
                "goal": "speed",
                "status": "rejected",
                "old_score": 4.5,
                "new_score": 4.9,
                "delta": 0.4,
                "delta_pct": 8.9,
                "new_provenance": {"hardware_key": "b200"},
            }
        ],
    )

    outputs = render_expectation_rejection_ledger(
        repo_root=repo_root,
        run_dir=run_dir,
        comparison_run_dirs=[comparison_run_dir_a, comparison_run_dir_b],
        threshold_pct=25.0,
    )

    rows = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert len(rows) == 2
    by_target = {row["target"]: row for row in rows}
    assert by_target["ch01:gemm"]["comparison_score"] == 1.18
    assert by_target["ch04:dataparallel"]["comparison_score"] == 4.9
