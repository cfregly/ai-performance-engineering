from __future__ import annotations

import json
import os
from pathlib import Path

import core.benchmark.bench_commands as bench_commands
import core.harness.run_benchmarks as run_benchmarks_module
import pytest
from cli.aisp import app
from core.analysis.history_index import resolve_history_entry_path, update_history_index
from core.analysis.regressions import compare_suite_summaries
from core.analysis.trends import build_trend_snapshot
from core.benchmark.suites.tier1 import (
    _confirm_speedup_regressions,
    _tier1_baseline_eligible,
    build_tier1_suite_summary,
    load_tier1_suite,
    run_tier1_suite,
)
from typer.testing import CliRunner


def _write_result_payload(
    path: Path,
    *,
    block_scaling_speedup: float,
    flash_speedup: float,
    kv_speedup: float = 1.58,
    kv_memory_savings_pct: float = 49.7,
    llama_speedup: float = 2.49,
) -> None:
    payload = {
        "timestamp": "2026-03-08T00:00:00Z",
        "results": [
            {
                "chapter": "labs_block_scaling",
                "benchmarks": [
                    {
                        "example": "block_scaling",
                        "status": "succeeded",
                        "baseline_time_ms": 0.1074,
                        "best_speedup": block_scaling_speedup,
                        "best_optimization": "hardware_block_scaled",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 512.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/block_scaling/baseline_block_scaling.py",
                        "nsys_rep": "artifacts/block_scaling.nsys-rep",
                    }
                ],
            },
            {
                "chapter": "labs_flashattention4",
                "benchmarks": [
                    {
                        "example": "flashattention4_alibi",
                        "status": "succeeded",
                        "baseline_time_ms": 5.5622,
                        "best_speedup": flash_speedup,
                        "best_optimization": "flashattention4_alibi",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 1024.0,
                        "best_memory_savings_pct": 18.5,
                        "baseline_file": "labs/flashattention4/baseline_flashattention4.py",
                        "ncu_json": "artifacts/flashattention4_alibi.json",
                    }
                ],
            },
            {
                "chapter": "labs_persistent_decode",
                "benchmarks": [
                    {
                        "example": "persistent_decode",
                        "status": "succeeded",
                        "baseline_time_ms": 1.4107,
                        "best_speedup": 11.93,
                        "best_optimization": "graphs",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 256.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/persistent_decode/baseline_persistent_decode.py",
                    }
                ],
            },
            {
                "chapter": "labs_kv_optimization",
                "benchmarks": [
                    {
                        "example": "kv_standard",
                        "status": "succeeded",
                        "baseline_time_ms": 1585.6,
                        "best_speedup": kv_speedup,
                        "best_optimization": "kv_standard",
                        "optimization_goal": "memory",
                        "baseline_memory_mb": 32140.0,
                        "best_memory_savings_pct": kv_memory_savings_pct,
                        "baseline_file": "labs/kv_optimization/baseline_kv_standard.py",
                    }
                ],
            },
            {
                "chapter": "ch04",
                "benchmarks": [
                    {
                        "example": "gradient_fusion",
                        "status": "succeeded",
                        "baseline_time_ms": 1.0,
                        "best_speedup": 1.21,
                        "best_optimization": "gradient_fusion",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 64.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "ch04/baseline_gradient_fusion.py",
                    }
                ],
            },
            {
                "chapter": "labs_real_world_models",
                "benchmarks": [
                    {
                        "example": "llama_3_1_8b",
                        "status": "succeeded",
                        "baseline_time_ms": 13.143,
                        "best_speedup": llama_speedup,
                        "best_optimization": "optimized_llama_3_1_8b",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 4096.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/real_world_models/baseline_llama_3_1_8b.py",
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_tier1_suite_summary_and_history_artifacts(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    result_json = tmp_path / "results.json"
    _write_result_payload(result_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    summary = build_tier1_suite_summary(result_json, suite, run_id="tier1_run_a")

    assert summary["suite_name"] == "tier1"
    assert summary["summary"]["target_count"] == 6
    assert summary["summary"]["succeeded"] == 6
    assert summary["summary"]["missing"] == 0
    assert summary["summary"]["median_speedup"] > 0
    assert summary["summary"]["representative_speedup"] == summary["summary"]["geomean_speedup"]

    block_scaling = next(
        target for target in summary["targets"] if target["key"] == "block_scaling"
    )
    assert block_scaling["best_speedup"] == 1.45
    assert block_scaling["best_optimized_time_ms"] == 0.1074 / 1.45
    assert block_scaling["artifacts"]["nsys_rep"] == (
        "tier1_run_a/artifacts/block_scaling.nsys-rep"
    )

    history_root = tmp_path / "history"
    run_dir = history_root / "tier1_run_a"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "summary.json"
    regression_md_path = run_dir / "regression_summary.md"
    regression_json_path = run_dir / "regression_summary.json"
    trend_snapshot_path = run_dir / "trend_snapshot.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    regression_md_path.write_text("# placeholder\n", encoding="utf-8")
    regression_json_path.write_text("{}", encoding="utf-8")
    trend_snapshot_path.write_text("{}", encoding="utf-8")

    updated_index = update_history_index(
        history_root=history_root,
        suite=suite,
        summary=summary,
        summary_path=summary_path,
        regression_summary_path=regression_md_path,
        regression_json_path=regression_json_path,
        trend_snapshot_path=trend_snapshot_path,
        run_accepted=True,
    )

    assert updated_index["suite_name"] == "tier1"
    assert updated_index["history_root"] == "."
    assert len(updated_index["runs"]) == 1
    assert updated_index["runs"][0]["summary_path"] == "tier1_run_a/summary.json"
    assert updated_index["runs"][0]["regression_json_path"] == (
        "tier1_run_a/regression_summary.json"
    )
    assert updated_index["runs"][0]["median_speedup"] == summary["summary"]["median_speedup"]
    assert (
        updated_index["runs"][0]["representative_speedup"]
        == summary["summary"]["representative_speedup"]
    )

    trend = build_trend_snapshot(updated_index)
    assert trend["run_count"] == 1
    assert trend["latest_run_id"] == "tier1_run_a"
    assert trend["best_speedup_seen"] == summary["summary"]["max_speedup"]
    assert trend["representative_speedup"] == summary["summary"]["representative_speedup"]
    assert trend["avg_median_speedup"] == summary["summary"]["median_speedup"]


def test_trend_snapshot_excludes_ineligible_evidence_from_canonical_headlines() -> None:
    trend = build_trend_snapshot(
        {
            "suite_name": "tier1",
            "runs": [
                {
                    "run_id": "tier1_anchor",
                    "run_accepted": True,
                    "baseline_eligible": True,
                    "avg_speedup": 2.0,
                    "median_speedup": 2.0,
                    "geomean_speedup": 2.0,
                    "representative_speedup": 2.0,
                    "max_speedup": 2.0,
                },
                {
                    "run_id": "tier1_external_import",
                    "run_accepted": False,
                    "baseline_eligible": False,
                    "avg_speedup": 1000.0,
                    "median_speedup": 1000.0,
                    "geomean_speedup": 1000.0,
                    "representative_speedup": 1000.0,
                    "max_speedup": 1000.0,
                },
            ],
        }
    )

    assert trend["run_count"] == 1
    assert trend["evidence_run_count"] == 2
    assert trend["best_speedup_seen"] == 2.0
    assert trend["latest_run_id"] == "tier1_anchor"
    assert trend["latest_evidence_run_id"] == "tier1_external_import"


def test_trend_snapshot_orders_runs_by_parsed_time_and_run_id() -> None:
    trend = build_trend_snapshot(
        {
            "suite_name": "tier1",
            "runs": [
                {
                    "run_id": "run_b",
                    "generated_at": "2026-08-17T00:00:00Z",
                    "run_accepted": True,
                    "baseline_eligible": True,
                },
                {
                    "run_id": "run_earlier",
                    "generated_at": "2026-08-17T00:30:00+01:00",
                    "run_accepted": False,
                    "baseline_eligible": False,
                },
                {
                    "run_id": "run_a",
                    "generated_at": "2026-08-17T02:00:00+02:00",
                    "run_accepted": True,
                    "baseline_eligible": True,
                },
            ],
        }
    )

    assert [row["run_id"] for row in trend["evidence_history"]] == [
        "run_earlier",
        "run_a",
        "run_b",
    ]
    assert trend["latest_run_id"] == "run_b"
    assert trend["latest_evidence_run_id"] == "run_b"


def test_build_tier1_suite_summary_uses_portable_evidence_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = load_tier1_suite()
    run_id = "tier1_portable"
    run_root = tmp_path / "artifacts" / "runs" / run_id
    result_json = run_root / "results" / "benchmark_test_results.json"
    result_json.parent.mkdir(parents=True)
    _write_result_payload(result_json, block_scaling_speedup=1.45, flash_speedup=14.45)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    block_scaling = payload["results"][0]["benchmarks"][0]
    block_scaling["nsys_rep"] = str(run_root / "profiles" / "block_scaling.nsys-rep")
    block_scaling["baseline_file"] = str(Path.cwd() / "labs" / "block_scaling" / "baseline.py")
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    report_path = run_root / "report.md"
    report_path.write_text("# report\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    summary = build_tier1_suite_summary(
        result_json,
        suite,
        run_id=run_id,
        manifest_path=manifest_path,
        report_path=report_path,
        evidence_artifact_name="tier1-evidence-123-1",
    )

    target = next(item for item in summary["targets"] if item["key"] == "block_scaling")
    assert summary["source_result_json"] == ("tier1_portable/results/benchmark_test_results.json")
    assert summary["source_manifest_json"] == "tier1_portable/manifest.json"
    assert summary["source_markdown_report"] == "tier1_portable/report.md"
    assert summary["source_git_commit"] == "a" * 40
    assert summary["evidence_artifact_name"] == "tier1-evidence-123-1"
    assert target["baseline_file"] == "labs/block_scaling/baseline.py"
    assert target["artifacts"]["nsys_rep"] == ("tier1_portable/profiles/block_scaling.nsys-rep")
    assert str(tmp_path) not in json.dumps(summary)


def test_history_index_paths_survive_history_root_relocation(tmp_path: Path) -> None:
    history_root = tmp_path / "original" / "history"
    run_dir = history_root / "run_a"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    relocated_root = tmp_path / "relocated" / "history"
    relocated_root.parent.mkdir()
    history_root.rename(relocated_root)

    resolved = resolve_history_entry_path(
        relocated_root,
        "run_a/summary.json",
        run_id="run_a",
    )

    assert resolved == relocated_root / "run_a" / "summary.json"
    assert resolved.read_text(encoding="utf-8") == "{}"


def test_legacy_absolute_history_path_relocates_by_run_id(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    run_dir = history_root / "run_a"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    resolved = resolve_history_entry_path(
        history_root,
        Path("/retired/runner/workspace/history/run_a/summary.json"),
        run_id="run_a",
    )

    assert resolved == summary_path


def test_portable_history_path_rejects_escape(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()

    try:
        resolve_history_entry_path(history_root, "../outside.json", run_id="run_a")
    except ValueError as exc:
        assert "escapes" in str(exc)
        assert str(history_root) not in str(exc)
        assert "outside.json" not in str(exc)
    else:
        raise AssertionError("history path escape was accepted")


@pytest.mark.parametrize("run_id", ["../outside", "/tmp/outside", ".", "..", "bad/run"])
def test_benchmark_run_id_rejects_unsafe_components(run_id: str) -> None:
    with pytest.raises(ValueError, match="Run id must start"):
        bench_commands._validate_run_id(run_id)


def test_run_tier1_rejects_history_run_collision_before_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history_root = tmp_path / "history"
    (history_root / "tier1_existing").mkdir(parents=True)

    def _unexpected_execute(**kwargs):
        raise AssertionError("benchmark must not run after a history collision")

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _unexpected_execute)

    with pytest.raises(ValueError, match="Refusing to overwrite existing Tier-1 history run"):
        run_tier1_suite(
            history_root=history_root,
            bench_root=tmp_path,
            run_id="tier1_existing",
        )


def test_run_tier1_rejects_evidence_run_collision_before_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    (artifacts_root / "tier1_existing").mkdir(parents=True)

    def _unexpected_execute(**kwargs):
        raise AssertionError("benchmark must not run after an evidence collision")

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _unexpected_execute)

    with pytest.raises(ValueError, match="Refusing to overwrite existing Tier-1 evidence run"):
        run_tier1_suite(
            history_root=tmp_path / "history",
            bench_root=tmp_path,
            artifacts_dir=str(artifacts_root),
            run_id="tier1_existing",
        )


def test_compare_suite_summaries_detects_speedup_regression_and_new_targets(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_result_payload(baseline_json, block_scaling_speedup=1.60, flash_speedup=12.0)
    _write_result_payload(current_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")
    llama_baseline = next(
        target
        for target in baseline_summary["targets"]
        if target["target"] == "labs/real_world_models:llama_3_1_8b"
    )
    llama_current = next(
        target
        for target in current_summary["targets"]
        if target["target"] == "labs/real_world_models:llama_3_1_8b"
    )
    llama_baseline["best_speedup"] = 2.49
    llama_baseline["best_optimized_time_ms"] = 13.143 / 2.49
    llama_current["best_speedup"] = 2.20
    llama_current["best_optimized_time_ms"] = 13.143 / 2.20

    comparison = compare_suite_summaries(current_summary, baseline_summary)

    assert comparison["baseline_run_id"] == "tier1_old"
    assert any(
        row["target"] == "labs/real_world_models:llama_3_1_8b" and row["reason"] == "speedup"
        for row in comparison["regressions"]
    )
    assert any(
        row["target"] == "labs/flashattention4:flashattention4_alibi" and row["reason"] == "speedup"
        for row in comparison["improvements"]
    )


def test_compare_suite_summaries_detects_common_mode_optimized_latency_regression() -> None:
    baseline = {
        "run_id": "tier1_anchor",
        "targets": [
            {
                "target": "ch01:demo",
                "status": "succeeded",
                "optimization_goal": "performance",
                "best_speedup": 2.0,
                "best_optimized_time_ms": 50.0,
            }
        ],
    }
    current = {
        "run_id": "tier1_slower",
        "targets": [
            {
                "target": "ch01:demo",
                "status": "succeeded",
                "optimization_goal": "performance",
                "best_speedup": 2.0,
                "best_optimized_time_ms": 100.0,
            }
        ],
    }

    comparison = compare_suite_summaries(current, baseline)

    assert any(
        row["target"] == "ch01:demo" and row["reason"] == "optimized_latency"
        for row in comparison["regressions"]
    )


def test_subthreshold_decline_does_not_move_anchor_before_cumulative_regression() -> None:
    def _summary(run_id: str, speedup: float, optimized_ms: float) -> dict:
        return {
            "run_id": run_id,
            "targets": [
                {
                    "target": "ch01:demo",
                    "status": "succeeded",
                    "optimization_goal": "performance",
                    "best_speedup": speedup,
                    "best_optimized_time_ms": optimized_ms,
                }
            ],
        }

    anchor = _summary("tier1_anchor", 2.0, 50.0)
    first_drift = _summary("tier1_first_drift", 1.92, 52.0)
    cumulative_drift = _summary("tier1_cumulative_drift", 1.84, 54.5)

    first_comparison = compare_suite_summaries(first_drift, anchor)
    assert not first_comparison["regressions"]
    assert first_comparison["anchor_declines"]
    assert not _tier1_baseline_eligible(
        run_accepted=True,
        comparison=first_comparison,
        accept_comparison=False,
    )

    cumulative_comparison = compare_suite_summaries(cumulative_drift, anchor)
    assert cumulative_comparison["regressions"]


def test_unratified_improvement_outlier_does_not_replace_anchor() -> None:
    def _summary(run_id: str, speedup: float, optimized_ms: float) -> dict:
        return {
            "run_id": run_id,
            "targets": [
                {
                    "target": "ch01:demo",
                    "status": "succeeded",
                    "optimization_goal": "performance",
                    "best_speedup": speedup,
                    "best_optimized_time_ms": optimized_ms,
                }
            ],
        }

    anchor = _summary("tier1_anchor", 2.0, 50.0)
    outlier = _summary("tier1_outlier", 2.2, 45.45)
    normal = _summary("tier1_normal", 2.0, 50.0)

    outlier_comparison = compare_suite_summaries(outlier, anchor)
    assert outlier_comparison["improvements"]
    assert not _tier1_baseline_eligible(
        run_accepted=True,
        comparison=outlier_comparison,
        accept_comparison=False,
    )

    normal_comparison = compare_suite_summaries(normal, anchor)
    assert not normal_comparison["regressions"]


def test_build_tier1_suite_summary_raises_clear_error_for_malformed_results(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    result_json = tmp_path / "results.json"
    result_json.write_text("[]", encoding="utf-8")

    try:
        build_tier1_suite_summary(result_json, suite, run_id="tier1_bad")
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected malformed result JSON to raise ValueError")

    assert "tier-1 benchmark result JSON" in message
    assert str(result_json) not in message


def test_compare_suite_summaries_ignores_small_absolute_speedup_drift(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_result_payload(baseline_json, block_scaling_speedup=1.60, flash_speedup=12.0)
    _write_result_payload(current_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")

    comparison = compare_suite_summaries(current_summary, baseline_summary)

    assert not any(
        row["target"] == "labs/block_scaling:block_scaling" and row["reason"] == "speedup"
        for row in comparison["regressions"]
    )


def test_compare_suite_summaries_uses_memory_goal_for_kv_target(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_result_payload(
        baseline_json,
        block_scaling_speedup=1.60,
        flash_speedup=12.0,
        kv_speedup=1.58,
        kv_memory_savings_pct=49.7,
    )
    _write_result_payload(
        current_json,
        block_scaling_speedup=1.60,
        flash_speedup=12.0,
        kv_speedup=1.10,
        kv_memory_savings_pct=49.7,
    )

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")

    comparison = compare_suite_summaries(current_summary, baseline_summary)

    assert not any(
        row["target"] == "labs/kv_optimization:kv_standard" and row["reason"] == "speedup"
        for row in comparison["regressions"]
    )
    assert not any(
        row["target"] == "labs/kv_optimization:kv_standard" and row["reason"] == "speedup"
        for row in comparison["improvements"]
    )


def test_compare_suite_summaries_tracks_memory_regression_for_memory_goal_target(
    tmp_path: Path,
) -> None:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_result_payload(
        baseline_json,
        block_scaling_speedup=1.60,
        flash_speedup=12.0,
        kv_speedup=1.58,
        kv_memory_savings_pct=49.7,
    )
    _write_result_payload(
        current_json,
        block_scaling_speedup=1.60,
        flash_speedup=12.0,
        kv_speedup=1.10,
        kv_memory_savings_pct=42.0,
    )

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")

    comparison = compare_suite_summaries(current_summary, baseline_summary)

    assert any(
        row["target"] == "labs/kv_optimization:kv_standard" and row["reason"] == "memory_savings"
        for row in comparison["regressions"]
    )


def test_compare_suite_summaries_detects_common_mode_optimized_memory_regression() -> None:
    baseline = {
        "run_id": "tier1_memory_anchor",
        "targets": [
            {
                "target": "labs/kv_optimization:kv_standard",
                "status": "succeeded",
                "optimization_goal": "memory",
                "baseline_memory_mb": 1024.0,
                "best_memory_savings_pct": 50.0,
                "best_optimized_memory_mb": 512.0,
            }
        ],
    }
    current = {
        "run_id": "tier1_memory_slower",
        "targets": [
            {
                "target": "labs/kv_optimization:kv_standard",
                "status": "succeeded",
                "optimization_goal": "memory",
                "baseline_memory_mb": 2048.0,
                "best_memory_savings_pct": 50.0,
                "best_optimized_memory_mb": 1024.0,
            }
        ],
    }

    comparison = compare_suite_summaries(current, baseline)

    assert any(
        row["target"] == "labs/kv_optimization:kv_standard"
        and row["reason"] == "optimized_memory"
        for row in comparison["regressions"]
    )


def _run_llama_speedup_recheck(
    tmp_path: Path,
    monkeypatch,
    *,
    recheck_status: str | None,
    recheck_speedup: float = 2.55,
) -> dict[str, object]:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    recheck_json = tmp_path / "recheck.json"
    _write_result_payload(
        baseline_json,
        block_scaling_speedup=1.60,
        flash_speedup=12.0,
        llama_speedup=2.49,
    )
    _write_result_payload(
        current_json,
        block_scaling_speedup=1.45,
        flash_speedup=14.45,
        llama_speedup=2.20,
    )
    _write_result_payload(
        recheck_json,
        block_scaling_speedup=1.62,
        flash_speedup=14.45,
        llama_speedup=recheck_speedup,
    )
    recheck_payload = json.loads(recheck_json.read_text(encoding="utf-8"))
    llama_result = next(
        chapter
        for chapter in recheck_payload["results"]
        if chapter["chapter"] == "labs_real_world_models"
    )
    if recheck_status is None:
        llama_result["benchmarks"] = []
    else:
        llama_result["benchmarks"][0]["status"] = recheck_status
    recheck_json.write_text(json.dumps(recheck_payload, indent=2), encoding="utf-8")

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")
    comparison = compare_suite_summaries(current_summary, baseline_summary)

    recheck_result_path = tmp_path / "recheck_results.json"
    recheck_result_path.write_text(recheck_json.read_text(encoding="utf-8"), encoding="utf-8")
    recheck_manifest_path = tmp_path / "recheck_manifest.json"
    recheck_manifest_path.write_text("{}", encoding="utf-8")
    recheck_markdown_path = tmp_path / "recheck_report.md"
    recheck_markdown_path.write_text("# recheck\n", encoding="utf-8")

    def _fake_execute_benchmarks(**kwargs):
        assert kwargs["targets"] == ["labs/real_world_models:llama_3_1_8b"]
        return {
            "run_id": "tier1_new__recheck__llama_3_1_8b",
            "output_json": str(recheck_result_path),
            "manifest_path": str(recheck_manifest_path),
            "output_markdown": str(recheck_markdown_path),
        }

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _fake_execute_benchmarks)

    return _confirm_speedup_regressions(
        comparison=comparison,
        current_summary=current_summary,
        previous_summary=baseline_summary,
        suite=suite,
        suite_run_dir=tmp_path / "suite_run",
        bench_root=None,
        execution_run_id="tier1_new",
        profile_type="minimal",
        output_format="both",
        suite_timeout=14400,
        timeout_multiplier=3.0,
        validity_profile="strict",
        allow_portable_expectations_update=False,
        reproducible=False,
        cold_start=False,
        force_synchronize=False,
        iterations=None,
        warmup=None,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
        artifacts_dir=str(tmp_path / "artifacts"),
        log_level="INFO",
        log_file=None,
        single_gpu=False,
        accept_regressions=False,
        update_expectations=False,
        allow_mixed_provenance=False,
        ncu_metric_set="minimal",
        ncu_replay_mode="kernel",
        pm_sampling_interval=None,
        nsys_timeout_seconds=None,
        ncu_timeout_seconds=None,
        launch_via="python",
        nproc_per_node=None,
        nnodes=None,
        rdzv_backend=None,
        rdzv_endpoint=None,
        torchrun_env=None,
        target_extra_args=None,
        verify_input=True,
        verify_output=True,
        llm_analysis=False,
        force_llm=False,
        llm_provider=None,
        apply_llm_patches=False,
        rebenchmark_llm_patches=False,
        patch_strategy="ast",
        llm_patch_retries=2,
        use_llm_cache=True,
        llm_explain=False,
    )


def test_confirm_speedup_regressions_suppresses_unconfirmed_noise(
    tmp_path: Path,
    monkeypatch,
) -> None:
    updated = _run_llama_speedup_recheck(tmp_path, monkeypatch, recheck_status="succeeded")

    assert not any(
        row["target"] == "labs/real_world_models:llama_3_1_8b" and row["reason"] == "speedup"
        for row in updated["regressions"]
    )
    assert any(
        row["target"] == "labs/real_world_models:llama_3_1_8b"
        and row["suppression_reason"] == "recheck_not_regressed"
        for row in updated["suppressed_regressions"]
    )
    assert updated["rechecks"][0]["confirmed_regression"] is False
    assert updated["regression_rechecks_path"] == "suite_run/regression_rechecks.json"


def test_confirm_speedup_regressions_keeps_regression_when_recheck_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    updated = _run_llama_speedup_recheck(tmp_path, monkeypatch, recheck_status="failed_runtime")

    assert any(
        row["target"] == "labs/real_world_models:llama_3_1_8b" and row["reason"] == "speedup"
        for row in updated["regressions"]
    )
    assert not updated["suppressed_regressions"]
    assert updated["rechecks"][0]["recheck_summary"]["status"] == "failed_runtime"
    assert updated["rechecks"][0]["confirmed_regression"] is True


def test_confirm_speedup_regressions_keeps_regression_when_recheck_target_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    updated = _run_llama_speedup_recheck(tmp_path, monkeypatch, recheck_status=None)

    assert any(
        row["target"] == "labs/real_world_models:llama_3_1_8b" and row["reason"] == "speedup"
        for row in updated["regressions"]
    )
    assert not updated["suppressed_regressions"]
    assert updated["rechecks"][0]["recheck_summary"]["status"] == "missing"
    assert updated["rechecks"][0]["confirmed_regression"] is True


def test_confirm_speedup_regressions_keeps_regression_when_recheck_metrics_are_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    updated = _run_llama_speedup_recheck(
        tmp_path,
        monkeypatch,
        recheck_status="succeeded",
        recheck_speedup=0.0,
    )

    assert any(
        row["target"] == "labs/real_world_models:llama_3_1_8b" and row["reason"] == "speedup"
        for row in updated["regressions"]
    )
    assert not updated["suppressed_regressions"]
    assert updated["rechecks"][0]["recheck_summary"]["status"] == "succeeded"
    assert updated["rechecks"][0]["recheck_metrics_valid"] is False
    assert updated["rechecks"][0]["confirmed_regression"] is True


def test_recheck_suppression_does_not_advance_original_regressed_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison = _run_llama_speedup_recheck(
        tmp_path,
        monkeypatch,
        recheck_status="succeeded",
        recheck_speedup=2.49,
    )

    assert comparison["suppressed_regressions"]
    assert not comparison["regressions"]
    assert _tier1_baseline_eligible(
        run_accepted=True,
        comparison=comparison,
        accept_comparison=False,
    ) is False
    assert _tier1_baseline_eligible(
        run_accepted=True,
        comparison=comparison,
        accept_comparison=True,
    ) is False


def test_run_tier1_rejects_preexisting_suppressed_anchor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = load_tier1_suite()
    history_root = tmp_path / "history"
    prior_run_id = "tier1_suppressed_anchor"
    prior_run_dir = history_root / prior_run_id
    prior_run_dir.mkdir(parents=True)

    prior_result = tmp_path / "prior_results.json"
    current_result = tmp_path / "current_results.json"
    _write_result_payload(prior_result, block_scaling_speedup=1.60, flash_speedup=12.0)
    _write_result_payload(current_result, block_scaling_speedup=1.61, flash_speedup=12.1)
    prior_summary = build_tier1_suite_summary(prior_result, suite, run_id=prior_run_id)
    (prior_run_dir / "summary.json").write_text(
        json.dumps(prior_summary, indent=2),
        encoding="utf-8",
    )
    (prior_run_dir / "regression_summary.md").write_text("# suppressed\n", encoding="utf-8")
    (prior_run_dir / "regression_summary.json").write_text(
        json.dumps(
            {
                "baseline_run_id": "tier1_older_anchor",
                "anchor_declines": [],
                "missing_targets": [],
                "regressions": [],
                "suppressed_regressions": [
                    {
                        "target": "labs/real_world_models:llama_3_1_8b",
                        "reason": "speedup",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (history_root / "index.json").write_text(
        json.dumps(
            {
                "suite_name": suite.name,
                "suite_version": suite.version,
                "history_root": ".",
                "runs": [
                    {
                        "run_id": prior_run_id,
                        "generated_at": "2026-08-16T00:00:00Z",
                        "summary_path": f"{prior_run_id}/summary.json",
                        "regression_summary_path": (
                            f"{prior_run_id}/regression_summary.md"
                        ),
                        "regression_json_path": (
                            f"{prior_run_id}/regression_summary.json"
                        ),
                        "run_accepted": True,
                        "baseline_eligible": True,
                        "baseline_acceptance": "accept_history_anchor",
                        "baseline_acceptance_actor": "reviewer",
                        "baseline_acceptance_note": "legacy acceptance",
                        "baseline_acceptance_workflow_run": (
                            "https://github.com/example/repo/actions/runs/1"
                        ),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    source_commit = "a" * 40
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"git": {"commit": source_commit, "dirty": False}}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"
    report_path.write_text("# report\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_SHA", source_commit)

    def _fake_execute_benchmarks(**kwargs):
        return {
            "run_id": "tier1_current",
            "output_json": str(current_result),
            "manifest_path": str(manifest_path),
            "output_markdown": str(report_path),
            "total_failed": 0,
        }

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _fake_execute_benchmarks)

    result = run_tier1_suite(
        history_root=history_root,
        bench_root=tmp_path,
        profile_type="minimal",
        output_format="json",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="tier1_current",
    )

    assert result["comparison"]["baseline_run_id"] is None
    assert result["history_integrity_failed"] is True
    assert any("suppressed regressions" in warning for warning in result["warnings"])


def test_run_tier1_suite_surfaces_history_warnings(tmp_path: Path, monkeypatch) -> None:
    result_json = tmp_path / "results.json"
    _write_result_payload(result_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    history_root = tmp_path / "history"
    history_root.mkdir()
    bad_index = history_root / "index.json"
    bad_index.write_text("{not-json", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "report.md"
    report_path.write_text("# report\n", encoding="utf-8")

    benchmark_called = False

    def _fake_execute_benchmarks(**kwargs):
        nonlocal benchmark_called
        benchmark_called = True
        return {
            "run_id": "tier1_history_warning_demo",
            "output_json": str(result_json),
            "manifest_path": str(manifest_path),
            "output_markdown": str(report_path),
            "total_failed": 0,
        }

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _fake_execute_benchmarks)

    with pytest.raises(ValueError, match="JSONDecodeError"):
        run_tier1_suite(
            history_root=history_root,
            bench_root=tmp_path,
            profile_type="minimal",
            output_format="json",
            artifacts_dir=str(tmp_path / "artifacts"),
            run_id="tier1_history_warning_demo",
        )

    assert benchmark_called is False
    assert bad_index.read_text(encoding="utf-8") == "{not-json"


def test_tier1_doc_mentions_current_targets_and_artifacts() -> None:
    suite = load_tier1_suite()
    doc_path = Path("docs/tier1_benchmark_suite.md")
    text = doc_path.read_text(encoding="utf-8")

    assert "## Current Tier-1 Targets" in text
    assert "## Artifact Contract" in text
    assert "`artifacts/history/tier1/index.json`" in text
    assert "`artifacts/history/tier1/<run_id>/summary.json`" in text

    for target in suite.targets:
        assert f"`{target.target}`" in text


def test_execute_benchmarks_defaults_bench_root_to_repo_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", False)

    result = bench_commands._execute_benchmarks(
        targets=["ch04:gradient_fusion"],
        bench_root=None,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="tier1_default_root_smoke",
        exit_on_failure=False,
    )

    assert Path(result["bench_root"]) == Path(bench_commands.__file__).resolve().parents[2]
    assert result["run_id"] == "tier1_default_root_smoke"
    assert result["error"] == "Benchmark dependencies missing"


def test_execute_benchmarks_sets_owner_run_id_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", False)
    monkeypatch.delenv("AISP_BENCHMARK_OWNER_RUN_ID", raising=False)

    bench_commands._execute_benchmarks(
        targets=["ch04:gradient_fusion"],
        bench_root=None,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="tier1_owner_env_smoke",
        exit_on_failure=False,
    )

    assert os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] == "tier1_owner_env_smoke"


def test_execute_benchmarks_batch_preflight_is_structured_and_defers_external_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    synthetic_issue = "synthetic batch preflight failure"

    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", True)
    monkeypatch.setattr(bench_commands, "dump_environment_and_capabilities", lambda: None)
    monkeypatch.setattr(
        bench_commands,
        "resolve_target_chapters",
        lambda targets, bench_root: (
            [bench_root / "labs" / "trtllm_phi_3_5_moe", bench_root / "ch04"],
            {},
        ),
    )

    def _fake_preflight(*args, **kwargs):
        captured.update(kwargs)
        return [synthetic_issue]

    monkeypatch.setattr(
        bench_commands,
        "_preflight_target_coverage_and_assets",
        _fake_preflight,
    )

    result = bench_commands._execute_benchmarks(
        targets=[
            "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
            "ch04:gradient_fusion",
        ],
        bench_root=tmp_path,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="batch_preflight_smoke",
        exit_on_failure=False,
        enforce_external_assets=False,
    )

    assert captured["enforce_external_assets"] is False
    assert result["preflight_failed"] is True
    assert result["preflight_issues"] == [synthetic_issue]
    assert result["total_failed"] == 1
    assert result["error"] == f"Benchmark preflight failed: {synthetic_issue}"

    captured.clear()
    monkeypatch.setattr(
        bench_commands,
        "resolve_target_chapters",
        lambda targets, bench_root: (
            [bench_root / "labs" / "trtllm_phi_3_5_moe"],
            {},
        ),
    )
    direct_result = bench_commands._execute_benchmarks(
        targets=["labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe"],
        bench_root=tmp_path,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="direct_preflight_smoke",
        exit_on_failure=False,
        enforce_external_assets=True,
    )

    assert captured["enforce_external_assets"] is True
    assert direct_result["preflight_failed"] is True


def test_external_asset_preflight_is_deferred_for_mixed_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chapter_dir = (
        Path(run_benchmarks_module.__file__).resolve().parents[2] / "labs" / "trtllm_phi_3_5_moe"
    )
    filters = {"labs/trtllm_phi_3_5_moe": {"trtllm_phi_3_5_moe"}}
    missing_model = tmp_path / "missing-model"
    missing_engine = tmp_path / "missing-engine"

    monkeypatch.setattr(
        run_benchmarks_module,
        "_resolve_phi35_model_path",
        lambda override: missing_model,
    )
    monkeypatch.setattr(
        run_benchmarks_module,
        "_resolve_phi35_engine_path",
        lambda override: missing_engine,
    )

    direct_issues = run_benchmarks_module._preflight_target_coverage_and_assets(
        [chapter_dir],
        filters,
        only_cuda=False,
        only_python=False,
        target_extra_args={},
        enforce_external_assets=True,
    )
    batch_issues = run_benchmarks_module._preflight_target_coverage_and_assets(
        [chapter_dir],
        filters,
        only_cuda=False,
        only_python=False,
        target_extra_args={},
        enforce_external_assets=False,
    )

    assert any("missing model assets" in issue for issue in direct_issues)
    assert any("missing TensorRT-LLM engine artifacts" in issue for issue in direct_issues)
    assert batch_issues == []


def test_bench_run_defaults_bench_root_to_repo_root(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_benchmarks(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _fake_execute_benchmarks)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--targets",
            "labs/flashattention4:flashattention4_alibi",
            "--profile",
            "none",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            "bench_run_default_root_smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert Path(captured["bench_root"]) == Path(bench_commands.__file__).resolve().parents[2]
    assert captured["targets"] == ["labs/flashattention4:flashattention4_alibi"]
    assert captured["ncu_replay_mode"] is None


def test_bench_run_passes_explicit_ncu_replay_mode(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_benchmarks(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(bench_commands, "_execute_benchmarks", _fake_execute_benchmarks)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--targets",
            "labs/flashattention4:flashattention4_alibi",
            "--profile",
            "none",
            "--ncu-replay-mode",
            "application",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            "bench_run_explicit_replay_smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["ncu_replay_mode"] == "application"


def _complete_tier1_cli_result(tmp_path: Path, *, run_id: str = "tier1_smoke") -> dict:
    return {
        "run_accepted": True,
        "baseline_eligible": True,
        "execution": {"run_id": run_id, "total_failed": 0},
        "summary": {
            "summary": {
                "target_count": 1,
                "succeeded": 1,
                "failed": 0,
                "skipped": 0,
                "missing": 0,
            }
        },
        "comparison": {"regressions": [], "missing_targets": []},
        "summary_path": tmp_path / "summary.json",
        "regression_summary_path": tmp_path / "regressions.json",
        "trend_snapshot_path": tmp_path / "trend.json",
        "history_root": tmp_path / "history",
    }


def test_run_tier1_defaults_ncu_replay_mode_to_none(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_tier1_suite(**kwargs):
        captured.update(kwargs)
        return _complete_tier1_cli_result(tmp_path)

    monkeypatch.setattr("core.benchmark.suites.tier1.run_tier1_suite", _fake_run_tier1_suite)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run-tier1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["ncu_replay_mode"] is None


def test_run_tier1_passes_explicit_ncu_replay_mode(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_tier1_suite(**kwargs):
        captured.update(kwargs)
        return _complete_tier1_cli_result(tmp_path)

    monkeypatch.setattr("core.benchmark.suites.tier1.run_tier1_suite", _fake_run_tier1_suite)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run-tier1",
            "--ncu-replay-mode",
            "application",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["ncu_replay_mode"] == "application"


def test_run_tier1_passes_expectation_write_flags(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_tier1_suite(**kwargs):
        captured.update(kwargs)
        return _complete_tier1_cli_result(tmp_path)

    monkeypatch.setattr("core.benchmark.suites.tier1.run_tier1_suite", _fake_run_tier1_suite)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run-tier1",
            "--accept-regressions",
            "--update-expectations",
            "--allow-mixed-provenance",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["accept_regressions"] is True
    assert captured["update_expectations"] is True
    assert captured["allow_mixed_provenance"] is True


@pytest.mark.parametrize("option", ["--accept-history-anchor", "--acceptance-note"])
def test_run_tier1_cli_does_not_advertise_post_benchmark_promotion_options(
    option: str,
) -> None:
    args = ["bench", "run-tier1", option]
    if option == "--acceptance-note":
        args.append("reviewed")

    result = CliRunner().invoke(app, args)

    assert result.exit_code != 0
    assert "No such option" in result.stdout


def test_run_tier1_cli_exits_nonzero_when_suite_summary_reports_failures(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_run_tier1_suite(**kwargs):
        return {
            "execution": {"run_id": "tier1_failed_smoke", "total_failed": 0},
            "summary": {
                "summary": {
                    "target_count": 6,
                    "failed": 1,
                    "skipped": 0,
                    "succeeded": 5,
                    "missing": 0,
                }
            },
            "comparison": {"regressions": [], "missing_targets": []},
            "summary_path": tmp_path / "summary.json",
            "regression_summary_path": tmp_path / "regressions.json",
            "trend_snapshot_path": tmp_path / "trend.json",
            "history_root": tmp_path / "history",
        }

    monkeypatch.setattr("core.benchmark.suites.tier1.run_tier1_suite", _fake_run_tier1_suite)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bench",
            "run-tier1",
        ],
    )

    assert result.exit_code == 1, result.stdout
    assert '"run_id": "tier1_failed_smoke"' in result.stdout


def test_run_tier1_cli_exits_nonzero_when_baseline_metrics_are_ineligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ineligible_result = _complete_tier1_cli_result(
        tmp_path,
        run_id="tier1_invalid_metrics",
    )
    ineligible_result["baseline_eligible"] = False
    ineligible_result["run_accepted"] = False
    monkeypatch.setattr(
        "core.benchmark.suites.tier1.run_tier1_suite",
        lambda **kwargs: ineligible_result,
    )

    result = CliRunner().invoke(app, ["bench", "run-tier1"])

    assert result.exit_code == 1, result.stdout
    assert '"run_id": "tier1_invalid_metrics"' in result.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.pop("summary"),
        lambda result: result["summary"].pop("summary"),
        lambda result: result["summary"]["summary"].update(target_count=0, succeeded=0),
        lambda result: result["summary"]["summary"].update(skipped=1, succeeded=0),
        lambda result: result["summary"]["summary"].update(missing=1, succeeded=0),
        lambda result: result["summary"]["summary"].update(failed=-1),
        lambda result: result["summary"]["summary"].update(succeeded=0),
        lambda result: result["execution"].update(total_failed=1),
        lambda result: result.pop("comparison"),
        lambda result: result["comparison"].update(regressions=[{"target": "ch01:demo"}]),
        lambda result: result["comparison"].update(missing_targets=[{"target": "ch01:old"}]),
        lambda result: result.update(history_integrity_failed=True),
        lambda result: result.update(run_accepted=False, baseline_eligible=False),
    ],
)
def test_tier1_result_failure_count_fails_closed(
    tmp_path: Path,
    mutation,
) -> None:
    result = _complete_tier1_cli_result(tmp_path)
    mutation(result)

    assert bench_commands._tier1_result_failure_count(result) > 0


def test_tier1_result_failure_count_allows_explicitly_accepted_comparison(
    tmp_path: Path,
) -> None:
    result = _complete_tier1_cli_result(tmp_path)
    result["comparison"] = {
        "regressions": [{"target": "ch01:demo"}],
        "missing_targets": [{"target": "ch01:old"}],
    }

    assert (
        bench_commands._tier1_result_failure_count(
            result,
            allow_comparison_regressions=True,
        )
        == 0
    )


def test_tier1_result_failure_count_accepts_run_that_does_not_advance_baseline(
    tmp_path: Path,
) -> None:
    result = _complete_tier1_cli_result(tmp_path)
    result["baseline_eligible"] = False

    assert bench_commands._tier1_result_failure_count(result) == 0


@pytest.mark.parametrize(
    "comparison",
    [
        {},
        {"regressions": [], "missing_targets": None},
        {"regressions": None, "missing_targets": []},
    ],
)
def test_tier1_result_failure_count_rejects_malformed_accepted_comparison(
    tmp_path: Path,
    comparison: dict,
) -> None:
    result = _complete_tier1_cli_result(tmp_path)
    result["comparison"] = comparison

    assert (
        bench_commands._tier1_result_failure_count(
            result,
            allow_comparison_regressions=True,
        )
        > 0
    )
