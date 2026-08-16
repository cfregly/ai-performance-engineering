from __future__ import annotations

import contextlib
import getpass
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import mcp.mcp_server as mcp_server
from core.api import handlers
from core.benchmark import e2e_sweep


def test_discover_benchmark_e2e_inventory_buckets_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        e2e_sweep,
        "_iter_discovered_targets",
        lambda _root: [
            {"target": "ch01:demo", "chapter": "ch01", "example": "demo", "bench_type": "python", "multi_gpu": False},
            {"target": "ch02:dist", "chapter": "ch02", "example": "dist", "bench_type": "python", "multi_gpu": True},
            {"target": "labs/foo:cuda_demo", "chapter": "labs/foo", "example": "cuda_demo", "bench_type": "cuda", "multi_gpu": False},
        ],
    )

    payload = e2e_sweep.discover_benchmark_e2e_inventory()

    assert payload["counts"] == {"total": 3, "single_gpu": 2, "multi_gpu": 1}
    assert payload["single_gpu"] == ["ch01:demo", "labs/foo:cuda_demo"]
    assert payload["multi_gpu"] == ["ch02:dist"]


def test_run_benchmark_e2e_sweep_derives_stage_ids_and_skips_duplicate_fabric(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    cluster_run_dir = tmp_path / "cluster_run"
    manifest_path = cluster_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    class _Tier1SuiteDefinitionLike:
        pass

    cluster_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [_Tier1SuiteDefinitionLike()],
            "total_failed": 0,
            "total_skipped": 0,
        },
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _fake_cluster_eval(**kwargs):
        cluster_calls.append(kwargs)
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_001",
        run_full_sweep=False,
        cluster_preset="fabric-systems",
        run_fabric=True,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    stages = {stage["name"]: stage for stage in result["stages"]}
    assert stages["tier1"]["run_id"] == "e2e_001__tier1"
    assert stages["cluster"]["run_id"] == "e2e_001__cluster"
    assert stages["fabric"]["run_id"] == "e2e_001__fabric"
    assert stages["fabric"]["status"] == "skipped_duplicate"
    assert result["overall_status"] == "succeeded"
    assert Path(result["progress_path"]).exists()
    assert "--skip-render-localhost-report" in cluster_calls[0]["extra_args"]
    progress_payload = json.loads(Path(result["progress_path"]).read_text(encoding="utf-8"))
    assert progress_payload["run_id"] == "e2e_001"
    assert progress_payload["current"]["metrics"]["run_state"] == "completed"
    assert progress_payload["current"]["metrics"]["overall_status"] == "succeeded"
    json.dumps(result)
    assert Path(result["manifest_path"]).exists()


def test_run_benchmark_e2e_sweep_detects_terminal_tier1_failure_from_nested_execution_output(
    tmp_path: Path, monkeypatch
) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    output_json = tmp_path / "tier1_run" / "results" / "benchmark_test_results.json"
    for path in (summary_path, regression_summary_path, trend_snapshot_path, output_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "targets": [
                    {
                        "key": "block_scaling",
                        "target": "labs/block_scaling:block_scaling",
                        "status": "failed_no_speedup",
                        "best_speedup": 1.749,
                        "optimization_goal": "speed",
                    }
                ],
                "summary": {"failed": 1, "skipped": 0, "succeeded": 0},
            }
        ),
        encoding="utf-8",
    )
    regression_summary_path.write_text("# regressions\n", encoding="utf-8")
    trend_snapshot_path.write_text("{}", encoding="utf-8")
    output_json.write_text(
        json.dumps(
            {
                "timestamp": "2026-03-29T06:58:44Z",
                "results": [
                    {
                        "chapter": "labs_block_scaling",
                        "benchmarks": [
                            {
                                "example": "block_scaling",
                                "status": "failed_no_speedup",
                                "best_speedup": 1.749,
                                "optimization_goal": "speed",
                                "minimum_required_speedup": 1.75,
                                "error": "Best speedup 1.749x below required 1.75x threshold for speed-goal benchmark",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {
                "run_id": kwargs["run_id"],
                "output_json": str(output_json),
                "total_failed": 1,
                "total_skipped": 0,
            },
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_tier1_nested_failure",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    tier1_stage = next(stage for stage in result["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "failed"
    assert result["overall_status"] == "failed"
    assert tier1_stage["attempts"][-1]["status"] == "failed"
    assert tier1_stage["attempts"][-1]["benchmark_summary"]["failed_benchmarks"] == [
        {
            "target": "labs_block_scaling:block_scaling",
            "status": "failed_no_speedup",
            "error": "Best speedup 1.749x below required 1.75x threshold for speed-goal benchmark",
            "best_speedup": 1.749,
            "optimization_goal": "speed",
            "minimum_required_speedup": 1.75,
        }
    ]

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id="e2e_tier1_nested_failure", repo_root=tmp_path)
    aggregate_targets = {entry["target"] for entry in status["aggregate_failures"]}
    assert aggregate_targets == {"labs_block_scaling:block_scaling"}
    assert status["ledgers"]["summary"]["unresolved_count"] == 1


def test_run_benchmark_e2e_sweep_marks_partial_when_multi_gpu_bucket_is_skipped(tmp_path: Path, monkeypatch) -> None:
    output_json = tmp_path / "single" / "results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 1, "multi_gpu": 1},
            "single_gpu": ["ch01:demo"],
            "multi_gpu": ["ch02:dist"],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_execute_benchmarks",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "output_json": str(output_json),
            "total_failed": 0,
            "total_skipped": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_partial",
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "partial"
    assert "multi-GPU bucket skipped" in " ".join(full_stage.get("issues", []))
    assert full_stage["result"]["buckets"]["multi_gpu"]["status"] == "skipped"
    assert result["overall_status"] == "partial"


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_not_configured_scorecard(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "fabric"
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    scorecard_path = structured_dir / "e2e_fabric__fabric_fabric_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "families": {
                    "nvlink": {"completeness": "runtime_verified"},
                    "infiniband": {"completeness": "not_configured"},
                },
                "summary": {"configured_management_planes": 0},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert result["overall_status"] == "partial"
    assert fabric_stage["artifacts"]["fabric_scorecard"]["degraded_families"] == [
        {"family": "infiniband", "completeness": "not_configured"}
    ]


def test_run_benchmark_e2e_sweep_marks_fabric_partial_for_partial_scorecard_status(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "fabric_partial"
    structured_dir = run_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    scorecard_path = structured_dir / "e2e_fabric_partial__fabric_fabric_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "families": {
                    "nvlink": {"completeness": "runtime_verified"},
                    "spectrum-x": {"completeness": "not_present"},
                },
                "summary": {"runtime_verified_families": 1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_cluster_fabric_eval",
        lambda **kwargs: {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
        },
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_partial",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    fabric_stage = next(stage for stage in result["stages"] if stage["name"] == "fabric")
    assert fabric_stage["status"] == "partial"
    assert "fabric completeness is partial" in " ".join(fabric_stage.get("issues", []))
    assert result["overall_status"] == "partial"


def test_run_benchmark_e2e_sweep_mirrors_cluster_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_cluster_progress__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_cluster_eval(**kwargs):
        child_progress_path = cluster_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(run_id=kwargs["run_id"], progress_path=child_progress_path)
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=2,
                total_phases=10,
                step="vllm_serve_sweep",
                step_detail="completed 2/10 suite steps",
                percent_complete=20.0,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_cluster_progress", tmp_path) / "progress.json").read_text(encoding="utf-8")
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_cluster_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "cluster:vllm_serve_sweep"
    assert payload["current"]["step_detail"] == "completed 2/10 suite steps"
    assert payload["current"]["percent_complete"] == 20.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_cluster_progress__cluster"


def test_run_benchmark_e2e_sweep_heartbeats_summary_during_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    cluster_run_dir = tmp_path / "cluster" / "runs" / "e2e_cluster_heartbeat__cluster"
    manifest_path = cluster_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)
    monkeypatch.setattr(e2e_sweep, "_STATE_HEARTBEAT_SECONDS", 0.01)

    def _fake_cluster_eval(**kwargs):
        child_progress_path = cluster_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(run_id=kwargs["run_id"], progress_path=child_progress_path)
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=2,
                total_phases=10,
                step="vllm_serve_sweep",
                step_detail="completed 2/10 suite steps",
                percent_complete=20.0,
            )
        )
        run_dir = e2e_sweep.e2e_run_dir("e2e_cluster_heartbeat", tmp_path)
        summary_path = run_dir / "summary.json"
        before = summary_path.stat().st_mtime_ns
        time.sleep(0.05)
        after = summary_path.stat().st_mtime_ns
        observed["before"] = before
        observed["after"] = after
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(cluster_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-cluster"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_common_eval", _fake_cluster_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_cluster_heartbeat",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=True,
        run_fabric=False,
    )

    assert result["success"] is True
    assert observed["after"] > observed["before"]


def test_run_benchmark_e2e_sweep_mirrors_fabric_stage_progress(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}
    fabric_run_dir = tmp_path / "cluster" / "runs" / "e2e_fabric_progress__fabric"
    manifest_path = fabric_run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_STAGE_PROGRESS_POLL_SECONDS", 0.01)

    def _fake_fabric_eval(**kwargs):
        child_progress_path = fabric_run_dir / "progress" / "run_progress.json"
        recorder = e2e_sweep.ProgressRecorder(run_id=kwargs["run_id"], progress_path=child_progress_path)
        recorder.emit(
            e2e_sweep.ProgressEvent(
                phase="cluster_eval_suite",
                phase_index=5,
                total_phases=12,
                step="build_fabric_eval",
                step_detail="completed 4/12 suite steps",
                percent_complete=33.3333,
            )
        )
        time.sleep(0.05)
        observed["payload"] = json.loads(
            (e2e_sweep.e2e_run_dir("e2e_fabric_progress", tmp_path) / "progress.json").read_text(encoding="utf-8")
        )
        return {
            "success": True,
            "run_id": kwargs["run_id"],
            "run_dir": str(fabric_run_dir),
            "manifest_path": str(manifest_path),
            "returncode": 0,
            "command": ["fake-fabric"],
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_cluster_fabric_eval", _fake_fabric_eval)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fabric_progress",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=True,
    )

    assert result["overall_status"] == "succeeded"
    payload = observed["payload"]
    assert payload["current"]["step"] == "fabric:build_fabric_eval"
    assert payload["current"]["step_detail"] == "completed 4/12 suite steps"
    assert payload["current"]["percent_complete"] == 33.3333
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_fabric_progress__fabric"


def test_emit_live_progress_includes_child_stage_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    recorder = e2e_sweep.ProgressRecorder(run_id="e2e_progress", progress_path=progress_path)
    stages = [
        {
            "name": "tier1",
            "enabled": True,
            "run_id": "e2e_progress__tier1",
            "status": "running",
            "attempts": [{"run_id": "e2e_progress__tier1", "status": "running"}],
        },
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": "e2e_progress__full_sweep",
            "status": "planned",
            "attempts": [],
        },
    ]

    e2e_sweep._emit_live_progress(
        recorder,
        stages=stages,
        run_state="running",
        overall_status="running",
        artifact_paths={"summary_path": tmp_path / "summary.json"},
        child_progress={
            "step": "ch04:gradient_fusion",
            "step_detail": "optimized timing (optimized_gradient_fusion)",
            "percent_complete": 50.0,
        },
        child_stage_name="tier1",
        child_run_id="e2e_progress__tier1",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["current"]["step"] == "tier1:ch04:gradient_fusion"
    assert payload["current"]["step_detail"] == "optimized timing (optimized_gradient_fusion)"
    assert payload["current"]["percent_complete"] == 25.0
    assert payload["current"]["metrics"]["current_stage_run_id"] == "e2e_progress__tier1"
    assert payload["current"]["metrics"]["child_progress"]["percent_complete"] == 50.0


def test_run_benchmark_e2e_sweep_writes_progress_and_checkpoint_before_stage_invocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, dict[str, object]] = {}
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _fake_tier1(**kwargs):
        run_dir = e2e_sweep.e2e_run_dir("e2e_checkpoint", tmp_path)
        observed["progress"] = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        observed["summary"] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        observed["checkpoint"] = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        return {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [],
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _fake_tier1)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_checkpoint",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["overall_status"] == "succeeded"
    assert observed["progress"]["run_id"] == "e2e_checkpoint"
    assert observed["progress"]["current"]["metrics"]["run_state"] == "running"
    assert observed["progress"]["current"]["metrics"]["overall_status"] == "running"
    assert observed["summary"]["run_state"] == "running"
    assert observed["summary"]["overall_status"] == "running"
    assert observed["summary"]["current_stage"] == "tier1"
    assert observed["summary"]["current_stage_run_id"] == "e2e_checkpoint__tier1"
    assert observed["checkpoint"]["run_id"] == "e2e_checkpoint"
    assert observed["checkpoint"]["run_state"] == "running"
    assert observed["checkpoint"]["overall_status"] == "running"
    assert observed["checkpoint"]["current_stage"] == "tier1"
    assert observed["checkpoint"]["current_stage_run_id"] == "e2e_checkpoint__tier1"
    assert observed["checkpoint"]["stages"][0]["name"] == "tier1"
    assert observed["checkpoint"]["stages"][0]["status"] == "running"


def test_run_benchmark_e2e_sweep_persists_aborted_state_on_unhandled_exception(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tier1 exploded")),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "tier1 exploded" in result["error"]
    assert summary_payload["run_state"] == "aborted"
    assert summary_payload["overall_status"] == "aborted"
    assert summary_payload["resume_available"] is True
    assert checkpoint_payload["run_state"] == "aborted"
    assert checkpoint_payload["resume_available"] is True
    assert manifest_payload["checkpoint"]["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_persists_aborted_state_on_sighup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    if not hasattr(signal, "SIGHUP"):
        return

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _raise_hup(**_kwargs):
        signal.raise_signal(signal.SIGHUP)
        raise AssertionError("SIGHUP handler should have interrupted execution")

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _raise_hup)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_sighup_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_sighup_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "received SIGHUP" in result["error"]
    assert result["crash"]["signal"] == "SIGHUP"
    assert summary_payload["run_state"] == "aborted"
    assert checkpoint_payload["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_sigterm_abort_bypasses_inner_exception_handlers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def _raise_term_inside_inner_exception_handler(**_kwargs):
        try:
            signal.raise_signal(signal.SIGTERM)
        except Exception as exc:  # pragma: no cover - regression guard
            return {"swallowed": str(exc)}
        raise AssertionError("SIGTERM handler should have aborted execution")

    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", _raise_term_inside_inner_exception_handler)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_sigterm_abort",
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        auto_resume=False,
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_sigterm_abort", tmp_path)
    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    assert result["run_state"] == "aborted"
    assert result["overall_status"] == "aborted"
    assert result["resume_available"] is True
    assert "received SIGTERM" in result["error"]
    assert result["crash"]["signal"] == "SIGTERM"
    assert summary_payload["run_state"] == "aborted"
    assert checkpoint_payload["run_state"] == "aborted"
    tier1_stage = next(stage for stage in summary_payload["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "aborted"
    assert tier1_stage["attempts"][-1]["status"] == "aborted"


def test_run_benchmark_e2e_sweep_resume_requires_explicit_run_id() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "requires an explicit run_id" in result["error"]


def test_run_benchmark_e2e_sweep_resume_rejects_contract_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_mismatch", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "none",
                    "run_tier1": False,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "full_sweep", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_mismatch",
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        profile_type="minimal",
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert result["resume_available"] is True
    assert "Resume contract mismatch" in result["error"]
    assert "profile_type" in result["error"]


def test_run_benchmark_e2e_sweep_resume_allows_extending_suite_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_suite_timeout", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "run_state": "aborted",
                "overall_status": "aborted",
                "success": False,
                "resume_available": True,
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": False,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                    "suite_timeout": 14400,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "full_sweep", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "planned", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_suite_timeout",
        resume=True,
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        profile_type="minimal",
        suite_timeout=0,
    )

    assert result["success"] is True
    assert result["run_state"] == "completed"
    assert result["overall_status"] == "succeeded"
    assert result["contract"]["suite_timeout"] == 0


def test_run_benchmark_e2e_sweep_resume_marks_superseded_running_attempt_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary_path = tmp_path / "tier1" / "summary.json"
    regression_summary_path = tmp_path / "tier1" / "regression_summary.md"
    trend_snapshot_path = tmp_path / "tier1" / "trend.json"
    history_root = tmp_path / "history"
    for path in (summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    history_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: {
            "execution": {"run_id": kwargs["run_id"], "total_failed": 0},
            "summary_path": str(summary_path),
            "regression_summary_path": str(regression_summary_path),
            "trend_snapshot_path": str(trend_snapshot_path),
            "history_root": str(history_root),
            "suite_definitions": [],
            "total_failed": 0,
            "total_skipped": 0,
        },
    )

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_running", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": True,
                    "run_full_sweep": False,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_running__tier1",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_running__tier1",
                                "status": "running",
                            }
                        ],
                    },
                    {"name": "full_sweep", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__full_sweep", "attempts": []},
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_running__fabric", "attempts": []},
                ],
                "frozen_plan": {"full_sweep": {"single_gpu_targets": [], "multi_gpu_targets": []}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_running",
        resume=True,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
    )

    tier1_stage = next(stage for stage in result["stages"] if stage["name"] == "tier1")
    assert tier1_stage["status"] == "succeeded"
    assert len(tier1_stage["attempts"]) == 2
    assert tier1_stage["attempts"][0]["status"] == "aborted"
    assert "resume superseded unfinished attempt" in tier1_stage["attempts"][0]["issues"]
    assert tier1_stage["attempts"][1]["status"] == "succeeded"


def test_normalize_stale_running_resume_state_marks_dead_orchestrator_aborted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda _pid: False)

    resume_state = {
        "run_state": "running",
        "overall_status": "running",
        "success": False,
        "resume_available": False,
        "orchestrator_pid": 999999,
        "stages": [
            {
                "name": "tier1",
                "enabled": True,
                "status": "running",
                "run_id": "stale_e2e__tier1",
                "attempts": [
                    {
                        "run_id": "stale_e2e__tier1",
                        "status": "running",
                    }
                ],
            },
            {
                "name": "full_sweep",
                "enabled": False,
                "status": "skipped",
                "run_id": "stale_e2e__full_sweep",
                "attempts": [],
            },
        ],
    }

    reason = e2e_sweep._normalize_stale_running_resume_state(
        resume_state,
        repo_root=tmp_path,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert reason == "orchestrator process 999999 exited without finalizing run state"
    assert resume_state["run_state"] == "aborted"
    assert resume_state["overall_status"] == "aborted"
    assert resume_state["resume_available"] is True
    assert resume_state["stages"][0]["status"] == "aborted"
    assert resume_state["stages"][0]["attempts"][0]["status"] == "aborted"
    assert reason in resume_state["stages"][0]["attempts"][0]["issues"]


def test_run_benchmark_e2e_sweep_resume_resolves_current_targets_for_remaining_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": ["ch13:torchao_quantization", "ch14:cublas_vs_cutlass"],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(e2e_sweep, "_invoke_run_tier1_suite", lambda **kwargs: {})

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch14"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_unit_map", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": False,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__tier1", "attempts": []},
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume_unit_map__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_unit_map__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                                "units": ["ch13", "ch14"],
                                "completed_units": [],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_unit_map__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": ["ch13:torchao_quantization", "ch14:cutlass"],
                        "single_gpu_units": ["ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_unit_map",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert result["overall_status"] == "succeeded"
    assert len(execute_calls) == 1
    assert execute_calls[0]["targets"] == ["ch13:torchao_quantization", "ch14:cublas_vs_cutlass"]


def test_run_benchmark_e2e_sweep_resume_reruns_partial_unit_with_resume_attempt_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    tier1_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 4, "single_gpu": 4, "multi_gpu": 0},
            "single_gpu": [
                "ch12:done",
                "ch13:torchao_quantization",
                "ch13:training_speed",
                "ch14:after",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "_invoke_run_tier1_suite",
        lambda **kwargs: tier1_calls.append(kwargs) or {},
    )

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch13"}),
                    json.dumps({"event_type": "chapter_start", "chapter": "ch14"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "ch14"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "chapter": "ch13",
                            "benchmarks": [
                                {"example": "torchao_quantization", "status": "succeeded"},
                                {"example": "training_speed", "status": "succeeded"},
                            ],
                        },
                        {
                            "chapter": "ch14",
                            "benchmarks": [
                                {"example": "after", "status": "succeeded"},
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": True,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                    "accept_regressions": False,
                    "update_expectations": False,
                    "allow_mixed_provenance": False,
                    "allow_portable_expectations_update": False,
                    "timeout_multiplier": 3.0,
                    "suite_timeout": 14400,
                    "timeout_seconds": None,
                    "iterations": None,
                    "warmup": None,
                    "gpu_sm_clock_mhz": None,
                    "gpu_mem_clock_mhz": None,
                    "ncu_metric_set": "minimal",
                    "ncu_replay_mode": None,
                    "nsys_timeout_seconds": None,
                    "ncu_timeout_seconds": None,
                },
                "stages": [
                    {
                        "name": "tier1",
                        "enabled": True,
                        "status": "succeeded",
                        "run_id": "e2e_resume__tier1",
                        "attempts": [{"run_id": "e2e_resume__tier1", "status": "succeeded"}],
                    },
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "aborted",
                        "run_id": "e2e_resume__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "aborted",
                                "targets": [
                                    "ch12:done",
                                    "ch13:torchao_quantization",
                                    "ch13:training_speed",
                                    "ch14:after",
                                ],
                                "units": ["ch12", "ch13", "ch14"],
                                "completed_units": ["ch12"],
                                "active_unit": "ch13",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "planned", "run_id": "e2e_resume__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "planned", "run_id": "e2e_resume__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "ch12:done",
                            "ch13:torchao_quantization",
                            "ch13:training_speed",
                            "ch14:after",
                        ],
                        "single_gpu_units": ["ch12", "ch13", "ch14"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume",
        resume=True,
        run_tier1=True,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert tier1_calls == []
    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == [
        "ch13:torchao_quantization",
        "ch13:training_speed",
        "ch14:after",
    ]

    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["status"] == "succeeded"
    assert len(full_stage["attempts"]) == 2
    assert full_stage["attempts"][0]["run_id"] == "e2e_resume__full_sweep__single"
    assert full_stage["attempts"][1]["run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert full_stage["attempts"][1]["completed_units"] == ["ch13", "ch14"]
    assert full_stage["result"]["buckets"]["single_gpu"]["latest_attempt_run_id"] == "e2e_resume__full_sweep__single__resume1"
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_resume_canonicalizes_lab_unit_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execute_calls: list[dict[str, object]] = []
    runs_root = tmp_path / "bench_runs"

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 2, "single_gpu": 2, "multi_gpu": 0},
            "single_gpu": [
                "labs/async_input_pipeline:async_input_pipeline",
                "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
            ],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)

    def _fake_execute(**kwargs):
        execute_calls.append(kwargs)
        paths = e2e_sweep._benchmark_run_event_paths(
            kwargs["run_id"],
            repo_root=tmp_path,
            artifacts_dir=str(runs_root),
        )
        paths["events"].parent.mkdir(parents=True, exist_ok=True)
        paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["progress"].parent.mkdir(parents=True, exist_ok=True)
        paths["events"].write_text(
            "\n".join(
                [
                    json.dumps({"event_type": "run_start", "targets": kwargs["targets"]}),
                    json.dumps({"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}),
                    json.dumps({"event_type": "chapter_end", "chapter": "labs_trtllm_phi_3_5_moe"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        paths["output_json"].write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "chapter": "labs_trtllm_phi_3_5_moe",
                            "benchmarks": [
                                {"example": "trtllm_phi_3_5_moe", "status": "succeeded"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        paths["progress"].write_text("{}", encoding="utf-8")
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(paths["output_json"]),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    prior_paths = e2e_sweep._benchmark_run_event_paths(
        "e2e_resume_labs__full_sweep__single",
        repo_root=tmp_path,
        artifacts_dir=str(runs_root),
    )
    prior_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    prior_paths["events"].write_text(
        "\n".join(
            [
                json.dumps({"event_type": "run_start", "targets": [
                    "labs/async_input_pipeline:async_input_pipeline",
                    "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                ]}),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_async_input_pipeline"}),
                json.dumps({"event_type": "chapter_end", "chapter": "labs_async_input_pipeline"}),
                json.dumps({"event_type": "chapter_start", "chapter": "labs_trtllm_phi_3_5_moe"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prior_paths["output_json"].write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs_async_input_pipeline",
                        "benchmarks": [
                            {"example": "async_input_pipeline", "status": "succeeded"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior_paths["progress"].write_text("{}", encoding="utf-8")

    run_dir = e2e_sweep.e2e_run_dir("e2e_resume_labs", tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T00:00:00Z",
                "run_state": "running",
                "orchestrator_pid": 999999,
                "contract": {
                    "profile_type": "minimal",
                    "run_tier1": False,
                    "run_full_sweep": True,
                    "run_cluster": False,
                    "run_fabric": False,
                    "cluster_preset": "common-answer-fast",
                    "hosts": ["localhost"],
                    "labels": ["localhost"],
                    "ssh_user": getpass.getuser(),
                    "ssh_key": None,
                    "bench_root": str(tmp_path),
                    "validity_profile": "strict",
                    "single_gpu": False,
                },
                "stages": [
                    {"name": "tier1", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__tier1", "attempts": []},
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "status": "running",
                        "run_id": "e2e_resume_labs__full_sweep",
                        "attempts": [
                            {
                                "run_id": "e2e_resume_labs__full_sweep__single",
                                "bucket": "single_gpu",
                                "status": "running",
                                "targets": [
                                    "labs/async_input_pipeline:async_input_pipeline",
                                    "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                                ],
                                "units": ["labs/async_input_pipeline", "labs/trtllm_phi_3_5_moe"],
                                "completed_units": [],
                                "active_unit": "labs/async_input_pipeline",
                            }
                        ],
                    },
                    {"name": "cluster", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__cluster", "attempts": []},
                    {"name": "fabric", "enabled": False, "status": "skipped", "run_id": "e2e_resume_labs__fabric", "attempts": []},
                ],
                "frozen_plan": {
                    "full_sweep": {
                        "single_gpu_targets": [
                            "labs/async_input_pipeline:async_input_pipeline",
                            "labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe",
                        ],
                        "single_gpu_units": ["labs/async_input_pipeline", "labs/trtllm_phi_3_5_moe"],
                        "multi_gpu_targets": [],
                        "multi_gpu_units": [],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_resume_labs",
        resume=True,
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        artifacts_dir=str(runs_root),
    )

    assert len(execute_calls) == 1
    assert execute_calls[0]["run_id"] == "e2e_resume_labs__full_sweep__single__resume1"
    assert execute_calls[0]["targets"] == ["labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe"]
    assert execute_calls[0]["enforce_external_assets"] is False
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert full_stage["attempts"][0]["status"] == "aborted"
    assert full_stage["attempts"][0]["completed_units"] == ["labs/async_input_pipeline"]
    assert full_stage["attempts"][0]["active_unit"] == "labs/trtllm_phi_3_5_moe"
    assert full_stage["attempts"][1]["completed_units"] == ["labs/trtllm_phi_3_5_moe"]
    assert result["overall_status"] == "succeeded"


def test_run_benchmark_e2e_sweep_rejects_portable_expectation_writes() -> None:
    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_invalid",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        validity_profile="portable",
        update_expectations=True,
        dry_run=True,
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "allow-portable-expectations-update" in result["error"]


def test_run_benchmark_e2e_sweep_rejects_non_local_hosts_without_ssh_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {"counts": {"total": 0, "single_gpu": 0, "multi_gpu": 0}, "single_gpu": [], "multi_gpu": [], "targets": []},
    )
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_bad_hosts",
        run_tier1=False,
        run_full_sweep=False,
        run_cluster=False,
        run_fabric=False,
        hosts=["gpu-node-1"],
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["success"] is False
    assert result["overall_status"] == "failed"
    assert "Non-local hosts require explicit ssh_user and ssh_key" in result["error"]


def test_benchmark_e2e_sweep_handler_returns_async_ticket(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.run_benchmark_e2e_sweep",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"]},
    )

    ticket = handlers.benchmark_e2e_sweep(
        {
            "async": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
            "run_id": "handler_async_e2e",
        }
    )

    assert ticket["status"] == "queued"
    assert ticket["run_id"] == "handler_async_e2e"
    assert ticket["run_dir"].endswith("artifacts/e2e_runs/handler_async_e2e")
    assert ticket["progress_path"].endswith("artifacts/e2e_runs/handler_async_e2e/progress.json")
    assert ticket["actions"]["preferred_mcp_tool"] == "benchmark_e2e_status"
    assert ticket["preferred_progress_source"]["kind"] == "normalized_e2e_status"
    assert ticket["actions"]["status_api_path"].endswith("/api/benchmark/e2e-status?run_id=handler_async_e2e")

    record = handlers.JobStore.get().get_status(ticket["job_id"])
    if record is not None:
        handlers.JobStore.get().update_job(ticket["job_id"], status="completed")


def test_tool_benchmark_e2e_sweep_delegates_to_handler(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_sweep",
        lambda params: {"success": True, "run_id": "tool_e2e", "echo": params.get("dry_run", False)},
    )

    payload = mcp_server.tool_benchmark_e2e_sweep({"dry_run": True})

    assert payload["success"] is True
    assert payload["run_id"] == "tool_e2e"
    assert payload["echo"] is True


def test_benchmark_e2e_sweep_handler_passes_resume_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {"success": True, "run_id": kwargs["run_id"], "resume": kwargs["resume"]}

    monkeypatch.setattr("core.benchmark.e2e_sweep.run_benchmark_e2e_sweep", _fake_run)

    payload = handlers.benchmark_e2e_sweep(
        {
            "run_id": "handler_resume_e2e",
            "resume": True,
            "run_tier1": False,
            "run_full_sweep": False,
            "run_cluster": False,
            "run_fabric": False,
        }
    )

    assert payload["success"] is True
    assert payload["resume"] is True
    assert captured["resume"] is True


def test_run_benchmark_e2e_sweep_uses_full_sweep_suite_timeout_for_bucket_runs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    output_json = tmp_path / "artifacts" / "runs" / "e2e_fs_timeout__full_sweep__single" / "results" / "benchmark_test_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": []}), encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        e2e_sweep,
        "discover_benchmark_e2e_inventory",
        lambda _root=None: {
            "counts": {"total": 1, "single_gpu": 1, "multi_gpu": 0},
            "single_gpu": ["ch10:atomic_reduction"],
            "multi_gpu": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(e2e_sweep, "detect_expectation_key", lambda: "test_gpu")
    monkeypatch.setattr(
        e2e_sweep,
        "detect_execution_environment",
        lambda: SimpleNamespace(kind="bare_metal", virtualized=False, dmi_product_name="test-box"),
    )
    monkeypatch.setattr(e2e_sweep, "_benchmark_queue_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(e2e_sweep, "_visible_gpu_count", lambda **kwargs: 1)
    monkeypatch.setattr(
        e2e_sweep,
        "watch_benchmark_e2e_sweep_run",
        lambda **kwargs: {"success": True, "watcher_pid": 777, "watch_status_path": str(tmp_path / "watcher.json")},
    )

    def _fake_execute(**kwargs):
        captured.update(kwargs)
        return {
            "run_id": kwargs["run_id"],
            "output_json": str(output_json),
            "total_failed": 0,
            "total_skipped": 0,
        }

    monkeypatch.setattr(e2e_sweep, "_invoke_execute_benchmarks", _fake_execute)

    result = e2e_sweep.run_benchmark_e2e_sweep(
        run_id="e2e_fs_timeout",
        run_tier1=False,
        run_full_sweep=True,
        run_cluster=False,
        run_fabric=False,
        suite_timeout=14400,
        full_sweep_suite_timeout=0,
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    assert result["success"] is True
    assert captured["suite_timeout"] == 0
    full_stage = next(stage for stage in result["stages"] if stage["name"] == "full_sweep")
    assert "--suite-timeout" in full_stage["attempts"][0]["command"]
    assert "0" in full_stage["attempts"][0]["command"]


def test_inspect_benchmark_e2e_sweep_run_detects_live_progress_and_child_events(tmp_path: Path, monkeypatch) -> None:
    run_id = "e2e_live_status"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        f"{run_id}__full_sweep__single",
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps({"event_type": "example_end", "chapter": "ch10", "example": "cluster_multicast", "status": "succeeded"})
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")
    watcher_status_path = e2e_sweep.e2e_watcher_status_path(run_dir)
    watcher_status_path.write_text(json.dumps({"watcher_pid": 456, "watch_state": "watching", "auto_resume_count": 0}), encoding="utf-8")

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch10",
                    "completed_units": ["ch06"],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:10:00+00:00",
            "step": "full_sweep/single_gpu:ch10:cluster_multicast",
            "step_detail": "ncu profiling (optimized)",
            "percent_complete": 42.0,
            "elapsed_seconds": 100.0,
            "eta_seconds": 200.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": f"{run_id}__full_sweep__single",
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(json.dumps({"ts": "2026-03-27T16:10:00Z", "event": "stage_started", "stage": "full_sweep"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) in {123, 456})

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["success"] is True
    assert status["inferred_state"] == "running_live"
    assert status["progress_source"]["kind"] == "live_child_progress"
    assert status["progress_source"]["preferred_mcp_tool"] == "benchmark_e2e_status"
    assert status["current"]["child_artifacts"]["events_path"].endswith("benchmark_events.jsonl")
    assert status["current"]["recent_child_events"][0]["event_type"] == "example_end"
    assert status["watcher"]["watcher_pid"] == 456
    assert status["actions"]["status_api_path"].endswith(f"/api/benchmark/e2e-status?run_id={run_id}")
    assert status["actions"]["dashboard_path"].endswith(f"/e2e?run_id={run_id}")


def test_inspect_benchmark_e2e_sweep_run_downgrades_stale_active_watcher_state(tmp_path: Path, monkeypatch) -> None:
    run_id = "e2e_dead_watcher_state"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "run_state": "aborted",
        "overall_status": "aborted",
        "updated_at": "2026-03-29T15:48:37Z",
        "resume_available": True,
        "contract": {"auto_resume": True},
        "stages": [],
        "orchestrator_pid": 999,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-29T15:48:37+00:00",
                    "metrics": {
                        "run_state": "aborted",
                        "overall_status": "aborted",
                        "orchestrator_pid": 999,
                        "stages": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    e2e_sweep.e2e_watcher_status_path(run_dir).write_text(
        json.dumps({"watcher_pid": 456, "watch_state": "resuming", "auto_resume_count": 1}),
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)

    assert status["watcher"]["stored_watch_state"] == "resuming"
    assert status["watcher"]["watch_state"] == "stale_dead"
    assert status["watcher"]["watcher_live"] is False
    assert (
        "stored watcher watch_state `resuming` corrected to `stale_dead` because the watcher pid is not live"
        in status["notes"]
    )
    assert "watcher_state=stale_dead" in rendered
    assert "stored_watcher_state=resuming" in rendered


def test_inspect_benchmark_e2e_sweep_run_syncs_active_issue_ledger_from_live_failures(tmp_path: Path, monkeypatch) -> None:
    run_id = "e2e_live_issue_sync"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_run_id = f"{run_id}__full_sweep__single"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps(
            {
                "event_type": "example_end",
                "timestamp": "2026-03-27T16:20:34.180782",
                "run_id": child_run_id,
                "chapter": "ch13",
                "example": "memory_profiling",
                "status": "failed_no_speedup",
                "best_speedup": 1.0,
                "best_memory_savings_pct": 10.1673,
                "optimization_goal": "speed",
                "error": "Best speedup 1.00x below required 1.05x threshold for speed-goal benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")

    existing_ledger = {
        "summary": {"run_id": run_id, "issue_count": 1, "resolved_count": 1, "unresolved_count": 0},
        "rows": [
            {
                "issue_id": "tier1_goalaware_regression_summary",
                "stage": "tier1",
                "status": "resolved",
                "symptom": "Existing manual issue",
                "root_cause": "Fixed already.",
                "fixes": [],
                "verification": {"pytest": "pytest -q tests/test_tier1_suite.py"},
                "evidence_paths": {},
            }
        ],
    }
    (run_dir / "active_issue_ledger.json").write_text(json.dumps(existing_ledger), encoding="utf-8")
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch13",
                    "completed_units": ["ch06"],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:21:00+00:00",
            "step": "full_sweep/single_gpu:ch13:memory_profiling",
            "step_detail": "timing (optimized)",
            "percent_complete": 48.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["current"]["reported_failures"][0]["target"] == "ch13:memory_profiling"
    assert any(entry["target"] == "ch13:memory_profiling" for entry in status["aggregate_failures"])
    assert status["ledgers"]["active_issue_ledger_json"].endswith("active_issue_ledger.json")

    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    assert synced_ledger["summary"]["run_id"] == run_id
    assert synced_ledger["summary"]["issue_count"] == 2
    assert synced_ledger["summary"]["resolved_count"] == 1
    assert synced_ledger["summary"]["unresolved_count"] == 1
    assert synced_ledger["summary"]["reported_issue_count"] == 1
    assert synced_ledger["summary"]["issue_group_count"] == 1
    issue_ids = [row["issue_id"] for row in synced_ledger["rows"]]
    assert "tier1_goalaware_regression_summary" in issue_ids
    assert "reported_full_sweep_single_gpu_ch13_memory_profiling" in issue_ids
    assert synced_ledger["summary"]["issue_group_count"] == 1
    assert synced_ledger["issue_groups"][0]["signature"] == "Best speedup 1.00x below required 1.05x threshold for speed-goal benchmark"
    markdown = (run_dir / "active_issue_ledger.md").read_text(encoding="utf-8")
    assert "ch13:memory_profiling" in markdown


def test_inspect_benchmark_e2e_sweep_run_preserves_local_no_speedup_threshold_from_child_events(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_local_threshold_issue_sync"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    child_run_id = f"{run_id}__full_sweep__single"
    child_paths = e2e_sweep._benchmark_run_event_paths(
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
    )
    child_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["events"].write_text(
        json.dumps(
            {
                "event_type": "example_end",
                "timestamp": "2026-03-27T16:20:34.180782",
                "run_id": child_run_id,
                "chapter": "ch06",
                "example": "launch_bounds",
                "status": "failed_no_speedup",
                "best_speedup": 1.006,
                "best_memory_savings_pct": 0.0,
                "optimization_goal": "speed",
                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child_paths["output_json"].parent.mkdir(parents=True, exist_ok=True)
    child_paths["output_json"].write_text(json.dumps({"results": []}), encoding="utf-8")

    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch06",
                    "completed_units": [],
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": True,
        "stages": stage_payload,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-27T16:21:00+00:00",
            "step": "full_sweep/single_gpu:ch06:launch_bounds",
            "step_detail": "timing (optimized)",
            "percent_complete": 48.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(
        run_id=run_id,
        repo_root=tmp_path,
        artifacts_dir=str(artifacts_dir),
        recent_events_limit=5,
    )

    assert status["current"]["reported_failures"][0]["target"] == "ch06:launch_bounds"
    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["issue_groups"][0]["signature"] == (
        "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark"
    )
    row = next(item for item in synced_ledger["rows"] if item["issue_id"] == "reported_full_sweep_single_gpu_ch06_launch_bounds")
    assert row["symptom"].endswith(
        "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark"
    )


def test_inspect_benchmark_e2e_sweep_run_detects_stale_running_and_builds_resume_command(tmp_path: Path, monkeypatch) -> None:
    run_id = "e2e_stale_status"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "run_tier1": True,
        "run_full_sweep": True,
        "run_cluster": False,
        "run_fabric": False,
        "validity_profile": "portable",
        "suite_timeout": 14400,
        "full_sweep_suite_timeout": 0,
        "auto_resume": True,
        "max_auto_resumes": 3,
        "watch_poll_interval_seconds": 15,
    }
    payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-27T16:00:00Z",
        "resume_available": False,
        "contract": contract,
        "stages": [],
        "orchestrator_pid": 999,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-27T16:10:00+00:00",
                    "metrics": {
                        "run_state": "running",
                        "overall_status": "running",
                        "orchestrator_pid": 999,
                        "stages": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["inferred_state"] == "running_stale"
    assert "--resume" in status["actions"]["resume_command"]
    assert "--full-sweep-suite-timeout" in status["actions"]["resume_command"]
    assert status["resume_available"] is True
    assert status["stored_resume_available"] is False
    assert "stored resume_available was false; corrected to true for stale running package" in status["notes"]


def test_inspect_benchmark_e2e_sweep_run_preserves_terminal_full_sweep_failures_from_all_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_full_sweep_failures"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T04:55:27Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": [
                    {
                        "run_id": f"{run_id}__full_sweep__single",
                        "bucket": "single_gpu",
                        "status": "aborted",
                        "artifacts": {
                            "events_path": str(tmp_path / "events_single.jsonl"),
                            "output_json": str(tmp_path / "single_results.json"),
                            "progress_path": str(tmp_path / "single_progress.json"),
                            "run_dir": str(tmp_path / "single_run"),
                        },
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch06:launch_bounds",
                                    "status": "failed_no_speedup",
                                    "error": "Best speedup 1.01x below required 1.05x threshold",
                                }
                            ],
                            "skipped_benchmarks": [],
                        },
                    },
                    {
                        "run_id": f"{run_id}__full_sweep__single__resume1",
                        "bucket": "single_gpu",
                        "status": "failed",
                        "artifacts": {
                            "events_path": str(tmp_path / "events_resume.jsonl"),
                            "output_json": str(tmp_path / "resume_results.json"),
                            "progress_path": str(tmp_path / "resume_progress.json"),
                            "run_dir": str(tmp_path / "resume_run"),
                        },
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch13:regional_compile",
                                    "status": "failed_no_speedup",
                                    "error": "Best speedup 1.05x below required 1.05x threshold",
                                }
                            ],
                            "skipped_benchmarks": [],
                        },
                    },
                    {
                        "run_id": f"{run_id}__full_sweep__multi",
                        "bucket": "multi_gpu",
                        "status": "skipped",
                    },
                ],
            }
        ],
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 1204046
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-28T04:55:27+00:00",
            "metrics": {
                "run_state": "completed",
                "overall_status": "failed",
                "orchestrator_pid": 1204046,
                "stages": [
                    {
                        "name": "full_sweep",
                        "enabled": True,
                        "run_id": f"{run_id}__full_sweep",
                        "status": "failed",
                        "attempts": [
                            {
                                "run_id": f"{run_id}__full_sweep__multi",
                                "bucket": "multi_gpu",
                                "status": "skipped",
                            }
                        ],
                    }
                ],
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    existing_ledger = {
        "summary": {"run_id": run_id, "issue_count": 1, "resolved_count": 1, "unresolved_count": 0},
        "rows": [
            {
                "issue_id": "tier1_goalaware_regression_summary",
                "stage": "tier1",
                "status": "resolved",
                "symptom": "Existing manual issue",
                "root_cause": "Fixed already.",
                "fixes": [],
                "verification": {"pytest": "pytest -q tests/test_tier1_suite.py"},
                "evidence_paths": {},
            }
        ],
    }
    (run_dir / "active_issue_ledger.json").write_text(json.dumps(existing_ledger), encoding="utf-8")
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    aggregate_targets = {entry["target"] for entry in status["aggregate_failures"]}
    assert "ch06:launch_bounds" in aggregate_targets
    assert "ch13:regional_compile" in aggregate_targets
    surfaced_targets = {entry["target"] for entry in status["current"]["reported_failures"]}
    assert surfaced_targets == aggregate_targets

    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    issue_ids = {row["issue_id"] for row in synced_ledger["rows"]}
    assert "reported_full_sweep_single_gpu_ch06_launch_bounds" in issue_ids
    assert "reported_full_sweep_single_gpu_ch13_regional_compile" in issue_ids
    assert synced_ledger["summary"]["issue_count"] == 3
    assert status["ledgers"]["summary"]["issue_count"] == 3
    assert status["ledgers"]["summary"]["resolved_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 2
    assert status["ledgers"]["summary"]["active_issue_count"] == 2
    assert status["ledgers"]["summary"]["historical_issue_count"] == 0
    assert status["issue_groups"]

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "reported_failures=2" in rendered
    assert "active_issue_counts=issues=2 unresolved=2" in rendered
    assert "ledger_totals=issues=3 resolved=1 unresolved=2" in rendered
    assert "issue_groups=" in rendered
    assert "ch06:launch_bounds[failed_no_speedup]" in rendered
    assert "ch13:regional_compile[failed_no_speedup]" in rendered


def test_inspect_benchmark_e2e_sweep_run_recovers_tier1_failures_from_nested_suite_result(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_tier1_nested_failure"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_json = tmp_path / "tier1_run" / "results" / "benchmark_test_results.json"
    summary_path = tmp_path / "history" / run_id / "summary.json"
    regression_summary_path = tmp_path / "history" / run_id / "regression_summary.md"
    trend_snapshot_path = tmp_path / "history" / run_id / "trend.json"
    for path in (output_json, summary_path, regression_summary_path, trend_snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs_block_scaling",
                        "benchmarks": [
                            {
                                "example": "block_scaling",
                                "status": "failed_no_speedup",
                                "best_speedup": 1.7492851550300643,
                                "best_memory_savings_pct": 0.0,
                                "optimization_goal": "speed",
                                "minimum_required_speedup": 1.75,
                                "error": "Best speedup 1.75x below required 1.75x threshold for speed-goal benchmark",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "targets": [
                    {
                        "target": "labs/block_scaling:block_scaling",
                        "status": "failed_no_speedup",
                        "best_speedup": 1.7492851550300643,
                        "best_memory_savings_pct": 0.0,
                        "optimization_goal": "speed",
                    }
                ],
                "summary": {"failed": 1, "skipped": 0, "succeeded": 5},
            }
        ),
        encoding="utf-8",
    )
    regression_summary_path.write_text("# regressions\n", encoding="utf-8")
    trend_snapshot_path.write_text("{}", encoding="utf-8")

    stage_payload = [
        {
            "name": "tier1",
            "enabled": True,
            "run_id": f"{run_id}__tier1",
            "status": "succeeded",
            "attempts": [
                {
                    "run_id": f"{run_id}__tier1",
                    "status": "succeeded",
                    "result": {
                        "execution": {
                            "run_id": f"{run_id}__tier1",
                            "output_json": str(output_json),
                            "total_failed": 1,
                            "total_skipped": 0,
                        },
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(regression_summary_path),
                        "trend_snapshot_path": str(trend_snapshot_path),
                    },
                    "artifacts": {
                        "summary_path": str(summary_path),
                        "regression_summary_path": str(regression_summary_path),
                        "trend_snapshot_path": str(trend_snapshot_path),
                    },
                }
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "succeeded",
        "updated_at": "2026-03-29T06:58:44Z",
        "resume_available": False,
        "stages": stage_payload,
        "contract": {"run_tier1": True},
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 777
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T06:58:44+00:00",
            "metrics": {
                "run_state": "completed",
                "overall_status": "succeeded",
                "orchestrator_pid": 777,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert {entry["target"] for entry in status["aggregate_failures"]} == {"labs_block_scaling:block_scaling"}
    assert {entry["target"] for entry in status["current"]["reported_failures"]} == {"labs_block_scaling:block_scaling"}
    assert status["stages"][0]["status"] == "failed"
    assert status["stages"][0]["stored_status"] == "succeeded"
    assert status["overall_status"] == "failed"
    assert status["stored_overall_status"] == "succeeded"
    assert status["ledgers"]["summary"]["unresolved_count"] == 1


def test_inspect_benchmark_e2e_sweep_run_preserves_resolved_reported_issue_rows(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_completed_preserve_resolved_reported"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_attempts = [
        {
            "run_id": f"{run_id}__full_sweep__single",
            "bucket": "single_gpu",
            "status": "failed",
            "artifacts": {
                "events_path": str(tmp_path / "events_single.jsonl"),
                "output_json": str(tmp_path / "single_results.json"),
            },
            "benchmark_summary": {
                "failed_benchmarks": [
                    {
                        "target": "ch13:regional_compile",
                        "status": "failed_no_speedup",
                        "error": "Best speedup 1.05x below required 1.05x threshold",
                    }
                ],
                "skipped_benchmarks": [],
            },
        },
        {
            "run_id": f"{run_id}__full_sweep__multi",
            "bucket": "multi_gpu",
            "status": "skipped",
        },
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T05:30:00Z",
        "resume_available": False,
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": stage_attempts,
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-28T05:30:00+00:00",
                    "metrics": {
                        "run_state": "completed",
                        "overall_status": "failed",
                        "stages": [
                            {
                                "name": "full_sweep",
                                "enabled": True,
                                "run_id": f"{run_id}__full_sweep",
                                "status": "failed",
                                "attempts": [{"run_id": f"{run_id}__full_sweep__multi", "bucket": "multi_gpu", "status": "skipped"}],
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "active_issue_ledger.json").write_text(
        json.dumps(
            {
                "summary": {"run_id": run_id, "issue_count": 1, "resolved_count": 1, "unresolved_count": 0},
                "rows": [
                    {
                        "issue_id": "reported_full_sweep_single_gpu_ch13_regional_compile",
                        "stage": "full_sweep/single_gpu",
                        "status": "resolved",
                        "symptom": "regional_compile rerun passed at 1.08x",
                        "root_cause": "Initial full sweep result was a noisy borderline measurement.",
                        "fixes": ["Re-ran ch13:regional_compile on an uncontended GPU and confirmed 1.08x."],
                        "verification": {"command": "python -m cli.aisp bench run --targets ch13:regional_compile --profile minimal --validity-profile portable --single-gpu"},
                        "evidence_paths": {"rerun_output_json": str(tmp_path / "rerun.json")},
                        "resolved_at": "2026-03-28T06:10:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "active_issue_ledger.md").write_text("# Active Issue Ledger\n", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert {entry["target"] for entry in status["aggregate_failures"]} == {"ch13:regional_compile"}
    synced_ledger = json.loads((run_dir / "active_issue_ledger.json").read_text(encoding="utf-8"))
    assert synced_ledger["schema_version"] == "1.0"
    assert synced_ledger["preferred_collection_key"] == "rows"
    assert synced_ledger["collection_aliases"] == {"issues": "rows"}
    row = next(item for item in synced_ledger["rows"] if item["issue_id"] == "reported_full_sweep_single_gpu_ch13_regional_compile")
    assert row["status"] == "resolved"
    assert row["root_cause"] == "Initial full sweep result was a noisy borderline measurement."
    assert row["verification"]["command"].endswith("--single-gpu")
    assert row["evidence_paths"]["rerun_output_json"].endswith("rerun.json")
    assert status["ledgers"]["summary"]["resolved_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 0
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 0
    assert status["current"]["reported_failures"] == []
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "reported_failures=" not in rendered


def test_inspect_benchmark_e2e_sweep_run_splits_historical_reported_rows_from_active_counts_while_live(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_live_resume_historical_rows"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    child_run_id = f"{run_id}__full_sweep__single__resume2"
    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "failed",
                    "benchmark_summary": {
                        "failed_benchmarks": [
                            {
                                "target": "ch06:launch_bounds",
                                "status": "failed_no_speedup",
                                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
                            }
                        ],
                        "skipped_benchmarks": [],
                    },
                },
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch18",
                    "completed_units": ["ch17"],
                },
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-29T17:00:00Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": stage_payload,
    }
    checkpoint_payload = dict(summary_payload)
    checkpoint_payload["orchestrator_pid"] = 123
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T17:00:05+00:00",
            "step": "full_sweep/single_gpu:ch18:flexdecoding",
            "step_detail": "optimized timing",
            "percent_complete": 66.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 123,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    benchmark_paths = e2e_sweep._benchmark_run_event_paths(  # type: ignore[attr-defined]
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    benchmark_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-29T17:00:04Z",
                        "event_type": "example_end",
                        "run_id": child_run_id,
                        "chapter": "ch17",
                        "example": "dynamic_routing",
                        "status": "succeeded",
                        "best_speedup": 33.98,
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: int(pid or 0) == 123)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["current"]["reported_failures"] == []
    assert status["issue_groups"] == []
    assert status["historical_issue_groups"]
    assert status["ledgers"]["summary"]["issue_count"] == 1
    assert status["ledgers"]["summary"]["unresolved_count"] == 1
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 1

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "active_issue_counts=issues=0 unresolved=0" in rendered
    assert "historical_issue_counts=issues=1 unresolved=1" in rendered
    assert "ledger_totals=issues=1 resolved=0 unresolved=1" in rendered
    assert "historical_issue_groups=1" in rendered


def test_inspect_benchmark_e2e_sweep_run_keeps_historical_rows_historical_when_running_stale_with_child_progress(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "e2e_stale_with_live_child_progress"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    child_run_id = f"{run_id}__full_sweep__single__resume2"
    stage_payload = [
        {
            "name": "full_sweep",
            "enabled": True,
            "run_id": f"{run_id}__full_sweep",
            "status": "running",
            "attempts": [
                {
                    "run_id": f"{run_id}__full_sweep__single",
                    "bucket": "single_gpu",
                    "status": "failed",
                    "benchmark_summary": {
                        "failed_benchmarks": [
                            {
                                "target": "ch06:launch_bounds",
                                "status": "failed_no_speedup",
                                "error": "Best speedup 1.006x below required 1.007x threshold for speed-goal benchmark",
                            }
                        ],
                        "skipped_benchmarks": [],
                    },
                },
                {
                    "run_id": child_run_id,
                    "bucket": "single_gpu",
                    "status": "running",
                    "active_unit": "ch18",
                    "completed_units": ["ch17"],
                },
            ],
        }
    ]
    summary_payload = {
        "run_id": run_id,
        "run_state": "running",
        "overall_status": "running",
        "updated_at": "2026-03-29T17:00:00Z",
        "resume_available": False,
        "contract": {"run_full_sweep": True, "full_sweep_suite_timeout": 0},
        "stages": stage_payload,
        "orchestrator_pid": 999,
    }
    checkpoint_payload = dict(summary_payload)
    progress_payload = {
        "run_id": run_id,
        "current": {
            "timestamp": "2026-03-29T17:00:05+00:00",
            "step": "full_sweep/single_gpu:ch18:flexdecoding",
            "step_detail": "optimized timing",
            "percent_complete": 66.0,
            "metrics": {
                "run_state": "running",
                "overall_status": "running",
                "current_stage": "full_sweep",
                "current_stage_run_id": child_run_id,
                "current_bucket": "single_gpu",
                "orchestrator_pid": 999,
                "stages": stage_payload,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps(progress_payload), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    benchmark_paths = e2e_sweep._benchmark_run_event_paths(  # type: ignore[attr-defined]
        child_run_id,
        repo_root=tmp_path,
        artifacts_dir=None,
    )
    benchmark_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    benchmark_paths["progress"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["events"].parent.mkdir(parents=True, exist_ok=True)
    benchmark_paths["progress"].write_text(
        json.dumps(
            {
                "run_id": child_run_id,
                "current": {
                    "timestamp": "2026-03-29T17:00:06+00:00",
                    "phase": "optimized_timing",
                    "step": "ch18:flexdecoding",
                    "step_detail": "optimized timing",
                    "percent_complete": 12.0,
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark_paths["events"].write_text(
        json.dumps(
            {
                "timestamp": "2026-03-29T17:00:04Z",
                "event_type": "example_end",
                "run_id": child_run_id,
                "chapter": "ch17",
                "example": "dynamic_routing",
                "status": "succeeded",
                "best_speedup": 33.98,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    assert status["inferred_state"] == "running_stale"
    assert status["current"]["reported_failures"] == []
    assert status["issue_groups"] == []
    assert status["historical_issue_groups"]
    assert status["ledgers"]["summary"]["active_issue_count"] == 0
    assert status["ledgers"]["summary"]["historical_issue_count"] == 1

    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "active_issue_counts=issues=0 unresolved=0" in rendered
    assert "historical_issue_counts=issues=1 unresolved=1" in rendered


def test_inspect_benchmark_e2e_sweep_run_groups_cascade_failures(tmp_path: Path, monkeypatch) -> None:
    run_id = "e2e_grouped_cascade"
    run_dir = e2e_sweep.e2e_run_dir(run_id, tmp_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "run_id": run_id,
        "run_state": "completed",
        "overall_status": "failed",
        "updated_at": "2026-03-28T05:30:00Z",
        "resume_available": False,
        "stages": [
            {
                "name": "full_sweep",
                "enabled": True,
                "run_id": f"{run_id}__full_sweep",
                "status": "failed",
                "attempts": [
                    {
                        "run_id": f"{run_id}__full_sweep__single",
                        "bucket": "single_gpu",
                        "status": "failed",
                        "benchmark_summary": {
                            "failed_benchmarks": [
                                {
                                    "target": "ch07:matmul_tiled",
                                    "status": "failed_error",
                                    "error": "Baseline execution failed: received SIGHUP",
                                },
                                {
                                    "target": "ch07:memory_access",
                                    "status": "failed_error",
                                    "error": "Baseline execution failed: [Errno 5] Input/output error",
                                },
                            ],
                            "skipped_benchmarks": [],
                        },
                    }
                ],
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(summary_payload), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "current": {
                    "timestamp": "2026-03-28T05:30:00+00:00",
                    "metrics": {
                        "run_state": "completed",
                        "overall_status": "failed",
                        "stages": summary_payload["stages"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    status = e2e_sweep.inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=tmp_path)

    groups = status["issue_groups"]
    assert len(groups) == 2
    signatures = {group["signature_key"] for group in groups}
    assert "received_sighup" in signatures
    assert "errno5_input_output_error" in signatures
    errno_group = next(group for group in groups if group["signature_key"] == "errno5_input_output_error")
    assert errno_group["cascade_from"] == "received_sighup"
    rendered = e2e_sweep.render_benchmark_e2e_status_text(status)
    assert "issue_group_preview=" in rendered


def test_watch_benchmark_e2e_sweep_run_launches_detached_watcher(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_id = "e2e_watch_launch"
    run_dir = e2e_sweep.e2e_run_dir(run_id, repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_sweep, "_repo_root", lambda: repo_root)

    captured: dict[str, object] = {}

    class DummyProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def _fake_popen(cmd, cwd=None, stdout=None, stderr=None, start_new_session=None):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["stdout_name"] = getattr(stdout, "name", None)
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        return DummyProc(54321)

    monkeypatch.setattr(e2e_sweep.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(e2e_sweep, "_pid_is_live", lambda pid: False)

    result = e2e_sweep.watch_benchmark_e2e_sweep_run(run_id=run_id, repo_root=repo_root, poll_interval_seconds=5, max_auto_resumes=2)

    assert result["success"] is True
    assert result["watcher_pid"] == 54321
    assert "--run-id" in captured["cmd"]
    assert "--poll-interval-seconds" in captured["cmd"]
    assert "--max-auto-resumes" in captured["cmd"]
    assert captured["cwd"] == str(repo_root)
    assert captured["start_new_session"] is True
    assert captured["stdout_name"].endswith(f"{run_id}_watcher.launch.log")


def test_benchmark_e2e_status_handler_and_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.inspect_benchmark_e2e_sweep_run",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"], "inferred_state": "running_live"},
    )

    payload = handlers.benchmark_e2e_status({"run_id": "handler_status"})
    assert payload["run_id"] == "handler_status"
    assert payload["inferred_state"] == "running_live"

    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_status",
        lambda params: {"success": True, "run_id": params.get("run_id"), "source": "tool"},
    )
    tool_payload = mcp_server.tool_benchmark_e2e_status({"run_id": "tool_status"})
    assert tool_payload["source"] == "tool"


def test_benchmark_e2e_watch_handler_and_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.watch_benchmark_e2e_sweep_run",
        lambda **kwargs: {"success": True, "run_id": kwargs["run_id"], "watcher_pid": 123},
    )
    monkeypatch.setattr(
        "core.benchmark.e2e_sweep.resolve_latest_e2e_run_id",
        lambda **kwargs: "latest_watch",
    )

    payload = handlers.benchmark_e2e_watch({})
    assert payload["run_id"] == "latest_watch"
    assert payload["watcher_pid"] == 123

    monkeypatch.setattr(
        "core.api.handlers.benchmark_e2e_watch",
        lambda params: {"success": True, "run_id": params.get("run_id"), "source": "tool-watch"},
    )
    tool_payload = mcp_server.tool_benchmark_e2e_watch({"run_id": "tool_watch"})
    assert tool_payload["source"] == "tool-watch"
