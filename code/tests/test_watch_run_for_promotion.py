from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cluster.scripts import watch_run_for_promotion as watcher
from cluster.scripts import write_manifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_canonical_run(cluster_root: Path, run_id: str, label: str) -> Path:
    run_dir = cluster_root / "runs" / run_id
    structured = run_dir / "structured"
    raw = run_dir / "raw" / f"{run_id}_suite"
    figures = run_dir / "figures"
    structured.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    (cluster_root / "docs").mkdir(parents=True, exist_ok=True)
    (cluster_root / "docs" / "advanced-runbook.md").write_text("# runbook\n", encoding="utf-8")
    (cluster_root / "docs" / "field-report-template.md").write_text("# template\n", encoding="utf-8")

    _write_json(run_dir / "manifest.json", {"run_id": run_id})
    _write_json(
        structured / f"{run_id}_suite_steps.json",
        [
            {"name": "bootstrap_nodes", "exit_code": 0},
            {"name": "preflight_services", "exit_code": 0},
            {"name": "discovery", "exit_code": 0},
            {"name": "quick_friction_all_nodes", "exit_code": 0},
            {"name": "monitoring_expectations_all_nodes", "exit_code": 0},
            {"name": "hang_triage_bundle", "exit_code": 0},
            {"name": "connectivity_probe", "exit_code": 0},
            {"name": "nccl_single_node", "exit_code": 0},
            {"name": "nccl_env_sensitivity", "exit_code": 0},
            {"name": "vllm_serve_sweep", "exit_code": 0},
            {"name": "vllm_request_rate_sweep", "exit_code": 0},
            {"name": "gemm_sanity", "exit_code": 0},
            {"name": "gpu_stream_all_nodes", "exit_code": 0},
            {"name": "fp4_checks", "exit_code": 0},
            {"name": "allreduce_stability", "exit_code": 0},
            {"name": "allreduce_latency_comp", "exit_code": 0},
            {"name": "allgather_control_plane", "exit_code": 0},
            {"name": "nccl_alltoall_single_node", "exit_code": 0},
            {"name": "nccl_algo_comparison", "exit_code": 0},
            {"name": "train_step_single_node", "exit_code": 0},
            {"name": "fio_all_nodes", "exit_code": 0},
            {"name": "nvbandwidth_all_nodes", "exit_code": 0},
            {"name": "build_cluster_scorecard", "exit_code": 0},
            {"name": "build_mlperf_alignment", "exit_code": 0},
            {"name": "analyze_benchmark_coverage", "exit_code": 0},
            {"name": "validate_required_artifacts", "exit_code": 0},
            {"name": "manifest_refresh", "exit_code": 0},
        ],
    )
    _write_json(structured / f"{run_id}_benchmark_coverage_analysis.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_cluster_scorecard.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_mlperf_alignment.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_gpu_stream.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_fio.json", {"status": "ok"})
    _write_json(structured / f"{run_id}_{label}_nvbandwidth.json", {"status": "ok"})
    (structured / f"{run_id}_{label}_vllm_serve_sweep.csv").write_text("ok\n", encoding="utf-8")
    (structured / f"{run_id}_{label}_vllm_serve_request_rate_sweep.csv").write_text("ok\n", encoding="utf-8")
    (structured / f"{run_id}_{label}_gemm_gpu_sanity.csv").write_text("ok\n", encoding="utf-8")
    (raw / "suite.log").write_text("ok\n", encoding="utf-8")
    (figures / f"{run_id}_cluster_story_dashboard.png").write_text("png\n", encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("cleanup_blocked", [False, True])
def test_watch_run_for_promotion_publishes_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_blocked: bool
) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-07_localhost_modern_profile_r99_full20b"
    run_dir = _seed_canonical_run(cluster_root, run_id, "localhost")
    status_filename = f"{run_id}_postrun_promote_watch_status.json"
    legacy_status = run_dir / "raw" / status_filename
    _write_json(legacy_status, {"status": "waiting"})
    legacy_contents = legacy_status.read_bytes()
    monkeypatch.setattr(write_manifest, "_git_provenance", lambda worktree: {
        "commit": "a" * 40, "dirty": False, "probe_ok": True, "errors": [],
    })
    manifest, manifest_path = write_manifest.build_manifest_payload(
        cluster_root=cluster_root,
        run_id=run_id,
        run_dir=run_dir,
        include_figures=True,
        hosts=["localhost"],
        labels=["localhost"],
        finalize=True,
    )
    _write_json(manifest_path, manifest)
    assert f"raw/{status_filename}" in manifest["summary"]["sha256"]
    stale = cluster_root / "runs/2026-03-06_comparison/raw/reference.json"
    if cleanup_blocked:
        _write_json(stale, {"retained": True})
        (cluster_root / "field-report.md").write_text("[Reference](runs/2026-03-06_comparison/raw/reference.json)\n")

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "watch_run_for_promotion.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--pid",
            "999999",
            "--poll-interval-seconds",
            "0.01",
            "--skip-render-localhost-report",
            "--skip-validate-localhost-report",
            *(["--cleanup"] if cleanup_blocked else []),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == (1 if cleanup_blocked else 0), proc.stderr or proc.stdout

    watch_status = json.loads((run_dir / "progress" / status_filename).read_text(encoding="utf-8"))
    assert watch_status["status"] == "published"
    assert watch_status["published"] is True
    assert watch_status["success"] == (not cleanup_blocked)
    if cleanup_blocked:
        assert "Cleanup planning failed" in watch_status["error"]
        assert stale.is_file()
    assert (cluster_root / "published/current/manifest.json").read_bytes() == manifest_path.read_bytes()
    assert (run_dir / "progress" / f"{run_id}_promote.log").is_file()
    assert legacy_status.read_bytes() == legacy_contents
    assert not (run_dir / "raw" / f"{run_id}_promote.log").exists()


def test_watch_run_for_promotion_marks_failed_run_without_promoting(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-07_localhost_modern_profile_r98_full20b"
    _seed_canonical_run(cluster_root, run_id, "localhost")

    steps_path = cluster_root / "runs" / run_id / "structured" / f"{run_id}_suite_steps.json"
    steps = json.loads(steps_path.read_text(encoding="utf-8"))
    for step in steps:
        if step["name"] == "fp4_checks":
            step["exit_code"] = 1
            break
    _write_json(steps_path, steps)

    script = Path(__file__).resolve().parents[1] / "cluster" / "scripts" / "watch_run_for_promotion.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--run-id",
            run_id,
            "--pid",
            "999999",
            "--poll-interval-seconds",
            "0.01",
            "--skip-render-localhost-report",
            "--skip-validate-localhost-report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1

    watch_status = json.loads(
        (cluster_root / "runs" / run_id / "progress" / f"{run_id}_postrun_promote_watch_status.json").read_text(encoding="utf-8")
    )
    assert watch_status["status"] == "run_failed"
    assert "fp4_checks" in watch_status["failed_steps"]
    assert not (cluster_root / "published" / "current").exists()


def test_watch_run_records_promotion_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "2026-03-07_watcher_outcome"
    run_dir = _seed_canonical_run(tmp_path / "cluster", run_id, "localhost")
    status_path, _, promotion_log = watcher.watcher_receipt_paths(run_dir)
    monkeypatch.setattr(sys, "argv", [
        "watch_run_for_promotion.py", "--repo-root", str(tmp_path), "--run-id", run_id,
        "--pid", "999999", "--cleanup",
    ])
    monkeypatch.setattr(watcher, "_process_alive", lambda pid: False)

    def promote(cmd: list[str], *, timeout: int, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, timeout, output=b"", stderr=b"partial diagnostic")

    monkeypatch.setattr(watcher, "run_promotion_process", promote)
    assert watcher.main() == 1
    status = json.loads(status_path.read_text())
    log = json.loads(promotion_log.read_text())
    assert status["success"] is False
    assert status["status"] == "promote_timeout"
    assert status["published"] is None
    assert "timed out" in status["error"]
    assert log["stderr"] == "partial diagnostic"


@pytest.mark.parametrize("contents", ['[{"name":', '{"name": "discovery"}', '[{"exit_code": 0}]'])
def test_watch_run_records_invalid_ledger(tmp_path, monkeypatch, contents):
    run_id = "2026-03-07_invalid_ledger"
    run_dir = _seed_canonical_run(tmp_path / "cluster", run_id, "localhost")
    status_path, _, promotion_log = watcher.watcher_receipt_paths(run_dir)
    (run_dir / "structured" / f"{run_id}_suite_steps.json").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "watch_run_for_promotion.py", "--repo-root", str(tmp_path), "--run-id", run_id, "--pid", "999999",
    ])
    monkeypatch.setattr(watcher, "_process_alive", lambda pid: False)

    assert watcher.main() == 1
    status = json.loads(status_path.read_text())
    assert status["status"] == "run_failed"
    assert status["failed_steps"] == ["suite_steps_invalid"]
    assert "Unable to read suite steps" in status["error"]
    assert not promotion_log.exists()


def test_watcher_accepts_external_runs_directory_without_following_selected_run_symlinks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runs = tmp_path / "dedicated-runs"
    run_dir = runs / "local-run"
    run_dir.mkdir(parents=True)
    cluster = repo_root / "cluster"
    cluster.mkdir(parents=True)
    (cluster / "runs").symlink_to(runs, target_is_directory=True)
    assert watcher.watcher_run_dir(repo_root, run_dir.name) == run_dir.resolve()

    (runs / "escaped-run").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Missing or invalid run directory"):
        watcher.watcher_run_dir(repo_root, "escaped-run")
