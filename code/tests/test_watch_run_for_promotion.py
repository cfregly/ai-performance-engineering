from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cluster.scripts.write_manifest import build_manifest_payload


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


def test_watch_run_for_promotion_publishes_completed_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    cluster_root = repo_root / "cluster"
    run_id = "2026-03-07_localhost_modern_profile_r99_full20b"
    run_dir = _seed_canonical_run(cluster_root, run_id, "localhost")
    status_filename = f"{run_id}_postrun_promote_watch_status.json"
    legacy_status = run_dir / "raw" / status_filename
    _write_json(legacy_status, {"status": "waiting"})
    legacy_contents = legacy_status.read_bytes()
    manifest, manifest_path = build_manifest_payload(
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
    assert proc.returncode == 0, proc.stderr or proc.stdout

    watch_status = json.loads((run_dir / "progress" / status_filename).read_text(encoding="utf-8"))
    assert watch_status["status"] == "published"
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
