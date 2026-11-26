#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_STEPS = [
    "bootstrap_nodes",
    "preflight_services",
    "discovery",
    "quick_friction_all_nodes",
    "monitoring_expectations_all_nodes",
    "hang_triage_bundle",
    "connectivity_probe",
    "nccl_single_node",
    "nccl_env_sensitivity",
    "vllm_serve_sweep",
    "vllm_request_rate_sweep",
    "gemm_sanity",
    "gpu_stream_all_nodes",
    "fp4_checks",
    "allreduce_stability",
    "allreduce_latency_comp",
    "allgather_control_plane",
    "nccl_alltoall_single_node",
    "nccl_algo_comparison",
    "train_step_single_node",
    "fio_all_nodes",
    "nvbandwidth_all_nodes",
    "build_cluster_scorecard",
    "build_mlperf_alignment",
    "analyze_benchmark_coverage",
    "validate_required_artifacts",
    "manifest_refresh",
]

DEFAULT_REQUIRED_FILES = [
    "manifest.json",
    "structured/{run_id}_suite_steps.json",
    "structured/{run_id}_benchmark_coverage_analysis.json",
    "structured/{run_id}_cluster_scorecard.json",
    "structured/{run_id}_mlperf_alignment.json",
    "structured/{run_id}_localhost_vllm_serve_sweep.csv",
    "structured/{run_id}_localhost_vllm_serve_request_rate_sweep.csv",
    "structured/{run_id}_localhost_gemm_gpu_sanity.csv",
    "structured/{run_id}_localhost_gpu_stream.json",
    "structured/{run_id}_localhost_fio.json",
    "structured/{run_id}_localhost_nvbandwidth.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a cluster run to exit, verify its artifacts, and promote it. "
            "Watcher receipts are written to the run's progress directory."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run id under cluster/runs/<run_id>.")
    parser.add_argument("--pid", required=True, type=int, help="PID to wait on before evaluation.")
    parser.add_argument("--repo-root", default=None, help="Override repo root for tests or custom layouts.")
    parser.add_argument("--label", default="localhost", help="Host label for localhost report rendering.")
    parser.add_argument(
        "--publish-report-path",
        default="cluster/field-report-localhost.md",
        help="Published localhost report path.",
    )
    parser.add_argument(
        "--publish-notes-path",
        default="cluster/field-report-localhost-notes.md",
        help="Published localhost notes path.",
    )
    parser.add_argument(
        "--skip-render-localhost-report",
        action="store_true",
        help="Skip rendering localhost markdown while promoting.",
    )
    parser.add_argument(
        "--skip-validate-localhost-report",
        action="store_true",
        help="Skip staged localhost report validation before publication.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup after promotion using this run id as canonical.",
    )
    parser.add_argument(
        "--allow-run-id",
        action="append",
        default=[],
        help="Additional run ids retained during cleanup/validation.",
    )
    parser.add_argument(
        "--required-step",
        action="append",
        default=[],
        help="Required suite step name. Defaults to the full canonical modern-llm set.",
    )
    parser.add_argument(
        "--required-file",
        action="append",
        default=[],
        help="Required file relative to cluster/runs/<run_id>/, with {run_id} expansion.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="How often to check whether the PID has exited.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Promotion subprocess timeout.",
    )
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be a positive integer")
    return args


def _repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root:
        return Path(args.repo_root).resolve()
    return Path(__file__).resolve().parents[2]


def watcher_run_dir(repo_root: Path, run_id: str) -> Path:
    if run_id in ("", ".", "..") or Path(run_id).name != run_id:
        raise ValueError("Run ID must name one directory under cluster/runs.")
    runs = repo_root / "cluster" / "runs"
    run_dir = runs / run_id
    if not run_dir.is_dir() or run_dir.resolve() != run_dir:
        raise ValueError(f"Missing or invalid run directory: {run_dir}")
    return run_dir


def watcher_receipt_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    """Return mutable watcher receipts outside the run's manifest inventory."""
    progress = run_dir / "progress"
    run_id = run_dir.name
    paths = (
        progress / f"{run_id}_postrun_promote_watch_status.json",
        progress / f"{run_id}_postrun_promote_watch.launch.log",
        progress / f"{run_id}_promote.log",
    )
    for path in (progress, *paths):
        if path.is_symlink():
            raise ValueError(f"Watcher receipts require a real path at {path}")
    return paths


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_required_files(run_dir: Path, run_id: str, configured: list[str]) -> list[Path]:
    items = configured if configured else DEFAULT_REQUIRED_FILES
    paths: list[Path] = []
    for item in items:
        rendered = item.format(run_id=run_id)
        candidate = Path(rendered)
        paths.append(candidate if candidate.is_absolute() else run_dir / candidate)
    return paths


def _evaluate_run(run_dir: Path, run_id: str, required_steps: list[str], required_files: list[Path]) -> dict[str, Any]:
    steps_path = run_dir / "structured" / f"{run_id}_suite_steps.json"
    if not steps_path.exists():
        return {
            "ok": False,
            "failed_steps": ["suite_steps_missing"],
            "missing_steps": required_steps,
            "missing_files": [str(path) for path in required_files if not path.exists()],
        }

    steps = json.loads(steps_path.read_text(encoding="utf-8"))
    status_by_name = {step["name"]: step.get("exit_code") for step in steps}
    failed_steps = [name for name, exit_code in status_by_name.items() if exit_code != 0]
    missing_steps = [name for name in required_steps if name not in status_by_name]
    missing_files = [str(path) for path in required_files if not path.exists()]
    return {
        "ok": not failed_steps and not missing_steps and not missing_files,
        "failed_steps": failed_steps,
        "missing_steps": missing_steps,
        "missing_files": missing_files,
    }


def _run_promote(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    script = Path(__file__).resolve().with_name("promote_run.py")
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(repo_root),
        "--run-id",
        args.run_id,
        "--label",
        args.label,
        "--publish-report-path",
        args.publish_report_path,
        "--publish-notes-path",
        args.publish_notes_path,
    ]
    if args.skip_render_localhost_report:
        cmd.append("--skip-render-localhost-report")
    if args.skip_validate_localhost_report:
        cmd.append("--skip-validate-localhost-report")
    if args.cleanup:
        cmd.append("--cleanup")
    for run_id in args.allow_run_id:
        if run_id:
            cmd.extend(["--allow-run-id", run_id])

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout_seconds if args.timeout_seconds and args.timeout_seconds > 0 else None,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }


def main() -> int:
    args = parse_args()
    repo_root = _repo_root(args)
    try:
        run_dir = watcher_run_dir(repo_root, args.run_id)
        status_path, _, promote_log_path = watcher_receipt_paths(run_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    required_steps = args.required_step or list(DEFAULT_REQUIRED_STEPS)
    required_files = _resolve_required_files(run_dir, args.run_id, args.required_file)

    _write_json(
        status_path,
        {
            "status": "waiting",
            "run_id": args.run_id,
            "pid": args.pid,
            "started_at": _iso_now(),
            "poll_interval_seconds": args.poll_interval_seconds,
        },
    )

    while _process_alive(args.pid):
        time.sleep(max(args.poll_interval_seconds, 0.1))

    finished_at = _iso_now()
    evaluation = _evaluate_run(run_dir, args.run_id, required_steps, required_files)
    if not evaluation["ok"]:
        _write_json(
            status_path,
            {
                "status": "run_failed",
                "run_id": args.run_id,
                "pid": args.pid,
                "finished_at": finished_at,
                **evaluation,
            },
        )
        return 1

    _write_json(
        status_path,
        {
            "status": "promoting",
            "run_id": args.run_id,
            "pid": args.pid,
            "finished_at": finished_at,
        },
    )

    promote_result = _run_promote(args, repo_root)
    promote_log_path.write_text(
        json.dumps(promote_result, indent=2) + "\n",
        encoding="utf-8",
    )
    if not promote_result["success"]:
        _write_json(
            status_path,
            {
                "status": "promote_failed",
                "run_id": args.run_id,
                "pid": args.pid,
                "finished_at": finished_at,
                "promote_log_path": str(promote_log_path),
                **evaluation,
                "promote_returncode": promote_result["returncode"],
            },
        )
        return 1

    _write_json(
        status_path,
        {
            "status": "published",
            "run_id": args.run_id,
            "pid": args.pid,
            "finished_at": finished_at,
            "promote_log_path": str(promote_log_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
