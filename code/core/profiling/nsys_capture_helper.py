#!/usr/bin/env python3
"""Run a one-shot Nsight Systems capture from a clean helper process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.profiling.nsight_automation import NsightAutomation


def _apply_owner_markers_from_args(args: argparse.Namespace) -> None:
    if getattr(args, "aisp_owner_run_id", None):
        os.environ["AISP_BENCHMARK_OWNER_RUN_ID"] = str(args.aisp_owner_run_id)
    if getattr(args, "aisp_owner_pid", None):
        os.environ["AISP_BENCHMARK_OWNER_PID"] = str(args.aisp_owner_pid)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--aisp-owner-run-id", help="Optional owner run id marker for strict foreign-process filtering.")
    parser.add_argument("--aisp-owner-pid", help="Optional owner pid marker for strict foreign-process filtering.")
    parser.add_argument("--payload", required=True, help="Path to JSON payload.")
    parser.add_argument("--result", required=True, help="Path to JSON result file.")
    args = parser.parse_args()
    _apply_owner_markers_from_args(args)

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    automation = NsightAutomation(Path(payload["output_dir"]))
    report = automation.profile_nsys(
        command=list(payload["command"]),
        output_name=str(payload["output_name"]),
        trace_cuda=True,
        trace_nvtx=True,
        trace_osrt=True,
        full_timeline=bool(payload.get("full_timeline", False)),
        trace_forks=bool(payload.get("trace_forks", False)),
        preset=str(payload.get("preset", "light")),
        timeout_seconds=payload.get("timeout_seconds"),
        wait_mode=str(payload.get("wait_mode", "primary")),
        finalize_grace_seconds=20.0,
        force_lineinfo=True,
        extra_env=dict(payload.get("extra_env") or {}),
        sanitize_python_startup=bool(payload.get("sanitize_python_startup", True)),
        capture_range_cuda_profiler_api=bool(
            payload.get("capture_range_cuda_profiler_api", False)
        ),
        cuda_graph_trace=payload.get("cuda_graph_trace"),
    )
    result = {
        "report": str(report) if report else None,
        "last_error": automation.last_error,
    }
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
