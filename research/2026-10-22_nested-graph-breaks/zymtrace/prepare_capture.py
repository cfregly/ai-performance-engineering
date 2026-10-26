#!/usr/bin/env python3
"""Build a non-executing Zymtrace capture plan for the nested-graph benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "nested-graph-breaks-zymtrace-capture-plan/v1"
ZYMTRACE_VERSION = "26.10.1"
CASE = "nested_break_depth8"
ARMS = (("disabled", 0), ("enabled", 1))
DEFAULT_TIMEOUT_SECONDS = 240
MAX_TIMEOUT_SECONDS = 300
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
    r"-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
PROJECT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class PlanError(ValueError):
    """Raised when a dry-run input is unsafe or incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_file(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise PlanError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_file():
        raise PlanError(f"{label} is not a regular file: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PlanError(f"{label} is not readable: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PlanError(f"{label} is not executable: {resolved}")
    return resolved


def _absolute_executable_invocation(path: Path, label: str) -> Path:
    """Validate the target while retaining a virtualenv or shim invocation path."""
    if not path.is_absolute():
        raise PlanError(f"{label} must be an absolute path")
    invocation = Path(os.path.abspath(path))
    if not invocation.is_file():
        raise PlanError(f"{label} is not a regular file: {invocation}")
    if not os.access(invocation, os.R_OK | os.X_OK):
        raise PlanError(f"{label} is not readable and executable: {invocation}")
    return invocation


def _new_absolute_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise PlanError(f"{label} must be an absolute path")
    resolved = path.resolve(strict=False)
    if resolved.exists():
        raise PlanError(f"{label} must not already exist: {resolved}")
    if not resolved.parent.is_dir():
        raise PlanError(f"{label} parent does not exist: {resolved.parent}")
    return resolved


def _validate_collection_agent(value: str) -> str:
    endpoint = value.strip()
    if (
        not endpoint
        or any(character.isspace() for character in endpoint)
        or "://" in endpoint
        or "/" in endpoint
        or "@" in endpoint
        or endpoint in {"ZYMTRACE_URL", "<host>:8080", "example.com:443"}
    ):
        raise PlanError(
            "collection agent must be an explicit host:port without a scheme, path, or credentials"
        )
    try:
        host, port_text = endpoint.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise PlanError("collection agent must end in a numeric port") from exc
    if not host or not 1 <= port <= 65535:
        raise PlanError("collection agent host or port is invalid")
    return endpoint


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    if not args.repo_root.is_absolute() or not repo_root.is_dir():
        raise PlanError("repo root must be an existing absolute directory")

    wrapper = _absolute_file(
        repo_root / "code/core/scripts/profiling/profile.sh",
        "repository profile wrapper",
        executable=True,
    )
    benchmark = _absolute_file(
        repo_root / "research/2026-10-22_nested-graph-breaks/benchmark.py",
        "nested-graph benchmark",
    )
    python = _absolute_executable_invocation(args.python, "Python interpreter")
    injection_library = _absolute_file(
        args.injection_library, "Zymtrace CUDA injection library"
    )
    if injection_library.stat().st_size == 0:
        raise PlanError("Zymtrace CUDA injection library must not be empty")
    owner_checkpoint = _absolute_file(
        args.runtime_owner_checkpoint, "runtime owner checkpoint"
    )
    if owner_checkpoint.stat().st_size == 0:
        raise PlanError("runtime owner checkpoint must not be empty")
    run_root = _new_absolute_directory(args.run_root, "run root")
    collection_agent = _validate_collection_agent(args.collection_agent)

    gpu_uuid = args.expected_gpu_uuid.strip()
    if not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise PlanError("expected GPU UUID must be a full GPU-... UUID")
    project = args.zymtrace_project.strip()
    if not PROJECT_RE.fullmatch(project):
        raise PlanError(
            "Zymtrace project must use only letters, digits, dot, underscore, or dash"
        )
    if not 60 <= args.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise PlanError(
            f"timeout seconds must be between 60 and {MAX_TIMEOUT_SECONDS}"
        )

    arms: list[dict[str, Any]] = []
    for arm_name, enabled in ARMS:
        arm_root = run_root / arm_name
        environment = {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "TORCHINDUCTOR_CACHE_DIR": str(arm_root / "torchinductor-cache"),
            "TRITON_CACHE_DIR": str(arm_root / "triton-cache"),
            "ZYMTRACE_CUDA_INJECTION64_PATH": str(injection_library),
            "ZYMTRACE_CUDAPROFILER__ENABLE_PC_SAMPLING": "false",
            "ZYMTRACE_CUDAPROFILER__PRINT_STATS": "true",
            "ZYMTRACE_CUDAPROFILER__QUIET": "false",
            "ZYMTRACE_CUDAPROFILER__STACK_TRACE_SAMPLING__ENABLED": "true",
            "ZYMTRACE_CUDAPROFILER__STACK_TRACE_SAMPLING__TARGET": "200",
            "ZYMTRACE_CUDAPROFILER__STACK_TRACE_SAMPLING__WINDOW_SIZE": "1",
        }
        argv = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{args.timeout_seconds}s",
            "env",
            *(f"{name}={value}" for name, value in environment.items()),
            str(wrapper),
            str(benchmark),
            "--arch",
            "sm_100",
            "--tool",
            "zymtrace",
            "--output-root",
            str(arm_root / "profile-wrapper"),
            "--python",
            str(python),
            "--",
            "--case",
            CASE,
            "--enabled",
            str(enabled),
            "--seed",
            "0",
            "--out",
            str(arm_root / "benchmark-result"),
            "--expected-gpu-uuid",
            gpu_uuid,
        ]
        arms.append(
            {
                "arm": arm_name,
                "enabled": enabled,
                "run_root": str(arm_root),
                "environment": environment,
                "argv": argv,
                "shell_command": shlex.join(argv),
                "expected_local_outputs": [
                    str(arm_root / "benchmark-result/result.json"),
                    str(arm_root / "profile-wrapper/<timestamp>_benchmark/zymtrace_launch_manifest.json"),
                    "controller-owned stdout log",
                    "controller-owned stderr log",
                ],
            }
        )

    return {
        "schema": SCHEMA,
        "status": "dry_run_only",
        "not_executed": True,
        "zymtrace": {
            "backend_and_profiler_version_pin": ZYMTRACE_VERSION,
            "collection_agent_precondition": {
                "value": collection_agent,
                "tls_mode": args.tls_mode,
                "passed_to_workload": False,
                "verification_required": (
                    "Confirm this value in the separately owned host profiler."
                ),
            },
            "project_precondition": {
                "value": project,
                "passed_to_workload": False,
                "verification_required": (
                    "Confirm this project in the separately owned host profiler."
                ),
            },
            "profiler_agent_precondition": (
                "The separately owned Zymtrace profiler must already be running, "
                "reporting to this collection agent, and using this project."
            ),
            "authentication": {
                "token_value_included": False,
                "profiler_environment_name_if_backend_auth_is_enabled": "ZYMTRACE_AUTH_TOKEN",
            },
        },
        "target_lane": {
            "gpu": "NVIDIA B200 (sm_100)",
            "cuda": "13.x",
            "pytorch": "2.14",
            "qualification": (
                "compatibility smoke required because official docs do not publish "
                "an exact B200, CUDA 13.x, and PyTorch 2.14 compatibility row"
            ),
        },
        "capture_policy": {
            "case": CASE,
            "seed": 0,
            "arms": [name for name, _ in ARMS],
            "timeout_seconds_per_arm": args.timeout_seconds,
            "maximum_total_wall_seconds_including_kill_grace": len(ARMS)
            * (args.timeout_seconds + 30),
            "pc_sampling_enabled": False,
            "kernel_launch_sampling_enabled": True,
            "kernel_launch_sampling_target_per_thread_per_second": 200,
            "benchmark_chrome_profile_enabled": False,
            "purpose": "visibility only, not a latency or acceptance measurement",
            "qualification_use_prohibited": True,
            "experiment_matrix_coverage": {
                "cases": "1 of 3",
                "seeds": "1 of 4",
                "arms": "2 of 2",
                "cells": "2 of 24",
            },
        },
        "runtime_ownership": {
            "must_be_rechecked_immediately_before_execution": True,
            "checkpoint": _path_record(owner_checkpoint),
        },
        "source": {
            "repo_root": str(repo_root),
            "benchmark": _path_record(benchmark),
            "profile_wrapper": _path_record(wrapper),
            "plan_builder": _path_record(Path(__file__).resolve()),
            "python": str(python),
            "injection_library": _path_record(injection_library),
        },
        "run_root": str(run_root),
        "arms": arms,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--injection-library", required=True, type=Path)
    parser.add_argument("--runtime-owner-checkpoint", required=True, type=Path)
    parser.add_argument("--collection-agent", required=True)
    parser.add_argument("--zymtrace-project", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument(
        "--tls-mode", choices=("tls", "disabled"), default="tls"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        plan = build_plan(_parse_args(argv))
    except PlanError as exc:
        raise SystemExit(f"capture plan rejected: {exc}") from exc
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
