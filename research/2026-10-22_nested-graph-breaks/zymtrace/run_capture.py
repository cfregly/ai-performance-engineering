#!/usr/bin/env python3
"""Run one bounded Zymtrace 26.10.1 visibility capture and retain its proof."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import prepare_capture


SCHEMA = "nested-graph-breaks-zymtrace-capture/v1"
BACKEND_BASE = "http://127.0.0.1:18080"
COLLECTION_AGENT = "127.0.0.1:18080"
PROJECTS_PATH = "/public/api/v1/projects/search"
CONFIG_GET_PATH = "/public/api/v1/agent/config/get"
CONFIG_SET_PATH = "/public/api/v1/agent/config/set"
VERSION = "26.10.1"
GLOBAL_LIMIT_SECONDS = 540
FINAL_RESERVE_SECONDS = 60
KILL_GRACE_SECONDS = 30
MIN_WORKER_SECONDS = 60
AGENT_READY_SECONDS = 60
HTTP_TIMEOUT_SECONDS = 5
EXPECTED_TORCH = "2.14.1+cu130"
CPU_AFFINITY_RE = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")
REQUIRED_SOURCE_PATHS = frozenset(
    {
        "research/2026-10-22_nested-graph-breaks/zymtrace/run_capture.py",
        "research/2026-10-22_nested-graph-breaks/zymtrace/prepare_capture.py",
        "research/2026-10-22_nested-graph-breaks/benchmark.py",
        "research/2026-10-22_nested-graph-breaks/counter_capture.py",
        "research/2026-10-22_nested-graph-breaks/profile_capture.py",
        "code/core/scripts/profiling/profile.sh",
    }
)


class CaptureError(RuntimeError):
    """Raised when a capture proof or safety condition fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_receipt(path: Path, raw: bytes) -> dict[str, Any]:
    path.write_bytes(raw)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _json_write(path: Path, value: Any) -> dict[str, Any]:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return _bytes_receipt(path, raw)


def _utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise CaptureError(f"timestamp must include a timezone: {value}")
    return parsed.astimezone(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_gpu(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def _is_competing_agent(
    executable: Path, profiler_binary: Path, _arguments: str = ""
) -> bool:
    """Use executable identity only, never ancestor or command arguments."""
    return executable == profiler_binary


def _config_parts(document: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    try:
        config = document["config"]
        version = config["version"]
        entries = config["config"]["entries"]
    except (KeyError, TypeError) as exc:
        raise CaptureError("invalid live-config response") from exc
    if isinstance(version, bool) or not isinstance(version, int) or not isinstance(entries, list):
        raise CaptureError("invalid live-config version or entries")
    return version, entries


def _worker_timeout(now: dt.datetime, hard_deadline: dt.datetime, requested: int) -> int:
    usable = math.floor((hard_deadline - now).total_seconds())
    usable -= FINAL_RESERVE_SECONDS + KILL_GRACE_SECONDS
    bounded = min(240, requested, usable)
    if bounded < MIN_WORKER_SECONDS:
        raise CaptureError("deadline leaves no safe worker window")
    return bounded


def _replace_timeout(argv: list[str], seconds: int) -> list[str]:
    command = list(argv)
    try:
        index = command.index("--kill-after=30s") + 1
    except ValueError as exc:
        raise CaptureError("builder command lacks the bounded timeout") from exc
    command[index] = f"{seconds}s"
    return command


def _validate_event_export(
    document: dict[str, Any], expected_pid: int, expected_gpu_uuid: str
) -> dict[str, Any]:
    try:
        strings = document["strings"]
        events = document["events"]
    except (KeyError, TypeError) as exc:
        raise CaptureError("raw CUDA export lacks strings or events") from exc
    accepted = []
    target_gpu = _normalize_gpu(expected_gpu_uuid)
    for row in events:
        try:
            duration = row["c"]
            pid = row["pid"]
            stream = row["sid"]
            gpu = strings[row["u"]]
            kind = strings[row["k"]]
        except (KeyError, IndexError, TypeError):
            continue
        if (
            not isinstance(duration, bool)
            and isinstance(duration, (int, float))
            and math.isfinite(duration)
            and duration > 0
            and pid == expected_pid
            and stream is not None
            and kind == "cuda"
            and _normalize_gpu(gpu) == target_gpu
        ):
            accepted.append(row)
    if not accepted:
        raise CaptureError(
            "no positive finite CUDA row matches the worker PID, stream, and GPU UUID"
        )
    return {
        "raw_cuda_rows": len(events),
        "matching_cuda_rows": len(accepted),
        "source_pid": expected_pid,
        "expected_gpu_uuid": expected_gpu_uuid,
        "pid_scope": "export validation only because PID-scoped live-config CEL is not documented",
    }


class SystemOps:
    """Small injectable operating-system boundary used by the controller."""

    def now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def request(self, method: str, path: str, payload: Any) -> tuple[int, bytes]:
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            BACKEND_BASE + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=HTTP_TIMEOUT_SECONDS
            ) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(argv, **kwargs)

    def popen(self, argv: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        return subprocess.Popen(argv, **kwargs)

    def killpg(self, pgid: int, sig: signal.Signals) -> None:
        os.killpg(pgid, sig)

    def process_group_alive(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise CaptureError(
                f"cannot inspect owned process group {pgid}"
            ) from exc
        return True

    def hostname(self) -> str:
        return socket.gethostname()

    def competing_agents(self, profiler_binary: Path) -> list[int]:
        matches = []
        expected = profiler_binary.resolve()
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            raise CaptureError("/proc is required to prove no competing profiler agent")
        for process_dir in proc_root.iterdir():
            if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
                continue
            try:
                executable = (process_dir / "exe").resolve(strict=True)
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise CaptureError(
                    f"cannot inspect process {process_dir.name} executable"
                ) from exc
            except OSError as exc:
                raise CaptureError(
                    f"failed to inspect process {process_dir.name} executable"
                ) from exc
            if _is_competing_agent(executable, expected):
                matches.append(int(process_dir.name))
        return matches


def _request_json(ops: SystemOps, method: str, path: str, payload: Any) -> tuple[bytes, Any]:
    status, raw = ops.request(method, path, payload)
    if status != 200:
        raise CaptureError(f"{method} {path} returned HTTP {status}")
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{method} {path} returned invalid JSON") from exc


def _project_matches(document: dict[str, Any], slug: str) -> list[dict[str, Any]]:
    try:
        projects = document["projects"]
    except (KeyError, TypeError) as exc:
        raise CaptureError("project search response lacks projects") from exc
    return [
        project
        for project in projects
        if project.get("slug") == slug and not project.get("is_deleted", False)
    ]


def _append_rule(
    ops: SystemOps,
    run_root: Path,
    rule: dict[str, Any],
) -> dict[str, Any]:
    raw, before = _request_json(ops, "GET", CONFIG_GET_PATH, {"configs": []})
    _bytes_receipt(run_root / "agent-config-before.json", raw)
    version, entries = _config_parts(before)
    if rule in entries:
        raise CaptureError("task live-config rule already exists")
    body = {
        "config": {"config": {"entries": [*entries, rule]}, "version": version}
    }
    _json_write(run_root / "agent-config-enable-request.json", body)
    _request_json(ops, "POST", CONFIG_SET_PATH, body)
    after_raw, after = _request_json(ops, "GET", CONFIG_GET_PATH, {"configs": []})
    _bytes_receipt(run_root / "agent-config-enable-after.json", after_raw)
    after_version, after_entries = _config_parts(after)
    if after_version <= version or after_entries.count(rule) != 1:
        raise CaptureError("task live-config rule enable readback failed")
    for entry in entries:
        if after_entries.count(entry) < entries.count(entry):
            raise CaptureError("live-config enable did not preserve existing entries")
    return {"enabled_version": after_version, "rule": rule}


def _restore_rule(
    ops: SystemOps,
    run_root: Path,
    rule: dict[str, Any],
) -> dict[str, Any]:
    raw, current = _request_json(ops, "GET", CONFIG_GET_PATH, {"configs": []})
    _bytes_receipt(run_root / "agent-config-restore-before.json", raw)
    version, entries = _config_parts(current)
    if entries.count(rule) != 1:
        raise CaptureError("refusing restore because exact owned rule is not present once")
    desired = list(entries)
    desired.remove(rule)
    body = {"config": {"config": {"entries": desired}, "version": version}}
    _json_write(run_root / "agent-config-restore-request.json", body)
    _request_json(ops, "POST", CONFIG_SET_PATH, body)
    after_raw, after = _request_json(ops, "GET", CONFIG_GET_PATH, {"configs": []})
    _bytes_receipt(run_root / "agent-config-restore-after.json", after_raw)
    after_version, after_entries = _config_parts(after)
    if after_version <= version or rule in after_entries or after_entries != desired:
        raise CaptureError("owned live-config rule restore readback failed")
    return {"restored_version": after_version, "preserved_entries": len(desired)}


def _stop_agent(
    ops: SystemOps, process: subprocess.Popen[Any], log: list[dict[str, Any]]
) -> None:
    if process.poll() is not None:
        log.append({"owned_pgid": process.pid, "final_returncode": process.returncode})
        if process.returncode not in (0, -signal.SIGINT):
            raise CaptureError(f"owned profiler exited early with {process.returncode}")
        return
    for sig, wait_seconds in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        ops.killpg(process.pid, sig)
        log.append({"signal": sig.name, "owned_pgid": process.pid})
        try:
            process.wait(timeout=wait_seconds)
            log.append(
                {"owned_pgid": process.pid, "final_returncode": process.returncode}
            )
            return
        except subprocess.TimeoutExpired:
            continue
    raise CaptureError("owned profiler process group survived SIGKILL")


def _run_owned_worker(
    ops: SystemOps,
    argv: list[str],
    stdout: Any,
    stderr: Any,
    timeout_seconds: int,
    cleanup_log: list[dict[str, Any]],
) -> tuple[int, int]:
    process = ops.popen(
        argv,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    if timed_out or ops.process_group_alive(process.pid):
        for sig, polls in ((signal.SIGTERM, 20), (signal.SIGKILL, 20)):
            if not ops.process_group_alive(process.pid):
                break
            ops.killpg(process.pid, sig)
            cleanup_log.append(
                {"kind": "worker", "owned_pgid": process.pid, "signal": sig.name}
            )
            for _ in range(polls):
                if not ops.process_group_alive(process.pid):
                    break
                ops.sleep(0.1)
        if ops.process_group_alive(process.pid):
            raise CaptureError("owned worker process group survived SIGKILL")
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as exc:
                raise CaptureError("owned worker launcher was not reaped") from exc
        cleanup_log.append(
            {"kind": "worker", "owned_pgid": process.pid, "group_empty": True}
        )
        if timed_out:
            raise CaptureError("owned worker exceeded its controller timeout")
        raise CaptureError("worker launcher exited with live descendants")
    cleanup_log.append(
        {"kind": "worker", "owned_pgid": process.pid, "group_empty": True}
    )
    return process.returncode, process.pid


def _wait_agent_ready(
    ops: SystemOps, process: subprocess.Popen[Any], log_path: Path
) -> None:
    deadline = ops.now() + dt.timedelta(seconds=AGENT_READY_SECONDS)
    while ops.now() < deadline:
        if process.poll() is not None:
            raise CaptureError(f"owned profiler died before readiness: {process.returncode}")
        data = log_path.read_bytes() if log_path.exists() else b""
        if b"Attached tracer program" in data and b"Watching for live config updates" in data:
            ops.sleep(1)
            if process.poll() is not None:
                raise CaptureError("owned profiler died during config grace")
            return
        ops.sleep(0.1)
    raise CaptureError("owned profiler did not reach both readiness markers")


def _verify_pins(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, str]:
    actual = {
        "profiler_binary": _sha256(args.profiler_binary),
        "injection_library": plan["source"]["injection_library"]["sha256"],
        "runtime_owner_checkpoint": plan["runtime_ownership"]["checkpoint"]["sha256"],
        "benchmark": plan["source"]["benchmark"]["sha256"],
        "profile_wrapper": plan["source"]["profile_wrapper"]["sha256"],
        "plan_builder": plan["source"]["plan_builder"]["sha256"],
    }
    expected = {
        "profiler_binary": args.profiler_sha256,
        "injection_library": args.injection_library_sha256,
        "runtime_owner_checkpoint": args.owner_checkpoint_sha256,
        "benchmark": args.benchmark_sha256,
        "profile_wrapper": args.profile_wrapper_sha256,
        "plan_builder": args.plan_builder_sha256,
    }
    for name, digest in actual.items():
        if digest != expected[name].lower():
            raise CaptureError(f"{name} SHA-256 pin mismatch")
    return actual


def _verify_source_closure(
    args: argparse.Namespace, ops: SystemOps
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", args.repo_commit):
        raise CaptureError("repository commit must be a full hexadecimal object ID")
    if bool(args.source_manifest) != bool(args.source_manifest_sha256):
        raise CaptureError(
            "--source-manifest and --source-manifest-sha256 must be supplied together"
        )
    if not args.source_manifest:
        git_result = ops.run(
            ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        repo_commit = git_result.stdout.strip().lower()
        if repo_commit != args.repo_commit.lower():
            raise CaptureError("repository commit pin mismatch")
        return {"mode": "git", "repo_commit": repo_commit}

    if not args.source_manifest.is_absolute():
        raise CaptureError("source manifest path must be absolute")
    manifest_path = args.source_manifest.resolve()
    if not manifest_path.is_file():
        raise CaptureError("source manifest is not an existing file")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if manifest_sha != args.source_manifest_sha256.lower():
        raise CaptureError("source manifest SHA-256 pin mismatch")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureError("source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"repo_commit", "files"}:
        raise CaptureError("source manifest must contain exactly repo_commit and files")
    if (
        not isinstance(manifest["repo_commit"], str)
        or manifest["repo_commit"].lower() != args.repo_commit.lower()
    ):
        raise CaptureError("source manifest repository commit pin mismatch")
    if not isinstance(manifest["files"], list):
        raise CaptureError("source manifest files must be a list")
    repo_root = args.repo_root.resolve()
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in manifest["files"]:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise CaptureError("source manifest file record has unexpected fields")
        relative = record["path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise CaptureError("source manifest path is not a POSIX relative path")
        lexical = Path(relative)
        if lexical.is_absolute() or ".." in lexical.parts or relative in seen:
            raise CaptureError("source manifest path escapes the repository or is duplicated")
        seen.add(relative)
        candidate = (repo_root / lexical).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise CaptureError("source manifest path escapes the repository") from exc
        if not candidate.is_file():
            raise CaptureError(f"source manifest file is missing: {relative}")
        candidate_raw = candidate.read_bytes()
        actual = {
            "path": relative,
            "bytes": len(candidate_raw),
            "sha256": hashlib.sha256(candidate_raw).hexdigest(),
        }
        if actual != record:
            raise CaptureError(f"source manifest file changed: {relative}")
        verified.append(actual)
    missing = sorted(REQUIRED_SOURCE_PATHS - seen)
    if missing:
        raise CaptureError(f"source manifest lacks required files: {', '.join(missing)}")
    closure = {
        "repo_commit": args.repo_commit.lower(),
        "files": sorted(verified, key=lambda item: item["path"]),
    }
    canonical = json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()
    return {
        "mode": "archive_manifest",
        "repo_commit": args.repo_commit.lower(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "source_closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(verified),
    }


def _verify_worker_result(
    result_path: Path, plan: dict[str, Any], expected_gpu_uuid: str
) -> dict[str, Any]:
    try:
        result = json.loads(result_path.read_bytes())
        runtime = result["runtime"]
        device = result["device"]
        provenance = result["provenance"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"invalid worker result: {result_path}") from exc
    pid = runtime.get("pid")
    if result.get("status") != "pass":
        raise CaptureError("worker result status is not pass")
    if runtime.get("torch_version") != EXPECTED_TORCH:
        raise CaptureError(f"worker Torch runtime is not {EXPECTED_TORCH}")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CaptureError("worker result lacks a valid source PID")
    for field in ("expected_gpu_uuid", "observed_uuid", "cuda_visible_devices"):
        if _normalize_gpu(device.get(field)) != _normalize_gpu(expected_gpu_uuid):
            raise CaptureError(f"worker result {field} does not match assigned GPU")
    if provenance.get("worker_source_sha256") != plan["source"]["benchmark"]["sha256"]:
        raise CaptureError("worker source hash does not match the planned benchmark")
    return result


def _export_arm(
    ops: SystemOps,
    root: Path,
    project_id: str,
    project_slug: str,
    arm: dict[str, Any],
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    export_root = root / "zymtrace-export"
    export_root.mkdir()
    event_filter = {
        "start_ts": arm["start_ms"] - 1000,
        "end_ts": arm["end_ms"] + 1000,
        "event_kind": "cuda",
    }
    event_body = {
        "filter": event_filter,
        "approx_limit": 100000,
        "exclude_idle_frames": True,
    }
    raw, events = _request_json(
        ops,
        "POST",
        f"/public/api/v1/projects/{project_id}/events/raw",
        event_body,
    )
    event_receipt = _bytes_receipt(export_root / "cuda-events.raw.json", raw)
    flame_body = {
        "filter": event_filter,
        "num_samples": 100000,
        "synthetic_frames": ["gpu", "thread", "main_exe", "script_name"],
    }
    flame_raw, flame = _request_json(
        ops,
        "POST",
        f"/public/api/v1/project/{project_id}/flamegraph/json",
        flame_body,
    )
    if (
        not isinstance(flame, dict)
        or not isinstance(flame.get("fg"), dict)
        or not flame["fg"].get("nodes")
    ):
        raise CaptureError("flamegraph export has no native JSON nodes")
    flame_receipt = _bytes_receipt(export_root / "cuda-flamegraph.raw.json", flame_raw)
    audit = _validate_event_export(events, arm["source_pid"], expected_gpu_uuid)
    audit.update(
        {
            "project_slug": project_slug,
            "project_id": project_id,
            "filter": event_filter,
            "filter_clock_units": "Unix epoch milliseconds, inclusive start and exclusive end",
            "raw_events": event_receipt,
            "raw_flamegraph": flame_receipt,
            "performance_claim": False,
            "visibility_only": True,
        }
    )
    _json_write(export_root / "audit.json", audit)
    return audit


def execute(args: argparse.Namespace, ops: SystemOps | None = None) -> dict[str, Any]:
    ops = ops or SystemOps()
    if args.backend_base != BACKEND_BASE or args.collection_agent != COLLECTION_AGENT:
        raise CaptureError("controller is pinned to the local 127.0.0.1:18080 backend")
    if args.tls_mode != "disabled":
        raise CaptureError("controller requires --tls-mode disabled")
    if not args.allow_expiring_highres_rule:
        raise CaptureError("explicit --allow-expiring-highres-rule is required")
    if not CPU_AFFINITY_RE.fullmatch(args.cpu_affinity):
        raise CaptureError("CPU affinity must be an explicit taskset CPU list")
    if args.timeout_seconds > 240:
        raise CaptureError("worker timeout must be at most 240 seconds")
    if (
        not args.profiler_binary.is_absolute()
        or not args.profiler_binary.is_file()
        or not os.access(args.profiler_binary, os.X_OK)
    ):
        raise CaptureError("profiler binary must be an existing absolute executable")
    plan = prepare_capture.build_plan(args)
    run_root = Path(plan["run_root"])
    run_root.mkdir(mode=0o750)
    actual_start = ops.now()
    supplied_hard = _utc(args.hard_cleanup_deadline)
    hard_deadline = min(
        supplied_hard, actual_start + dt.timedelta(seconds=GLOBAL_LIMIT_SECONDS)
    )
    handback: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PARTIAL",
        "purpose": "Zymtrace visibility only, not performance qualification",
        "actual_start": _iso(actual_start),
        "supplied_hard_cleanup_deadline": _iso(supplied_hard),
        "effective_hard_cleanup_deadline": _iso(hard_deadline),
        "project": args.zymtrace_project,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "cpu_affinity": args.cpu_affinity,
        "arms": [],
        "commands": [],
        "cleanup": [],
        "errors": [],
    }
    agent = None
    agent_log_handle = None
    rule = None
    worker_ran = False
    agent_flushed = False
    handback_path = run_root / "HANDOFF-PARTIAL.json"
    try:
        if actual_start >= hard_deadline:
            raise CaptureError("hard cleanup deadline has already passed")
        not_before = _utc(args.not_before)
        latest_start = _utc(args.latest_gpu_start)
        if not_before > latest_start:
            raise CaptureError("not-before is later than latest GPU start")
        if ops.now() < not_before:
            wait = (not_before - ops.now()).total_seconds()
            if ops.now() + dt.timedelta(seconds=wait + FINAL_RESERVE_SECONDS) >= hard_deadline:
                raise CaptureError("not-before wait would consume the cleanup reserve")
            ops.sleep(wait)
        if ops.now() > latest_start:
            raise CaptureError("latest GPU start has passed")
        handback["source_closure"] = _verify_source_closure(args, ops)
        handback["repo_commit"] = args.repo_commit.lower()
        handback["hash_pins"] = _verify_pins(args, plan)
        if ops.competing_agents(args.profiler_binary):
            raise CaptureError("a competing Zymtrace profiler agent is already running")
        gpu_check = ops.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
        )
        visible_gpus = [line.strip() for line in gpu_check.stdout.splitlines() if line.strip()]
        if sum(_normalize_gpu(item) == _normalize_gpu(args.expected_gpu_uuid) for item in visible_gpus) != 1:
            raise CaptureError("assigned GPU UUID is not present exactly once")
        backend = ops.run(
            ["docker", "ps", "--filter",
             "label=com.docker.compose.project=aiperf-pr38-zymtrace",
             "--format", "{{.Names}}|{{.Image}}|{{.ID}}"],
            text=True, capture_output=True, check=True,
        )
        _bytes_receipt(run_root / "backend-container-images.txt", backend.stdout.encode())
        images = [line.split("|")[1] for line in backend.stdout.splitlines()
                  if len(line.split("|")) == 3 and "zymtrace" in line.split("|")[1]]
        if len(images) < 3 or any(image.rsplit(":", 1)[-1] != VERSION for image in images):
            raise CaptureError("running Zymtrace container image tags do not prove version 26.10.1")
        handback["backend_version_source"] = "running container image tags and IDs"
        projects_raw, projects = _request_json(
            ops, "POST", PROJECTS_PATH, {"pagination": {"page_size": 100, "page": 1}}
        )
        _bytes_receipt(run_root / "projects-preflight.raw.json", projects_raw)
        if _project_matches(projects, args.zymtrace_project):
            raise CaptureError("capture project is not fresh")
        agent_argv = [
            "taskset", "--cpu-list", args.cpu_affinity, str(args.profiler_binary),
            f"-collection-agent={COLLECTION_AGENT}", "-disable-tls",
            f"-project={args.zymtrace_project}", "-tracers=python,cuda",
            "-enable-gpu-metrics=true", "-nvml-auto-scan", "-reporter-interval=1s",
            "-collect-metrics-interval=5s", "-export-metrics-interval=5s",
            "-refresh-config-interval=1s",
        ]
        agent_log_path = run_root / "zymtrace-profiler.log"
        agent_log_handle = agent_log_path.open("wb")
        handback["commands"].append({"kind": "profiler", "argv": agent_argv})
        agent = ops.popen(
            agent_argv,
            stdout=agent_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_agent_ready(ops, agent, agent_log_path)
        project_raw, project_document = _request_json(
            ops, "POST", PROJECTS_PATH, {"pagination": {"page_size": 100, "page": 1}}
        )
        _bytes_receipt(run_root / "projects-after-agent.raw.json", project_raw)
        matches = _project_matches(project_document, args.zymtrace_project)
        if len(matches) != 1:
            raise CaptureError("agent did not create exactly one fresh capture project")
        project_id = matches[0].get("id")
        if not isinstance(project_id, str) or not project_id:
            raise CaptureError("fresh capture project lacks an ID")
        expires = hard_deadline
        rule = {
            "expression": "host.name == " + json.dumps(ops.hostname()),
            "project": project_id,
            "expires_at": _iso(expires),
            "settings": [{"setting": {"HighResProfiling": True}}],
        }
        handback["high_resolution_scope"] = {
            "host": ops.hostname(),
            "project_id": project_id,
            "expires_at": _iso(expires),
            "pid_live_config_scope": "unsupported because no PID CEL field is documented",
            "pid_export_scope": "required and validated per worker",
        }
        handback["high_resolution_enable"] = _append_rule(ops, run_root, rule)
        ops.sleep(1)
        for arm_index, arm_plan in enumerate(plan["arms"]):
            if arm_index == 0 and ops.now() > latest_start:
                raise CaptureError("latest GPU start passed before the first worker")
            seconds = _worker_timeout(ops.now(), hard_deadline, args.timeout_seconds)
            argv = [
                "taskset", "--cpu-list", args.cpu_affinity,
                *_replace_timeout(arm_plan["argv"], seconds),
            ]
            arm_root = Path(arm_plan["run_root"])
            arm_root.mkdir(parents=True)
            stdout_path = arm_root / "worker.stdout.log"
            stderr_path = arm_root / "worker.stderr.log"
            arm_start = ops.now()
            label = (
                "without nested graph-break support"
                if arm_plan["arm"] == "disabled"
                else "with nested graph-break support"
            )
            command_record = {
                "kind": "worker",
                "arm": arm_plan["arm"],
                "label": label,
                "argv": argv,
                "shell_command": shlex.join(argv),
                "timeout_seconds": seconds,
            }
            handback["commands"].append(command_record)
            result_path = arm_root / "benchmark-result/result.json"
            arm_record = {
                "arm": arm_plan["arm"],
                "label": label,
                "exit_code": None,
                "start_ms": int(arm_start.timestamp() * 1000),
                "end_ms": None,
                "clock_units": "Unix epoch milliseconds",
                "source_pid": None,
                "result_status": None,
                "result_path": str(result_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            handback["arms"].append(arm_record)
            _json_write(run_root / f"{arm_plan['arm']}-DISPATCH.json", arm_record)
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                worker_ran = True
                returncode, worker_pgid = _run_owned_worker(
                    ops,
                    argv,
                    stdout,
                    stderr,
                    seconds + KILL_GRACE_SECONDS + 5,
                    handback["cleanup"],
                )
            arm_end = ops.now()
            arm_record.update(
                {
                    "exit_code": returncode,
                    "owned_launcher_pgid": worker_pgid,
                    "end_ms": int(arm_end.timestamp() * 1000),
                    "stdout_sha256": _sha256(stdout_path),
                    "stderr_sha256": _sha256(stderr_path),
                }
            )
            result = _verify_worker_result(result_path, plan, args.expected_gpu_uuid)
            arm_record.update(
                {
                    "source_pid": result["runtime"]["pid"],
                    "result_status": result["status"],
                    "torch_version": result["runtime"]["torch_version"],
                    "assigned_gpu_uuid": result["device"]["observed_uuid"],
                    "result_sha256": _sha256(result_path),
                }
            )
            if returncode != 0:
                raise CaptureError(f"{label} worker exited {returncode}")
        ops.sleep(5)
        agent_flushed = True
        _stop_agent(ops, agent, handback["cleanup"])
        agent = None
        agent_log_handle.close()
        agent_log_handle = None
        final_projects_raw, final_projects = _request_json(
            ops, "POST", PROJECTS_PATH, {"pagination": {"page_size": 100, "page": 1}}
        )
        _bytes_receipt(run_root / "projects-export.raw.json", final_projects_raw)
        matches = _project_matches(final_projects, args.zymtrace_project)
        if len(matches) != 1 or matches[0].get("id") != project_id:
            raise CaptureError("capture project identity changed before export")
        for arm in handback["arms"]:
            arm["export"] = _export_arm(
                ops,
                Path(plan["run_root"]) / arm["arm"],
                project_id,
                args.zymtrace_project,
                arm,
                args.expected_gpu_uuid,
            )
        handback["status"] = "COMPLETE"
    except BaseException as exc:
        handback["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if agent is not None:
            try:
                if worker_ran and not agent_flushed:
                    ops.sleep(5)
                    agent_flushed = True
                _stop_agent(ops, agent, handback["cleanup"])
            except BaseException as exc:
                handback["errors"].append(f"agent cleanup: {type(exc).__name__}: {exc}")
        if agent_log_handle is not None:
            agent_log_handle.close()
        agent_log_path = run_root / "zymtrace-profiler.log"
        if agent_log_path.is_file():
            handback["profiler_log"] = {
                "path": str(agent_log_path),
                "sha256": _sha256(agent_log_path),
                "bytes": agent_log_path.stat().st_size,
            }
        for arm in handback["arms"]:
            for field in ("stdout", "stderr"):
                path = Path(arm[field])
                if path.is_file():
                    arm.setdefault(f"{field}_sha256", _sha256(path))
        if rule is not None:
            try:
                handback["high_resolution_restore"] = _restore_rule(
                    ops, run_root, rule
                )
            except BaseException as exc:
                handback["errors"].append(f"rule cleanup: {type(exc).__name__}: {exc}")
        handback["actual_end"] = _iso(ops.now())
        if ops.now() > hard_deadline:
            handback["errors"].append("effective hard cleanup deadline exceeded")
        if handback["errors"]:
            handback["status"] = "PARTIAL"
        final_path = run_root / f"HANDOFF-{handback['status']}.json"
        _json_write(final_path, handback)
        if handback_path != final_path and handback_path.exists():
            handback_path.unlink()
    return handback


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--injection-library", required=True, type=Path)
    parser.add_argument("--runtime-owner-checkpoint", required=True, type=Path)
    parser.add_argument("--collection-agent", default=COLLECTION_AGENT)
    parser.add_argument("--zymtrace-project", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--tls-mode", choices=("tls", "disabled"), default="disabled")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--backend-base", default=BACKEND_BASE)
    parser.add_argument("--profiler-binary", required=True, type=Path)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--profiler-sha256", required=True)
    parser.add_argument("--injection-library-sha256", required=True)
    parser.add_argument("--owner-checkpoint-sha256", required=True)
    parser.add_argument("--benchmark-sha256", required=True)
    parser.add_argument("--profile-wrapper-sha256", required=True)
    parser.add_argument("--plan-builder-sha256", required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--latest-gpu-start", required=True)
    parser.add_argument("--hard-cleanup-deadline", required=True)
    parser.add_argument("--allow-expiring-highres-rule", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    def terminate(signum: int, frame: Any) -> None:
        raise CaptureError(f"external termination signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = execute(args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
