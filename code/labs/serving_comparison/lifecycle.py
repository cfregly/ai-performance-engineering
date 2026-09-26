"""Owned local process lifecycle with fixed-fleet GPU custody checks."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .schema import ArmProfile, ComparisonProfile, ConfigError


@dataclass
class ActiveProcess:
    process: asyncio.subprocess.Process
    process_group_id: int
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    allocation_evidence: dict[str, Any]


def _run_nvidia_smi(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise ConfigError("engine lifecycle requires nvidia-smi on the serving host") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConfigError("nvidia-smi timed out while checking fixed-fleet custody") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ConfigError(f"nvidia-smi failed while checking fixed-fleet custody: {detail}")
    return completed.stdout


def _gpu_uuid_map() -> dict[str, str]:
    output = _run_nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    mapping: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            raise ConfigError("nvidia-smi returned an invalid GPU identity row")
        index, uuid = parts
        mapping[index] = uuid
        mapping[uuid] = uuid
    return mapping


def _compute_apps() -> list[dict[str, Any]]:
    output = _run_nvidia_smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"])
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip() or "No running processes found" in line:
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            raise ConfigError("nvidia-smi returned an invalid compute process row")
        try:
            pid = int(parts[0])
        except ValueError as exc:
            raise ConfigError("nvidia-smi returned a nonnumeric compute process id") from exc
        rows.append({"pid": pid, "gpu_uuid": parts[1]})
    return rows


def _process_tree() -> dict[int, set[int]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ConfigError("ps failed while checking serving process ownership")
    children: dict[int, set[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, parent = (int(part) for part in parts)
        children.setdefault(parent, set()).add(pid)
    return children


def _descendants(root_pid: int) -> set[int]:
    children = _process_tree()
    owned = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, set()):
            if child not in owned:
                owned.add(child)
                pending.append(child)
    return owned


def _resolved_gpu_uuids(profile: ComparisonProfile) -> set[str]:
    mapping = _gpu_uuid_map()
    missing = sorted(gpu_id for gpu_id in profile.gpu_ids if gpu_id not in mapping)
    if missing:
        raise ConfigError(f"fixed GPU ids are not visible to nvidia-smi: {missing}")
    return {mapping[gpu_id] for gpu_id in profile.gpu_ids}


def require_fleet_idle(profile: ComparisonProfile) -> dict[str, Any]:
    """Require no compute process on the declared fleet before an engine arm starts."""
    if profile.mode != "engine":
        return {
            "evidence": "process_only",
            "fixed_gpu_ids": list(profile.gpu_ids),
            "compute_processes": [],
            "observed_unix_s": time.time(),
        }
    gpu_uuids = _resolved_gpu_uuids(profile)
    occupants = [row for row in _compute_apps() if row["gpu_uuid"] in gpu_uuids]
    if occupants:
        raise ConfigError(
            "fixed GPU fleet is not idle before activation: "
            + ", ".join(f"pid={row['pid']} gpu={row['gpu_uuid']}" for row in occupants)
        )
    return {
        "evidence": "nvidia_smi",
        "fixed_gpu_ids": list(profile.gpu_ids),
        "resolved_gpu_uuids": sorted(gpu_uuids),
        "compute_processes": [],
        "observed_unix_s": time.time(),
    }


def _active_allocation(
    profile: ComparisonProfile, arm: ArmProfile, root_pid: int
) -> dict[str, Any]:
    owned = _descendants(root_pid)
    if profile.mode != "engine":
        return {
            "evidence": "process_only",
            "root_pid": root_pid,
            "owned_pids": sorted(owned),
            "fixed_gpu_ids": list(profile.gpu_ids),
            "observed_unix_s": time.time(),
        }
    gpu_uuids = _resolved_gpu_uuids(profile)
    occupants = [row for row in _compute_apps() if row["gpu_uuid"] in gpu_uuids]
    foreign = [row for row in occupants if row["pid"] not in owned]
    if foreign:
        raise ConfigError(f"{arm.arm_id} fixed GPU fleet has foreign compute processes: {foreign}")
    occupied_uuids = {row["gpu_uuid"] for row in occupants}
    if occupied_uuids != gpu_uuids:
        missing = sorted(gpu_uuids - occupied_uuids)
        raise ConfigError(f"{arm.arm_id} did not allocate every fixed-fleet GPU: {missing}")
    return {
        "evidence": "nvidia_smi_process_ownership",
        "root_pid": root_pid,
        "owned_pids": sorted(owned),
        "fixed_gpu_ids": list(profile.gpu_ids),
        "resolved_gpu_uuids": sorted(gpu_uuids),
        "compute_processes": occupants,
        "observed_unix_s": time.time(),
    }


async def start_arm(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    artifact_dir: Path,
) -> ActiveProcess:
    """Start one declared argv without a shell and verify readiness and GPU custody."""
    idle_evidence = require_fleet_idle(profile)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = (artifact_dir / "server.stdout.log").open("wb")
    stderr_handle = (artifact_dir / "server.stderr.log").open("wb")
    environment = os.environ.copy()
    for name in arm.lifecycle["env_passthrough"]:
        if name not in os.environ:
            stdout_handle.close()
            stderr_handle.close()
            raise ConfigError(f"{arm.arm_id} required environment variable is unset: {name}")
    environment.update(arm.lifecycle["environment"])
    try:
        process = await asyncio.create_subprocess_exec(
            *arm.lifecycle["start_command"],
            cwd=arm.lifecycle["working_directory"],
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        stdout_handle.close()
        stderr_handle.close()
        raise ConfigError(f"{arm.arm_id} launch failed: {exc}") from exc
    process_group_id = process.pid
    deadline = time.monotonic() + arm.lifecycle["timeout_s"]
    last_error = "ready endpoint has not responded"
    try:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise ConfigError(
                    f"{arm.arm_id} serving process exited before readiness with code {process.returncode}"
                )
            try:
                response = await client.get(arm.lifecycle["ready_url"])
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(0.1)
                continue
            if response.status_code < 400:
                allocation = _active_allocation(profile, arm, process.pid)
                allocation["idle_before_start"] = idle_evidence
                return ActiveProcess(
                    process=process,
                    process_group_id=process_group_id,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    allocation_evidence=allocation,
                )
            last_error = f"ready endpoint returned HTTP {response.status_code}"
            await asyncio.sleep(0.1)
        raise ConfigError(f"{arm.arm_id} did not become ready: {last_error}")
    except BaseException:
        await _terminate_process(
            process,
            process_group_id,
            float(arm.lifecycle["shutdown_timeout_s"]),
        )
        stdout_handle.close()
        stderr_handle.close()
        raise


async def _terminate_process(
    process: asyncio.subprocess.Process,
    process_group_id: int,
    timeout_s: float,
) -> None:
    def group_exists() -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    if not group_exists():
        if process.returncode is None:
            await process.wait()
        return
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while group_exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if group_exists():
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        kill_deadline = time.monotonic() + min(timeout_s, 5.0)
        while group_exists() and time.monotonic() < kill_deadline:
            await asyncio.sleep(0.05)
    if group_exists():
        raise ConfigError(f"owned process group {process_group_id} remained alive after SIGKILL")
    if process.returncode is None:
        await process.wait()


async def stop_arm(
    active: ActiveProcess, arm: ArmProfile, profile: ComparisonProfile
) -> dict[str, Any]:
    """Stop only the process group launched by this tool and prove fleet release."""
    await _terminate_process(
        active.process,
        active.process_group_id,
        float(arm.lifecycle["shutdown_timeout_s"]),
    )
    active.stdout_handle.close()
    active.stderr_handle.close()
    evidence = require_fleet_idle(profile)
    evidence.update(
        {
            "stopped_root_pid": active.process.pid,
            "returncode": active.process.returncode,
            "observed_unix_s": time.time(),
        }
    )
    return evidence
