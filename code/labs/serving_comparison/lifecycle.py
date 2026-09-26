"""Owned local process lifecycle with fixed-fleet GPU custody checks."""

from __future__ import annotations

import asyncio
import json
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


class _AllocationPendingError(RuntimeError):
    """The owned process group has not allocated the complete fixed fleet yet."""


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


def _application_clock_evidence(profile: ComparisonProfile) -> dict[str, Any]:
    mapping = _gpu_uuid_map()
    missing = sorted(gpu_id for gpu_id in profile.gpu_ids if gpu_id not in mapping)
    if missing:
        raise ConfigError(f"fixed GPU ids are not visible to nvidia-smi: {missing}")
    output = _run_nvidia_smi(
        [
            "--query-gpu=uuid,clocks.applications.graphics",
            "--format=csv,noheader,nounits",
        ]
    )
    observed_by_uuid: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            raise ConfigError("nvidia-smi returned an invalid application clock row")
        try:
            observed_by_uuid[parts[0]] = int(parts[1])
        except ValueError as exc:
            raise ConfigError("nvidia-smi returned a nonnumeric application clock") from exc
    rows: list[dict[str, Any]] = []
    for gpu_id in profile.gpu_ids:
        gpu_uuid = mapping[gpu_id]
        expected_mhz = profile.application_clocks_mhz[gpu_id]
        observed_mhz = observed_by_uuid.get(gpu_uuid)
        if observed_mhz != expected_mhz:
            raise ConfigError(
                f"GPU {gpu_id} application clock mismatch: "
                f"expected {expected_mhz} MHz, observed {observed_mhz}"
            )
        rows.append(
            {
                "gpu_id": gpu_id,
                "gpu_uuid": gpu_uuid,
                "expected_mhz": expected_mhz,
                "observed_mhz": observed_mhz,
            }
        )
    return {"source": "nvidia_smi_applications_graphics", "gpus": rows}


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
    clock_evidence = _application_clock_evidence(profile)
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
        "application_clocks": clock_evidence,
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
        raise _AllocationPendingError(
            f"{arm.arm_id} has not allocated every fixed-fleet GPU: {missing}"
        )
    return {
        "evidence": "nvidia_smi_process_ownership",
        "root_pid": root_pid,
        "owned_pids": sorted(owned),
        "fixed_gpu_ids": list(profile.gpu_ids),
        "resolved_gpu_uuids": sorted(gpu_uuids),
        "compute_processes": occupants,
        "application_clocks": _application_clock_evidence(profile),
        "observed_unix_s": time.time(),
    }


async def start_arm(
    client: Any,
    arm: ArmProfile,
    profile: ComparisonProfile,
    artifact_dir: Path,
) -> ActiveProcess:
    """Start one declared argv without a shell and verify readiness and GPU custody."""
    launch_binding: dict[str, Any] | None = None
    start_command = arm.lifecycle["start_command"]
    if profile.mode == "engine" and arm.architecture == "prefill_decode":
        from .provenance import bind_pd_launch

        if arm.pd_provenance is None:
            raise ConfigError(f"{arm.arm_id} has no P/D provenance")
        launch_binding = bind_pd_launch(
            engine=arm.engine,
            endpoint=arm.endpoint,
            lifecycle=arm.lifecycle,
            provenance=arm.pd_provenance,
            model=profile.model,
        )
        start_command = launch_binding["launcher_argv"]
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
            *start_command,
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
    readiness_urls = list(
        dict.fromkeys(
            [*arm.lifecycle["ready_urls"], *(probe["url"] for probe in arm.identity_probes)]
        )
    )
    last_error = "readiness endpoints have not responded"
    try:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise ConfigError(
                    f"{arm.arm_id} serving process exited before readiness with code {process.returncode}"
                )
            all_ready = True
            for url in readiness_urls:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    all_ready = False
                    break
                try:
                    response = await client.get(url, timeout=min(2.0, remaining_s))
                except Exception as exc:
                    last_error = f"readiness endpoint {url} failed: {exc}"
                    all_ready = False
                    break
                if response.status_code >= 400:
                    last_error = f"readiness endpoint {url} returned HTTP {response.status_code}"
                    all_ready = False
                    break
            if all_ready:
                try:
                    allocation = _active_allocation(profile, arm, process.pid)
                except _AllocationPendingError as exc:
                    last_error = str(exc)
                    await asyncio.sleep(0.1)
                    continue
                if launch_binding is not None:
                    from .provenance import verify_pd_allocation

                    stdout_handle.flush()
                    allocation["pd_launch_binding"] = verify_pd_allocation(
                        launch_binding,
                        stdout_path=artifact_dir / "server.stdout.log",
                        allocation=allocation,
                        gpu_uuid_map=_gpu_uuid_map(),
                        descendants=_descendants,
                    )
                allocation["idle_before_start"] = idle_evidence
                return ActiveProcess(
                    process=process,
                    process_group_id=process_group_id,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    allocation_evidence=allocation,
                )
            await asyncio.sleep(0.1)
        raise ConfigError(f"{arm.arm_id} did not become ready: {last_error}")
    except BaseException as primary_error:
        cleanup_errors: list[str] = []
        try:
            await _terminate_process(
                process,
                process_group_id,
                float(arm.lifecycle["shutdown_timeout_s"]),
            )
        except Exception as exc:
            cleanup_errors.append(f"process termination failed: {exc}")
        cleanup_evidence: dict[str, Any] | None = None
        try:
            cleanup_evidence = require_fleet_idle(profile)
        except Exception as exc:
            cleanup_errors.append(f"fleet release check failed: {exc}")
        cleanup_payload = {
            "arm_id": arm.arm_id,
            "cleanup_complete": not cleanup_errors,
            "errors": cleanup_errors,
            "released_allocation": cleanup_evidence,
        }
        try:
            (artifact_dir / "startup-cleanup.json").write_text(
                json.dumps(cleanup_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            cleanup_errors.append(f"cleanup evidence write failed: {exc}")
        stdout_handle.close()
        stderr_handle.close()
        if cleanup_errors:
            raise ConfigError(
                f"{arm.arm_id} startup failed and cleanup proof was incomplete: "
                + "; ".join(cleanup_errors)
            ) from primary_error
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
    cleanup_errors: list[str] = []
    try:
        await _terminate_process(
            active.process,
            active.process_group_id,
            float(arm.lifecycle["shutdown_timeout_s"]),
        )
    except Exception as exc:
        cleanup_errors.append(f"process termination failed: {exc}")
    for name, handle in (
        ("stdout", active.stdout_handle),
        ("stderr", active.stderr_handle),
    ):
        try:
            handle.close()
        except OSError as exc:
            cleanup_errors.append(f"{name} log close failed: {exc}")
    evidence: dict[str, Any] | None = None
    try:
        evidence = require_fleet_idle(profile)
    except Exception as exc:
        cleanup_errors.append(f"fleet release check failed: {exc}")
    if cleanup_errors:
        raise ConfigError(
            f"{arm.arm_id} cleanup proof was incomplete: " + "; ".join(cleanup_errors)
        )
    assert evidence is not None
    evidence.update(
        {
            "stopped_root_pid": active.process.pid,
            "returncode": active.process.returncode,
            "observed_unix_s": time.time(),
        }
    )
    return evidence
