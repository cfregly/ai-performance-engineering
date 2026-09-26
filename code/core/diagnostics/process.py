"""Run a bounded local command and clean up only processes it owns."""

from __future__ import annotations

import math
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

_OWNER_ENV = "AISP_DIAGNOSTIC_PROCESS_OWNER"


def _finite_interval(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}")
    return result


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    create_time: float


def _identity(process: psutil.Process) -> _ProcessIdentity | None:
    try:
        return _ProcessIdentity(process.pid, process.create_time())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def _resolve(identity: _ProcessIdentity) -> psutil.Process | None:
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.create_time:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


class _OwnedProcesses:
    """Retain process identities even after a child changes parent or session."""

    def __init__(self, root: psutil.Process, owner_token: str) -> None:
        self._owner_token = owner_token
        self._lock = threading.Lock()
        self._identities: dict[int, _ProcessIdentity] = {}
        self.remember(root)

    def remember(self, process: psutil.Process) -> None:
        identity = _identity(process)
        if identity is None:
            return
        with self._lock:
            self._identities[identity.pid] = identity

    def snapshot(self) -> list[_ProcessIdentity]:
        with self._lock:
            return list(self._identities.values())

    def discover_descendants(self) -> None:
        for identity in self.snapshot():
            process = _resolve(identity)
            if process is None:
                continue
            try:
                children = process.children(recursive=True)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            for child in children:
                self.remember(child)

    def discover_owner_token(self) -> None:
        """Recover owned children that reparented or started a new session."""
        for process in psutil.process_iter():
            try:
                if process.environ().get(_OWNER_ENV) == self._owner_token:
                    self.remember(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

    def live(self) -> list[psutil.Process]:
        return [process for identity in self.snapshot() if (process := _resolve(identity))]


def _track_descendants(owned: _OwnedProcesses, stop: threading.Event, poll_seconds: float) -> None:
    while not stop.is_set():
        owned.discover_descendants()
        stop.wait(poll_seconds)


def _signal_live(
    owned: _OwnedProcesses,
    action: str,
    errors: list[str],
) -> list[int]:
    sent: list[int] = []
    for process in owned.live():
        try:
            if action == "sigint":
                process.send_signal(signal.SIGINT)
            elif action == "terminate":
                process.terminate()
            elif action == "kill":
                process.kill()
            else:  # pragma: no cover - internal programming error
                raise ValueError(f"Unknown process action: {action}")
            sent.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
            errors.append(f"{action} pid {process.pid}: {type(exc).__name__}")
    return sent


def _wait_for_owned_exit(owned: _OwnedProcesses, timeout_seconds: float) -> None:
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while owned.live() and time.monotonic() < deadline:
        time.sleep(min(0.02, max(deadline - time.monotonic(), 0.0)))


def _drain_root(
    process: psutil.Popen[Any],
    timeout_seconds: float,
) -> tuple[str, str] | None:
    try:
        stdout, stderr = process.communicate(timeout=max(timeout_seconds, 0.001))
        return stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        return None


def run_owned_process(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    interrupt_grace_seconds: float = 2.0,
    terminate_grace_seconds: float = 1.0,
    poll_seconds: float = 0.02,
) -> dict[str, Any]:
    """Run ``argv`` and reap its retained local process identities after timeout."""
    timeout_seconds = _finite_interval(timeout_seconds, "timeout_seconds", positive=True)
    interrupt_grace_seconds = _finite_interval(
        interrupt_grace_seconds, "interrupt_grace_seconds", positive=False
    )
    terminate_grace_seconds = _finite_interval(
        terminate_grace_seconds, "terminate_grace_seconds", positive=False
    )
    poll_seconds = _finite_interval(poll_seconds, "poll_seconds", positive=True)
    command = [str(value) for value in argv]
    owner_token = uuid.uuid4().hex
    child_env = dict(os.environ if env is None else env)
    child_env[_OWNER_ENV] = owner_token
    started_at = time.time()
    started_monotonic = time.monotonic()
    try:
        process: psutil.Popen[Any] = psutil.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return {
            "command": shlex.join(command),
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "timeout_seconds": timeout_seconds,
            "timeout_hit": False,
            "duration_seconds": round(time.time() - started_at, 2),
            "error": str(exc),
        }

    owned = _OwnedProcesses(process, owner_token)
    stop_tracking = threading.Event()
    tracker = threading.Thread(
        target=_track_descendants,
        args=(owned, stop_tracking, poll_seconds),
        daemon=True,
    )
    tracker.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "command": shlex.join(command),
            "returncode": process.returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timeout_seconds": timeout_seconds,
            "timeout_hit": False,
            "duration_seconds": round(time.monotonic() - started_monotonic, 2),
        }
    except subprocess.TimeoutExpired:
        owned.discover_descendants()
        owned.discover_owner_token()
        cleanup_errors: list[str] = []
        sigint_sent = _signal_live(owned, "sigint", cleanup_errors)
        drained = _drain_root(process, interrupt_grace_seconds)
        _wait_for_owned_exit(owned, interrupt_grace_seconds if drained is not None else 0.0)

        owned.discover_descendants()
        owned.discover_owner_token()
        terminate_sent = _signal_live(owned, "terminate", cleanup_errors)
        if drained is None:
            drained = _drain_root(process, terminate_grace_seconds)
        _wait_for_owned_exit(owned, terminate_grace_seconds)

        owned.discover_descendants()
        owned.discover_owner_token()
        kill_sent = _signal_live(owned, "kill", cleanup_errors)
        if drained is None:
            drained = _drain_root(process, 1.0)
        _wait_for_owned_exit(owned, 1.0)
        survivors = sorted(process.pid for process in owned.live())
        stdout, stderr = drained or ("", "")
        process_returncode = process.poll()
        return {
            "command": shlex.join(command),
            "returncode": 124,
            "process_returncode": process_returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_seconds": timeout_seconds,
            "timeout_hit": True,
            "duration_seconds": round(time.monotonic() - started_monotonic, 2),
            "error": f"Timed out after {timeout_seconds}s",
            "termination": {
                "tracked_processes": len(owned.snapshot()),
                "sigint_sent": sorted(sigint_sent),
                "terminate_sent": sorted(terminate_sent),
                "kill_sent": sorted(kill_sent),
                "survivors": survivors,
                "errors": cleanup_errors,
            },
        }
    finally:
        stop_tracking.set()
        tracker.join(timeout=max(poll_seconds * 2, 0.1))
