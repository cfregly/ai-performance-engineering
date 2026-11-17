"""Run an owned process and let its cancellation cleanup finish."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil


class CooperativeProcessInterrupted(KeyboardInterrupt):
    """Cancellation with captured output after the owned group stops running."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("Cancelled after the owned process group stopped running")
        self.result = result


def _live_process_group_members(process_group_id: int) -> list[int]:
    """Return processes that can still execute in this owned POSIX group."""
    members = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) != process_group_id:
                continue
            if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                members.append(process.pid)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return sorted(members)


def run_cooperative_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Capture output without a deadline and forward one SIGINT on cancellation."""
    if os.name != "posix":
        raise RuntimeError("SKIPPED: cooperative subprocess execution requires POSIX process groups")
    if isinstance(command, (str, bytes)) or not command or any(not isinstance(value, str) for value in command):
        raise ValueError("command must be a nonempty argument vector of strings")
    argv = list(command)
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        return communicate_cooperatively(process, process_group_id=process.pid, started_at=started)
    except CooperativeProcessInterrupted:
        raise
    except KeyboardInterrupt:
        return communicate_cooperatively(
            process, process_group_id=process.pid, started_at=started,
            cancellation_requested=True,
        )


def communicate_cooperatively(
    process: subprocess.Popen[str],
    *,
    process_group_id: int,
    input: str | None = None,
    started_at: float | None = None,
    cancellation_requested: bool = False,
) -> dict[str, Any]:
    """Wait for the owned group, excluding zombies and descendants that leave it."""
    if os.name != "posix":
        raise RuntimeError("SKIPPED: cooperative subprocess execution requires POSIX process groups")
    if isinstance(process_group_id, bool) or process_group_id != process.pid:
        raise ValueError("Cooperative execution requires the child's explicitly owned process group")
    try:
        observed_group = os.getpgid(process.pid)
    except ProcessLookupError:
        # A quick child may exit before this query. Communicate still owns reaping and output.
        pass
    else:
        if observed_group != process_group_id:
            raise ValueError("Cooperative execution requires a process group owned by this child")
    started = time.perf_counter() if started_at is None else started_at
    argv = process.args
    if isinstance(argv, (str, bytes)):
        raise ValueError("Cooperative execution requires an argument vector")
    cancellation = None

    def request_cancellation() -> None:
        nonlocal cancellation
        if cancellation is not None:
            cancellation["repeated_interrupts"] += 1
            return
        cancellation = {
            "signal": "SIGINT", "process_group_id": process.pid,
            "status": "waiting_for_cleanup", "repeated_interrupts": 0,
        }
        try:
            os.killpg(process.pid, signal.SIGINT)
            cancellation["signal_forwarded"] = True
        except ProcessLookupError:
            cancellation["signal_forwarded"] = False

    if cancellation_requested:
        request_cancellation()
    communication_started = False
    while True:
        try:
            if communication_started:
                stdout, stderr = process.communicate()
            else:
                communication_started = True
                stdout, stderr = process.communicate(input=input)
            break
        except KeyboardInterrupt:
            request_cancellation()

    while True:
        try:
            if not _live_process_group_members(process_group_id):
                break
            # POSIX groups have no portable wait handle for non-child members.
            time.sleep(0.05)
        except KeyboardInterrupt:
            request_cancellation()
    if cancellation is not None:
        cancellation["status"] = "cleanup_completed"
    result = {
        "command": shlex.join(argv),
        "returncode": process.returncode,
        "stdout": stdout or "", "stderr": stderr or "",
        "duration_seconds": time.perf_counter() - started,
        "timeout_seconds": None, "timeout_hit": False,
        "cancelled": cancellation is not None,
        "execution": {
            "process_id": process.pid, "process_group_id": process.pid,
            "start_new_session": True, "live_process_group_members": [],
            "completion_scope": "process_group_quiescent",
        },
    }
    if cancellation is not None:
        result["cancellation"] = cancellation
        raise CooperativeProcessInterrupted(result)
    return result
