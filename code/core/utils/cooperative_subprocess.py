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


class CooperativeProcessInterrupted(KeyboardInterrupt):
    """Cancellation with the completed child's output and ownership receipt."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("Cancelled after the owned process completed cleanup")
        self.result = result


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
    cancellation = None
    try:
        stdout, stderr = process.communicate()
    except KeyboardInterrupt:
        cancellation = {
            "signal": "SIGINT", "process_group_id": process.pid,
            "status": "waiting_for_cleanup", "repeated_interrupts": 0,
        }
        try:
            os.killpg(process.pid, signal.SIGINT)
            cancellation["signal_forwarded"] = True
        except ProcessLookupError:
            cancellation["signal_forwarded"] = False
        while True:
            try:
                stdout, stderr = process.communicate()
                break
            except KeyboardInterrupt:
                # Another interrupt must not kill a child that is cleaning up.
                cancellation["repeated_interrupts"] += 1
        cancellation["status"] = "cleanup_completed"
    result = {
        "command": shlex.join(argv),
        "returncode": process.returncode,
        "stdout": stdout or "", "stderr": stderr or "",
        "duration_seconds": time.perf_counter() - started,
        "timeout_seconds": None, "timeout_hit": False,
        "cancelled": cancellation is not None,
        "execution": {"process_id": process.pid, "process_group_id": process.pid, "start_new_session": True},
    }
    if cancellation is not None:
        result["cancellation"] = cancellation
        raise CooperativeProcessInterrupted(result)
    return result
