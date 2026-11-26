"""Run a promotion in an owned process group so its children share its deadline."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def run_promotion_process(
    command: list[str], *, cwd: str | Path | None, timeout: float | None
) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The unreaped leader still owns this process group ID.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
