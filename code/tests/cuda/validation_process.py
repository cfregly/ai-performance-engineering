"""Bound a validation command and its compiler/sanitizer descendants on POSIX."""
from __future__ import annotations

from contextlib import suppress
import os
import signal
import subprocess


def run_command(command: list[str], timeout: float = 300) -> subprocess.CompletedProcess[str]:
    if os.name != "posix":
        raise RuntimeError("CUDA validation process containment requires a POSIX host")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            # Killing only Python/nvcc can leave Ninja, host compilers, or the
            # sanitizer's target alive. Each command owns a fresh process group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(command, 124, stdout,
                stderr + f"\nTIMEOUT after {timeout} seconds; process group terminated\n")
