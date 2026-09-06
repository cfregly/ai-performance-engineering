"""Profiler invocations must only clean up processes they own."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from core.scripts.harness import profile_harness
from core.scripts.harness.example_registry import _example
from core.scripts.harness.metrics_config import ProfilerOverrides


def test_ncu_dry_run_never_launches_process_cleanup(tmp_path, monkeypatch):
    def unexpected_run(*args, **kwargs):
        pytest.fail(f"Dry run executed an external command: {args!r}")

    monkeypatch.setattr(profile_harness.subprocess, "run", unexpected_run)
    result = profile_harness.run_ncu(
        _example(name="isolated", path="ch01/baseline_performance.py", description="test"),
        tmp_path,
        argparse.Namespace(dry_run=True),
        10,
        ProfilerOverrides(),
    )
    assert result.skipped and result.skip_reason == "dry-run"


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_profile_command_cleans_stubborn_descendants(tmp_path: Path, parent_exits: bool):
    child_pid_path = tmp_path / "child.pid"
    child = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent = (
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not pathlib.Path({str(child_pid_path)!r}).exists(): time.sleep(0.01)\n"
        + ("sys.exit(0)\n" if parent_exits else "time.sleep(60)\n")
    )
    try:
        exit_code, duration = profile_harness.run_command(
            [sys.executable, "-c", parent],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=3,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            dry_run=False,
        )
        assert child_pid_path.exists(), "Child never reached its ready state"
        child_pid = int(child_pid_path.read_text())
        deadline = time.monotonic() + 2
        while _alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _alive(child_pid), "Profiler left its child process running"
        assert exit_code == (0 if parent_exits else -1)
        assert 0 <= duration < 10
    finally:
        if child_pid_path.exists():
            pid = int(child_pid_path.read_text())
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_profile_command_reports_group_cleanup_failure(tmp_path: Path, monkeypatch):
    def denied_group_signal(*args, **kwargs):
        raise PermissionError("denied for regression")

    monkeypatch.setattr(profile_harness.os, "killpg", denied_group_signal)
    stderr_path = tmp_path / "stderr.log"
    exit_code, _ = profile_harness.run_command(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=3,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=stderr_path,
        dry_run=False,
    )

    assert exit_code != 0
    diagnostics = stderr_path.read_text(encoding="utf-8")
    assert "PermissionError: denied for regression" in diagnostics
    assert "process-group cleanup" in diagnostics


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_profile_command_preserves_interrupt_during_failed_cleanup(tmp_path: Path, monkeypatch):
    original_wait = subprocess.Popen.wait
    first_wait = True
    interrupted_pid = None

    def interrupt_once(process, *args, **kwargs):
        nonlocal first_wait, interrupted_pid
        if first_wait:
            first_wait = False
            interrupted_pid = process.pid
            raise KeyboardInterrupt("active interrupt")
        return original_wait(process, *args, **kwargs)

    def denied_group_signal(*args, **kwargs):
        raise PermissionError("denied during interrupt")

    monkeypatch.setattr(profile_harness.subprocess.Popen, "wait", interrupt_once)
    monkeypatch.setattr(profile_harness.os, "killpg", denied_group_signal)
    stderr_path = tmp_path / "stderr.log"

    with pytest.raises(KeyboardInterrupt, match="active interrupt"):
        profile_harness.run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=3,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=stderr_path,
            dry_run=False,
        )

    assert "PermissionError: denied during interrupt" in stderr_path.read_text(encoding="utf-8")
    assert interrupted_pid is not None
    assert not _alive(interrupted_pid), "Interrupted profiler process was not reaped"
