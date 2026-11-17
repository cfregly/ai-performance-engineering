"""CLI timeout parsing and plan coverage, without claiming GPU execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.benchmark.bench_commands import _execute_benchmarks


CODE_ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "run", *args],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("selected", [None, 0, 180, 1200])
def test_cli_plan_retains_explicit_execution_timeout(selected: int | None) -> None:
    args = ["--targets", "labs/nvshmem_device:nvshmem_device", "--dry-run", "--suite-timeout", "0"]
    if selected is not None:
        args.extend(["--execution-timeout-seconds", str(selected)])
    result = _cli(*args)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["execution_timeout_seconds"] == selected
    assert plan["suite_timeout"] == 0
    assert plan["nsys_timeout_seconds"] is None
    assert plan["ncu_timeout_seconds"] is None
    assert plan["validity_profile"] == "strict"


def test_cli_rejects_negative_execution_timeout() -> None:
    result = _cli("--execution-timeout-seconds", "-1", "--dry-run")
    assert result.returncode != 0
    assert "--execution-timeout-seconds" in result.stderr
    assert "0" in result.stderr


def test_cli_rejects_uncapped_torchrun_before_execution() -> None:
    result = _cli("--execution-timeout-seconds", "0", "--launch-via", "torchrun", "--dry-run")
    assert result.returncode != 0
    assert "owned subprocess" in result.stderr


@pytest.mark.parametrize("invalid", [-1, True, 0.5, "0"])
def test_python_executor_rejects_invalid_timeout_before_setup(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError), match="execution_timeout_seconds"):
        _execute_benchmarks(execution_timeout_seconds=invalid)
