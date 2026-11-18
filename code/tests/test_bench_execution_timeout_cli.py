"""CLI timeout parsing and plan coverage, without claiming GPU execution."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.benchmark.bench_commands import _execute_benchmarks
from core.benchmark import bench_commands


CODE_ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "run", *args],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    result.stdout = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout)
    result.stderr = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stderr)
    return result


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


@pytest.mark.parametrize("suite_args", [[], ["--suite-timeout", "12"], ["--timeout-seconds", "12"]])
def test_cli_rejects_uncapped_execution_with_finite_suite_before_execution(
    suite_args: list[str],
) -> None:
    result = _cli("--execution-timeout-seconds", "0", *suite_args)
    assert result.returncode != 0
    assert "--suite-timeout 0" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_suite_alias_zero_disables_the_effective_deadline() -> None:
    result = _cli("--execution-timeout-seconds", "0", "--timeout-seconds", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["suite_timeout"] == 0


@pytest.mark.parametrize("suite_timeout", [None, 12, 14400])
def test_python_executor_rejects_uncapped_execution_with_active_suite_before_setup(
    suite_timeout: int | None,
) -> None:
    with pytest.raises(ValueError, match="suite_timeout=0"):
        _execute_benchmarks(execution_timeout_seconds=0, suite_timeout=suite_timeout)


def test_python_executor_rejects_uncapped_execution_with_default_suite_before_setup() -> None:
    with pytest.raises(ValueError, match="suite_timeout=0"):
        _execute_benchmarks(execution_timeout_seconds=0)


def test_cli_rejects_uncapped_execution_with_timing_watchdog_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", "1")
    result = _cli("--execution-timeout-seconds", "0", "--suite-timeout", "0")
    assert result.returncode != 0
    assert "AISP_STALE_WATCH_TIMING" in result.stderr
    assert "incompatible" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("watchdog", ["1", "true", "yes", "on", " TRUE "])
def test_python_executor_rejects_uncapped_timing_watchdog_before_setup(
    monkeypatch: pytest.MonkeyPatch, watchdog: str,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Incompatible watchdog must fail before setup or CUDA work")

    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", watchdog)
    monkeypatch.setattr(bench_commands, "_parse_target_extra_args", forbidden)
    monkeypatch.setattr(bench_commands, "get_gpu_state", forbidden)
    with pytest.raises(ValueError, match="incompatible with AISP_STALE_WATCH_TIMING"):
        _execute_benchmarks(execution_timeout_seconds=0, suite_timeout=0)


@pytest.mark.parametrize("selected", [None, 19])
def test_cli_inherited_or_finite_deadline_keeps_timing_watchdog_available(
    monkeypatch: pytest.MonkeyPatch, selected: int | None,
) -> None:
    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", "1")
    args = ["--dry-run"]
    if selected is not None:
        args.extend(["--execution-timeout-seconds", str(selected)])
    result = _cli(*args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["execution_timeout_seconds"] == selected


@pytest.mark.parametrize("invalid", [-1, True, 0.5, "0"])
def test_python_executor_rejects_invalid_timeout_before_setup(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError), match="execution_timeout_seconds"):
        _execute_benchmarks(execution_timeout_seconds=invalid)
