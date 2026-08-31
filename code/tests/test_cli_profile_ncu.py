"""CLI tests for `aisp profile ncu` option plumbing.

These tests validate argument parsing and propagation at the Typer entrypoint.
The live Nsight execution path is covered separately by the longer integration
profiling workflow tests.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.aisp import app, profile_app


class DummyNsightAutomation:
    """Capture CLI arguments without running a real Nsight Compute session."""

    calls: list[dict[str, object]] = []

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.last_error: str | None = None
        self.last_run: dict[str, object] = {}

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    def profile_ncu(self, **kwargs: object) -> Path:
        type(self).calls.append(kwargs)
        output_name = str(kwargs.get("output_name", "profile_ncu"))
        metric_set = str(kwargs.get("metric_set", "full"))
        resolved_metric_set = "basic" if metric_set == "minimal" else metric_set
        output_path = self.output_root / f"{output_name}.ncu-rep"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake ncu report", encoding="utf-8")
        self.last_run = {
            "launch_skip": kwargs.get("launch_skip"),
            "launch_count": kwargs.get("launch_count"),
            "replay_mode": kwargs.get("replay_mode"),
            "metric_set_resolved": resolved_metric_set,
        }
        return output_path


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "test_kernel"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _invoke_profile_ncu(args: list[str]) -> tuple[object, dict[str, object]]:
    runner = CliRunner()
    DummyNsightAutomation.reset()
    with patch("core.profiling.nsight_automation.NsightAutomation", DummyNsightAutomation):
        result = runner.invoke(app, ["profile", "ncu", *args])
    assert len(DummyNsightAutomation.calls) == 1
    return result, DummyNsightAutomation.calls[0]


def test_cli_profile_ncu_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["profile", "ncu", "--help"])

    assert result.exit_code == 0, result.stdout
    profile_commands = {command.name: command for command in profile_app.registered_commands}
    signature = inspect.signature(profile_commands["ncu"].callback)
    option_declarations = {
        declaration
        for parameter in signature.parameters.values()
        for declaration in (getattr(parameter.default, "param_decls", None) or ())
    }
    assert {
        "--launch-skip",
        "--launch-count",
        "--metric-set",
        "--replay-mode",
    } <= option_declarations


def test_cli_profile_ncu_minimal_metric_set(fake_binary: Path, tmp_path: Path) -> None:
    result, call = _invoke_profile_ncu(
        [
            "--command",
            str(fake_binary),
            "--output-dir",
            str(tmp_path / "ncu_output"),
            "--output-name",
            "minimal_test",
            "--metric-set",
            "minimal",
            "--timeout",
            "60",
        ]
    )

    assert result.exit_code == 0, result.stdout
    assert call["command"] == [str(fake_binary)]
    assert call["metric_set"] == "minimal"
    assert call["timeout_seconds"] == 60
    assert "NCU report:" in result.stdout
    assert "Metric set: minimal (resolved: basic)" in result.stdout


def test_cli_profile_ncu_launch_limiting(fake_binary: Path, tmp_path: Path) -> None:
    result, call = _invoke_profile_ncu(
        [
            "--command",
            str(fake_binary),
            "--output-dir",
            str(tmp_path / "ncu_output"),
            "--output-name",
            "launch_limit_test",
            "--metric-set",
            "minimal",
            "--launch-skip",
            "2",
            "--launch-count",
            "1",
            "--timeout",
            "60",
        ]
    )

    assert result.exit_code == 0, result.stdout
    assert call["launch_skip"] == 2
    assert call["launch_count"] == 1
    assert call["replay_mode"] == "application"
    assert "Launch skip: 2" in result.stdout
    assert "Launch count: 1" in result.stdout


def test_cli_profile_ncu_kernel_replay_mode(fake_binary: Path, tmp_path: Path) -> None:
    result, call = _invoke_profile_ncu(
        [
            "--command",
            str(fake_binary),
            "--output-dir",
            str(tmp_path / "ncu_output"),
            "--output-name",
            "kernel_replay_test",
            "--metric-set",
            "minimal",
            "--replay-mode",
            "kernel",
            "--timeout",
            "60",
        ]
    )

    assert result.exit_code == 0, result.stdout
    assert call["replay_mode"] == "kernel"
    assert "Replay mode: kernel" in result.stdout


def test_cli_profile_ncu_kernel_filter(fake_binary: Path, tmp_path: Path) -> None:
    result, call = _invoke_profile_ncu(
        [
            "--command",
            str(fake_binary),
            "--output-dir",
            str(tmp_path / "ncu_output"),
            "--output-name",
            "kernel_filter_test",
            "--kernel-filter",
            "test_kernel",
            "--metric-set",
            "minimal",
            "--launch-skip",
            "1",
            "--launch-count",
            "1",
            "--timeout",
            "60",
        ]
    )

    assert result.exit_code == 0, result.stdout
    assert call["kernel_filter"] == "test_kernel"
    assert call["launch_skip"] == 1
    assert call["launch_count"] == 1
    assert "Kernel filter: test_kernel" in result.stdout


def test_cli_profile_ncu_script_mode(tmp_path: Path) -> None:
    script_path = tmp_path / "gpu_script.py"
    script_path.write_text("print('OK')\n", encoding="utf-8")

    result, call = _invoke_profile_ncu(
        [
            str(script_path),
            "--output-dir",
            str(tmp_path / "ncu_output"),
            "--output-name",
            "script_test",
            "--metric-set",
            "minimal",
            "--launch-skip",
            "5",
            "--launch-count",
            "1",
            "--timeout",
            "120",
        ]
    )

    assert result.exit_code == 0, result.stdout
    assert call["command"] == [sys.executable, str(script_path)]
    assert call["launch_skip"] == 5
    assert call["launch_count"] == 1
