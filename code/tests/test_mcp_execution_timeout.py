"""Check MCP timeout controls without running a GPU benchmark."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

import mcp.mcp_server as server
from core.utils.cooperative_subprocess import CooperativeProcessInterrupted


def _precheck(**overrides: Any) -> dict[str, Any]:
    return server.tool_run_benchmarks(
        {
            "targets": ["labs/multi_gpu_communication:jacobi"],
            "profile": "none",
            "run_id": "mcp_execution_timeout_validation",
            "include_context": False,
            "precheck_only": True,
            **overrides,
        }
    )


def test_timeout_schema_exposes_nullable_overrides() -> None:
    properties = server.TOOLS["run_benchmarks"].input_schema["properties"]
    for name in ("execution_timeout_seconds", "suite_timeout"):
        assert properties[name]["type"] == ["integer", "null"]
        assert properties[name]["minimum"] == 0
        assert properties[name]["default"] is None
    assert properties["timeout_seconds"]["type"] == ["integer", "null"]
    assert properties["timeout_seconds"]["default"] == 900


@pytest.mark.parametrize("value", [0, 19, None])
def test_stage_and_suite_overrides_reach_the_planned_cli(value: int | None) -> None:
    options = {"timeout_seconds": 0} if value == 0 else {}
    result = _precheck(execution_timeout_seconds=value, suite_timeout=value, **options)
    assert result["precheck_only"] is True
    assert result["execution_timeout_seconds"] == value
    assert result["suite_timeout"] == value
    args = result["planned_args"]
    for flag in ("--execution-timeout-seconds", "--suite-timeout"):
        if value is None:
            assert flag not in args
        else:
            assert args[args.index(flag) + 1] == str(value)


def test_omitted_stage_and_suite_values_keep_cli_defaults() -> None:
    result = _precheck()
    assert result["execution_timeout_seconds"] is None
    assert result["suite_timeout"] is None
    assert "--execution-timeout-seconds" not in result["planned_args"]
    assert "--suite-timeout" not in result["planned_args"]


@pytest.mark.parametrize("name", ["execution_timeout_seconds", "suite_timeout"])
@pytest.mark.parametrize("value", [-1, True, False, 1.5, "0"])
def test_invalid_overrides_fail_before_cuda_or_queue_work(
    monkeypatch: pytest.MonkeyPatch, name: str, value: Any
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid timeout must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks({name: value, "async": True})
    assert result["success"] is False
    assert result["error"] == f"{name} must be a nonnegative integer or null"


@pytest.mark.parametrize("suite_options", [{}, {"suite_timeout": None}, {"suite_timeout": 19}])
def test_uncapped_execution_rejects_inherited_or_finite_suite_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch, suite_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsafe deadlines must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks({
        "execution_timeout_seconds": 0, "timeout_seconds": 0, "async": True,
        **suite_options,
    })
    assert result["success"] is False
    assert "requires suite_timeout=0" in result["error"]


@pytest.mark.parametrize("outer_options", [
    {}, {"timeout_seconds": 19}, {"timeout_seconds": 600}, {"timeout_seconds": 900},
    {"timeout_seconds": -1}, {"timeout_seconds": False}, {"timeout_seconds": True},
    {"timeout_seconds": 0.0}, {"timeout_seconds": "0"},
])
def test_uncapped_execution_requires_explicit_uncapped_outer_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch, outer_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsafe deadlines must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks({
        "execution_timeout_seconds": 0, "suite_timeout": 0, "async": True,
        **outer_options,
    })
    assert result["success"] is False
    assert "explicit timeout_seconds=0 or null" in result["error"]


@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("outer_timeout", [0, None])
def test_uncapped_execution_plan_has_no_enclosing_deadline(
    monkeypatch: pytest.MonkeyPatch, test_mode: bool, outer_timeout: int | None,
) -> None:
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    result = _precheck(
        execution_timeout_seconds=0, suite_timeout=0, timeout_seconds=outer_timeout,
    )
    assert result["precheck_only"] is True
    assert result["timeout_seconds"] is None
    assert result["suite_timeout"] == 0


@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("value", [0, None])
def test_explicit_uncapped_outer_timeout_survives_test_mode(
    monkeypatch: pytest.MonkeyPatch, test_mode: bool, value: int | None
) -> None:
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    result = _precheck(timeout_seconds=value)
    assert result["timeout_seconds"] is None


@pytest.mark.parametrize("test_mode,expected", [(False, None), (True, 600)])
def test_omitted_outer_timeout_preserves_existing_mode_default(
    monkeypatch: pytest.MonkeyPatch, test_mode: bool, expected: int | None
) -> None:
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    assert _precheck()["timeout_seconds"] == expected


def test_positive_outer_timeout_is_retained() -> None:
    assert _precheck(timeout_seconds=23)["timeout_seconds"] == 23


def test_uncapped_bench_cli_runs_real_help_with_owned_process_receipt() -> None:
    result = server._run_bench_cli(["run", "--help"], timeout=None, uncapped=True)
    assert result["returncode"] == 0, result
    assert "--help" in re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result["stdout"])
    assert result["timeout_seconds"] is None
    assert result["timeout_hit"] is False
    assert result["cancelled"] is False
    assert result["execution"]["start_new_session"] is True
    assert result["execution"]["process_group_id"] > 0


def test_uncapped_bench_cli_preserves_a_real_cli_argument_failure() -> None:
    result = server._run_bench_cli(
        ["run", "--execution-timeout-seconds", "not-an-integer"], timeout=None, uncapped=True,
    )
    assert result["returncode"] != 0
    assert result["error"]
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result["stdout"] + result["stderr"])
    assert "not-an-integer" in output
    assert result["timeout_hit"] is False


@pytest.mark.parametrize("timeout", [None, 17, server._BENCH_CLI_TIMEOUT])
def test_finite_bench_cli_keeps_existing_runner_and_failure_output(
    monkeypatch: pytest.MonkeyPatch, timeout: int | None
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == timeout
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="deliberate child failure")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Finite execution must retain the existing runner")

    monkeypatch.setattr(server.subprocess, "run", run)
    monkeypatch.setattr(server, "run_cooperative_process", forbidden)
    result = server._run_bench_cli(["run"], timeout=timeout)
    assert result["returncode"] == 7
    assert result["error"] == "deliberate child failure"
    assert result["timeout_seconds"] == timeout
    assert result["timeout_hit"] is False


def test_uncapped_bench_cli_does_not_swallow_completed_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {"cancelled": True, "returncode": 130, "stdout": "cleanup completed"}

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise CooperativeProcessInterrupted(receipt)

    monkeypatch.setattr(server, "run_cooperative_process", interrupted)
    with pytest.raises(CooperativeProcessInterrupted) as caught:
        server._run_bench_cli(["run"], timeout=None, uncapped=True)
    assert caught.value.result is receipt


@pytest.mark.parametrize("tool_name", [
    "benchmark_explore", "benchmark_llm_patch_loop", "benchmark_deep_dive_compare",
])
@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("timeout_options", [{}, {"timeout_seconds": None}, {"timeout_seconds": 0}, {"timeout_seconds": 17}])
def test_composite_timeout_forwarding_preserves_absence_and_existing_test_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_name: str,
    test_mode: bool,
    timeout_options: dict[str, Any],
) -> None:
    class PlannedBenchmarkCall(Exception):
        """Stop at the benchmark handoff without simulating execution."""

    captured: dict[str, Any] = {}

    def capture(params: dict[str, Any]) -> None:
        captured.update(params)
        raise PlannedBenchmarkCall

    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    monkeypatch.setattr(server, "tool_run_benchmarks", capture)
    params = {
        "targets": ["ch01:example"], "include_context": False,
        "run_id": "timeout_forwarding", "output_dir": str(tmp_path / "runs"),
        **timeout_options,
    }
    if tool_name == "benchmark_explore":
        wrapper = tmp_path / "baseline_example.py"
        wrapper.write_text('binary_name = "baseline_example"\n', encoding="utf-8")
        params["path"] = str(wrapper)
        params["artifacts_dir"] = str(tmp_path / "artifacts")

    with pytest.raises(PlannedBenchmarkCall):
        getattr(server, f"tool_{tool_name}")(params)

    if tool_name == "benchmark_deep_dive_compare" and test_mode:
        expected = timeout_options.get("timeout_seconds")
        assert captured["timeout_seconds"] == (expected if expected and expected > 0 else 300)
    elif "timeout_seconds" in timeout_options:
        assert "timeout_seconds" in captured
        assert captured["timeout_seconds"] == timeout_options["timeout_seconds"]
    else:
        assert "timeout_seconds" not in captured
