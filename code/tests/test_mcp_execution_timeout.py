"""Check MCP timeout controls without running a GPU benchmark."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import psutil
import pytest

import core.utils.cooperative_subprocess as cooperative
import mcp.mcp_server as server
from core.jobs import JobStore
from core.utils.cooperative_subprocess import (
    CooperativeCleanupIncomplete,
    CooperativeProcessInterrupted,
)
from core.utils.execution_timeouts import (
    MAX_EXECUTION_TIMEOUT_SECONDS,
    MAX_SUBPROCESS_WAIT_SECONDS,
)


@pytest.fixture(autouse=True)
def isolated_owned_work_admission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    queue = tmp_path / "mcp-queue"
    store = JobStore()
    monkeypatch.setattr(server, "JOB_STORE", store)
    monkeypatch.setattr(server, "_QUEUE_DIR", queue)
    monkeypatch.setattr(server, "_QUEUE_LOG_PATH", queue / "queue.log")
    monkeypatch.setattr(server, "_QUEUE_SCRIPT_PATH", queue / "bench_queue.sh")
    monkeypatch.setattr(server, "_UNVERIFIABLE_CLEANUP", {})
    yield
    store._executor.shutdown(wait=True)


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
    assert properties["timeout_seconds"]["default"] is None
    deep_dive_timeout = server.TOOLS["benchmark_explore"].input_schema["properties"][
        "deep_dive_timeout_seconds"
    ]
    assert deep_dive_timeout["type"] == ["integer", "null"]
    assert deep_dive_timeout["default"] == 0


@pytest.mark.parametrize("tool_name", ["benchmark_variants", "optimize"])
def test_benchmark_alias_schemas_do_not_inject_an_outer_deadline(tool_name: str) -> None:
    timeout = server.TOOLS[tool_name].input_schema["properties"]["timeout_seconds"]
    assert timeout["type"] == ["integer", "null"]
    assert timeout["default"] is None


@pytest.mark.parametrize("tool_name", ["benchmark_variants", "optimize"])
def test_benchmark_aliases_preserve_an_omitted_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    captured: dict[str, Any] = {}

    def capture(params: dict[str, Any]) -> dict[str, Any]:
        captured.update(params)
        return {"success": True}

    monkeypatch.setattr(server, "tool_run_benchmarks", capture)
    if tool_name == "optimize":
        server.tool_optimize({"target": "ch01:example"})
    else:
        server.tool_benchmark_variants({"targets": ["ch01:example"]})

    assert "timeout_seconds" not in captured


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


def test_suite_deadline_above_execution_poll_limit_keeps_queue_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture_queue(
        name: str, runner: Any, params: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        captured.update(params)
        return {"job_id": "suite-boundary-handoff", "status": "queued"}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("This boundary test must not start GPU or benchmark work")

    monkeypatch.setattr(server, "_queue_job", capture_queue)
    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    result = server.tool_run_benchmarks(
        {
            "targets": ["ch01:example"],
            "execution_timeout_seconds": 19,
            "suite_timeout": MAX_EXECUTION_TIMEOUT_SECONDS + 1,
            "async": True,
            "timeout_seconds": 19,
            "run_id": "suite_boundary_handoff",
            "include_context": False,
        }
    )
    assert result["status"] == "queued"
    assert captured["execution_timeout_seconds"] == 19
    assert captured["suite_timeout"] == MAX_EXECUTION_TIMEOUT_SECONDS + 1


def test_execution_platform_limit_retains_its_diagnostic_before_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsupported execution wait must fail before GPU or queue work")

    monkeypatch.setattr(server, "_queue_job", forbidden)
    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    result = server.tool_run_benchmarks(
        {
            "execution_timeout_seconds": MAX_EXECUTION_TIMEOUT_SECONDS + 1,
            "async": True,
        }
    )
    assert result["success"] is False
    assert str(MAX_EXECUTION_TIMEOUT_SECONDS) in result["error"]
    assert "supported finite wait" in result["error"]
    assert "runtime-provenance allowance" in result["error"]


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "19"])
def test_invalid_outer_timeout_fails_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid outer timeout must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks({"timeout_seconds": value, "async": True})
    assert result == {
        "error": "timeout_seconds must be a nonnegative integer or null",
        "success": False,
    }


def test_outer_timeout_platform_limit_rejects_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsupported outer wait must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {"timeout_seconds": int(MAX_SUBPROCESS_WAIT_SECONDS) + 1, "async": True}
    )
    assert result["success"] is False
    assert "timeout_seconds" in result["error"]
    assert "supported wait" in result["error"]


@pytest.mark.parametrize("suite_options", [{}, {"suite_timeout": None}, {"suite_timeout": 19}])
def test_uncapped_execution_rejects_inherited_or_finite_suite_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    suite_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsafe deadlines must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {
            "execution_timeout_seconds": 0,
            "timeout_seconds": 0,
            "async": True,
            **suite_options,
        }
    )
    assert result["success"] is False
    assert "requires suite_timeout=0" in result["error"]


@pytest.mark.parametrize(
    "outer_options",
    [
        {},
        {"timeout_seconds": 19},
        {"timeout_seconds": 600},
        {"timeout_seconds": 900},
        {"timeout_seconds": -1},
        {"timeout_seconds": False},
        {"timeout_seconds": True},
        {"timeout_seconds": 0.0},
        {"timeout_seconds": "0"},
    ],
)
def test_uncapped_execution_requires_explicit_uncapped_outer_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    outer_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Unsafe deadlines must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {
            "execution_timeout_seconds": 0,
            "suite_timeout": 0,
            "async": True,
            **outer_options,
        }
    )
    assert result["success"] is False
    assert "explicit timeout_seconds=0 or null" in result["error"]


@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("watchdog", ["1", "true", "yes", "on", " TRUE "])
def test_uncapped_timing_watchdog_rejects_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    run_async: bool,
    watchdog: str,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Incompatible watchdog must fail before GPU or queue work")

    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", watchdog)
    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {
            "execution_timeout_seconds": 0,
            "suite_timeout": 0,
            "timeout_seconds": 0,
            "async": run_async,
        }
    )
    assert result["success"] is False
    assert "incompatible with AISP_STALE_WATCH_TIMING" in result["error"]


@pytest.mark.parametrize("outer_timeout", [0, None])
def test_async_uncapped_execution_rejects_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    outer_timeout: int | None,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Uncancellable async execution must fail before GPU or queue work")

    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", "0")
    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {
            "execution_timeout_seconds": 0,
            "suite_timeout": 0,
            "timeout_seconds": outer_timeout,
            "async": True,
        }
    )
    assert result["success"] is False
    assert "requires async=false" in result["error"]


@pytest.mark.parametrize("selected", [None, 19])
def test_inherited_or_finite_execution_keeps_async_queue_handoff(
    monkeypatch: pytest.MonkeyPatch,
    selected: int | None,
) -> None:
    captured: dict[str, Any] = {}

    def capture_queue(
        name: str, runner: Any, params: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        captured.update({"name": name, "runner": runner, "params": params})
        return {"job_id": "captured-queue-handoff", "status": "queued"}

    monkeypatch.setattr(server, "_queue_job", capture_queue)
    result = server.tool_run_benchmarks(
        {
            "targets": ["ch01:example"],
            "execution_timeout_seconds": selected,
            "async": True,
            "timeout_seconds": 19,
            "run_id": "finite_queue_handoff",
            "include_context": False,
        }
    )
    assert result["status"] == "queued"
    assert captured["name"] == "run_benchmarks"
    assert captured["params"]["execution_timeout_seconds"] == selected
    assert callable(captured["runner"])


@pytest.mark.parametrize("outer_options", [{}, {"timeout_seconds": None}, {"timeout_seconds": 0}])
def test_async_requires_a_positive_outer_timeout_before_cuda_or_queue(
    monkeypatch: pytest.MonkeyPatch,
    outer_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Uncancellable async execution must fail before GPU or queue work")

    monkeypatch.setattr(server, "_cuda_precheck", forbidden)
    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_run_benchmarks(
        {
            "targets": ["ch01:example"],
            "async": True,
            **outer_options,
        }
    )
    assert result == {
        "error": (
            "async=true requires an explicit positive timeout_seconds because "
            "a background worker cannot receive operator cancellation"
        ),
        "success": False,
    }


@pytest.mark.parametrize("timeout_options", [{}, {"timeout_seconds": None}, {"timeout_seconds": 0}])
@pytest.mark.parametrize(
    ("tool_name", "required_params"),
    [
        ("benchmark_variants", {"targets": ["ch01:example"]}),
        ("optimize", {"target": "ch01:example"}),
        ("benchmark_deep_dive_compare", {"targets": ["ch01:example"]}),
        ("benchmark_explore", {"path": "ch01/baseline_example.py"}),
        ("benchmark_llm_patch_loop", {"targets": ["ch01:example"]}),
    ],
)
def test_async_benchmark_workflows_require_a_positive_outer_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    required_params: dict[str, Any],
    timeout_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid async work must fail before queueing")

    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = getattr(server, f"tool_{tool_name}")(
        {
            **required_params,
            **timeout_options,
            "async": True,
            "include_context": False,
        }
    )
    assert result["success"] is False
    assert "requires an explicit positive timeout_seconds" in result["error"]


@pytest.mark.parametrize("deep_dive", ["always", "auto"])
@pytest.mark.parametrize(
    "deep_dive_timeout_options",
    [{}, {"deep_dive_timeout_seconds": None}, {"deep_dive_timeout_seconds": 0}],
)
def test_async_explore_requires_a_positive_possible_deep_dive_timeout_before_queue(
    monkeypatch: pytest.MonkeyPatch,
    deep_dive: str,
    deep_dive_timeout_options: dict[str, Any],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Uncancellable deep-dive work must fail before queueing")

    monkeypatch.setattr(server, "_queue_job", forbidden)
    result = server.tool_benchmark_explore(
        {
            "path": "ch01/baseline_example.py",
            "async": True,
            "timeout_seconds": 19,
            "deep_dive": deep_dive,
            "include_context": False,
            **deep_dive_timeout_options,
        }
    )
    assert result["success"] is False
    assert (
        result["error"]
        == f"async=true with deep_dive={deep_dive} requires an explicit positive "
        "deep_dive_timeout_seconds because the background worker may run a deep dive "
        "and cannot receive operator cancellation"
    )


@pytest.mark.parametrize(
    "deep_dive_options",
    [
        {"deep_dive": "always", "deep_dive_timeout_seconds": 23},
        {"deep_dive": "auto", "deep_dive_timeout_seconds": 23},
        {"deep_dive": "never"},
    ],
)
def test_async_explore_queues_only_with_bounded_possible_deep_dive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deep_dive_options: dict[str, Any],
) -> None:
    captured: dict[str, Any] = {}

    def capture_queue(
        name: str, runner: Any, params: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        captured.update({"name": name, "runner": runner, "params": params})
        return {"job_id": "bounded-explore", "status": "queued"}

    monkeypatch.setattr(server, "_queue_job", capture_queue)
    result = server.tool_benchmark_explore(
        {
            "path": "ch01/baseline_example.py",
            "async": True,
            "timeout_seconds": 19,
            "run_id": "bounded_explore",
            "artifacts_dir": str(tmp_path),
            "include_context": False,
            **deep_dive_options,
        }
    )
    assert result["status"] == "queued"
    assert captured["name"] == "benchmark_explore"
    assert callable(captured["runner"])
    assert captured["params"] == {
        "path": "ch01/baseline_example.py",
        "async": True,
        "timeout_seconds": 19,
        "run_id": "bounded_explore",
        "artifacts_dir": str(tmp_path),
        "include_context": False,
        **deep_dive_options,
    }


@pytest.mark.parametrize("selected", [None, 19])
def test_inherited_or_finite_deadline_keeps_timing_watchdog_available(
    monkeypatch: pytest.MonkeyPatch,
    selected: int | None,
) -> None:
    monkeypatch.setenv("AISP_STALE_WATCH_TIMING", "1")
    result = _precheck(execution_timeout_seconds=selected)
    assert result["precheck_only"] is True
    assert result["execution_timeout_seconds"] == selected


@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("outer_timeout", [0, None])
def test_uncapped_execution_plan_has_no_enclosing_deadline(
    monkeypatch: pytest.MonkeyPatch,
    test_mode: bool,
    outer_timeout: int | None,
) -> None:
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    result = _precheck(
        execution_timeout_seconds=0,
        suite_timeout=0,
        timeout_seconds=outer_timeout,
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


@pytest.mark.parametrize("test_mode", [False, True])
def test_omitted_outer_timeout_remains_uncapped(
    monkeypatch: pytest.MonkeyPatch, test_mode: bool
) -> None:
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    assert _precheck()["timeout_seconds"] is None


def test_positive_outer_timeout_is_retained() -> None:
    assert _precheck(timeout_seconds=23)["timeout_seconds"] == 23


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")),
    reason="POSIX waitid with WNOWAIT is required",
)
def test_uncapped_bench_cli_runs_real_help_with_owned_process_receipt() -> None:
    result = server._run_bench_cli(["run", "--help"], timeout=None, uncapped=True)
    assert result["returncode"] == 0, result
    assert "--help" in re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result["stdout"])
    assert result["timeout_seconds"] is None
    assert result["timeout_hit"] is False
    assert result["cancelled"] is False
    assert result["execution"]["start_new_session"] is True
    assert result["execution"]["process_group_id"] > 0


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")),
    reason="POSIX waitid with WNOWAIT is required",
)
def test_uncapped_bench_cli_preserves_a_real_cli_argument_failure() -> None:
    result = server._run_bench_cli(
        ["run", "--execution-timeout-seconds", "not-an-integer"],
        timeout=None,
        uncapped=True,
    )
    assert result["returncode"] != 0
    assert result["error"]
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result["stdout"] + result["stderr"])
    assert "not-an-integer" in output
    assert result["timeout_hit"] is False


@pytest.mark.parametrize("timeout", [17, server._BENCH_CLI_TIMEOUT])
def test_finite_bench_cli_keeps_existing_runner_and_failure_output(
    monkeypatch: pytest.MonkeyPatch, timeout: int
) -> None:
    import core.diagnostics.process as diagnostic

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["timeout_seconds"] == timeout
        return {
            "command": " ".join(command),
            "returncode": 7,
            "stdout": "",
            "stderr": "deliberate child failure",
            "timeout_seconds": timeout,
            "timeout_hit": False,
            "execution": {"process_group_id": 123, "completion_scope": "process_tree_quiescent"},
        }

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Finite execution must retain the existing runner")

    monkeypatch.setattr(diagnostic, "run_owned_process", run)
    monkeypatch.setattr(server, "run_cooperative_process", forbidden)
    result = server._run_bench_cli(["run"], timeout=timeout)
    assert result["returncode"] == 7
    assert result["error"] == "deliberate child failure"
    assert result["timeout_seconds"] == timeout
    assert result["timeout_hit"] is False


def test_finite_bench_cli_keeps_timeout_output_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.diagnostics.process as diagnostic

    def expired(command: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "command": " ".join(command),
            "returncode": 124,
            "stdout": "partial stdout",
            "stderr": "partial stderr",
            "timeout_seconds": kwargs["timeout_seconds"],
            "timeout_hit": True,
            "error": "Timed out after 17s",
            "execution": {"process_group_id": 123, "completion_scope": "process_tree_quiescent"},
        }

    monkeypatch.setattr(diagnostic, "run_owned_process", expired)
    result = server._run_bench_cli(["run"], timeout=17)
    assert result["timeout_hit"] is True
    assert result["timeout_seconds"] == 17
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"


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


def test_uncapped_bench_cli_retains_ordinary_helper_error_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "returncode": 0,
        "stdout": "retained stdout",
        "stderr": "retained stderr",
        "timeout_seconds": None,
        "timeout_hit": False,
        "execution": {"process_group_id": 123, "completion_scope": "process_group_quiescent"},
    }
    original = PermissionError("group observation denied")
    original.cooperative_process_result = receipt

    def failed(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(server, "run_cooperative_process", failed)
    result = server._run_bench_cli(["run"], timeout=None, uncapped=True)
    assert result["stdout"] == receipt["stdout"]
    assert result["stderr"] == receipt["stderr"]
    assert result["execution"] == receipt["execution"]
    assert result["returncode"] == 0
    assert result["error"] == str(original)
    assert result["error_type"] == "PermissionError"
    assert result["success"] is False
    assert result["timeout_hit"] is False
    assert server._extract_exit_code(result) == 1
    assert server._benchmark_next_steps(result)[0]["tool"] == "status"


@pytest.mark.parametrize("legacy_exception", [False, True])
def test_incomplete_helper_cleanup_blocks_the_next_owned_launch(
    monkeypatch: pytest.MonkeyPatch,
    legacy_exception: bool,
) -> None:
    receipt = {
        "returncode": 0,
        "stdout": "retained stdout",
        "stderr": "retained stderr",
        "execution": {"process_group_id": 123, "completion_scope": "cleanup_incomplete"},
    }
    original = (
        PermissionError("unverified group")
        if legacy_exception
        else CooperativeCleanupIncomplete(receipt)
    )
    original.cooperative_process_result = receipt

    def failed(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(server, "run_cooperative_process", failed)
    with pytest.raises(CooperativeCleanupIncomplete) as caught:
        server._run_bench_cli(["run"], timeout=None, uncapped=True)
    assert caught.value.result == receipt
    barrier = json.loads(server._cleanup_barrier_path().read_text())
    assert barrier["event"] == "aisp_cleanup_incomplete"
    assert cooperative.validate_cleanup_incomplete_receipt(barrier["receipt"]) == receipt

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("A blocked MCP process must not launch another owned child")

    monkeypatch.setattr(server, "run_cooperative_process", forbidden)
    with pytest.raises(CooperativeCleanupIncomplete):
        server._run_bench_cli(["run", "--help"], timeout=None, uncapped=True)

    monkeypatch.setattr(server, "JOB_STORE", JobStore())
    with pytest.raises(server.OwnedWorkAdmissionBlocked) as restarted:
        server._run_bench_cli(["run", "--help"], timeout=None, uncapped=True)
    assert restarted.value.result["admission_blocked"] is True
    assert restarted.value.result["blocking_cleanup"] == receipt
    server.JOB_STORE._executor.shutdown(wait=True)


def test_successful_child_cannot_poison_admission_with_stderr_event() -> None:
    receipt = {
        "returncode": 1,
        "stdout": "nested stdout",
        "stderr": "nested stderr",
        "execution": {"process_group_id": 123, "completion_scope": "cleanup_incomplete"},
    }
    nonce = "successful-child-cleanup-event-nonce"
    stream = io.StringIO()
    cooperative.emit_cleanup_incomplete_event(receipt, nonce=nonce, stream=stream)
    result = {
        "returncode": 0,
        "stdout": "ordinary success",
        "stderr": stream.getvalue(),
        "execution": {
            "process_group_id": 456,
            "completion_scope": "process_tree_quiescent",
            "cleanup_event_nonce": nonce,
        },
    }

    assert server._check_child_cleanup_result(result) is result
    assert not server._cleanup_barrier_path().exists()
    assert server.JOB_STORE.incomplete_cleanup_receipt() is None


def test_uncapped_bench_cli_propagates_helper_error_without_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PermissionError("failure before any retained process receipt")

    def failed(*args: Any, **kwargs: Any) -> None:
        raise original

    monkeypatch.setattr(server, "run_cooperative_process", failed)
    with pytest.raises(PermissionError) as caught:
        server._run_bench_cli(["run"], timeout=None, uncapped=True)
    assert caught.value is original


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="owned subprocess support requires waitid")
def test_real_bench_cli_retains_output_after_failed_final_group_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\nprint('real CLI startup stderr', file=sys.stderr, flush=True)\n",
        encoding="utf-8",
    )
    environment = server._subprocess_env()
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + environment["PYTHONPATH"]
    monkeypatch.setattr(server, "_subprocess_env", lambda: environment)
    original = psutil.AccessDenied(msg="final group observation denied")
    processes: list[subprocess.Popen[str]] = []
    communicate = cooperative.communicate_cooperatively
    query_group = cooperative._live_process_group_members

    def observe(process: subprocess.Popen[str], **kwargs: Any) -> dict[str, Any]:
        processes.append(process)
        return communicate(process, **kwargs)

    def fail_after_pipe_eof(process_group_id: int) -> list[int]:
        if processes and processes[-1].stdout.closed and processes[-1].stderr.closed:
            raise original
        return query_group(process_group_id)

    monkeypatch.setattr(cooperative, "communicate_cooperatively", observe)
    monkeypatch.setattr(cooperative, "_live_process_group_members", fail_after_pipe_eof)
    with pytest.raises(CooperativeCleanupIncomplete) as caught:
        server._run_bench_cli(["run", "--help"], timeout=None, uncapped=True)
    result = caught.value.result
    stdout = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result["stdout"])
    assert "--help" in stdout
    assert "real CLI startup stderr" in result["stderr"]
    assert caught.value.__cause__ is original
    assert result["execution"]["coordinator_error"]["type"] == "AccessDenied"
    assert result["execution"]["completion_scope"] == "cleanup_incomplete"
    assert result["execution"]["process_group_id"] > 0
    assert result["timeout_hit"] is False


@pytest.mark.parametrize(
    "tool_name",
    [
        "benchmark_explore",
        "benchmark_llm_patch_loop",
        "benchmark_deep_dive_compare",
    ],
)
@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize(
    "timeout_options",
    [{}, {"timeout_seconds": None}, {"timeout_seconds": 0}, {"timeout_seconds": 17}],
)
def test_composite_timeout_forwarding_preserves_absence_including_test_mode(
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
        "targets": ["ch01:example"],
        "include_context": False,
        "run_id": "timeout_forwarding",
        "output_dir": str(tmp_path / "runs"),
        **timeout_options,
    }
    if tool_name == "benchmark_explore":
        wrapper = tmp_path / "baseline_example.py"
        wrapper.write_text('binary_name = "baseline_example"\n', encoding="utf-8")
        params["path"] = str(wrapper)
        params["artifacts_dir"] = str(tmp_path / "artifacts")

    with pytest.raises(PlannedBenchmarkCall):
        getattr(server, f"tool_{tool_name}")(params)

    if "timeout_seconds" in timeout_options:
        assert "timeout_seconds" in captured
        assert captured["timeout_seconds"] == timeout_options["timeout_seconds"]
    else:
        assert "timeout_seconds" not in captured
