"""CPU-only orchestration tests for the bounded B200 dependency probe."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

import pytest

from core.scripts.b200_dependency_execution_probe import (
    CHECK_SPECS,
    MAX_DEADLINE_SECONDS,
    STATUS_FAIL,
    STATUS_NOT_RUN,
    STATUS_PASS,
    CheckResult,
    _terminate_process_group,
    build_parser,
    execute_probe,
)


def _fake_result(spec, *, status: str = STATUS_PASS, diagnostic: str | None = None):
    return CheckResult(
        name=spec.name,
        coverage=spec.coverage,
        status=status,
        duration_ms=1.0,
        timeout_seconds=spec.timeout_seconds,
        details={"fake_execution": True} if status == STATUS_PASS else {},
        diagnostic=diagnostic,
    )


def test_execute_probe_collects_every_required_check_as_structured_json(tmp_path) -> None:
    observed: list[tuple[str, float]] = []

    def run_check(spec, *, code_root, timeout_seconds):
        assert code_root == tmp_path
        observed.append((spec.name, timeout_seconds))
        return _fake_result(spec)

    report = execute_probe(
        tmp_path,
        deadline_seconds=MAX_DEADLINE_SECONDS,
        run_check=run_check,
    )

    assert report["status"] == STATUS_PASS
    assert sum(spec.timeout_seconds for spec in CHECK_SPECS) < MAX_DEADLINE_SECONDS
    assert [name for name, _ in observed] == [spec.name for spec in CHECK_SPECS]
    assert all(timeout > 0 for _, timeout in observed)
    assert report["summary"] == {
        "required": len(CHECK_SPECS),
        "passed": len(CHECK_SPECS),
        "failed": 0,
        "not_run": 0,
    }
    decoded = json.loads(json.dumps(report))
    assert decoded["checks"][0]["details"] == {"fake_execution": True}


def test_execute_probe_keeps_failures_and_runs_remaining_checks(tmp_path) -> None:
    observed: list[str] = []

    def run_check(spec, *, code_root, timeout_seconds):
        del code_root, timeout_seconds
        observed.append(spec.name)
        if spec.name == "torchvision_cuda_nms":
            return _fake_result(
                spec,
                status=STATUS_FAIL,
                diagnostic="NotImplementedError: torchvision::nms lacks a CUDA implementation",
            )
        return _fake_result(spec)

    report = execute_probe(
        tmp_path,
        deadline_seconds=MAX_DEADLINE_SECONDS,
        run_check=run_check,
    )

    assert observed == [spec.name for spec in CHECK_SPECS]
    assert report["status"] == STATUS_FAIL
    assert report["summary"] == {
        "required": len(CHECK_SPECS),
        "passed": len(CHECK_SPECS) - 1,
        "failed": 1,
        "not_run": 0,
    }
    failed = [check for check in report["checks"] if check["status"] == STATUS_FAIL]
    assert failed == [
        {
            "name": "torchvision_cuda_nms",
            "coverage": CHECK_SPECS[1].coverage,
            "status": STATUS_FAIL,
            "duration_ms": 1.0,
            "timeout_seconds": CHECK_SPECS[1].timeout_seconds,
            "details": {},
            "diagnostic": ("NotImplementedError: torchvision::nms lacks a CUDA implementation"),
            "worker_stderr": "",
        }
    ]


def test_execute_probe_turns_runner_exceptions_into_structured_failures(tmp_path) -> None:
    observed: list[str] = []

    def run_check(spec, *, code_root, timeout_seconds):
        del code_root, timeout_seconds
        observed.append(spec.name)
        if spec.name == "torchaudio_cuda_resample":
            raise RuntimeError("worker protocol broke")
        return _fake_result(spec)

    report = execute_probe(
        tmp_path,
        deadline_seconds=MAX_DEADLINE_SECONDS,
        run_check=run_check,
    )

    assert observed == [spec.name for spec in CHECK_SPECS]
    assert report["status"] == STATUS_FAIL
    assert report["summary"]["failed"] == 1
    assert report["checks"][2]["diagnostic"] == (
        "check runner failed: RuntimeError: worker protocol broke"
    )


def test_execute_probe_marks_remaining_checks_not_run_after_deadline(tmp_path) -> None:
    now = 0.0

    def clock() -> float:
        return now

    def run_check(spec, *, code_root, timeout_seconds):
        nonlocal now
        del code_root, timeout_seconds
        now = MAX_DEADLINE_SECONDS + 1.0
        return _fake_result(spec)

    report = execute_probe(
        tmp_path,
        deadline_seconds=MAX_DEADLINE_SECONDS,
        run_check=run_check,
        clock=clock,
    )

    assert report["status"] == STATUS_FAIL
    assert report["summary"] == {
        "required": len(CHECK_SPECS),
        "passed": 1,
        "failed": 0,
        "not_run": len(CHECK_SPECS) - 1,
    }
    assert [check["status"] for check in report["checks"]] == [
        STATUS_PASS,
        STATUS_NOT_RUN,
        STATUS_NOT_RUN,
        STATUS_NOT_RUN,
    ]


def test_parser_rejects_deadlines_over_hard_limit() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--deadline-seconds", "90.001"])


def test_process_group_termination_respects_the_absolute_deadline() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "print('ready',flush=True);time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == "ready\n"
    started = time.monotonic()
    stdout, stderr = _terminate_process_group(
        process,
        deadline=started + 0.2,
    )
    elapsed = time.monotonic() - started

    assert process.returncode == -signal.SIGKILL
    assert stdout == ""
    assert stderr == ""
    assert elapsed < 1.0
