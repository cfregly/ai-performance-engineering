"""Check supported finite deadlines against real CPU wait primitives."""

import ctypes
import os
from pathlib import Path
import re
import select
import subprocess
import sys

import pytest

from core.utils.execution_timeouts import (
    MAX_EXECUTION_TIMEOUT_SECONDS,
    MAX_SUBPROCESS_WAIT_SECONDS,
    MAX_THREAD_WAIT_SECONDS,
    RUNTIME_PROVENANCE_TIMEOUT_SECONDS,
    validate_execution_timeout_seconds,
    validate_subprocess_wait_timeout,
    validate_thread_wait_timeout,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [None, 0, 7, 1200, MAX_EXECUTION_TIMEOUT_SECONDS])
def test_supported_operator_values_remain_exact(value) -> None:
    assert validate_execution_timeout_seconds(value) == value


@pytest.mark.parametrize("value", [0, 0.125, 7])
def test_supported_resolved_wait_values_remain_exact(value) -> None:
    assert validate_subprocess_wait_timeout(
        value, additional_seconds=RUNTIME_PROVENANCE_TIMEOUT_SECONDS,
    ) == value


@pytest.mark.parametrize("value", [MAX_EXECUTION_TIMEOUT_SECONDS + 1, 10 ** 100])
def test_unsupported_operator_wait_is_rejected_before_config_or_launch(value: int) -> None:
    with pytest.raises(ValueError, match="execution_timeout_seconds.*supported finite wait"):
        validate_execution_timeout_seconds(value)


def test_resolved_finite_wait_accounts_for_the_existing_allowance() -> None:
    maximum = MAX_EXECUTION_TIMEOUT_SECONDS
    assert validate_subprocess_wait_timeout(
        maximum, additional_seconds=RUNTIME_PROVENANCE_TIMEOUT_SECONDS,
    ) == maximum
    with pytest.raises(ValueError, match="supported wait"):
        validate_subprocess_wait_timeout(
            maximum + 1, additional_seconds=RUNTIME_PROVENANCE_TIMEOUT_SECONDS,
        )


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 10 ** 100])
def test_invalid_resolved_waits_fail_before_subprocess_use(value) -> None:
    with pytest.raises(ValueError, match="Subprocess timeout"):
        validate_subprocess_wait_timeout(value)


def test_thread_wait_boundary_is_validated_without_the_subprocess_poll_limit() -> None:
    assert validate_thread_wait_timeout(MAX_THREAD_WAIT_SECONDS) == MAX_THREAD_WAIT_SECONDS
    with pytest.raises(ValueError, match="Thread timeout.*supported wait"):
        validate_thread_wait_timeout(MAX_THREAD_WAIT_SECONDS + 1)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), 10 ** 100])
def test_invalid_thread_waits_fail_before_executor_use(value) -> None:
    with pytest.raises(ValueError, match="Thread timeout"):
        validate_thread_wait_timeout(value)


@pytest.mark.skipif(not hasattr(select, "poll"), reason="This platform does not use poll")
def test_poll_boundary_is_an_actual_platform_constraint_on_a_ready_pipe() -> None:
    maximum_milliseconds = (1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)) - 1
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, b"ready")
        poller = select.poll()
        poller.register(read_descriptor, select.POLLIN)
        assert poller.poll(maximum_milliseconds) == [(read_descriptor, select.POLLIN)]
        with pytest.raises(OverflowError):
            poller.poll(maximum_milliseconds + 1)
        assert MAX_SUBPROCESS_WAIT_SECONDS <= maximum_milliseconds / 1000
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


def test_largest_supported_operator_wait_reads_a_real_completed_cpu_child() -> None:
    with subprocess.Popen(
        [sys.executable, "-c", "print('CPU wait boundary')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as child:
        assert child.wait() == 0
        stdout, stderr = child.communicate(
            timeout=MAX_EXECUTION_TIMEOUT_SECONDS + RUNTIME_PROVENANCE_TIMEOUT_SECONDS,
        )
    assert stdout == "CPU wait boundary\n"
    assert stderr == ""


@pytest.mark.parametrize("launch_via", ["python", "torchrun"])
def test_real_cli_rejects_unsupported_wait_before_benchmark_dispatch(launch_via: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "bench", "run",
         "--targets", "nonexistent_capability_fixture",
         "--execution-timeout-seconds", str(MAX_EXECUTION_TIMEOUT_SECONDS + 1),
         "--launch-via", launch_via],
        cwd=CODE_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    plain_error = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stderr)
    assert "execution_timeout_seconds exceeds" in " ".join(plain_error.split())
    assert "SUBPROCESS DISPATCH" not in result.stdout + result.stderr
