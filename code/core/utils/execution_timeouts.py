"""Validate an operator's exact benchmark execution deadline override."""

from __future__ import annotations

import ctypes
import math
import os
import select
import threading


# Keep the existing allowance shared with the harness without importing CUDA code.
RUNTIME_PROVENANCE_TIMEOUT_SECONDS = 15.0

# subprocess uses PollSelector on POSIX when poll is available. Its timeout is
# rounded up to milliseconds and passed as a signed C int. Threaded waits also
# have the platform limit exposed by threading.TIMEOUT_MAX.
_POLL_TIMEOUT_MAX_MILLISECONDS = (
    (1 << (ctypes.sizeof(ctypes.c_int) * 8 - 1)) - 1
    if os.name == "posix" and hasattr(select, "poll") else None
)
MAX_SUBPROCESS_WAIT_SECONDS = min(
    threading.TIMEOUT_MAX,
    _POLL_TIMEOUT_MAX_MILLISECONDS / 1000
    if _POLL_TIMEOUT_MAX_MILLISECONDS is not None else threading.TIMEOUT_MAX,
)
MAX_THREAD_WAIT_SECONDS = threading.TIMEOUT_MAX
MAX_EXECUTION_TIMEOUT_SECONDS = math.floor(
    MAX_SUBPROCESS_WAIT_SECONDS - RUNTIME_PROVENANCE_TIMEOUT_SECONDS
)


def validate_thread_wait_timeout(value: int | float) -> int | float:
    """Check a resolved future wait against the platform thread limit."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or value > MAX_THREAD_WAIT_SECONDS
        or not math.isfinite(value)
    ):
        raise ValueError(
            "Thread timeout must be finite, nonnegative, and within this "
            f"platform's supported wait of {MAX_THREAD_WAIT_SECONDS} seconds"
        )
    return value


def validate_subprocess_wait_timeout(
    value: int | float,
    *,
    additional_seconds: int | float = 0,
) -> int | float:
    """Check a resolved wait against platform timer limits before process launch."""
    for component in (value, additional_seconds):
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or component < 0
            or component > MAX_SUBPROCESS_WAIT_SECONDS
            or not math.isfinite(component)
        ):
            raise ValueError(
                "Subprocess timeout must be finite, nonnegative, and within this "
                f"platform's supported wait of {MAX_SUBPROCESS_WAIT_SECONDS} seconds"
            )
    combined = value + additional_seconds
    if (
        combined > threading.TIMEOUT_MAX
        or (
            _POLL_TIMEOUT_MAX_MILLISECONDS is not None
            and math.ceil(combined * 1000) > _POLL_TIMEOUT_MAX_MILLISECONDS
        )
    ):
        raise ValueError(
            "Subprocess timeout including its allowance exceeds this platform's "
            f"supported wait of {MAX_SUBPROCESS_WAIT_SECONDS} seconds"
        )
    return value


def validate_execution_timeout_seconds(value: int | None) -> int | None:
    """Preserve zero and reject finite overrides the platform cannot wait for."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("execution_timeout_seconds must be a nonnegative integer or None")
    if value > MAX_EXECUTION_TIMEOUT_SECONDS:
        raise ValueError(
            "execution_timeout_seconds exceeds this platform's supported finite wait. "
            f"The maximum is {MAX_EXECUTION_TIMEOUT_SECONDS} seconds with the existing "
            f"{RUNTIME_PROVENANCE_TIMEOUT_SECONDS:g}-second runtime-provenance allowance"
        )
    return value


def execution_deadline_seconds(value: int | float | None) -> int | float | None:
    """Translate an explicit zero to an unbounded execution wait."""
    if value is not None and not isinstance(value, bool) and value == 0:
        return None
    return value


def validate_execution_timeout_mode(
    value: int | None,
    *,
    execution_mode: str,
    launch_via: str,
    owned_subprocess: bool = False,
) -> None:
    """Reject uncapped execution when no subprocess owns cancellation cleanup."""
    if validate_execution_timeout_seconds(value) != 0:
        return
    if launch_via != "python" or (execution_mode != "subprocess" and not owned_subprocess):
        raise ValueError(
            "execution_timeout_seconds=0 requires owned subprocess execution with "
            "launch_via='python'. Direct thread and torchrun modes are unsupported."
        )
