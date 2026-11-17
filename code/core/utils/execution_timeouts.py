"""Validate an operator's exact benchmark execution deadline override."""

from __future__ import annotations


def validate_execution_timeout_seconds(value: int | None) -> int | None:
    """Keep an omitted override or accept a nonnegative integer, including zero."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("execution_timeout_seconds must be a nonnegative integer or None")
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
