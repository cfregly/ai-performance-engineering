"""Run an owned process and let its cancellation cleanup finish."""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import select
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import psutil

_CLEANUP_EVENT_NONCE_ENV = "AISP_CLEANUP_EVENT_NONCE"


@dataclass
class DeferredSigintState:
    """Record interrupts deferred until an owned child has been registered."""

    interrupt_count: int = 0

    @property
    def observed(self) -> bool:
        return self.interrupt_count > 0


@dataclass
class _OwnershipHandoffState:
    """Record ownership completion independently of caller exception objects."""

    committed: bool = False
    result: dict[str, Any] | None = None

    def commit(self, result: dict[str, Any]) -> None:
        self.result = result
        self.committed = True


@contextmanager
def _count_sigint_during_owned_state_change() -> Iterator[DeferredSigintState]:
    """Count Ctrl-C without interrupting an owned-process state commit."""
    if threading.current_thread() is not threading.main_thread():
        yield DeferredSigintState()
        return
    previous_handler = signal.getsignal(signal.SIGINT)
    if previous_handler == signal.SIG_IGN:
        raise RuntimeError("Owned process state changes require SIGINT not to be ignored")
    state = DeferredSigintState()

    def record_interrupt(signum: int, frame: Any) -> None:
        del signum, frame
        state.interrupt_count += 1

    signal.signal(signal.SIGINT, record_interrupt)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, previous_handler)


@contextmanager
def defer_sigint_during_owned_launch() -> Iterator[DeferredSigintState]:
    """Defer Ctrl-C until the caller has stored the new child's identity."""
    with _count_sigint_during_owned_state_change() as state:
        yield state


def prepare_cleanup_event_env(
    env: Mapping[str, str] | None,
) -> tuple[dict[str, str], str]:
    """Provision a launch-specific token for trusted terminal cleanup events."""
    child_env = dict(os.environ if env is None else env)
    nonce = secrets.token_hex(32)
    child_env[_CLEANUP_EVENT_NONCE_ENV] = nonce
    return child_env, nonce


def consume_cleanup_event_nonce() -> str | None:
    """Remove a wrapper's event token before application code can inherit it."""
    nonce = os.environ.pop(_CLEANUP_EVENT_NONCE_ENV, None)
    return nonce if isinstance(nonce, str) and nonce else None


class CooperativeProcessInterrupted(KeyboardInterrupt):
    """An operator interrupt with captured output and the observed cleanup state."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("Operator interrupt with retained process diagnostics")
        self.result = result


class CooperativeCleanupIncomplete(BaseException):
    """Signal that cleanup of an owned worker could not be established."""

    outer_process_result: dict[str, Any]
    cleanup_protocol_error: str

    def __init__(self, result: dict[str, Any]):
        group = result["execution"]["process_group_id"]
        super().__init__(f"Cannot continue after incomplete cleanup of owned process group {group}")
        self.result = result
        self.cooperative_process_result = result


class CooperativeCoordinatorError(RuntimeError):
    """A coordinator failure whose exception object could not retain the cleanup receipt."""

    def __init__(self, result: dict[str, Any], original_error: BaseException):
        super().__init__(
            "Owned-process cleanup completed after a coordinator error, but the "
            f"{type(original_error).__name__} exception could not retain the cleanup receipt"
        )
        self.result = result
        self.cooperative_process_result = result
        self.original_error = original_error


def _exception_message(error: BaseException) -> tuple[str, int]:
    try:
        return str(error), 0
    except KeyboardInterrupt:
        return f"<KeyboardInterrupt while rendering {type(error).__name__}>", 1
    except BaseException as rendering_error:
        return (
            f"<{type(rendering_error).__name__} while rendering {type(error).__name__}>",
            0,
        )


def _attach_process_result(
    error: BaseException,
    result: dict[str, Any],
) -> tuple[bool, int]:
    try:
        setattr(error, "cooperative_process_result", result)  # noqa: B010
    except BaseException as attachment_error:
        message, rendering_interrupts = _exception_message(attachment_error)
        attachment_interrupts = (
            1 if isinstance(attachment_error, KeyboardInterrupt) else 0
        ) + rendering_interrupts
        result["execution"]["receipt_attachment_error"] = {
            "type": type(attachment_error).__name__,
            "message": message,
        }
        return False, attachment_interrupts
    return True, 0


def _record_post_cleanup_interrupts(
    result: dict[str, Any],
    count: int,
) -> None:
    if count <= 0:
        return
    execution = result["execution"]
    cancellation = result.get("cancellation")
    if not isinstance(cancellation, dict):
        cancellation = {
            "signal": "SIGINT",
            "process_group_id": execution["process_group_id"],
            "reason": "interrupt",
            "signal_forwarded": False,
            "signal_not_forwarded_reason": "process_group_already_quiescent",
        }
        result["cancellation"] = cancellation
    previous = cancellation.get("interrupt_count", 0)
    if isinstance(previous, bool) or not isinstance(previous, int):
        previous = 0
    total = previous + count
    cancellation["operator_interrupt_observed"] = True
    cancellation["interrupt_count"] = total
    cancellation["repeated_interrupts"] = max(total - 1, 0)
    cancellation["status"] = (
        "cleanup_incomplete"
        if execution.get("completion_scope") == "cleanup_incomplete"
        else "cleanup_completed"
    )
    result["cancelled"] = True


def _exception_instance_attributes(error: BaseException) -> dict[str, Any]:
    try:
        values = object.__getattribute__(error, "__dict__")
    except BaseException:
        return {}
    return values if isinstance(values, dict) else {}


def _exception_link(error: BaseException, name: str) -> BaseException | None:
    try:
        descriptor = vars(BaseException)[name]
        linked = descriptor.__get__(error, type(error))
    except BaseException:
        return None
    return linked if isinstance(linked, BaseException) else None


def process_receipt_from_error(error: BaseException) -> dict[str, Any] | None:
    """Read retained receipts without invoking user-defined exception properties."""
    attributes = _exception_instance_attributes(error)
    for name in (
        "result",
        "cooperative_process_result",
        "outer_process_result",
    ):
        receipt = attributes.get(name)
        if isinstance(receipt, dict):
            return receipt
    return None


def validate_cleanup_incomplete_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValueError("Incomplete cleanup event requires a receipt object")
    execution = receipt.get("execution")
    if not isinstance(execution, dict) or execution.get("completion_scope") != "cleanup_incomplete":
        raise ValueError("Incomplete cleanup event requires an incomplete execution receipt")
    group = execution.get("process_group_id")
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
        raise ValueError("Incomplete cleanup event requires a positive process group ID")
    if not all(isinstance(receipt.get(name), str) for name in ("stdout", "stderr")):
        raise ValueError("Incomplete cleanup event requires captured stdout and stderr")
    return receipt


def emit_cleanup_incomplete_event(
    receipt: dict[str, Any],
    *,
    nonce: str | None,
    stream: TextIO | None = None,
) -> None:
    """Carry terminal cleanup state across a process boundary on one JSON line."""
    payload = {
        "event": "aisp_cleanup_incomplete",
        "nonce": nonce,
        "receipt": validate_cleanup_incomplete_receipt(receipt),
    }
    print(json.dumps(payload), file=sys.stderr if stream is None else stream, flush=True)


class _MalformedAuthenticatedCleanupEventError(ValueError):
    """The launch nonce matched, but the event did not contain a valid receipt."""


def parse_cleanup_incomplete_event(
    stderr: str,
    *,
    expected_nonce: str,
) -> dict[str, Any] | None:
    """Read a launch-authenticated cleanup event without trusting ordinary stderr."""
    if not isinstance(expected_nonce, str) or not expected_nonce:
        raise ValueError("Cleanup event parsing requires a launch-specific nonce")
    for line in reversed(stderr.splitlines()):
        try:
            payload = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(payload, dict) or payload.get("event") != "aisp_cleanup_incomplete":
            continue
        nonce = payload.get("nonce")
        if not isinstance(nonce, str) or not nonce.isascii():
            continue
        if not hmac.compare_digest(nonce, expected_nonce):
            continue
        try:
            return validate_cleanup_incomplete_receipt(payload.get("receipt"))
        except ValueError as error:
            raise _MalformedAuthenticatedCleanupEventError(str(error)) from error
    return None


def cleanup_incomplete_receipt_from_result(result: Any) -> dict[str, Any] | None:
    """Extract coordinator proof or an authenticated nested cleanup event."""
    if not isinstance(result, dict):
        return None
    execution = result.get("execution")
    if isinstance(execution, dict) and execution.get("completion_scope") == "cleanup_incomplete":
        return validate_cleanup_incomplete_receipt(result)
    nonce = execution.get("cleanup_event_nonce") if isinstance(execution, dict) else None
    if not isinstance(nonce, str) or not nonce:
        return None
    return parse_cleanup_incomplete_event(
        str(result.get("stderr", "")),
        expected_nonce=nonce,
    )


def authenticated_cleanup_receipt_or_raise(result: dict[str, Any]) -> dict[str, Any] | None:
    """Preserve the outer process evidence if a trusted event has an invalid receipt."""
    try:
        return cleanup_incomplete_receipt_from_result(result)
    except _MalformedAuthenticatedCleanupEventError as protocol_error:
        execution = result["execution"]
        message = str(protocol_error)
        receipt = {
            **result,
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
            "execution": {
                **execution,
                "outer_completion_scope": execution.get("completion_scope"),
                "completion_scope": "cleanup_incomplete",
                "live_process_group_members": None,
                "cleanup_protocol_scope": "authenticated_nested_event",
                "cleanup_protocol_error": message,
            },
            "outer_execution": dict(execution),
            "cleanup_protocol_error": message,
        }
        fatal = CooperativeCleanupIncomplete(validate_cleanup_incomplete_receipt(receipt))
        fatal.outer_process_result = result
        fatal.cleanup_protocol_error = message
        raise fatal from protocol_error


def cleanup_incomplete_receipt_from_error(error: BaseException) -> dict[str, Any] | None:
    """Find terminal cleanup through wrapper exceptions without losing its scope."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        attributes = _exception_instance_attributes(current)
        receipt = attributes.get("cleanup_incomplete_receipt")
        if receipt is not None:
            return validate_cleanup_incomplete_receipt(receipt)
        for name in ("result", "cooperative_process_result", "outer_process_result"):
            receipt = cleanup_incomplete_receipt_from_result(attributes.get(name))
            if receipt is not None:
                return receipt
        for previous in (
            _exception_link(current, "__context__"),
            _exception_link(current, "__cause__"),
        ):
            if isinstance(previous, BaseException):
                pending.append(previous)
    return None


def operator_interrupt_from_error(error: BaseException) -> bool:
    """Report whether an exception chain or retained receipt observed Ctrl-C."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, KeyboardInterrupt):
            return True
        attributes = _exception_instance_attributes(current)
        for name in ("cleanup_incomplete_receipt", "result", "cooperative_process_result"):
            receipt = attributes.get(name)
            cancellation = receipt.get("cancellation") if isinstance(receipt, dict) else None
            if (
                isinstance(cancellation, dict)
                and cancellation.get("operator_interrupt_observed") is True
            ):
                return True
        for previous in (
            _exception_link(current, "__context__"),
            _exception_link(current, "__cause__"),
        ):
            if isinstance(previous, BaseException):
                pending.append(previous)
    return False


def _live_process_group_members(process_group_id: int) -> list[int]:
    """Return processes that can still execute in this owned POSIX group."""
    members = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) != process_group_id:
                continue
            if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                members.append(process.pid)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return sorted(members)


def run_cooperative_process(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    input: str | None = None,
    input_factory: Callable[[subprocess.Popen[str]], str] | None = None,
    preexec_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Capture output without a deadline and forward one SIGINT on cancellation."""
    if os.name != "posix":
        raise RuntimeError(
            "SKIPPED: cooperative subprocess execution requires POSIX process groups"
        )
    require_cooperative_process_support()
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(value, str) for value in command)
    ):
        raise ValueError("command must be a nonempty argument vector of strings")
    if input is not None and input_factory is not None:
        raise ValueError("input and input_factory are mutually exclusive")
    argv = list(command)
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    launch: DeferredSigintState | None = None
    ownership_handoff = _OwnershipHandoffState()
    child_env, cleanup_event_nonce = prepare_cleanup_event_env(env)
    launch_context = (
        defer_sigint_during_owned_launch()
        if threading.current_thread() is threading.main_thread()
        else nullcontext(DeferredSigintState())
    )
    try:
        with launch_context as launch:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=child_env,
                stdin=subprocess.PIPE if input is not None or input_factory is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=preexec_fn,
            )
        if launch.observed:
            return communicate_cooperatively(
                process,
                process_group_id=process.pid,
                started_at=started,
                cancellation_requested=True,
                cleanup_event_nonce=cleanup_event_nonce,
                _ownership_handoff_state=ownership_handoff,
            )
        process_input = input_factory(process) if input_factory is not None else input
        return communicate_cooperatively(
            process,
            process_group_id=process.pid,
            input=process_input,
            started_at=started,
            cleanup_event_nonce=cleanup_event_nonce,
            _ownership_handoff_state=ownership_handoff,
        )
    except BaseException as error:
        if ownership_handoff.committed:
            result = ownership_handoff.result
            if result is not None and not isinstance(
                error,
                (CooperativeProcessInterrupted, CooperativeCleanupIncomplete),
            ):
                attached, attachment_interrupts = _attach_process_result(
                    error,
                    result,
                )
                interrupt_count = attachment_interrupts + int(isinstance(error, KeyboardInterrupt))
                _record_post_cleanup_interrupts(result, interrupt_count)
                if attachment_interrupts:
                    raise CooperativeProcessInterrupted(result) from error
                if not attached:
                    if isinstance(error, KeyboardInterrupt):
                        raise CooperativeProcessInterrupted(result) from error
                    raise CooperativeCoordinatorError(result, error) from error
            raise
        if launch is not None and launch.observed and process is None:
            raise KeyboardInterrupt from error
        if process is not None:
            if isinstance(error, KeyboardInterrupt):
                return communicate_cooperatively(
                    process,
                    process_group_id=process.pid,
                    started_at=started,
                    cancellation_requested=True,
                    cleanup_event_nonce=cleanup_event_nonce,
                    _ownership_handoff_state=ownership_handoff,
                )
            if launch is not None and launch.observed:
                return communicate_cooperatively(
                    process,
                    process_group_id=process.pid,
                    started_at=started,
                    cancellation_requested=True,
                    abort_error=error,
                    cleanup_event_nonce=cleanup_event_nonce,
                    _ownership_handoff_state=ownership_handoff,
                )
            communicate_cooperatively(
                process,
                process_group_id=process.pid,
                started_at=started,
                abort_error=error,
                cleanup_event_nonce=cleanup_event_nonce,
                _ownership_handoff_state=ownership_handoff,
            )
        raise


def require_cooperative_process_support() -> None:
    """Reject unsupported ownership semantics before launching a worker."""
    if not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")):
        raise RuntimeError("SKIPPED: owned subprocess cleanup requires waitid with WNOWAIT")
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        raise RuntimeError("SKIPPED: cooperative cancellation requires SIGINT not to be ignored")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError(
            "SKIPPED: owned subprocess cleanup requires the default SIGCHLD disposition and a sole child reaper"
        )


def communicate_cooperatively(
    process: subprocess.Popen[str],
    *,
    process_group_id: int,
    input: str | None = None,
    started_at: float | None = None,
    cancellation_requested: bool = False,
    abort_error: BaseException | None = None,
    cleanup_event_nonce: str | None = None,
    _ownership_handoff_state: _OwnershipHandoffState | None = None,
) -> dict[str, Any]:
    """Drain output and own the group until cancellation or leader-exit cleanup ends.

    A completed leader ends the command's lifetime. Residual group members are
    killed before reaping that leader, which reserves its PID and group identity.
    Cancellation instead forwards one SIGINT and waits for cooperative cleanup.
    Neither path imposes an execution deadline. The caller must be the sole
    reaper for this child. Descendants that leave the group are outside this
    ownership contract.
    """
    if os.name != "posix":
        raise RuntimeError(
            "SKIPPED: cooperative subprocess execution requires POSIX process groups"
        )
    require_cooperative_process_support()
    waitid: Any = getattr(os, "waitid")
    if isinstance(process_group_id, bool) or process_group_id != process.pid:
        raise ValueError(
            "Cooperative execution requires the child's explicitly owned process group"
        )
    ownership_error: Exception | None = None
    try:
        observed_group = os.getpgid(process.pid)
    except ProcessLookupError:
        # A quick child may exit before this query. Communicate still owns reaping and output.
        pass
    except Exception as error:
        ownership_error = error
    else:
        if observed_group != process_group_id:
            raise ValueError("Cooperative execution requires a process group owned by this child")
    started = time.perf_counter() if started_at is None else started_at
    raw_argv = process.args
    if (
        isinstance(raw_argv, (str, bytes))
        or not isinstance(raw_argv, Sequence)
        or any(not isinstance(value, str) for value in raw_argv)
    ):
        raise ValueError("Cooperative execution requires an argument vector")
    argv = [value for value in raw_argv if isinstance(value, str)]
    cancellation: dict[str, Any] | None = None
    pending_error = abort_error
    completion_cleanup: dict[str, Any] | None = None
    already_reaped = process.returncode is not None
    operator_interrupts = int(cancellation_requested or isinstance(abort_error, KeyboardInterrupt))
    signal_attempted = False
    group_quiescent = False
    captured_bytes: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    capture_errors: list[str] = []
    cleanup_incomplete = False
    group_observation_failed = False
    lifecycle_observation_failed = False
    ownership_lost = False
    leader_exited = already_reaped
    leader_status: Any | None = None
    abort_cleanup = False
    capture_limit: dict[str, Any] | None = None

    def request_cancellation(*, interrupts: int = 0) -> None:
        nonlocal cancellation, ownership_lost, operator_interrupts, signal_attempted
        forwarding_error = None
        pending_interrupts = interrupts
        while True:
            deferred = DeferredSigintState()
            try:
                with _count_sigint_during_owned_state_change() as deferred:
                    operator_interrupts += pending_interrupts
                    pending_interrupts = 0
                    if cancellation is None:
                        cancellation = {
                            "signal": "SIGINT",
                            "process_group_id": process.pid,
                            "status": "waiting_for_cleanup",
                            "repeated_interrupts": 0,
                            "reason": "coordinator_error"
                            if pending_error is not None
                            else "interrupt",
                            "signal_forwarded": False,
                        }
                    cancellation["operator_interrupt_observed"] = operator_interrupts > 0
                    cancellation["interrupt_count"] = operator_interrupts
                    cancellation["repeated_interrupts"] = max(operator_interrupts - 1, 0)
                    if not signal_attempted and forwarding_error is None:
                        signal_attempted = True
                        if ownership_lost:
                            raise RuntimeError(
                                "Cannot signal after losing the owned leader identity"
                            )
                        if already_reaped or process.returncode is not None:
                            if not group_quiescent and _live_process_group_members(
                                process_group_id
                            ):
                                ownership_lost = True
                                raise RuntimeError(
                                    "Cannot signal a reaped leader's live process group safely"
                                )
                            cancellation["signal_not_forwarded_reason"] = "leader_reaped"
                        else:
                            try:
                                os.killpg(process.pid, signal.SIGINT)
                                cancellation["signal_forwarded"] = True
                            except ProcessLookupError:
                                pass
            except KeyboardInterrupt:
                pending_interrupts += 1
                continue
            except Exception as error:
                forwarding_error = error
                if cancellation is not None:
                    cancellation["signal_error"] = f"{type(error).__name__}: {error}"
            pending_interrupts += deferred.interrupt_count
            if pending_interrupts:
                continue
            if forwarding_error is not None:
                raise forwarding_error
            return

    def retain_error(error: BaseException) -> None:
        nonlocal pending_error, operator_interrupts
        message, rendering_interrupts = _exception_message(error)
        operator_interrupts += rendering_interrupts
        capture_errors.append(f"{type(error).__name__}: {message}")
        if pending_error is None:
            pending_error = error

    def capture_stream_chunk(name: str, stream: TextIO) -> bytes | None:
        """Read and retain one pipe chunk before delivering a pending SIGINT."""
        deferred = DeferredSigintState()
        try:
            with _count_sigint_during_owned_state_change() as deferred:
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    return None
                if chunk:
                    captured_bytes[name].extend(chunk)
                return chunk
        finally:
            if deferred.observed:
                request_cancellation(interrupts=deferred.interrupt_count)

    def write_stream_chunk(stream: TextIO, payload: bytes) -> int:
        """Commit one pipe write before delivering a pending SIGINT."""
        deferred = DeferredSigintState()
        try:
            with _count_sigint_during_owned_state_change() as deferred:
                return os.write(stream.fileno(), payload)
        finally:
            if deferred.observed:
                request_cancellation(interrupts=deferred.interrupt_count)

    def final_nonblocking_capture(
        streams: Sequence[tuple[str, TextIO]],
        *,
        reason: str,
    ) -> None:
        """Retain currently available bytes, then close pipes outside ownership."""
        nonlocal capture_limit
        streams_without_eof: list[str] = []
        first_error: Exception | None = None
        for name, stream in streams:
            if stream.closed:
                continue
            if name not in captured_bytes:
                try:
                    stream.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
                continue
            reached_eof = False
            try:
                os.set_blocking(stream.fileno(), False)
                while True:
                    chunk = capture_stream_chunk(name, stream)
                    if chunk is None:
                        break
                    if not chunk:
                        reached_eof = True
                        break
            except Exception as error:
                if first_error is None:
                    first_error = error
            finally:
                try:
                    stream.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            if not reached_eof:
                streams_without_eof.append(name)
        if streams_without_eof:
            capture_limit = {
                "reason": reason,
                "strategy": "final_nonblocking_drain_then_close",
                "streams_closed_before_eof": sorted(set(streams_without_eof)),
                "bytes_retained": {
                    name: len(captured_bytes[name]) for name in sorted(set(streams_without_eof))
                },
            }
        if first_error is not None:
            raise first_error

    def group_members() -> list[int]:
        nonlocal group_observation_failed
        try:
            return _live_process_group_members(process_group_id)
        except Exception:
            group_observation_failed = True
            raise

    def wait_for_quiescence() -> None:
        nonlocal group_quiescent
        while True:
            try:
                if group_quiescent or not group_members():
                    group_quiescent = True
                    # Reap last so the leader reserves this group's identity.
                    reap_leader()
                    return
                # POSIX groups have no portable wait handle for non-child members.
                time.sleep(0.05)
            except KeyboardInterrupt:
                request_cancellation(interrupts=1)

    def reap_leader() -> None:
        nonlocal already_reaped, leader_status, ownership_lost, lifecycle_observation_failed
        if already_reaped:
            return
        reaping_error: BaseException | None = None
        pending_interrupts = 0
        while not already_reaped and reaping_error is None:
            deferred = DeferredSigintState()
            try:
                with _count_sigint_during_owned_state_change() as deferred:
                    if leader_status is None:
                        leader_status = waitid(
                            os.P_PID,
                            process.pid,
                            os.WEXITED | os.WNOWAIT,
                        )
                    if leader_status.si_code == os.CLD_EXITED:
                        process.returncode = leader_status.si_status
                    elif leader_status.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
                        process.returncode = -leader_status.si_status
                    else:
                        raise RuntimeError(f"Unexpected owned leader wait status: {leader_status}")
                    try:
                        waitid(os.P_PID, process.pid, os.WEXITED)
                    except ChildProcessError as error:
                        process.returncode = None
                        ownership_lost = True
                        lifecycle_observation_failed = True
                        reaping_error = RuntimeError(
                            "The owned leader was reaped by another caller before its status was retained"
                        )
                        reaping_error.__cause__ = error
                    else:
                        already_reaped = True
            except KeyboardInterrupt:
                pending_interrupts += 1
            except BaseException as error:
                lifecycle_observation_failed = True
                reaping_error = error
            pending_interrupts += deferred.interrupt_count
            if pending_interrupts:
                request_cancellation(interrupts=pending_interrupts)
                pending_interrupts = 0
        if reaping_error is not None:
            raise reaping_error

    try:
        if ownership_error is not None:
            ownership_lost = True
            raise ownership_error
        if cancellation_requested or pending_error is not None:
            request_cancellation()
        input_encoding: Any = (
            getattr(process.stdin, "encoding", "utf-8") if process.stdin is not None else "utf-8"
        )
        input_bytes = (input or "").encode(input_encoding)
        input_offset = 0
        # Raw I/O observes leader exit even when descendants retain capture pipes.
        # This selector interval observes lifecycle events. It is not a deadline.
        with selectors.DefaultSelector() as selector:
            streams = [("stdout", process.stdout), ("stderr", process.stderr)]
            if process.stdin is not None:
                streams.append(("stdin", process.stdin))
            for name, stream in streams:
                if stream is None or stream.closed:
                    continue
                if name == "stdin" and not input_bytes:
                    stream.close()
                    continue
                os.set_blocking(stream.fileno(), False)
                selector.register(
                    stream,
                    selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ,
                    name,
                )
            while True:
                try:
                    if not leader_exited:
                        try:
                            status = waitid(
                                os.P_PID,
                                process.pid,
                                os.WEXITED | os.WNOWAIT | os.WNOHANG,
                            )
                        except ChildProcessError as error:
                            ownership_lost = True
                            raise RuntimeError(
                                "The owned leader was reaped by another caller before cleanup"
                            ) from error
                        except Exception:
                            lifecycle_observation_failed = True
                            raise
                        if status is not None:
                            leader_status = status
                            leader_exited = True
                    if leader_exited:
                        members = group_members()
                        if cancellation is None and members and completion_cleanup is None:
                            if already_reaped:
                                ownership_lost = True
                                raise RuntimeError(
                                    "Cannot clean a reaped leader's live process group safely"
                                )
                            os.killpg(process_group_id, signal.SIGKILL)
                            completion_cleanup = {
                                "reason": "leader_exited",
                                "signal": "SIGKILL",
                                "process_ids": members,
                            }
                        if not members:
                            group_quiescent = True
                            final_streams: list[tuple[str, TextIO]] = []
                            for key in list(selector.get_map().values()):
                                selected_stream = cast(TextIO, key.fileobj)
                                stream_name = cast(str, key.data)
                                selector.unregister(selected_stream)
                                final_streams.append((stream_name, selected_stream))
                            final_nonblocking_capture(
                                final_streams,
                                reason="owned_process_group_quiescent_before_capture_eof",
                            )
                            break
                    if leader_exited and not selector.get_map():
                        break
                    for key, _ in selector.select(0.05):
                        selected_stream = cast(TextIO, key.fileobj)
                        stream_name = cast(str, key.data)
                        try:
                            if stream_name == "stdin":
                                input_offset += write_stream_chunk(
                                    selected_stream,
                                    input_bytes[input_offset : input_offset + 4096],
                                )
                                if input_offset == len(input_bytes):
                                    selector.unregister(selected_stream)
                                    selected_stream.close()
                            else:
                                chunk = capture_stream_chunk(stream_name, selected_stream)
                                if chunk is None:
                                    continue
                                if not chunk:
                                    selector.unregister(selected_stream)
                                    selected_stream.close()
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            # Early stdin closure must not discard child diagnostics.
                            selector.unregister(selected_stream)
                            selected_stream.close()
                except KeyboardInterrupt:
                    request_cancellation(interrupts=1)
        wait_for_quiescence()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            try:
                request_cancellation(interrupts=1)
            except Exception as cleanup_error:
                retain_error(cleanup_error)
                cleanup_incomplete = True
        else:
            retain_error(error)
        while True:
            try:
                abort_cleanup = True
                # An actual coordinator failure ends normal I/O. This distinct cleanup
                # path preserves its error and never retries the failed selector setup.
                try:
                    if ownership_lost:
                        raise RuntimeError("Cannot signal after losing the owned leader identity")
                    if cancellation is not None and "signal_error" in cancellation:
                        raise RuntimeError(
                            "Cannot complete cooperative cleanup after SIGINT forwarding failed"
                        )
                    request_cancellation()
                except BaseException as cleanup_error:
                    if isinstance(cleanup_error, KeyboardInterrupt):
                        raise
                    retain_error(cleanup_error)
                    cleanup_incomplete = True
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except Exception as cleanup_error:
                        retain_error(cleanup_error)
                    finally:
                        process.stdin = None
                remaining: dict[int, tuple[str, TextIO]] = {}
                for name in ("stdout", "stderr"):
                    stream = getattr(process, name)
                    if stream is None or stream.closed:
                        continue
                    try:
                        descriptor = stream.fileno()
                        remaining[descriptor] = (name, stream)
                        try:
                            os.set_blocking(stream.fileno(), False)
                        except Exception as cleanup_error:
                            retain_error(cleanup_error)
                            cleanup_incomplete = True
                    except Exception as cleanup_error:
                        retain_error(cleanup_error)
                        cleanup_incomplete = True
                        try:
                            stream.close()
                        except Exception as close_error:
                            retain_error(close_error)

                if cleanup_incomplete or group_observation_failed:
                    reason = (
                        "owned_process_group_observation_failed_before_capture_eof"
                        if group_observation_failed
                        else "owned_process_cleanup_incomplete_before_capture_eof"
                    )
                    try:
                        final_nonblocking_capture(list(remaining.values()), reason=reason)
                    except Exception as cleanup_error:
                        retain_error(cleanup_error)
                        cleanup_incomplete = True
                    if group_observation_failed:
                        cleanup_incomplete = True
                        # A previously observed exit can still be reaped safely. Do not
                        # retry a failed group query or claim its descendants are gone.
                        if leader_exited:
                            try:
                                reap_leader()
                            except Exception as cleanup_error:
                                retain_error(cleanup_error)
                else:
                    while True:
                        try:
                            if not group_members():
                                group_quiescent = True
                                final_nonblocking_capture(
                                    list(remaining.values()),
                                    reason="owned_process_group_quiescent_before_capture_eof",
                                )
                                remaining.clear()
                                wait_for_quiescence()
                                break
                            if remaining:
                                readable, _, _ = select.select(
                                    list(remaining),
                                    [],
                                    [],
                                    0.05,
                                )
                                for descriptor in readable:
                                    name, stream = remaining[descriptor]
                                    chunk = capture_stream_chunk(name, stream)
                                    if chunk is None:
                                        continue
                                    if not chunk:
                                        del remaining[descriptor]
                                        stream.close()
                            else:
                                time.sleep(0.05)
                        except KeyboardInterrupt:
                            try:
                                request_cancellation(interrupts=1)
                            except Exception as cleanup_error:
                                retain_error(cleanup_error)
                                cleanup_incomplete = True
                                break
                        except Exception as cleanup_error:
                            retain_error(cleanup_error)
                            cleanup_incomplete = True
                            try:
                                final_nonblocking_capture(
                                    list(remaining.values()),
                                    reason=(
                                        "owned_process_group_observation_failed_before_capture_eof"
                                        if group_observation_failed
                                        else "owned_process_abort_capture_failed_before_eof"
                                    ),
                                )
                            except Exception as capture_error:
                                retain_error(capture_error)
                            if group_observation_failed and leader_exited:
                                try:
                                    reap_leader()
                                except Exception as reap_error:
                                    retain_error(reap_error)
                            break
                cleanup_incomplete = cleanup_incomplete or lifecycle_observation_failed
                break
            except KeyboardInterrupt:
                try:
                    request_cancellation(interrupts=1)
                except Exception as cleanup_error:
                    retain_error(cleanup_error)
                    cleanup_incomplete = True

    def build_result() -> dict[str, Any]:
        nonlocal cancellation, operator_interrupts
        captured: dict[str, str] = {}
        undecodable_streams: dict[str, str] = {}
        for name in ("stdout", "stderr"):
            stream = getattr(process, name)
            encoding = stream.encoding if stream is not None else "utf-8"
            raw = bytes(captured_bytes[name])
            try:
                captured[name] = raw.decode(encoding)
            except UnicodeDecodeError as error:
                captured[name] = raw.decode(encoding, errors="replace")
                undecodable_streams[name] = base64.b64encode(raw).decode("ascii")
                capture_errors.append(f"{name} decoding failed: {error}")
            captured[name] = captured[name].replace("\r\n", "\n").replace("\r", "\n")
        coordinator_error = None
        if pending_error is not None:
            message, rendering_interrupts = _exception_message(pending_error)
            operator_interrupts += rendering_interrupts
            coordinator_error = {
                "type": type(pending_error).__name__,
                "message": message,
            }
        if operator_interrupts and cancellation is None:
            # An ownership failure can prevent forwarding the requested signal.
            cancellation = {
                "signal": "SIGINT",
                "process_group_id": process.pid,
                "reason": "interrupt",
                "signal_forwarded": False,
            }
        if cancellation is not None:
            cancellation["status"] = (
                "cleanup_incomplete" if cleanup_incomplete else "cleanup_completed"
            )
            cancellation["operator_interrupt_observed"] = bool(operator_interrupts)
            cancellation["interrupt_count"] = operator_interrupts
            cancellation["repeated_interrupts"] = max(operator_interrupts - 1, 0)
        result: dict[str, Any] = {
            "command": shlex.join(argv),
            "returncode": process.returncode,
            "stdout": captured["stdout"] or "",
            "stderr": captured["stderr"] or "",
            "duration_seconds": time.perf_counter() - started,
            "timeout_seconds": None,
            "timeout_hit": False,
            "cancelled": cancellation is not None,
            "execution": {
                "process_id": process.pid,
                "process_group_id": process.pid,
                "start_new_session": True,
                "live_process_group_members": None if cleanup_incomplete else [],
                "completion_scope": "cleanup_incomplete"
                if cleanup_incomplete
                else "process_group_quiescent",
            },
        }
        if cleanup_event_nonce is not None:
            result["execution"]["cleanup_event_nonce"] = cleanup_event_nonce
        if completion_cleanup is not None:
            result["execution"]["completion_cleanup"] = completion_cleanup
        if capture_errors:
            result["execution"]["capture_errors"] = capture_errors
        if capture_limit is not None:
            result["execution"]["capture_limit"] = capture_limit
        if abort_cleanup:
            result["execution"]["abort_cleanup"] = "raw_pipe_drain"
        if undecodable_streams:
            result["execution"]["undecodable_streams_base64"] = undecodable_streams
            result["raw_streams_base64"] = undecodable_streams
        if cancellation is not None:
            result["cancellation"] = cancellation
        if coordinator_error is not None:
            result["execution"]["coordinator_error"] = coordinator_error
        return result

    while True:
        try:
            result = build_result()
            if _ownership_handoff_state is not None:
                _ownership_handoff_state.commit(result)
            if pending_error is not None:
                result_attached, attachment_interrupts = _attach_process_result(
                    pending_error,
                    result,
                )
                operator_interrupts += attachment_interrupts
                _record_post_cleanup_interrupts(result, attachment_interrupts)
            else:
                result_attached = True
            if operator_interrupts:
                raise CooperativeProcessInterrupted(result) from pending_error
            if cleanup_incomplete:
                raise CooperativeCleanupIncomplete(result) from pending_error
            if pending_error is not None:
                if not result_attached:
                    raise CooperativeCoordinatorError(
                        result,
                        pending_error,
                    ) from pending_error
                raise pending_error
            return result
        except CooperativeProcessInterrupted:
            raise
        except KeyboardInterrupt:
            try:
                request_cancellation(interrupts=1)
            except Exception as cleanup_error:
                retain_error(cleanup_error)
                cleanup_incomplete = True
