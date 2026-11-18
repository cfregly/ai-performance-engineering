"""Run a bounded local command and clean up only processes it owns."""

from __future__ import annotations

import math
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from core.utils.cooperative_subprocess import (
    CooperativeCleanupIncomplete,
    CooperativeProcessInterrupted,
    DeferredSigintState,
    defer_sigint_during_owned_launch,
    prepare_cleanup_event_env,
)

_OWNER_ENV = "AISP_DIAGNOSTIC_PROCESS_OWNER"


def _finite_interval(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}")
    return result


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    create_time: float


def _identity(process: psutil.Process) -> _ProcessIdentity | None:
    try:
        return _ProcessIdentity(process.pid, process.create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def _resolve(identity: _ProcessIdentity) -> psutil.Process | None:
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.create_time:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


class _OwnedProcesses:
    """Retain process identities even after a child changes parent or session."""

    def __init__(self, owner_token: str, root: psutil.Process | None = None) -> None:
        self._owner_token = owner_token
        self._lock = threading.Lock()
        self._identities: dict[int, _ProcessIdentity] = {}
        self._observation_errors: set[str] = set()
        if root is not None:
            self.remember(root)

    def record_observation_error(self, operation: str, pid: int, error: BaseException) -> None:
        detail = f"{operation} pid {pid}: {type(error).__name__}: {error}"
        with self._lock:
            self._observation_errors.add(detail)

    def observation_errors(self) -> list[str]:
        with self._lock:
            return sorted(self._observation_errors)

    def remember(self, process: psutil.Process) -> None:
        try:
            identity = _identity(process)
        except psutil.AccessDenied as error:
            self.record_observation_error("identify", process.pid, error)
            return
        if identity is None:
            return
        with self._lock:
            self._identities[identity.pid] = identity

    def snapshot(self) -> list[_ProcessIdentity]:
        with self._lock:
            return list(self._identities.values())

    def discover_descendants(self) -> None:
        for identity in self.snapshot():
            try:
                process = _resolve(identity)
            except psutil.AccessDenied as error:
                self.record_observation_error("resolve for descendant discovery", identity.pid, error)
                continue
            if process is None:
                continue
            try:
                children = process.children(recursive=True)
            except psutil.AccessDenied as error:
                self.record_observation_error("discover descendants", identity.pid, error)
                continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.Error, OSError) as error:
                self.record_observation_error("discover descendants", identity.pid, error)
                continue
            for child in children:
                self.remember(child)

    def discover_owner_token(self) -> None:
        """Recover owned children that reparented or started a new session."""
        for process in psutil.process_iter():
            try:
                tokens = process.environ().get(_OWNER_ENV, "").split(",")
                if self._owner_token in tokens:
                    self.remember(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

    def live(self) -> list[psutil.Process]:
        live: list[psutil.Process] = []
        for identity in self.snapshot():
            try:
                process = _resolve(identity)
            except psutil.AccessDenied as error:
                self.record_observation_error("resolve", identity.pid, error)
                continue
            if process is not None:
                live.append(process)
        return live


def _track_descendants(owned: _OwnedProcesses, stop: threading.Event, poll_seconds: float) -> None:
    while not stop.is_set():
        try:
            owned.discover_descendants()
        except Exception as error:
            owned.record_observation_error("descendant tracker", 0, error)
            return
        stop.wait(poll_seconds)


def _signal_live(
    owned: _OwnedProcesses,
    action: str,
    errors: list[str],
) -> list[int]:
    sent: list[int] = []
    for process in owned.live():
        try:
            if action == "sigint":
                process.send_signal(signal.SIGINT)
            elif action == "terminate":
                process.terminate()
            elif action == "kill":
                process.kill()
            else:  # pragma: no cover - internal programming error
                raise ValueError(f"Unknown process action: {action}")
            sent.append(process.pid)
        except psutil.AccessDenied as exc:
            errors.append(f"{action} pid {process.pid}: {type(exc).__name__}")
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return sent


def _wait_for_owned_exit(owned: _OwnedProcesses, timeout_seconds: float) -> None:
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while owned.live() and time.monotonic() < deadline:
        time.sleep(min(0.02, max(deadline - time.monotonic(), 0.0)))


def _drain_root(
    process: psutil.Popen[Any],
    timeout_seconds: float,
) -> tuple[str, str] | None:
    try:
        stdout, stderr = process.communicate(timeout=max(timeout_seconds, 0.001))
        return stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        return None


def run_owned_process(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    input: str | None = None,
    input_factory: Callable[[psutil.Popen[Any]], str] | None = None,
    preexec_fn: Callable[[], None] | None = None,
    interrupt_grace_seconds: float = 2.0,
    terminate_grace_seconds: float = 1.0,
    poll_seconds: float = 0.02,
) -> dict[str, Any]:
    """Run ``argv`` and verify that every tracked local process has stopped."""
    timeout_seconds = _finite_interval(timeout_seconds, "timeout_seconds", positive=True)
    interrupt_grace_seconds = _finite_interval(
        interrupt_grace_seconds, "interrupt_grace_seconds", positive=False
    )
    terminate_grace_seconds = _finite_interval(
        terminate_grace_seconds, "terminate_grace_seconds", positive=False
    )
    poll_seconds = _finite_interval(poll_seconds, "poll_seconds", positive=True)
    if input is not None and input_factory is not None:
        raise ValueError("input and input_factory are mutually exclusive")
    command = [str(value) for value in argv]
    owner_token = uuid.uuid4().hex
    child_env, cleanup_event_nonce = prepare_cleanup_event_env(env)
    inherited_owner_tokens = [
        token for token in child_env.get(_OWNER_ENV, "").split(",") if token
    ]
    child_env[_OWNER_ENV] = ",".join([*inherited_owner_tokens, owner_token])
    started_monotonic = time.monotonic()
    process: psutil.Popen[Any] | None = None
    owned = _OwnedProcesses(owner_token)
    stop_tracking = threading.Event()
    tracker: threading.Thread | None = None
    launch: DeferredSigintState | None = None
    launch_context = (
        defer_sigint_during_owned_launch()
        if threading.current_thread() is threading.main_thread()
        else nullcontext(DeferredSigintState())
    )

    def _cleanup_result_once(
        *,
        reason: str,
        timeout_hit: bool,
        operator_interrupt: bool = False,
        pending_error: BaseException | None = None,
        captured_output: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        assert process is not None
        cleanup_errors: list[str] = []
        owned.discover_descendants()
        owned.discover_owner_token()
        sigint_sent = _signal_live(owned, "sigint", cleanup_errors)
        drained = captured_output
        if drained is None:
            drained = _drain_root(process, interrupt_grace_seconds)
        _wait_for_owned_exit(owned, interrupt_grace_seconds if drained is not None else 0.0)

        owned.discover_descendants()
        owned.discover_owner_token()
        terminate_sent = _signal_live(owned, "terminate", cleanup_errors)
        if drained is None:
            drained = _drain_root(process, terminate_grace_seconds)
        _wait_for_owned_exit(owned, terminate_grace_seconds)

        owned.discover_descendants()
        owned.discover_owner_token()
        kill_sent = _signal_live(owned, "kill", cleanup_errors)
        if drained is None:
            drained = _drain_root(process, 1.0)
        _wait_for_owned_exit(owned, 1.0)
        owned.discover_descendants()
        owned.discover_owner_token()
        survivors = sorted(item.pid for item in owned.live())
        cleanup_errors = sorted({*cleanup_errors, *owned.observation_errors()})
        stdout, stderr = drained or ("", "")
        incomplete = bool(survivors or cleanup_errors or drained is None)
        result: dict[str, Any] = {
            "command": shlex.join(command),
            "returncode": 124 if timeout_hit else (130 if operator_interrupt else process.poll()),
            "process_returncode": process.poll(),
            "stdout": stdout,
            "stderr": stderr,
            "timeout_seconds": timeout_seconds,
            "timeout_hit": timeout_hit,
            "duration_seconds": round(time.monotonic() - started_monotonic, 2),
            "execution": {
                "process_id": process.pid,
                "process_group_id": process.pid,
                "start_new_session": True,
                "cleanup_event_nonce": cleanup_event_nonce,
                "live_process_group_members": None if cleanup_errors else survivors,
                "completion_scope": "cleanup_incomplete" if incomplete else "process_tree_quiescent",
            },
            "termination": {
                "reason": reason,
                "tracked_processes": len(owned.snapshot()),
                "sigint_sent": sorted(sigint_sent),
                "terminate_sent": sorted(terminate_sent),
                "kill_sent": sorted(kill_sent),
                "survivors": survivors,
                "errors": cleanup_errors,
            },
        }
        if timeout_hit:
            result["error"] = f"Timed out after {timeout_seconds}s"
        if operator_interrupt:
            result["cancelled"] = True
            result["cancellation"] = {
                "reason": "interrupt",
                "operator_interrupt_observed": True,
                "signal": "SIGINT",
            }
        if pending_error is not None:
            result["execution"]["coordinator_error"] = {
                "type": type(pending_error).__name__,
                "message": str(pending_error),
            }
        return result

    def cleanup_result(
        *,
        reason: str,
        timeout_hit: bool,
        operator_interrupt: bool = False,
        pending_error: BaseException | None = None,
        captured_output: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        cleanup_context = (
            defer_sigint_during_owned_launch()
            if threading.current_thread() is threading.main_thread()
            else nullcontext(DeferredSigintState())
        )
        result: dict[str, Any] | None = None
        deferred_interrupts = DeferredSigintState()
        commit_interrupts = 0
        while True:
            try:
                if result is None:
                    with cleanup_context as deferred_interrupts:
                        result = _cleanup_result_once(
                            reason=reason,
                            timeout_hit=timeout_hit,
                            operator_interrupt=operator_interrupt,
                            pending_error=pending_error,
                            captured_output=captured_output,
                        )
                interrupt_count = (
                    int(operator_interrupt)
                    + deferred_interrupts.interrupt_count
                    + commit_interrupts
                )
                if interrupt_count:
                    result["returncode"] = 130
                    result["cancelled"] = True
                    result["cancellation"] = {
                        "reason": "interrupt",
                        "operator_interrupt_observed": True,
                        "interrupt_count": interrupt_count,
                        "repeated_interrupts": max(interrupt_count - 1, 0),
                        "signal": "SIGINT",
                    }
                return result
            except KeyboardInterrupt:
                if result is None:
                    raise
                commit_interrupts += 1

    launch_interrupted = False
    try:
        with launch_context as launch:
            process = psutil.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env=child_env,
                stdin=subprocess.PIPE if input is not None or input_factory is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=preexec_fn,
            )
            owned.remember(process)
        launch_interrupted = launch.observed
    except OSError as error:
        interrupted_during_launch = launch is not None and launch.observed
        if process is None:
            if interrupted_during_launch:
                raise KeyboardInterrupt from error
            return {
                "command": shlex.join(command),
                "returncode": -1,
                "stdout": "",
                "stderr": str(error),
                "timeout_seconds": timeout_seconds,
                "timeout_hit": False,
                "duration_seconds": round(time.monotonic() - started_monotonic, 2),
                "error": str(error),
                "error_type": type(error).__name__,
                "errno": error.errno,
            }
        result = cleanup_result(
            reason="launch_error",
            timeout_hit=False,
            operator_interrupt=interrupted_during_launch,
            pending_error=error,
        )
        error.cooperative_process_result = result
        if result.get("cancelled"):
            interrupted = CooperativeProcessInterrupted(result)
            if result["execution"]["completion_scope"] == "cleanup_incomplete":
                interrupted.cleanup_incomplete_receipt = result
            raise interrupted from error
        if result["execution"]["completion_scope"] == "cleanup_incomplete":
            raise CooperativeCleanupIncomplete(result) from error
        raise
    except BaseException as error:
        interrupted_during_launch = launch is not None and launch.observed
        if process is None:
            if interrupted_during_launch:
                raise KeyboardInterrupt from error
            raise
        result = cleanup_result(
            reason="launch_error",
            timeout_hit=False,
            operator_interrupt=interrupted_during_launch,
            pending_error=error,
        )
        error.cooperative_process_result = result
        if result.get("cancelled"):
            interrupted = CooperativeProcessInterrupted(result)
            if result["execution"]["completion_scope"] == "cleanup_incomplete":
                interrupted.cleanup_incomplete_receipt = result
            raise interrupted from error
        if result["execution"]["completion_scope"] == "cleanup_incomplete":
            raise CooperativeCleanupIncomplete(result) from error
        raise

    assert process is not None

    try:
        tracker = threading.Thread(
            target=_track_descendants,
            args=(owned, stop_tracking, poll_seconds),
            daemon=True,
        )
        tracker.start()
        if launch_interrupted:
            raise KeyboardInterrupt
        process_input = input_factory(process) if input_factory is not None else input
        stdout, stderr = process.communicate(input=process_input, timeout=timeout_seconds)
        owned.discover_descendants()
        owned.discover_owner_token()
        residual = [item for item in owned.live() if item.pid != process.pid]
        if residual or owned.observation_errors():
            result = cleanup_result(
                reason="leader_exit",
                timeout_hit=False,
                captured_output=(stdout or "", stderr or ""),
            )
            if result.get("cancelled"):
                interrupted = CooperativeProcessInterrupted(result)
                if result["execution"]["completion_scope"] == "cleanup_incomplete":
                    interrupted.cleanup_incomplete_receipt = result
                raise interrupted
            if result["execution"]["completion_scope"] == "cleanup_incomplete":
                raise CooperativeCleanupIncomplete(result)
            result["returncode"] = process.returncode
            result.pop("process_returncode", None)
            return result
        return {
            "command": shlex.join(command),
            "returncode": process.returncode,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timeout_seconds": timeout_seconds,
            "timeout_hit": False,
            "duration_seconds": round(time.monotonic() - started_monotonic, 2),
            "execution": {
                "process_id": process.pid,
                "process_group_id": process.pid,
                "start_new_session": True,
                "cleanup_event_nonce": cleanup_event_nonce,
                "live_process_group_members": [],
                "completion_scope": "process_tree_quiescent",
            },
        }
    except subprocess.TimeoutExpired as error:
        result = cleanup_result(reason="timeout", timeout_hit=True)
        if result.get("cancelled"):
            interrupted = CooperativeProcessInterrupted(result)
            if result["execution"]["completion_scope"] == "cleanup_incomplete":
                interrupted.cleanup_incomplete_receipt = result
            raise interrupted from error
        if result["execution"]["completion_scope"] == "cleanup_incomplete":
            raise CooperativeCleanupIncomplete(result) from error
        return result
    except KeyboardInterrupt as error:
        result = cleanup_result(
            reason="interrupt",
            timeout_hit=False,
            operator_interrupt=True,
            pending_error=error,
        )
        interrupted = CooperativeProcessInterrupted(result)
        if result["execution"]["completion_scope"] == "cleanup_incomplete":
            interrupted.cleanup_incomplete_receipt = result
        raise interrupted from error
    except (CooperativeCleanupIncomplete, CooperativeProcessInterrupted):
        raise
    except BaseException as error:
        result = cleanup_result(
            reason="coordinator_error",
            timeout_hit=False,
            pending_error=error,
        )
        error.cooperative_process_result = result
        if result.get("cancelled"):
            interrupted = CooperativeProcessInterrupted(result)
            if result["execution"]["completion_scope"] == "cleanup_incomplete":
                interrupted.cleanup_incomplete_receipt = result
            raise interrupted from error
        if result["execution"]["completion_scope"] == "cleanup_incomplete":
            raise CooperativeCleanupIncomplete(result) from error
        raise
    finally:
        stop_tracking.set()
        if tracker is not None and tracker.ident is not None:
            tracker.join()
