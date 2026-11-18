"""Run an owned process and let its cancellation cleanup finish."""

from __future__ import annotations

import base64
import os
import select
import selectors
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil


class CooperativeProcessInterrupted(KeyboardInterrupt):
    """Cancellation with captured output after the owned group stops running."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("Cancelled after the owned process group stopped running")
        self.result = result


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
) -> dict[str, Any]:
    """Capture output without a deadline and forward one SIGINT on cancellation."""
    if os.name != "posix":
        raise RuntimeError("SKIPPED: cooperative subprocess execution requires POSIX process groups")
    require_cooperative_process_support()
    if isinstance(command, (str, bytes)) or not command or any(not isinstance(value, str) for value in command):
        raise ValueError("command must be a nonempty argument vector of strings")
    argv = list(command)
    started = time.perf_counter()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        return communicate_cooperatively(process, process_group_id=process.pid, started_at=started)
    except CooperativeProcessInterrupted:
        raise
    except KeyboardInterrupt:
        return communicate_cooperatively(
            process, process_group_id=process.pid, started_at=started,
            cancellation_requested=True,
        )
    except BaseException as error:
        if not hasattr(error, "cooperative_process_result"):
            communicate_cooperatively(
                process, process_group_id=process.pid, started_at=started,
                abort_error=error,
            )
        raise


def require_cooperative_process_support() -> None:
    """Reject unsupported ownership semantics before launching a worker."""
    if not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")):
        raise RuntimeError("SKIPPED: owned subprocess cleanup requires waitid with WNOWAIT")


def communicate_cooperatively(
    process: subprocess.Popen[str],
    *,
    process_group_id: int,
    input: str | None = None,
    started_at: float | None = None,
    cancellation_requested: bool = False,
    abort_error: BaseException | None = None,
) -> dict[str, Any]:
    """Drain output and own the group until cancellation or leader-exit cleanup ends.

    A completed leader ends the command's lifetime. Residual group members are
    killed before reaping that leader, which reserves its PID and group identity.
    Cancellation instead forwards one SIGINT and waits for cooperative cleanup.
    Neither path imposes an execution deadline. Descendants that leave the group
    are outside this ownership contract.
    """
    if os.name != "posix":
        raise RuntimeError("SKIPPED: cooperative subprocess execution requires POSIX process groups")
    require_cooperative_process_support()
    if isinstance(process_group_id, bool) or process_group_id != process.pid:
        raise ValueError("Cooperative execution requires the child's explicitly owned process group")
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
    argv = process.args
    if isinstance(argv, (str, bytes)):
        raise ValueError("Cooperative execution requires an argument vector")
    cancellation: dict[str, Any] | None = None
    pending_error = abort_error
    completion_cleanup: dict[str, Any] | None = None
    already_reaped = process.returncode is not None
    captured_bytes: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    capture_errors: list[str] = []
    cleanup_incomplete = False
    group_observation_failed = False
    lifecycle_observation_failed = False
    ownership_lost = False
    leader_exited = already_reaped
    abort_cleanup = False

    def request_cancellation(*, interrupted: bool = False) -> None:
        nonlocal cancellation, ownership_lost
        if cancellation is not None:
            if interrupted:
                cancellation["repeated_interrupts"] += 1
            return
        cancellation = {
            "signal": "SIGINT", "process_group_id": process.pid,
            "status": "waiting_for_cleanup", "repeated_interrupts": 0,
            "reason": "coordinator_error" if pending_error is not None else "interrupt",
            "signal_forwarded": False,
        }
        if already_reaped:
            if _live_process_group_members(process_group_id):
                ownership_lost = True
                raise RuntimeError("Cannot signal a reaped leader's live process group safely")
            cancellation["signal_forwarded"] = False
        else:
            try:
                os.killpg(process.pid, signal.SIGINT)
                cancellation["signal_forwarded"] = True
            except ProcessLookupError:
                pass
            except Exception as error:
                cancellation["signal_error"] = f"{type(error).__name__}: {error}"
                raise

    def retain_error(error: BaseException) -> None:
        nonlocal pending_error
        capture_errors.append(f"{type(error).__name__}: {error}")
        if pending_error is None:
            pending_error = error

    def group_members() -> list[int]:
        nonlocal group_observation_failed
        try:
            return _live_process_group_members(process_group_id)
        except Exception:
            group_observation_failed = True
            raise

    def wait_for_quiescence() -> None:
        while True:
            try:
                if not group_members():
                    # Reap last so the leader reserves this group's identity.
                    process.wait()
                    return
                # POSIX groups have no portable wait handle for non-child members.
                time.sleep(0.05)
            except KeyboardInterrupt:
                request_cancellation(interrupted=True)

    try:
        if ownership_error is not None:
            ownership_lost = True
            raise ownership_error
        if cancellation_requested or pending_error is not None:
            request_cancellation()
        input_bytes = (input or "").encode(
            process.stdin.encoding if process.stdin is not None else "utf-8"
        )
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
                    stream, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name,
                )
            while True:
                try:
                    if not leader_exited:
                        try:
                            status = os.waitid(
                                os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG,
                            )
                        except ChildProcessError:
                            ownership_lost = True
                            raise RuntimeError("The owned leader was reaped by another caller before cleanup")
                        except Exception:
                            lifecycle_observation_failed = True
                            raise
                        leader_exited = status is not None
                    if leader_exited and cancellation is None and completion_cleanup is None:
                        members = group_members()
                        if members:
                            if already_reaped:
                                ownership_lost = True
                                raise RuntimeError("Cannot clean a reaped leader's live process group safely")
                            os.killpg(process_group_id, signal.SIGKILL)
                            completion_cleanup = {
                                "reason": "leader_exited", "signal": "SIGKILL",
                                "process_ids": members,
                            }
                    if leader_exited and not selector.get_map():
                        break
                    for key, _ in selector.select(0.05):
                        stream, name = key.fileobj, key.data
                        try:
                            if name == "stdin":
                                input_offset += os.write(stream.fileno(), input_bytes[input_offset:input_offset + 4096])
                                if input_offset == len(input_bytes):
                                    selector.unregister(stream)
                                    stream.close()
                            else:
                                chunk = os.read(stream.fileno(), 65536)
                                if chunk:
                                    captured_bytes[name].extend(chunk)
                                else:
                                    selector.unregister(stream)
                                    stream.close()
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            # Early stdin closure must not discard child diagnostics.
                            selector.unregister(stream)
                            stream.close()
                except KeyboardInterrupt:
                    request_cancellation(interrupted=True)
        wait_for_quiescence()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            request_cancellation(interrupted=True)
        else:
            retain_error(error)
        abort_cleanup = True
        # An actual coordinator failure ends normal I/O. This distinct cleanup
        # path preserves its error and never retries the failed selector setup.
        try:
            if ownership_lost:
                raise RuntimeError("Cannot signal after losing the owned leader identity")
            if cancellation is not None and "signal_error" in cancellation:
                raise RuntimeError("Cannot complete cooperative cleanup after SIGINT forwarding failed")
            request_cancellation()
        except BaseException as cleanup_error:
            retain_error(cleanup_error)
            cleanup_incomplete = True
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception as cleanup_error:
                retain_error(cleanup_error)
            finally:
                process.stdin = None
        if not cleanup_incomplete:
            remaining = {}
            try:
                for name in ("stdout", "stderr"):
                    stream = getattr(process, name)
                    if stream is not None and not stream.closed:
                        remaining[stream.fileno()] = (name, stream)
            except Exception as cleanup_error:
                retain_error(cleanup_error)
                cleanup_incomplete = True
            while remaining:
                try:
                    readable, _, _ = select.select(list(remaining), [], [])
                    for descriptor in readable:
                        name, stream = remaining[descriptor]
                        chunk = os.read(descriptor, 65536)
                        if chunk:
                            captured_bytes[name].extend(chunk)
                        else:
                            del remaining[descriptor]
                            stream.close()
                except KeyboardInterrupt:
                    request_cancellation(interrupted=True)
                except Exception as cleanup_error:
                    retain_error(cleanup_error)
                    cleanup_incomplete = True
                    break
            if not cleanup_incomplete and not group_observation_failed:
                try:
                    wait_for_quiescence()
                except Exception as cleanup_error:
                    retain_error(cleanup_error)
                    cleanup_incomplete = True
            elif group_observation_failed:
                cleanup_incomplete = True
                # A previously observed exit can still be reaped safely. Do not
                # retry a failed group query or claim its descendants are gone.
                if leader_exited:
                    try:
                        process.wait()
                    except Exception as cleanup_error:
                        retain_error(cleanup_error)
        cleanup_incomplete = cleanup_incomplete or lifecycle_observation_failed
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
    if cancellation is not None:
        cancellation["status"] = "cleanup_incomplete" if cleanup_incomplete else "cleanup_completed"
    result = {
        "command": shlex.join(argv),
        "returncode": process.returncode,
        "stdout": captured["stdout"] or "", "stderr": captured["stderr"] or "",
        "duration_seconds": time.perf_counter() - started,
        "timeout_seconds": None, "timeout_hit": False,
        "cancelled": cancellation is not None,
        "execution": {
            "process_id": process.pid, "process_group_id": process.pid,
            "start_new_session": True,
            "live_process_group_members": None if cleanup_incomplete else [],
            "completion_scope": "cleanup_incomplete" if cleanup_incomplete else "process_group_quiescent",
        },
    }
    if completion_cleanup is not None:
        result["execution"]["completion_cleanup"] = completion_cleanup
    if capture_errors:
        result["execution"]["capture_errors"] = capture_errors
    if abort_cleanup:
        result["execution"]["abort_cleanup"] = "raw_pipe_drain"
    if undecodable_streams:
        result["execution"]["undecodable_streams_base64"] = undecodable_streams
    if undecodable_streams:
        result["raw_streams_base64"] = undecodable_streams
    if cancellation is not None:
        result["cancellation"] = cancellation
    if pending_error is not None:
        pending_error.cooperative_process_result = result
        raise pending_error
    if cancellation is not None:
        raise CooperativeProcessInterrupted(result)
    return result
