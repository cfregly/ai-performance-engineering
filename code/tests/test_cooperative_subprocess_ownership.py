"""Exercise real CPU process ownership, output retention, and cancellation."""

from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
from typing import Iterator

import pytest
import psutil

import core.utils.cooperative_subprocess as cooperative_subprocess
from core.utils.cooperative_subprocess import (
    CooperativeProcessInterrupted,
    communicate_cooperatively,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "waitid"),
    reason="POSIX waitid with WNOWAIT is required",
)


def _release(connection: socket.socket) -> None:
    with suppress(BrokenPipeError, ConnectionResetError):
        connection.sendall(b"release\n")


@contextmanager
def _python_process(source: str, *, stdin: bool = False) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        yield process
    finally:
        if process.returncode is None:
            process.communicate()


@contextmanager
def _controlled_process(
    source: str, *, start_new_session: bool = True,
) -> Iterator[tuple[subprocess.Popen[str], socket.socket]]:
    connection, child_connection = socket.socketpair()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source, str(child_connection.fileno())],
            pass_fds=(child_connection.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=start_new_session,
        )
        child_connection.close()
        yield process, connection
    finally:
        _release(connection)
        if process is not None and process.returncode is None:
            process.communicate()
        connection.close()
        child_connection.close()


_WAIT_FOR_RELEASE = """
import socket
import sys
control = socket.socket(fileno=int(sys.argv[1]))
control.sendall(b"ready\\n")
control.recv(1024)
print("released", flush=True)
control.close()
"""


@pytest.mark.parametrize("invalid_group", ["boolean", "different_pid"])
def test_rejects_group_identity_that_does_not_match_child(invalid_group: str) -> None:
    with _controlled_process(_WAIT_FOR_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"ready\n"
            group = True if invalid_group == "boolean" else process.pid + 1
            with pytest.raises(ValueError, match="explicitly owned process group"):
                communicate_cooperatively(process, process_group_id=group)
            assert process.poll() is None


def test_rejects_child_that_inherited_another_process_group() -> None:
    with _controlled_process(_WAIT_FOR_RELEASE, start_new_session=False) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"ready\n"
            assert os.getpgid(process.pid) != process.pid
            with pytest.raises(ValueError, match="process group owned by this child"):
                communicate_cooperatively(process, process_group_id=process.pid)
            assert process.poll() is None


def test_reaped_quick_child_preserves_both_output_streams() -> None:
    source = "import sys\nprint('quick stdout')\nprint('quick stderr', file=sys.stderr)\nraise SystemExit(7)\n"
    with _python_process(source) as process:
        assert process.wait() == 7
        result = communicate_cooperatively(process, process_group_id=process.pid)
        assert result["stdout"] == "quick stdout\n"
        assert result["stderr"] == "quick stderr\n"
        assert result["returncode"] == 7
        assert result["cancelled"] is False
        assert result["timeout_seconds"] is None
        assert result["timeout_hit"] is False
        assert result["execution"]["process_group_id"] == process.pid


def test_stdin_is_delivered_until_eof() -> None:
    source = "import json, sys\nprint(json.dumps({'input': sys.stdin.read()}))\n"
    payload = "first line\nsecond line with unicode \u03bb\n"
    with _python_process(source, stdin=True) as process:
        result = communicate_cooperatively(process, process_group_id=process.pid, input=payload)
        assert json.loads(result["stdout"]) == {"input": payload}
        assert result["stderr"] == ""
        assert result["returncode"] == 0


@pytest.mark.parametrize("keep_capture_pipes", [False, True])
def test_completed_leader_cleans_residual_group_before_reaping(keep_capture_pipes: bool) -> None:
    source = f"""
import os, signal
ready_read, ready_write = os.pipe()
child = os.fork()
if child == 0:
    os.close(ready_read)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if not {keep_capture_pipes!r}:
        os.close(1)
        os.close(2)
    os.write(ready_write, b"ready")
    os.close(ready_write)
    while True:
        signal.pause()
os.close(ready_write)
os.read(ready_read, 5)
os.close(ready_read)
print(child, flush=True)
"""
    with _python_process(source) as process:
        result = communicate_cooperatively(process, process_group_id=process.pid)
        descendant = int(result["stdout"])
        assert result["returncode"] == 0
        assert result["cancelled"] is False
        cleanup = result["execution"]["completion_cleanup"]
        assert cleanup == {"reason": "leader_exited", "signal": "SIGKILL", "process_ids": [descendant]}
        assert cooperative_subprocess._live_process_group_members(process.pid) == []


def test_invalid_text_preserves_raw_capture_without_blocking_cleanup() -> None:
    import base64

    with _python_process("import os\nos.write(1, b'valid\\xffbytes')\n") as process:
        result = communicate_cooperatively(process, process_group_id=process.pid)
    assert result["stdout"] == "valid\ufffdbytes"
    assert base64.b64decode(result["execution"]["undecodable_streams_base64"]["stdout"]) == b"valid\xffbytes"


_INTERRUPT_AND_RELEASE = """
import signal
import socket
import sys
control = socket.socket(fileno=int(sys.argv[1]))
def interrupted(signum, frame):
    print("cleanup stdout before release", flush=True)
    print("cleanup stderr before release", file=sys.stderr, flush=True)
    control.sendall(b"cleanup ready\\n")
    control.recv(1024)
    print("cleanup stdout after release", flush=True)
    print("cleanup stderr after release", file=sys.stderr, flush=True)
    control.close()
    raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
if sys.stdin.readline() == "begin\\n":
    control.sendall(b"ready\\n")
control.recv(1024)
control.close()
"""


def test_cancellation_waits_for_cleanup_and_retains_output() -> None:
    with _controlled_process(_INTERRUPT_AND_RELEASE) as (process, connection):
        controller_errors: list[BaseException] = []

        def interrupt_after_stdin_delivery() -> None:
            try:
                with connection.makefile("rb") as messages:
                    assert messages.readline() == b"ready\n"
                    os.kill(os.getpid(), signal.SIGINT)
                    assert messages.readline() == b"cleanup ready\n"
                    assert process.poll() is None
            except BaseException as error:
                controller_errors.append(error)
            finally:
                _release(connection)

        controller = threading.Thread(target=interrupt_after_stdin_delivery)
        controller.start()
        try:
            with pytest.raises(CooperativeProcessInterrupted) as interrupted:
                communicate_cooperatively(process, process_group_id=process.pid, input="begin\n")
        finally:
            _release(connection)
            controller.join()
        assert not controller_errors
        result = interrupted.value.result
        assert result["stdout"] == "cleanup stdout before release\ncleanup stdout after release\n"
        assert result["stderr"] == "cleanup stderr before release\ncleanup stderr after release\n"
        assert result["returncode"] == 0
        assert result["cancelled"] is True
        assert result["timeout_seconds"] is None
        assert result["timeout_hit"] is False
        assert result["cancellation"]["signal"] == "SIGINT"
        assert result["cancellation"]["signal_forwarded"] is True
        assert result["cancellation"]["status"] == "cleanup_completed"
        assert process.poll() == 0


_DESCENDANT_AFTER_LEADER = """
import os
import signal
import socket
import sys
control = socket.socket(fileno=int(sys.argv[1]))
parent_release = int(sys.argv[2])
ready_read, ready_write = os.pipe()
def interrupted(signum, frame):
    print("leader cleanup", flush=True)
    control.close()
    raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
descendant = os.fork()
if descendant == 0:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.close(parent_release)
    os.close(ready_read)
    for descriptor in (0, 1, 2):
        os.close(descriptor)
    control.sendall(("descendant ready " + str(os.getpid()) + "\\n").encode())
    os.write(ready_write, b"ready")
    os.close(ready_write)
    control.recv(1024)
    control.sendall(b"descendant cleanup\\n")
    os._exit(0)
os.close(ready_write)
os.read(ready_read, 1024)
os.close(ready_read)
if sys.stdin.readline() == "begin\\n":
    control.sendall(b"leader ready\\n")
control.close()
os.read(parent_release, 1024)
os.close(parent_release)
"""


@contextmanager
def _descendant_process() -> Iterator[tuple[subprocess.Popen[str], socket.socket, int]]:
    connection, child_connection = socket.socketpair()
    release_read, release_write = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _DESCENDANT_AFTER_LEADER,
             str(child_connection.fileno()), str(release_read)],
            pass_fds=(child_connection.fileno(), release_read),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        child_connection.close()
        os.close(release_read)
        release_read = -1
        yield process, connection, release_write
    finally:
        _release(connection)
        with suppress(BrokenPipeError):
            os.write(release_write, b"release")
        child_connection.close()
        if process is not None:
            if process.returncode is None:
                process.communicate()
            while connection.recv(4096):
                pass
        connection.close()
        if release_read >= 0:
            os.close(release_read)
        os.close(release_write)


def test_cancellation_waits_for_descendant_that_closed_capture_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_group_query = cooperative_subprocess._live_process_group_members
    with _descendant_process() as (process, connection, parent_release):
        events: queue.Queue[str] = queue.Queue()
        descendant: list[int] = []
        first_events: list[str] = []
        controller_errors: list[BaseException] = []
        reported_live = threading.Event()
        observed_after_leader_exit: list[tuple[int, ...]] = []

        def observe_real_group_query(process_group_id: int) -> list[int]:
            members = real_group_query(process_group_id)
            leader_exited = (
                process.returncode is not None
                or os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None
            )
            if process_group_id == process.pid and leader_exited:
                observed_after_leader_exit.append(tuple(members))
            if (
                process_group_id == process.pid
                and leader_exited
                and descendant
                and descendant[0] in members
                and not reported_live.is_set()
            ):
                assert os.getpgid(descendant[0]) == process.pid
                reported_live.set()
                events.put("observed_live_descendant")
            return members

        # Observe the actual kernel-backed query without changing its result.
        monkeypatch.setattr(cooperative_subprocess, "_live_process_group_members", observe_real_group_query)

        def release_after_group_observation() -> None:
            try:
                with connection.makefile("rb") as messages:
                    ready = messages.readline().decode().strip().split()
                    assert ready[:2] == ["descendant", "ready"]
                    descendant.append(int(ready[2]))
                    assert os.getpgid(descendant[0]) == process.pid
                    assert messages.readline() == b"leader ready\n"
                    os.kill(os.getpid(), signal.SIGINT)
                    first_events.append(events.get())
                    _release(connection)
                    assert messages.readline() == b"descendant cleanup\n"
                    assert messages.readline() == b""
            except BaseException as error:
                controller_errors.append(error)
            finally:
                _release(connection)
                with suppress(BrokenPipeError):
                    os.write(parent_release, b"release")

        controller = threading.Thread(target=release_after_group_observation)
        controller.start()
        try:
            with pytest.raises(CooperativeProcessInterrupted) as interrupted:
                communicate_cooperatively(process, process_group_id=process.pid, input="begin\n")
        finally:
            events.put("returned")
            _release(connection)
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
                process.stdin = None
            with suppress(BrokenPipeError):
                os.write(parent_release, b"release")
            controller.join()
        assert not controller_errors
        assert first_events == ["observed_live_descendant"]
        assert descendant[0] in observed_after_leader_exit[0]
        assert observed_after_leader_exit[-1] == ()
        result = interrupted.value.result
        assert result["stdout"] == "leader cleanup\n"
        assert result["stderr"] == ""
        assert result["cancellation"]["status"] == "cleanup_completed"


_PENDING_CANCEL_AND_RELEASE = """
import signal
import socket
import sys
control = socket.socket(fileno=int(sys.argv[1]))
interruptions = 0
def interrupted(signum, frame):
    global interruptions
    interruptions += 1
    print("pending cancel signal " + str(interruptions), flush=True)
    print("pending cancel stderr", file=sys.stderr, flush=True)
    control.sendall(b"cleanup ready\\n")
    control.recv(1024)
    print("pending cancel cleanup " + str(interruptions), flush=True)
    control.close()
    raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
control.sendall(b"handler ready\\n")
control.recv(1024)
control.close()
"""


def test_pending_cancellation_forwards_once_and_waits_for_group_cleanup() -> None:
    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        controller_errors: list[BaseException] = []
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"

            def release_after_pending_cancellation() -> None:
                try:
                    assert messages.readline() == b"cleanup ready\n"
                    assert process.poll() is None
                except BaseException as error:
                    controller_errors.append(error)
                finally:
                    _release(connection)

            controller = threading.Thread(target=release_after_pending_cancellation)
            controller.start()
            try:
                with pytest.raises(CooperativeProcessInterrupted) as interrupted:
                    communicate_cooperatively(
                        process, process_group_id=process.pid, cancellation_requested=True,
                    )
            finally:
                _release(connection)
                controller.join()
        assert not controller_errors
        result = interrupted.value.result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["returncode"] == 0
        assert result["cancelled"] is True
        assert result["cancellation"]["signal"] == "SIGINT"
        assert result["cancellation"]["signal_forwarded"] is True
        assert result["cancellation"]["repeated_interrupts"] == 0
        assert result["cancellation"]["status"] == "cleanup_completed"
        assert result["execution"]["completion_scope"] == "process_group_quiescent"
        assert cooperative_subprocess._live_process_group_members(process.pid) == []


def test_coordinator_exception_retains_identity_output_and_cleanup() -> None:
    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        original = TimeoutError("coordinator operation failed")
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            controller = threading.Thread(target=lambda: (messages.readline(), _release(connection)))
            controller.start()
            try:
                with pytest.raises(TimeoutError) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid, abort_error=original)
            finally:
                _release(connection)
                controller.join()
        assert failure.value is original
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["cancellation"]["reason"] == "coordinator_error"
        assert result["cancellation"]["status"] == "cleanup_completed"
        assert cooperative_subprocess._live_process_group_members(process.pid) == []


@pytest.mark.parametrize("failure_stage", ["constructor", "registration", "input_encoding"])
def test_setup_failure_aborts_once_and_retains_child_diagnostics(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str,
) -> None:
    original = OSError("selector setup failed")
    setup_failures: list[str] = []
    real_selector = cooperative_subprocess.selectors.DefaultSelector

    if failure_stage == "constructor":
        def failing_constructor():
            setup_failures.append("constructor")
            raise original

        monkeypatch.setattr(cooperative_subprocess.selectors, "DefaultSelector", failing_constructor)
    elif failure_stage == "registration":
        class FailingRegistration(real_selector):
            def register(self, fileobj, events, data=None):
                if data == "stderr":
                    setup_failures.append("registration")
                    raise original
                return super().register(fileobj, events, data)

        monkeypatch.setattr(cooperative_subprocess.selectors, "DefaultSelector", FailingRegistration)

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            controller_errors: list[BaseException] = []

            def release_after_abort() -> None:
                try:
                    assert messages.readline() == b"cleanup ready\n"
                except BaseException as error:
                    controller_errors.append(error)
                finally:
                    _release(connection)

            controller = threading.Thread(target=release_after_abort)
            controller.start()
            try:
                expected_error = UnicodeEncodeError if failure_stage == "input_encoding" else OSError
                with pytest.raises(expected_error) as failure:
                    communicate_cooperatively(
                        process, process_group_id=process.pid,
                        input="\ud800" if failure_stage == "input_encoding" else None,
                    )
            finally:
                _release(connection)
                controller.join()
        assert not controller_errors
        if failure_stage != "input_encoding":
            assert failure.value is original
            assert setup_failures == [failure_stage]
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["returncode"] == 0
        assert result["cancellation"]["repeated_interrupts"] == 0
        assert result["cancellation"]["reason"] == "coordinator_error"
        assert result["cancellation"]["status"] == "cleanup_completed"
        assert result["execution"]["abort_cleanup"] == "raw_pipe_drain"
        assert cooperative_subprocess._live_process_group_members(process.pid) == []


def test_persistent_selector_failure_keeps_captured_prefix_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("selector cannot observe descriptors")
    real_selector = cooperative_subprocess.selectors.DefaultSelector
    select_calls: list[None] = []

    class FailingSelector(real_selector):
        def select(self, timeout=None):
            select_calls.append(None)
            if len(select_calls) > 1:
                raise original
            return super().select(timeout)

    monkeypatch.setattr(cooperative_subprocess.selectors, "DefaultSelector", FailingSelector)
    source = "print('captured before failure', flush=True)\n" + _PENDING_CANCEL_AND_RELEASE
    with _controlled_process(source) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            controller = threading.Thread(target=lambda: (messages.readline(), _release(connection)))
            controller.start()
            try:
                with pytest.raises(OSError) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid)
            finally:
                _release(connection)
                controller.join()
        assert failure.value is original
        assert len(select_calls) == 2
        result = failure.value.cooperative_process_result
        assert result["stdout"] == (
            "captured before failure\npending cancel signal 1\npending cancel cleanup 1\n"
        )
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["cancellation"]["repeated_interrupts"] == 0
        assert result["execution"]["completion_scope"] == "process_group_quiescent"
        assert result["returncode"] == 0


def test_failed_final_group_observation_reports_incomplete_cleanup_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = psutil.AccessDenied(msg="group observation denied")
    real_group_query = cooperative_subprocess._live_process_group_members
    failed_queries: list[int] = []
    source = "import sys\nprint('finished stdout')\nprint('finished stderr', file=sys.stderr)\n"
    with _python_process(source) as process:
        def fail_after_pipe_eof(process_group_id: int) -> list[int]:
            if process.stdout.closed and process.stderr.closed:
                failed_queries.append(process_group_id)
                raise original
            return real_group_query(process_group_id)

        monkeypatch.setattr(cooperative_subprocess, "_live_process_group_members", fail_after_pipe_eof)
        with pytest.raises(psutil.AccessDenied) as failure:
            communicate_cooperatively(process, process_group_id=process.pid)
        assert failure.value is original
        assert failed_queries == [process.pid]
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "finished stdout\n"
        assert result["stderr"] == "finished stderr\n"
        assert result["returncode"] == 0
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["execution"]["live_process_group_members"] is None
        assert result["cancellation"]["status"] == "cleanup_incomplete"
        assert real_group_query(process.pid) == []


def test_persistent_waitid_failure_preserves_error_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("waitid observation failed")
    calls: list[None] = []

    def failing_waitid(*args):
        calls.append(None)
        raise original

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            monkeypatch.setattr(cooperative_subprocess.os, "waitid", failing_waitid)
            controller = threading.Thread(target=lambda: (messages.readline(), _release(connection)))
            controller.start()
            try:
                with pytest.raises(OSError) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid)
            finally:
                _release(connection)
                controller.join()
        assert failure.value is original
        assert len(calls) == 1
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["returncode"] == 0
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["cancellation"]["status"] == "cleanup_incomplete"
        assert result["cancellation"]["repeated_interrupts"] == 0


@pytest.mark.parametrize("failed_operation", ["getpgid", "killpg"])
def test_failed_ownership_or_signal_reports_original_error_without_waiting_or_retrying(
    monkeypatch: pytest.MonkeyPatch, failed_operation: str,
) -> None:
    original = PermissionError("owned process operation denied")
    calls: list[None] = []

    def denied_operation(*args):
        calls.append(None)
        raise original

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            monkeypatch.setattr(cooperative_subprocess.os, failed_operation, denied_operation)
            with pytest.raises(PermissionError) as failure:
                communicate_cooperatively(
                    process, process_group_id=process.pid, cancellation_requested=True,
                )
        assert failure.value is original
        assert len(calls) == 1
        result = failure.value.cooperative_process_result
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["execution"]["live_process_group_members"] is None
        assert result["returncode"] is None
        assert process.poll() is None
        if failed_operation == "killpg":
            assert result["cancellation"]["signal_forwarded"] is False
            assert result["cancellation"]["status"] == "cleanup_incomplete"
        # The control socket releases this still-owned child in the fixture.
