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

import core.utils.cooperative_subprocess as cooperative_subprocess
from core.utils.cooperative_subprocess import (
    CooperativeProcessInterrupted,
    communicate_cooperatively,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")


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
        if process is not None:
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
            if process_group_id == process.pid and process.returncode is not None:
                observed_after_leader_exit.append(tuple(members))
            if (
                process_group_id == process.pid
                and process.returncode is not None
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
