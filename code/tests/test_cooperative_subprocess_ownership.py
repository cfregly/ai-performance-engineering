"""Exercise real CPU process ownership, output retention, and cancellation."""

from __future__ import annotations

import errno
import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress

import psutil
import pytest

import core.utils.cooperative_subprocess as cooperative_subprocess
from core.utils.cooperative_subprocess import (
    CooperativeCleanupIncomplete,
    CooperativeCoordinatorError,
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


@pytest.mark.parametrize("ignored_signal", ["SIGINT", "SIGCHLD"])
def test_ignored_signal_is_rejected_before_child_launch(tmp_path, ignored_signal: str) -> None:
    marker = tmp_path / "unexpected-child.txt"
    probe = """
import json
from pathlib import Path
import signal
import sys
from core.utils.cooperative_subprocess import run_cooperative_process
signal.signal(getattr(signal, sys.argv[1]), signal.SIG_IGN)
child = "from pathlib import Path; Path(" + repr(sys.argv[2]) + ").write_text('launched')"
try:
    run_cooperative_process([sys.executable, '-c', child])
except RuntimeError as error:
    print(json.dumps({'error': str(error), 'created': Path(sys.argv[2]).exists()}))
else:
    raise AssertionError('Ignored signal unexpectedly allowed owned child launch')
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, ignored_signal, str(marker)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert ignored_signal in result["error"]
    assert result["created"] is False
    assert not marker.exists()


def test_sigint_between_real_spawn_and_popen_return_is_deferred_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = cooperative_subprocess.subprocess.Popen
    spawned: list[subprocess.Popen[str]] = []
    previous_handler = signal.getsignal(signal.SIGINT)

    def spawn_then_interrupt(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        os.kill(os.getpid(), signal.SIGINT)
        return process

    monkeypatch.setattr(cooperative_subprocess.subprocess, "Popen", spawn_then_interrupt)
    with pytest.raises(CooperativeProcessInterrupted) as interrupted:
        cooperative_subprocess.run_cooperative_process(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )

    assert len(spawned) == 1
    assert interrupted.value.result["cancellation"]["operator_interrupt_observed"] is True
    assert interrupted.value.result["execution"]["completion_scope"] == "process_group_quiescent"
    assert signal.getsignal(signal.SIGINT) is previous_handler
    with pytest.raises(ProcessLookupError):
        os.kill(spawned[0].pid, 0)


def test_deferred_sigint_takes_priority_over_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    missing = tmp_path / "missing-cooperative-command"
    real_popen = cooperative_subprocess.subprocess.Popen
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt_then_fail(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(cooperative_subprocess.subprocess, "Popen", interrupt_then_fail)
    with pytest.raises(KeyboardInterrupt) as failure:
        cooperative_subprocess.run_cooperative_process([str(missing)])

    assert type(failure.value) is KeyboardInterrupt
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_post_spawn_launch_error_reaps_the_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("launch context failed after spawn")
    process_state: list[subprocess.Popen[str]] = []

    @contextmanager
    def fail_after_spawn():
        state = cooperative_subprocess.DeferredSigintState()
        yield state
        raise original

    real_popen = cooperative_subprocess.subprocess.Popen

    def remember_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        process_state.append(process)
        return process

    monkeypatch.setattr(
        cooperative_subprocess,
        "defer_sigint_during_owned_launch",
        fail_after_spawn,
    )
    monkeypatch.setattr(cooperative_subprocess.subprocess, "Popen", remember_process)
    with pytest.raises(RuntimeError) as failure:
        cooperative_subprocess.run_cooperative_process(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )

    assert failure.value is original
    result = failure.value.cooperative_process_result
    assert result["execution"]["completion_scope"] == "process_group_quiescent"
    assert result["execution"]["coordinator_error"] == {
        "type": "RuntimeError",
        "message": str(original),
    }
    assert len(process_state) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(process_state[0].pid, 0)


def test_input_factory_error_cannot_spoof_completed_cleanup() -> None:
    original = RuntimeError("input construction failed")
    original.cooperative_process_result = {"forged": True}
    process_state: list[subprocess.Popen[str]] = []

    def fail_input(process: subprocess.Popen[str]) -> str:
        process_state.append(process)
        raise original

    with pytest.raises(RuntimeError) as failure:
        cooperative_subprocess.run_cooperative_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_factory=fail_input,
        )

    assert failure.value is original
    result = failure.value.cooperative_process_result
    assert "forged" not in result
    assert result["execution"]["completion_scope"] == "process_group_quiescent"
    assert result["execution"]["coordinator_error"] == {
        "type": "RuntimeError",
        "message": str(original),
    }
    assert len(process_state) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(process_state[0].pid, 0)


@pytest.mark.parametrize(
    "terminal_error",
    [CooperativeProcessInterrupted, CooperativeCleanupIncomplete],
)
def test_input_factory_terminal_type_cannot_bypass_child_cleanup(
    terminal_error,
) -> None:
    forged_receipt = {
        "stdout": "",
        "stderr": "",
        "execution": {
            "process_group_id": 99999,
            "completion_scope": "cleanup_incomplete",
        },
    }
    forged = terminal_error(forged_receipt)
    process_state: list[subprocess.Popen[str]] = []

    def fail_input(process: subprocess.Popen[str]) -> str:
        process_state.append(process)
        raise forged

    with pytest.raises(terminal_error) as failure:
        cooperative_subprocess.run_cooperative_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_factory=fail_input,
        )

    assert len(process_state) == 1
    outer_receipt = (
        failure.value.result
        if isinstance(failure.value, CooperativeProcessInterrupted)
        else failure.value.cooperative_process_result
    )
    assert outer_receipt["execution"]["completion_scope"] == "process_group_quiescent"
    with pytest.raises(ProcessLookupError):
        os.kill(process_state[0].pid, 0)


def test_read_only_coordinator_error_uses_receipt_carrying_wrapper() -> None:
    class ReadOnlyReceiptError(RuntimeError):
        @property
        def cooperative_process_result(self):
            raise RuntimeError("receipt getter must not control ownership")

        @cooperative_process_result.setter
        def cooperative_process_result(self, value):
            del value
            raise AttributeError("receipt is read only")

    original = ReadOnlyReceiptError("read-only coordinator failure")
    with _python_process("print('completed before coordinator error')\n") as process:
        process.wait()
        with pytest.raises(CooperativeCoordinatorError) as failure:
            communicate_cooperatively(
                process,
                process_group_id=process.pid,
                abort_error=original,
            )

    assert failure.value.__cause__ is original
    assert failure.value.original_error is original
    assert failure.value.result["stdout"] == "completed before coordinator error\n"
    assert failure.value.result["execution"]["completion_scope"] == "process_group_quiescent"
    assert failure.value.result["execution"]["receipt_attachment_error"]["type"] == "AttributeError"
    assert cooperative_subprocess.cleanup_incomplete_receipt_from_error(failure.value) is None
    assert cooperative_subprocess.process_receipt_from_error(original) is None


@pytest.mark.parametrize("interrupt_stage", ["render", "attach"])
def test_interrupt_during_error_receipt_commit_keeps_operator_priority(
    interrupt_stage: str,
) -> None:
    class InterruptingError(RuntimeError):
        def __str__(self) -> str:
            if interrupt_stage == "render":
                raise KeyboardInterrupt("interrupt while rendering error")
            return "interrupting coordinator error"

        @property
        def cooperative_process_result(self):
            return None

        @cooperative_process_result.setter
        def cooperative_process_result(self, value):
            del value
            if interrupt_stage == "attach":
                raise KeyboardInterrupt("interrupt while attaching receipt")

    original = InterruptingError()
    with _python_process("print('completed before receipt interrupt')\n") as process:
        process.wait()
        with pytest.raises(CooperativeProcessInterrupted) as interrupted:
            communicate_cooperatively(
                process,
                process_group_id=process.pid,
                abort_error=original,
            )

    assert interrupted.value.__cause__ is original
    result = interrupted.value.result
    assert result["stdout"] == "completed before receipt interrupt\n"
    assert result["cancellation"]["operator_interrupt_observed"] is True
    assert result["cancellation"]["interrupt_count"] == 1
    assert result["execution"]["completion_scope"] == "process_group_quiescent"
    if interrupt_stage == "render":
        assert result["execution"]["coordinator_error"]["message"].startswith(
            "<KeyboardInterrupt while rendering"
        )
    else:
        assert result["execution"]["receipt_attachment_error"]["type"] == "KeyboardInterrupt"


def test_interrupt_after_completed_handoff_does_not_repeat_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_communicate = cooperative_subprocess.communicate_cooperatively
    calls = []

    def interrupt_after_completion(process, **kwargs):
        calls.append(kwargs.copy())
        result = actual_communicate(process, **kwargs)
        raise KeyboardInterrupt("interrupt after owned cleanup")

    monkeypatch.setattr(
        cooperative_subprocess,
        "communicate_cooperatively",
        interrupt_after_completion,
    )
    with pytest.raises(KeyboardInterrupt) as interrupted:
        cooperative_subprocess.run_cooperative_process(
            [sys.executable, "-c", "print('completed before interrupt')"],
        )

    assert len(calls) == 1
    receipt = interrupted.value.cooperative_process_result
    assert receipt["stdout"] == "completed before interrupt\n"
    assert receipt["execution"]["completion_scope"] == "process_group_quiescent"


@pytest.mark.parametrize("invalid_group", ["boolean", "different_pid"])
def test_rejects_group_identity_that_does_not_match_child(invalid_group: str) -> None:
    with (
        _controlled_process(_WAIT_FOR_RELEASE) as (process, connection),
        connection.makefile("rb") as messages,
    ):
        assert messages.readline() == b"ready\n"
        group = True if invalid_group == "boolean" else process.pid + 1
        with pytest.raises(ValueError, match="explicitly owned process group"):
            communicate_cooperatively(process, process_group_id=group)
        assert process.poll() is None


def test_rejects_child_that_inherited_another_process_group() -> None:
    with (
        _controlled_process(_WAIT_FOR_RELEASE, start_new_session=False) as (
            process,
            connection,
        ),
        connection.makefile("rb") as messages,
    ):
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


def test_interrupt_while_retaining_setup_error_still_owns_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingMessageError(OSError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("interrupt while retaining setup error")

    original = InterruptingMessageError()

    def failing_constructor():
        raise original

    monkeypatch.setattr(
        cooperative_subprocess.selectors,
        "DefaultSelector",
        failing_constructor,
    )
    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            controller = threading.Thread(
                target=lambda: (messages.readline(), _release(connection))
            )
            controller.start()
            try:
                with pytest.raises(CooperativeProcessInterrupted) as interrupted:
                    communicate_cooperatively(
                        process,
                        process_group_id=process.pid,
                    )
            finally:
                _release(connection)
                controller.join()

    assert interrupted.value.__cause__ is original
    result = interrupted.value.result
    assert result["cancellation"]["operator_interrupt_observed"] is True
    assert result["cancellation"]["status"] == "cleanup_completed"
    assert result["execution"]["completion_scope"] == "process_group_quiescent"
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
        with pytest.raises(CooperativeCleanupIncomplete) as failure:
            communicate_cooperatively(process, process_group_id=process.pid)
        assert failure.value.__cause__ is original
        assert failed_queries == [process.pid]
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "finished stdout\n"
        assert result["stderr"] == "finished stderr\n"
        assert result["returncode"] == 0
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["execution"]["live_process_group_members"] is None
        assert result["cancellation"]["status"] == "cleanup_incomplete"
        assert real_group_query(process.pid) == []
        event = io.StringIO()
        nonce = "cpu-cleanup-event-nonce"
        cooperative_subprocess.emit_cleanup_incomplete_event(result, nonce=nonce, stream=event)
        assert len(event.getvalue().splitlines()) == 1
        assert cooperative_subprocess.parse_cleanup_incomplete_event(
            "noise\n" + event.getvalue() + "trailer\n",
            expected_nonce=nonce,
        ) == result
        wrapper = RuntimeError("boundary wrapper")
        wrapper.__cause__ = failure.value
        failure.value.__context__ = wrapper
        assert cooperative_subprocess.cleanup_incomplete_receipt_from_error(wrapper) is result


def test_persistent_waitid_failure_preserves_error_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OSError("waitid observation failed")
    calls: list[int] = []

    def failing_waitid(idtype, pid, options):
        calls.append(options)
        raise original

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            monkeypatch.setattr(cooperative_subprocess.os, "waitid", failing_waitid)
            controller = threading.Thread(target=lambda: (messages.readline(), _release(connection)))
            controller.start()
            try:
                with pytest.raises(CooperativeCleanupIncomplete) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid)
            finally:
                _release(connection)
                controller.join()
        assert failure.value.__cause__ is original
        assert calls == [os.WEXITED | os.WNOWAIT | os.WNOHANG, os.WEXITED]
        result = failure.value.cooperative_process_result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["returncode"] is None
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["cancellation"]["status"] == "cleanup_incomplete"
        assert result["cancellation"]["repeated_interrupts"] == 0


@pytest.mark.parametrize("failed_operation", ["getpgid", "killpg"])
def test_requested_interrupt_preserves_ownership_or_signal_error_without_retrying(
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
            with pytest.raises(CooperativeProcessInterrupted) as failure:
                communicate_cooperatively(
                    process, process_group_id=process.pid, cancellation_requested=True,
                )
        assert failure.value.__cause__ is original
        assert len(calls) == 1
        result = failure.value.result
        assert result["cancellation"]["operator_interrupt_observed"] is True
        assert result["execution"]["coordinator_error"] == {
            "type": "PermissionError", "message": str(original),
        }
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["execution"]["live_process_group_members"] is None
        assert result["returncode"] is None
        assert process.poll() is None
        if failed_operation == "killpg":
            assert result["cancellation"]["signal_forwarded"] is False
            assert result["cancellation"]["status"] == "cleanup_incomplete"
        # The control socket releases this still-owned child in the fixture.


def test_operator_interrupt_takes_priority_during_coordinator_abort() -> None:
    original = RuntimeError("coordinator failed before operator cancellation")
    observed = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def observe_interrupt(signum, frame):
        observed.set()
        raise KeyboardInterrupt

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            controller_errors: list[BaseException] = []

            def interrupt_abort_cleanup() -> None:
                try:
                    assert messages.readline() == b"cleanup ready\n"
                    os.kill(os.getpid(), signal.SIGINT)
                    observed.wait()
                except BaseException as error:
                    controller_errors.append(error)
                finally:
                    _release(connection)

            signal.signal(signal.SIGINT, observe_interrupt)
            controller = threading.Thread(target=interrupt_abort_cleanup)
            controller.start()
            try:
                with pytest.raises(CooperativeProcessInterrupted) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid, abort_error=original)
            finally:
                _release(connection)
                controller.join()
                signal.signal(signal.SIGINT, previous_handler)
        assert not controller_errors
        assert failure.value.__cause__ is original
        result = failure.value.result
        assert original.cooperative_process_result is result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["stderr"] == "pending cancel stderr\n"
        assert result["cancellation"]["reason"] == "coordinator_error"
        assert result["cancellation"]["operator_interrupt_observed"] is True
        assert result["cancellation"]["interrupt_count"] == 1
        assert result["cancellation"]["repeated_interrupts"] == 0
        assert result["execution"]["coordinator_error"] == {
            "type": "RuntimeError", "message": str(original),
        }
        assert result["execution"]["completion_scope"] == "process_group_quiescent"


def test_real_interrupt_survives_failed_cleanup_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = psutil.AccessDenied(msg="cleanup observation failed after interrupt")
    real_group_query = cooperative_subprocess._live_process_group_members
    with _controlled_process(_INTERRUPT_AND_RELEASE) as (process, connection):
        controller_errors: list[BaseException] = []

        def fail_after_capture_closed(process_group_id):
            if process.stdout.closed and process.stderr.closed:
                raise original
            return real_group_query(process_group_id)

        monkeypatch.setattr(cooperative_subprocess, "_live_process_group_members", fail_after_capture_closed)
        with connection.makefile("rb") as messages:
            def interrupt_then_release() -> None:
                try:
                    assert messages.readline() == b"ready\n"
                    os.kill(os.getpid(), signal.SIGINT)
                    assert messages.readline() == b"cleanup ready\n"
                except BaseException as error:
                    controller_errors.append(error)
                finally:
                    _release(connection)

            controller = threading.Thread(target=interrupt_then_release)
            controller.start()
            try:
                with pytest.raises(CooperativeProcessInterrupted) as failure:
                    communicate_cooperatively(process, process_group_id=process.pid, input="begin\n")
            finally:
                _release(connection)
                controller.join()
        assert not controller_errors
        assert failure.value.__cause__ is original
        result = failure.value.result
        assert result["cancellation"]["operator_interrupt_observed"] is True
        assert result["cancellation"]["status"] == "cleanup_incomplete"
        assert result["execution"]["completion_scope"] == "cleanup_incomplete"
        assert result["stdout"] == "cleanup stdout before release\ncleanup stdout after release\n"
        assert result["stderr"] == "cleanup stderr before release\ncleanup stderr after release\n"


def test_interrupt_during_signal_forwarding_does_not_drop_or_repeat_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_killpg = os.killpg
    forwarded: list[int] = []
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def interrupt_before_forward(process_group_id, signum):
        forwarded.append(signum)
        os.kill(os.getpid(), signal.SIGINT)
        real_killpg(process_group_id, signum)

    with _controlled_process(_PENDING_CANCEL_AND_RELEASE) as (process, connection):
        with connection.makefile("rb") as messages:
            assert messages.readline() == b"handler ready\n"
            monkeypatch.setattr(cooperative_subprocess.os, "killpg", interrupt_before_forward)
            controller = threading.Thread(target=lambda: (messages.readline(), _release(connection)))
            controller.start()
            try:
                with pytest.raises(CooperativeProcessInterrupted) as failure:
                    communicate_cooperatively(
                        process, process_group_id=process.pid, cancellation_requested=True,
                    )
            finally:
                _release(connection)
                controller.join()
        assert forwarded == [signal.SIGINT]
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask
        result = failure.value.result
        assert result["stdout"] == "pending cancel signal 1\npending cancel cleanup 1\n"
        assert result["cancellation"]["signal_forwarded"] is True
        assert result["cancellation"]["interrupt_count"] == 2
        assert result["cancellation"]["repeated_interrupts"] == 1


@pytest.mark.parametrize("interrupt_stage", ["reap", "receipt"])
def test_interrupt_after_reap_keeps_output_without_signalling_released_identity(
    monkeypatch: pytest.MonkeyPatch, interrupt_stage: str,
) -> None:
    forwarded: list[int] = []
    source = "import sys\nprint('retained stdout')\nprint('retained stderr', file=sys.stderr)\n"
    with _python_process(source) as process:
        real_waitid = os.waitid
        real_join = cooperative_subprocess.shlex.join
        interrupted = False

        def interrupt_after_wait(idtype, pid, options):
            status = real_waitid(idtype, pid, options)
            if options == os.WEXITED:
                os.kill(os.getpid(), signal.SIGINT)
            return status

        def interrupt_while_building_receipt(argv):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGINT)
            return real_join(argv)

        monkeypatch.setattr(cooperative_subprocess.os, "killpg", lambda group, signum: forwarded.append(signum))
        if interrupt_stage == "reap":
            monkeypatch.setattr(cooperative_subprocess.os, "waitid", interrupt_after_wait)
        else:
            monkeypatch.setattr(cooperative_subprocess.shlex, "join", interrupt_while_building_receipt)
        with pytest.raises(CooperativeProcessInterrupted) as failure:
            communicate_cooperatively(process, process_group_id=process.pid)
        result = failure.value.result
        assert forwarded == []
        assert result["stdout"] == "retained stdout\n"
        assert result["stderr"] == "retained stderr\n"
        assert result["returncode"] == 0
        assert result["execution"]["completion_scope"] == "process_group_quiescent"
        assert result["cancellation"]["signal_forwarded"] is False
        assert result["cancellation"]["signal_not_forwarded_reason"] == "leader_reaped"
        assert result["cancellation"]["interrupt_count"] == 1


def test_repeated_interrupts_while_committing_result_are_all_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: list[int] = []
    receipt_attempts = 0
    source = "print('retained after repeated receipt interrupts')\n"
    with _python_process(source) as process:
        real_join = cooperative_subprocess.shlex.join

        def interrupt_two_receipt_attempts(argv):
            nonlocal receipt_attempts
            receipt_attempts += 1
            if receipt_attempts <= 2:
                os.kill(os.getpid(), signal.SIGINT)
            return real_join(argv)

        monkeypatch.setattr(
            cooperative_subprocess.os,
            "killpg",
            lambda group, signum: forwarded.append(signum),
        )
        monkeypatch.setattr(
            cooperative_subprocess.shlex,
            "join",
            interrupt_two_receipt_attempts,
        )
        with pytest.raises(CooperativeProcessInterrupted) as failure:
            communicate_cooperatively(process, process_group_id=process.pid)

        result = failure.value.result
        assert receipt_attempts == 3
        assert forwarded == []
        assert result["stdout"] == "retained after repeated receipt interrupts\n"
        assert result["returncode"] == 0
        assert result["execution"]["completion_scope"] == "process_group_quiescent"
        assert result["cancellation"]["interrupt_count"] == 2
        assert result["cancellation"]["repeated_interrupts"] == 1
        assert result["cancellation"]["signal_not_forwarded_reason"] == "leader_reaped"


def test_lost_final_reap_preserves_output_without_reporting_success(monkeypatch: pytest.MonkeyPatch) -> None:
    real_waitid = os.waitid
    original = ChildProcessError(errno.ECHILD, "Final status stolen by another reaper")
    final_statuses = []
    forwarded = []
    source = "import sys\nprint('finished stdout')\nprint('finished stderr', file=sys.stderr)\nsys.exit(7)\n"
    with _python_process(source) as process:
        def lose_final_status(idtype, pid, options):
            status = real_waitid(idtype, pid, options)
            if options == os.WEXITED:
                final_statuses.append(status)
                raise original
            return status

        monkeypatch.setattr(cooperative_subprocess.os, "waitid", lose_final_status)
        monkeypatch.setattr(cooperative_subprocess.os, "killpg", lambda group, signum: forwarded.append(signum))
        try:
            with pytest.raises(CooperativeCleanupIncomplete) as failure:
                communicate_cooperatively(process, process_group_id=process.pid)
            assert isinstance(failure.value.__cause__, RuntimeError)
            assert failure.value.__cause__.__cause__ is original
            assert len(final_statuses) == 1
            assert final_statuses[0].si_status == 7
            result = failure.value.cooperative_process_result
            assert result["stdout"] == "finished stdout\n"
            assert result["stderr"] == "finished stderr\n"
            assert result["returncode"] is None
            assert result["execution"]["completion_scope"] == "cleanup_incomplete"
            assert result["execution"]["live_process_group_members"] is None
            assert forwarded == []
        finally:
            # The seam consumed the real status. Restore only that observed handle state.
            if final_statuses:
                process.returncode = final_statuses[0].si_status


@pytest.mark.parametrize("receipt", [None, {"execution": {"completion_scope": "cleanup_incomplete", "process_group_id": True}}])
def test_malformed_terminal_event_is_not_ignored(receipt) -> None:
    nonce = "cpu-cleanup-event-nonce"
    payload = json.dumps({"event": "aisp_cleanup_incomplete", "nonce": nonce, "receipt": receipt})
    with pytest.raises(ValueError):
        cooperative_subprocess.parse_cleanup_incomplete_event(
            payload,
            expected_nonce=nonce,
        )


def test_terminal_event_requires_the_matching_launch_nonce() -> None:
    receipt = {
        "stdout": "captured stdout",
        "stderr": "captured stderr",
        "returncode": 1,
        "execution": {
            "process_group_id": 123,
            "completion_scope": "cleanup_incomplete",
        },
    }
    nonce = "matching-cpu-cleanup-event-nonce"
    stream = io.StringIO()
    cooperative_subprocess.emit_cleanup_incomplete_event(receipt, nonce=nonce, stream=stream)
    payload = stream.getvalue()

    assert cooperative_subprocess.parse_cleanup_incomplete_event(
        payload,
        expected_nonce="wrong-cpu-cleanup-event-nonce",
    ) is None
    assert cooperative_subprocess.parse_cleanup_incomplete_event(
        json.dumps({"event": "aisp_cleanup_incomplete", "receipt": receipt}),
        expected_nonce=nonce,
    ) is None
    assert cooperative_subprocess.parse_cleanup_incomplete_event(
        payload,
        expected_nonce=nonce,
    ) == receipt


def test_cleanup_event_nonce_is_removed_before_application_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AISP_CLEANUP_EVENT_NONCE", "one-time-cpu-event-nonce")
    assert cooperative_subprocess.consume_cleanup_event_nonce() == "one-time-cpu-event-nonce"
    assert "AISP_CLEANUP_EVENT_NONCE" not in os.environ


def test_terminal_receipt_lookup_stops_at_exception_cycles() -> None:
    first = RuntimeError("first wrapper")
    second = RuntimeError("second wrapper")
    first.__cause__ = second
    second.__context__ = first
    assert cooperative_subprocess.cleanup_incomplete_receipt_from_error(first) is None
