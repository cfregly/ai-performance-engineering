from __future__ import annotations

import base64
import errno
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

import core.diagnostics.process as diagnostic
from core.diagnostics.process import run_owned_process
from core.utils.cooperative_subprocess import (
    CooperativeCleanupIncomplete,
    CooperativeProcessInterrupted,
)


def _is_live(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_timeout_interrupts_detached_descendant_after_launcher_exits(tmp_path: Path) -> None:
    child_pid = tmp_path / "child-pid"
    interrupted = tmp_path / "interrupted"
    child_code = """
import os
import signal
import sys
import time
from pathlib import Path

pid_path = Path(sys.argv[1])
interrupted_path = Path(sys.argv[2])

def stop(_signum, _frame):
    interrupted_path.write_text("sigint")
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
pid_path.write_text(str(os.getpid()))
while True:
    time.sleep(1)
"""
    launcher_code = """
import subprocess
import sys
import time
from pathlib import Path

subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]],
    start_new_session=True,
)
pid_path = Path(sys.argv[2])
for _ in range(200):
    if pid_path.exists():
        break
    time.sleep(0.005)
"""

    result = run_owned_process(
        [sys.executable, "-c", launcher_code, child_code, str(child_pid), str(interrupted)],
        timeout_seconds=1.0,
        interrupt_grace_seconds=1.0,
    )

    pid = int(child_pid.read_text())
    assert result["timeout_hit"] is True
    assert result["returncode"] == 124
    assert result["process_returncode"] == 0
    assert result["error"].startswith("Timed out")
    assert interrupted.read_text() == "sigint"
    assert pid in result["termination"]["sigint_sent"]
    assert not result["termination"]["survivors"]
    assert not _is_live(pid)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("timeout_seconds", math.nan),
        ("timeout_seconds", math.inf),
        ("timeout_seconds", 0),
        ("interrupt_grace_seconds", math.nan),
        ("terminate_grace_seconds", -1),
        ("poll_seconds", math.inf),
    ],
)
def test_process_intervals_must_be_finite(argument: str, value: float) -> None:
    options = {"timeout_seconds": 1, argument: value}
    with pytest.raises(ValueError, match=argument):
        run_owned_process([sys.executable, "-c", "pass"], **options)


def test_launch_enoent_retains_machine_readable_error_fields(tmp_path: Path) -> None:
    missing = tmp_path / "missing-diagnostic-command"

    result = run_owned_process([str(missing)], timeout_seconds=1)

    assert result["returncode"] == -1
    assert result["error_type"] == "FileNotFoundError"
    assert result["errno"] == errno.ENOENT
    assert result["error"] == result["stderr"]


def test_finite_owned_launch_allows_inherited_ignored_sigint() -> None:
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = run_owned_process(
            [sys.executable, "-c", "print('finite launch completed')"],
            timeout_seconds=5,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    assert result["returncode"] == 0
    assert result["stdout"] == "finite launch completed\n"
    assert result["execution"]["completion_scope"] == "process_tree_quiescent"


def test_finite_owned_launch_retains_invalid_output_bytes() -> None:
    result = run_owned_process(
        [
            sys.executable,
            "-c",
            "import os\nos.write(1, b'finite\\xffstdout')\nos.write(2, b'finite\\xfestderr')\n",
        ],
        timeout_seconds=5,
    )

    assert result["returncode"] == 0
    assert result["stdout"] == "finite\ufffdstdout"
    assert result["stderr"] == "finite\ufffdstderr"
    raw = result["execution"]["undecodable_streams_base64"]
    assert base64.b64decode(raw["stdout"]) == b"finite\xffstdout"
    assert base64.b64decode(raw["stderr"]) == b"finite\xfestderr"
    assert result["raw_streams_base64"] == raw


def test_finite_output_decoder_uses_python_310_compatible_locale_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostic.locale, "getpreferredencoding", lambda _setlocale: "ascii")

    stdout, stderr, raw = diagnostic._decode_captured_output((b"ok\xff", b"error"))

    assert stdout == "ok�"
    assert stderr == "error"
    assert base64.b64decode(raw["stdout"]) == b"ok\xff"


def test_finite_owned_launch_encodes_text_input_for_binary_capture() -> None:
    payload = "finite input with unicode \u03bb\n"
    result = run_owned_process(
        [sys.executable, "-c", "import sys\nsys.stdout.write(sys.stdin.read())\n"],
        timeout_seconds=5,
        input=payload,
    )

    assert result["returncode"] == 0
    assert result["stdout"] == payload
    assert result["stderr"] == ""


def test_finite_timeout_cleanup_retains_invalid_output_and_verifies_exit() -> None:
    result = run_owned_process(
        [
            sys.executable,
            "-c",
            "import os, time\nos.write(1, b'before-timeout\\xff')\nos.write(2, b'before-timeout\\xfe')\ntime.sleep(30)\n",
        ],
        timeout_seconds=2,
        interrupt_grace_seconds=1,
    )

    assert result["timeout_hit"] is True
    assert result["returncode"] == 124
    assert result["execution"]["completion_scope"] == "process_tree_quiescent"
    assert result["termination"]["survivors"] == []
    raw = result["execution"]["undecodable_streams_base64"]
    assert b"before-timeout\xff" in base64.b64decode(raw["stdout"])
    assert b"before-timeout\xfe" in base64.b64decode(raw["stderr"])


def test_incomplete_finite_drain_retains_partial_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_drain = diagnostic._drain_root

    def retain_but_mark_incomplete(process, timeout_seconds, errors):
        attempt = real_drain(process, timeout_seconds, errors)
        return diagnostic._DrainResult(False, attempt.captured)

    monkeypatch.setattr(diagnostic, "_drain_root", retain_but_mark_incomplete)
    with pytest.raises(CooperativeCleanupIncomplete) as caught:
        run_owned_process(
            [
                sys.executable,
                "-c",
                "import os, time\nos.write(1, b'partial\\xff')\nos.write(2, b'error\\xfe')\ntime.sleep(30)\n",
            ],
            timeout_seconds=2,
            interrupt_grace_seconds=1,
        )

    result = caught.value.result
    assert result["stdout"] == "partial�"
    assert result["stderr"].startswith("error�")
    assert result["execution"]["completion_scope"] == "cleanup_incomplete"
    raw = result["execution"]["undecodable_streams_base64"]
    raw_stdout = base64.b64decode(raw["stdout"])
    raw_stderr = base64.b64decode(raw["stderr"])
    assert result["execution"]["capture_limit"]["bytes_retained"] == {
        "stdout": len(raw_stdout),
        "stderr": len(raw_stderr),
    }
    assert raw_stdout == b"partial\xff"
    assert raw_stderr.startswith(b"error\xfe")


def test_deferred_sigint_takes_priority_over_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-after-interrupt"
    real_popen = diagnostic.psutil.Popen
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt_then_fail(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(diagnostic.psutil, "Popen", interrupt_then_fail)
    with pytest.raises(KeyboardInterrupt) as failure:
        run_owned_process([str(missing)], timeout_seconds=1)

    assert type(failure.value) is KeyboardInterrupt
    assert isinstance(failure.value.__cause__, FileNotFoundError)
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_post_spawn_launch_error_reaps_the_registered_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("registration failed after spawn")
    real_remember = diagnostic._OwnedProcesses.remember
    spawned: list[psutil.Process] = []

    def remember_then_fail(owned, process):
        real_remember(owned, process)
        if not spawned:
            spawned.append(process)
            raise original

    monkeypatch.setattr(diagnostic._OwnedProcesses, "remember", remember_then_fail)
    with pytest.raises(RuntimeError) as failure:
        run_owned_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=30,
        )

    assert failure.value is original
    result = failure.value.cooperative_process_result
    assert result["termination"]["reason"] == "launch_error"
    assert result["execution"]["completion_scope"] == "process_tree_quiescent"
    assert len(spawned) == 1
    assert not _is_live(spawned[0].pid)


def test_timeout_ignores_exited_child_and_stops_live_root(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    root_code = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "pass"])
child.wait()
Path(sys.argv[1]).write_text(json.dumps({"root": os.getpid(), "child": child.pid}))
time.sleep(30)
"""

    result = run_owned_process(
        [sys.executable, "-c", root_code, str(state_path)],
        timeout_seconds=1.0,
        interrupt_grace_seconds=0.2,
        terminate_grace_seconds=0.2,
    )

    state = json.loads(state_path.read_text())
    assert result["timeout_hit"] is True
    assert state["root"] in result["termination"]["sigint_sent"]
    assert state["child"] not in result["termination"]["sigint_sent"]
    assert not result["termination"]["survivors"]
    assert not _is_live(state["root"])


def test_timeout_escalates_to_kill_for_uncooperative_process(tmp_path: Path) -> None:
    pid_path = tmp_path / "stubborn-pid"
    stubborn_code = """
import os
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    time.sleep(1)
"""

    result = run_owned_process(
        [sys.executable, "-c", stubborn_code, str(pid_path)],
        timeout_seconds=1.0,
        interrupt_grace_seconds=0.1,
        terminate_grace_seconds=0.1,
    )

    pid = int(pid_path.read_text())
    assert pid in result["termination"]["sigint_sent"]
    assert pid in result["termination"]["terminate_sent"]
    assert pid in result["termination"]["kill_sent"]
    assert not result["termination"]["survivors"]
    assert not _is_live(pid)


def test_access_denied_identity_remains_an_incomplete_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = diagnostic._OwnedProcesses("cpu-owner-token", psutil.Process(os.getpid()))

    def denied(_identity):
        raise psutil.AccessDenied(pid=os.getpid())

    monkeypatch.setattr(diagnostic, "_resolve", denied)
    assert owned.live() == []
    assert any("AccessDenied" in error for error in owned.observation_errors())


def test_no_such_identity_is_terminal_without_an_observation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = diagnostic._OwnedProcesses("cpu-owner-token", psutil.Process(os.getpid()))
    monkeypatch.setattr(diagnostic, "_resolve", lambda _identity: None)

    assert owned.live() == []
    assert owned.observation_errors() == []


def test_signal_exit_race_does_not_create_a_cleanup_error() -> None:
    class ExitedProcess:
        pid = 456

        def send_signal(self, _signum):
            raise psutil.NoSuchProcess(self.pid)

    class ExitedOwned:
        def live(self):
            return [ExitedProcess()]

    errors: list[str] = []
    assert diagnostic._signal_live(ExitedOwned(), "sigint", errors) == []
    assert errors == []


def test_repeated_sigint_during_cleanup_is_counted_and_process_is_reaped(tmp_path: Path) -> None:
    child_pid = tmp_path / "interrupt-child-pid"
    cleanup_started = tmp_path / "interrupt-cleanup-started"
    child_code = """
import os
import signal
import sys
import time
from pathlib import Path

pid_path = Path(sys.argv[1])
cleanup_path = Path(sys.argv[2])

def stop(_signum, _frame):
    cleanup_path.write_text("cleanup started")
    time.sleep(0.25)
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
pid_path.write_text(str(os.getpid()))
while True:
    time.sleep(1)
"""
    sender_finished = threading.Event()

    def send_interrupts() -> None:
        while not child_pid.exists():
            time.sleep(0.005)
        os.kill(os.getpid(), signal.SIGINT)
        while not cleanup_started.exists():
            time.sleep(0.005)
        os.kill(os.getpid(), signal.SIGINT)
        sender_finished.set()

    sender = threading.Thread(target=send_interrupts)
    sender.start()
    with pytest.raises(CooperativeProcessInterrupted) as interrupted:
        run_owned_process(
            [sys.executable, "-c", child_code, str(child_pid), str(cleanup_started)],
            timeout_seconds=30,
            interrupt_grace_seconds=1.0,
        )
    sender.join()

    pid = int(child_pid.read_text())
    result = interrupted.value.result
    assert sender_finished.is_set()
    assert result["cancellation"]["interrupt_count"] >= 2
    assert result["execution"]["completion_scope"] == "process_tree_quiescent"
    assert not _is_live(pid)


def test_owned_launch_defers_sigint_until_real_child_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = diagnostic.psutil.Popen
    spawned: list[psutil.Popen] = []

    def spawn_then_interrupt(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        os.kill(os.getpid(), signal.SIGINT)
        return process

    monkeypatch.setattr(diagnostic.psutil, "Popen", spawn_then_interrupt)
    with pytest.raises(CooperativeProcessInterrupted) as interrupted:
        run_owned_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=30,
        )

    assert len(spawned) == 1
    assert interrupted.value.result["cancellation"]["operator_interrupt_observed"] is True
    assert interrupted.value.result["execution"]["completion_scope"] == "process_tree_quiescent"
    assert not _is_live(spawned[0].pid)
