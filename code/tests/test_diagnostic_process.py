from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import psutil
import pytest

from core.diagnostics.process import run_owned_process


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
