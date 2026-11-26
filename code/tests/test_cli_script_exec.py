"""Keep script-form CLI startup in the same process with unchanged arguments."""

import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CODE_ROOT / "cli" / "aisp.py"


def test_script_bootstrap_replaces_the_process_and_preserves_arguments_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecObserved(BaseException):
        pass

    captured = []
    arguments = ["bench", "run", "--targets", "path with spaces:example", "--execution-timeout-seconds", "0"]
    inherited_path = os.pathsep.join(["/existing/python path", str(CODE_ROOT)])
    monkeypatch.setenv("PYTHONPATH", inherited_path)
    monkeypatch.setenv("AISP_EXEC_FIXTURE", "preserved")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])

    def observe_exec(executable, argv, environment):
        captured.append((executable, argv, environment))
        raise ExecObserved

    monkeypatch.setattr(os, "execve", observe_exec)
    with pytest.raises(ExecObserved):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    executable, argv, environment = captured[0]
    assert executable == sys.executable
    assert argv == [sys.executable, "-m", "cli.aisp", *arguments]
    assert environment["PYTHONPATH"] == os.pathsep.join([str(CODE_ROOT), "/existing/python path"])
    assert environment["AISP_EXEC_FIXTURE"] == "preserved"
    assert os.environ["PYTHONPATH"] == inherited_path


def test_real_script_help_starts_from_outside_the_code_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bench" in result.stdout
