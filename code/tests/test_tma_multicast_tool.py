"""CPU-only regression tests for the Chapter 10 TMA multicast demo tool."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = CODE_ROOT / "ch10" / "tma_multicast_tool.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tma_multicast_tool", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_binary_preserves_output_and_returns_marked_skip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()

    return_code = tool._run_demo_binary(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('SKIPPED: unsupported test architecture'); "
                "print('skip detail', file=sys.stderr); "
                "raise SystemExit(3)"
            ),
        ],
        cwd=tmp_path,
    )

    captured = capsys.readouterr()
    assert return_code == 3
    assert captured.out == "SKIPPED: unsupported test architecture\n"
    assert captured.err == "skip detail\n"


def test_demo_binary_rejects_unmarked_exit_code_three(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        tool._run_demo_binary(
            [sys.executable, "-c", "raise SystemExit(3)"],
            cwd=tmp_path,
        )

    assert exc_info.value.returncode == 3


def test_demo_binary_preserves_non_skip_failures_as_hard_failures(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        tool._run_demo_binary(
            [
                sys.executable,
                "-c",
                "print('SKIPPED: wrong exit code'); raise SystemExit(2)",
            ],
            cwd=tmp_path,
        )

    assert exc_info.value.returncode == 2


def test_build_failures_remain_hard_failures(tmp_path: Path) -> None:
    tool = _load_tool()

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        tool._run(
            [sys.executable, "-c", "raise SystemExit(2)"],
            cwd=tmp_path,
        )

    assert exc_info.value.returncode == 2


def test_main_returns_three_without_launching_cluster_after_baseline_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(tool, "_detect_sm_suffix", lambda: "_sm120")

    def skip_baseline(cmd: list[str], cwd: Path) -> int:
        launched_commands.append(cmd)
        return 3

    monkeypatch.setattr(tool, "_run_demo_binary", skip_baseline)

    assert tool.main(["--no-build"]) == 3
    assert launched_commands == [["./tma_multicast_baseline_sm120"]]
