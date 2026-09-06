from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.benchmark import bench_commands
from core.harness import run_benchmarks


def _child_visible_devices() -> str:
    return subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ.get('CUDA_VISIBLE_DEVICES', '<missing>'))"
            ),
        ],
        text=True,
    ).strip()


@pytest.mark.parametrize(
    ("original", "selected"),
    [(None, "0"), ("5,7", "5")],
)
def test_execute_benchmarks_scopes_single_gpu_visibility_for_child_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str | None,
    selected: str,
) -> None:
    if original is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", original)
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", False)

    observed: list[str] = []
    monkeypatch.setattr(
        bench_commands,
        "setup_logging",
        lambda **_kwargs: observed.append(_child_visible_devices()),
    )

    result = bench_commands._execute_benchmarks(
        targets=["ch04:gradient_fusion"],
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="device_visibility_return",
        single_gpu=True,
        exit_on_failure=False,
    )

    assert result["error"].startswith("Benchmark dependencies missing")
    assert observed == [selected]
    if original is None:
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    else:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == original


def test_execute_benchmarks_restores_visibility_after_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "setup_logging", lambda **_kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        bench_commands._execute_benchmarks(
            targets=["ch04:gradient_fusion"],
            output_format="json",
            profile_type="none",
            artifacts_dir=str(tmp_path / "artifacts"),
            run_id="device_visibility_exit",
            single_gpu=True,
            exit_on_failure=True,
        )

    assert exc_info.value.code == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,6"


def test_execute_benchmarks_restores_absent_visibility_after_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fail_after_selection(**_kwargs) -> None:
        assert _child_visible_devices() == "0"
        raise RuntimeError("fixture setup failure")

    monkeypatch.setattr(bench_commands, "setup_logging", _fail_after_selection)

    with pytest.raises(RuntimeError, match="fixture setup failure"):
        bench_commands._execute_benchmarks(
            targets=["ch04:gradient_fusion"],
            output_format="json",
            profile_type="none",
            artifacts_dir=str(tmp_path / "artifacts"),
            run_id="device_visibility_exception",
            single_gpu=True,
            exit_on_failure=False,
        )

    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_test_chapter_scopes_visibility_until_child_work_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    observed: list[str] = []

    def _capture_then_fail() -> None:
        observed.append(_child_visible_devices())
        raise RuntimeError("stop after environment capture")

    monkeypatch.setattr(
        run_benchmarks,
        "dump_environment_and_capabilities",
        _capture_then_fail,
    )

    with pytest.raises(RuntimeError, match="stop after environment capture"):
        run_benchmarks.test_chapter(
            tmp_path / "ch99",
            single_gpu=True,
            validity_profile="portable",
        )

    assert observed == ["GPU-first"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-first,GPU-second"
