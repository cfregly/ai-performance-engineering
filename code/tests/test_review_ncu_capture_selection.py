from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.harness import run_benchmarks
from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_config import ProfilerConfig


def test_filtered_minimal_ncu_command_captures_every_kernel_in_range() -> None:
    command = ProfilerConfig(preset="minimal", metric_set="minimal").get_ncu_command_for_target(
        output_path="/tmp/report",
        target_command=["python", "workload.py"],
        nvtx_includes=["compute_kernel:profile"],
    )

    assert command[command.index("--nvtx-include") + 1] == "compute_kernel:profile"
    assert "--nvtx" in command
    assert "--target-processes" in command
    assert "--launch-count" not in command


def _write_fake_ncu(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import subprocess
            import sys

            args = sys.argv[1:]
            if args == ["--version"]:
                print("fixture ncu")
                raise SystemExit(0)
            if "--import" in args:
                print('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"')
                if os.environ.get("FAKE_NCU_EMPTY_REPORT") != "1":
                    print('"0","fixture_kernel_a","gpu__time_duration.avg","us","10"')
                    print('"1","fixture_kernel_b","gpu__time_duration.avg","us","20"')
                raise SystemExit(0)

            output_prefix = Path(args[args.index("-o") + 1])
            target = args[args.index("-o") + 2:]
            Path(os.environ["FAKE_NCU_COMMAND_LOG"]).write_text(
                json.dumps(args), encoding="utf-8"
            )
            completed = subprocess.run(target, check=False)
            if completed.returncode == 0 or os.environ.get("FAKE_NCU_WRITE_PARTIAL") == "1":
                output_prefix.with_suffix(".ncu-rep").write_text(
                    "fixture report", encoding="utf-8"
                )
            raise SystemExit(completed.returncode)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_benchmark_fixture(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            import os
            from pathlib import Path

            def _record(value):
                marker = Path(os.environ["NCU_BENCHMARK_MARKER"])
                with marker.open("a", encoding="utf-8") as stream:
                    stream.write(value + "\\n")

            class FixtureBenchmark:
                def setup(self):
                    _record("setup")

                def benchmark_fn(self):
                    _record("workload")

            def get_benchmark():
                return FixtureBenchmark()
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("empty_report", "accepted"),
    [(False, True), (True, False)],
)
def test_python_ncu_profile_uses_measured_range_and_rejects_empty_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_report: bool,
    accepted: bool,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_ncu(fake_bin / "ncu")
    benchmark_path = tmp_path / "fixture_benchmark.py"
    _write_benchmark_fixture(benchmark_path)
    marker = tmp_path / "benchmark.marker"
    command_log = tmp_path / "ncu-command.json"

    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("NCU_BENCHMARK_MARKER", str(marker))
    monkeypatch.setenv("FAKE_NCU_COMMAND_LOG", str(command_log))
    monkeypatch.setenv("FAKE_NCU_EMPTY_REPORT", "1" if empty_report else "0")

    result = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path=benchmark_path,
        chapter_dir=tmp_path,
        output_dir=tmp_path / "profiles",
        config=BenchmarkConfig(
            profile_type="minimal",
            ncu_metric_set="minimal",
            validity_profile="portable",
        ),
        variant="baseline",
        output_stem="fixture",
    )

    command = json.loads(command_log.read_text(encoding="utf-8"))
    assert command[command.index("--nvtx-include") + 1] == "compute_kernel:profile"
    assert "--nvtx" in command
    assert "--launch-count" not in command
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "setup",
        "workload",
        "workload",
    ]

    assert (result is not None) is accepted
    if not accepted:
        detail = run_benchmarks._get_profile_failure_detail("ncu")
        assert detail is not None
        assert "contains no importable kernel metrics" in detail


def _mock_profile_result(
    tmp_path: Path,
    *,
    returncode: int | None,
) -> SimpleNamespace:
    stdout_log = tmp_path / "mock-ncu.stdout.log"
    stderr_log = tmp_path / "mock-ncu.stderr.log"
    stdout_log.write_text("", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")
    return SimpleNamespace(
        process=SimpleNamespace(returncode=returncode),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        timed_out=False,
        failure_warning=None,
    )


def test_python_ncu_rejects_missing_exit_status_even_with_fresh_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "fixture_benchmark.py"
    _write_benchmark_fixture(benchmark_path)

    def run_with_incomplete_status(**kwargs):
        command = kwargs["command"]
        Path(command[command.index("-o") + 1]).with_suffix(".ncu-rep").write_text(
            "fresh partial report",
            encoding="utf-8",
        )
        return _mock_profile_result(tmp_path, returncode=None)

    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(run_benchmarks, "_run_profile_subprocess", run_with_incomplete_status)
    monkeypatch.setattr(
        run_benchmarks,
        "extract_from_ncu_report",
        lambda _path: {"kernel_time_ms": 1.0},
    )

    result = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path=benchmark_path,
        chapter_dir=tmp_path,
        output_dir=tmp_path / "profiles",
        config=BenchmarkConfig(validity_profile="portable"),
    )

    assert result is None
    detail = run_benchmarks._get_profile_failure_detail("ncu")
    assert detail is not None
    assert "completed process exit status" in detail


def test_python_ncu_does_not_accept_stale_same_name_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "fixture_benchmark.py"
    _write_benchmark_fixture(benchmark_path)
    output_dir = tmp_path / "profiles"
    output_dir.mkdir()
    stale_report = output_dir / "fixture_benchmark__baseline.ncu-rep"
    stale_report.write_text("stale report", encoding="utf-8")

    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(
        run_benchmarks,
        "_run_profile_subprocess",
        lambda **_kwargs: _mock_profile_result(tmp_path, returncode=0),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "extract_from_ncu_report",
        lambda _path: {"kernel_time_ms": 1.0},
    )

    result = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path=benchmark_path,
        chapter_dir=tmp_path,
        output_dir=output_dir,
        config=BenchmarkConfig(validity_profile="portable"),
    )

    assert result is None
    assert stale_report.read_text(encoding="utf-8") == "stale report"
    detail = run_benchmarks._get_profile_failure_detail("ncu")
    assert detail is not None
    assert "without a fresh report" in detail


def test_native_ncu_rejects_unscoped_capture_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fixture-executable"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    launched = False

    def unexpected_launch(**_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("unscoped native NCU must fail before launch")

    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(run_benchmarks, "_run_profile_subprocess", unexpected_launch)

    result = run_benchmarks.profile_cuda_executable_ncu(
        executable,
        chapter_dir=tmp_path,
        output_dir=tmp_path / "profiles",
        config=BenchmarkConfig(validity_profile="portable"),
    )

    assert result is None
    assert not launched
    detail = run_benchmarks._get_profile_failure_detail("ncu")
    assert detail is not None
    assert "requires an explicit BenchmarkConfig.nsys_nvtx_include workload range" in detail


@pytest.mark.parametrize(
    ("target_exit", "empty_report", "write_partial", "accepted"),
    [
        (0, False, False, True),
        (0, True, False, False),
        (7, False, True, False),
    ],
)
def test_native_ncu_accepts_only_successful_importable_scoped_fresh_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_exit: int,
    empty_report: bool,
    write_partial: bool,
    accepted: bool,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_ncu(fake_bin / "ncu")
    executable = tmp_path / "fixture-executable"
    executable.write_text(f"#!/bin/sh\nexit {target_exit}\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    command_log = tmp_path / "native-ncu-command.json"

    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_NCU_COMMAND_LOG", str(command_log))
    monkeypatch.setenv("FAKE_NCU_EMPTY_REPORT", "1" if empty_report else "0")
    monkeypatch.setenv("FAKE_NCU_WRITE_PARTIAL", "1" if write_partial else "0")

    result = run_benchmarks.profile_cuda_executable_ncu(
        executable,
        chapter_dir=tmp_path,
        output_dir=tmp_path / "profiles",
        config=BenchmarkConfig(
            profile_type="minimal",
            ncu_metric_set="minimal",
            nsys_nvtx_include=["compute_kernel:fixture"],
            validity_profile="portable",
        ),
    )

    command = json.loads(command_log.read_text(encoding="utf-8"))
    assert command[command.index("--nvtx-include") + 1] == "compute_kernel:fixture"
    assert "--nvtx" in command
    assert "--launch-count" not in command
    assert (result is not None) is accepted
    if result is not None:
        assert result.is_file()
        assert result.name.startswith("fixture-executable__baseline__ncu_")
        assert result.name.endswith("_attempt1.ncu-rep")
    else:
        detail = run_benchmarks._get_profile_failure_detail("ncu")
        assert detail is not None
        expected = "no importable kernel metrics" if empty_report else "exited with code 7"
        assert expected in detail
        assert list((tmp_path / "profiles").glob("*.ncu-rep"))
