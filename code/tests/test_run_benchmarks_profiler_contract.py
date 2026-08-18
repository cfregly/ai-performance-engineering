import sys
import os
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os.path

import pytest

from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia, TorchrunLaunchSpec
from core.harness import run_benchmarks
from core.harness.run_benchmarks import (
    _attach_failure_metadata,
    _collect_required_profiler_failure_details,
    _build_torchrun_profile_command,
    _collect_required_profiler_failures,
    _format_required_profiler_failure,
    _resolve_profile_torchrun_spec,
    profile_python_benchmark,
    _run_profile_subprocess,
    _temporary_python_profile_launch,
)


def test_collect_required_profiler_failures_captures_baseline_and_optimized_failures() -> None:
    result_entry = {
        "baseline_profiler_statuses": {
            "nsys": "succeeded",
            "ncu": "failed",
            "torch": "skipped",
        }
    }
    best_opt = {
        "optimized_profiler_statuses": {
            "nsys": "failed",
            "ncu": "succeeded",
        }
    }

    failures = _collect_required_profiler_failures(
        result_entry,
        best_opt,
        profiling_requested=True,
    )

    assert failures == [
        "baseline:ncu:failed",
        "baseline:torch:skipped",
        "optimized:nsys:failed",
    ]


def test_collect_required_profiler_failures_ignores_disabled_profiling() -> None:
    failures = _collect_required_profiler_failures(
        {"baseline_profiler_statuses": {"nsys": "failed"}},
        {"optimized_profiler_statuses": {"ncu": "failed"}},
        profiling_requested=False,
    )

    assert failures == []


def test_format_required_profiler_failure_is_explicit() -> None:
    message = _format_required_profiler_failure(
        ["baseline:torch:failed", "optimized:nsys:skipped"]
    )

    assert message == (
        "Required profilers did not succeed: "
        "baseline:torch:failed, optimized:nsys:skipped"
    )


def test_collect_required_profiler_failure_details_returns_structured_errors() -> None:
    result_entry = {
        "baseline_profiler_statuses": {"nsys": "failed"},
        "baseline_profiler_errors": {"nsys": "no report artifact produced"},
    }
    best_opt = {
        "optimized_profiler_statuses": {"ncu": "skipped"},
        "optimized_profiler_errors": {"ncu": "Nsight Compute unavailable on current host."},
    }

    details = _collect_required_profiler_failure_details(
        result_entry,
        best_opt,
        profiling_requested=True,
    )

    assert details == {
        "baseline:nsys": "no report artifact produced",
        "optimized:ncu": "Nsight Compute unavailable on current host.",
    }


def test_format_required_profiler_failure_includes_detail_text() -> None:
    message = _format_required_profiler_failure(
        ["optimized:nsys:failed"],
        failure_details={"optimized:nsys": "no report artifact produced"},
    )

    assert message == (
        "Required profilers did not succeed: optimized:nsys:failed. "
        "Details: optimized:nsys: no report artifact produced"
    )


def test_attach_failure_metadata_promotes_child_failure_to_parent() -> None:
    result_entry = {
        "status": "failed_error",
        "error": "Baseline or optimization failed",
        "optimizations": [
            {
                "technique": "default",
                "status": "failed_error",
                "error": (
                    "Benchmark execution failed: ENVIRONMENT INVALID: "
                    "Foreign CUDA compute process(es) detected on benchmark GPU before run."
                ),
            }
        ],
    }

    _attach_failure_metadata(result_entry)

    assert result_entry["error"].startswith("Benchmark execution failed: ENVIRONMENT INVALID:")
    assert result_entry["failure_class"] == "environment_invalid"
    assert result_entry["failure_details"] == [
        {
            "scope": "optimization",
            "technique": "default",
            "status": "failed_error",
            "error": (
                "Benchmark execution failed: ENVIRONMENT INVALID: "
                "Foreign CUDA compute process(es) detected on benchmark GPU before run."
            ),
        }
    ]


def test_attach_failure_metadata_promotes_best_only_profiler_details_to_parent() -> None:
    result_entry = {
        "status": "failed_profiler",
        "error": (
            "Required profilers did not succeed: baseline:nsys:failed, "
            "optimized:nsys:failed. Details: baseline:nsys: Nsight Systems timed out "
            "after 120.0s; optimized:nsys: Nsight Systems timed out after 120.0s"
        ),
        "baseline_profiler_errors": {"nsys": "Nsight Systems timed out after 120.0s"},
        "optimizations": [
            {
                "technique": "optimized_pipeline_3stage",
                "status": "succeeded",
                "optimized_profiler_statuses": {
                    "nsys": "failed",
                    "ncu": "succeeded",
                    "torch": "succeeded",
                },
                "optimized_profiler_errors": {
                    "nsys": "Nsight Systems timed out after 120.0s",
                },
            }
        ],
    }

    _attach_failure_metadata(result_entry)

    assert result_entry["failure_class"] == "profiler_failed"
    assert result_entry["optimized_profiler_statuses"] == {
        "nsys": "failed",
        "ncu": "succeeded",
        "torch": "succeeded",
    }
    assert result_entry["optimized_profiler_errors"] == {
        "nsys": "Nsight Systems timed out after 120.0s",
    }


def test_temporary_python_profile_launch_uses_python_wrapper_by_default(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    chapter_dir = repo_root / "ch03"
    chapter_dir.mkdir(parents=True)
    benchmark = SimpleNamespace(profile_env_overrides={"AISP_TEST_OVERRIDE": "1"})

    with _temporary_python_profile_launch(
        "print('ok')\n",
        chapter_dir=chapter_dir,
        repo_root=repo_root,
        config=None,
        benchmark=benchmark,
    ) as (wrapper_path, command, env, use_torchrun):
        assert wrapper_path.exists()
        assert command == [sys.executable, str(wrapper_path)]
        assert use_torchrun is False
        assert env["AISP_TEST_OVERRIDE"] == "1"
        assert env["TORCH_DISABLE_ADDR2LINE"] == "1"
        pythonpath_entries = [entry for entry in env["PYTHONPATH"].split(os.pathsep) if entry]
        assert str(repo_root) in pythonpath_entries
        assert str(chapter_dir) in pythonpath_entries


def test_temporary_python_profile_launch_honors_torchrun_when_allowed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    chapter_dir = repo_root / "ch11"
    chapter_dir.mkdir(parents=True)
    config = BenchmarkConfig(
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=2,
        profile_env_overrides={"AISP_CONFIG_OVERRIDE": "1"},
    )

    with _temporary_python_profile_launch(
        "print('ok')\n",
        chapter_dir=chapter_dir,
        repo_root=repo_root,
        config=config,
        benchmark=SimpleNamespace(),
    ) as (wrapper_path, command, env, use_torchrun):
        assert wrapper_path.exists()
        assert os.path.basename(command[0]) == "torchrun" or command[:3] == [
            sys.executable,
            "-m",
            "torch.distributed.run",
        ]
        assert str(wrapper_path) in command
        assert use_torchrun is True
        assert env["AISP_CONFIG_OVERRIDE"] == "1"


def test_temporary_python_profile_launch_can_disable_torchrun(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    chapter_dir = repo_root / "ch11"
    chapter_dir.mkdir(parents=True)
    config = BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=2)

    with _temporary_python_profile_launch(
        "print('ok')\n",
        chapter_dir=chapter_dir,
        repo_root=repo_root,
        config=config,
        benchmark=SimpleNamespace(),
        allow_torchrun=False,
    ) as (wrapper_path, command, _env, use_torchrun):
        assert wrapper_path.exists()
        assert command == [sys.executable, str(wrapper_path)]
        assert use_torchrun is False


def test_resolve_profile_torchrun_spec_prefers_benchmark_profile_spec(tmp_path: Path) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    config = BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=1)

    class _Bench:
        def get_profile_torchrun_spec(self, *, profiler, config=None, output_path=None):
            assert profiler == "torch"
            assert output_path == tmp_path / "trace.json"
            return TorchrunLaunchSpec(
                script_path=script_path,
                script_args=["--torch-profile-output", str(output_path)],
            )

    spec = _resolve_profile_torchrun_spec(
        _Bench(),
        profiler="torch",
        config=config,
        output_path=tmp_path / "trace.json",
    )

    assert spec is not None
    assert spec.script_path == script_path
    assert spec.script_args == ["--torch-profile-output", str(tmp_path / "trace.json")]


def test_resolve_profile_torchrun_spec_falls_back_to_base_spec_for_nsys(tmp_path: Path) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    config = BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=1)

    class _Bench:
        def get_torchrun_spec(self, _config):
            return TorchrunLaunchSpec(script_path=script_path, script_args=["--skip-preflight"])

    spec = _resolve_profile_torchrun_spec(
        _Bench(),
        profiler="nsys",
        config=config,
    )

    assert spec is not None
    assert spec.script_path == script_path
    assert spec.script_args == ["--skip-preflight"]


def test_build_torchrun_profile_command_bypasses_launcher_for_single_process(tmp_path: Path) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    config = BenchmarkConfig(
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        nnodes="1",
        seed=123,
    )
    spec = TorchrunLaunchSpec(script_path=script_path, script_args=["--skip-preflight"])

    command, env = _build_torchrun_profile_command(config, spec=spec)

    assert command[:3] == [sys.executable, "-m", "core.harness.torchrun_wrapper"]
    assert "--aisp-target-script" in command
    assert str(script_path) in command
    assert command[-1] == "--skip-preflight"
    assert env["RANK"] == "0"
    assert env["WORLD_SIZE"] == "1"
    assert env["LOCAL_RANK"] == "0"


def test_build_torchrun_profile_command_keeps_launcher_for_multi_process(tmp_path: Path) -> None:
    script_path = tmp_path / "demo.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")
    config = BenchmarkConfig(
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=2,
        nnodes="1",
        seed=123,
    )
    spec = TorchrunLaunchSpec(script_path=script_path, script_args=["--skip-preflight"])

    command, _env = _build_torchrun_profile_command(config, spec=spec)

    assert "--nproc_per_node" in command
    assert "2" in command
    assert "-m" in command
    assert "core.harness.torchrun_wrapper" in command


def test_profile_python_benchmark_retries_direct_wrapper_once(tmp_path: Path) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"
    config = BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=1, nnodes="1")
    report_path = output_dir / "demo__baseline.nsys-rep"

    class _Bench:
        def get_profile_torchrun_spec(self, *, profiler, config=None, output_path=None):
            assert profiler == "nsys"
            return TorchrunLaunchSpec(script_path=bench_path, script_args=["--skip-preflight"])

    class _Automation:
        def __init__(self, _output_dir):
            self.calls = 0
            self.last_error = None

        def profile_nsys(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.last_error = (
                    "Nsight Systems exited successfully but no report artifact was produced "
                    f"at {report_path}"
                )
                return None
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("rep", encoding="utf-8")
            self.last_error = None
            return report_path

    automation = _Automation(output_dir)

    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", return_value=automation),
        patch.object(run_benchmarks.time, "sleep", return_value=None),
    ):
        report = profile_python_benchmark(
            _Bench(),
            bench_path,
            bench_path.parent,
            output_dir,
            config=config,
            variant="baseline",
            output_stem="demo",
        )

    assert report == report_path
    assert automation.calls == 2


def test_profile_python_benchmark_retries_direct_python_wrapper_once(tmp_path: Path) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"
    report_path = output_dir / "demo__baseline.nsys-rep"

    class _Automation:
        def __init__(self, _output_dir):
            self.calls = 0
            self.last_error = None

        def profile_nsys(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                self.last_error = (
                    "Nsight Systems exited successfully but no report artifact was produced "
                    f"at {report_path}"
                )
                return None
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("rep", encoding="utf-8")
            self.last_error = None
            return report_path

    automation = _Automation(output_dir)

    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", return_value=automation),
        patch.object(run_benchmarks.time, "sleep", return_value=None),
    ):
        report = profile_python_benchmark(
            SimpleNamespace(),
            bench_path,
            bench_path.parent,
            output_dir,
            config=BenchmarkConfig(),
            variant="baseline",
            output_stem="demo",
        )

    assert report == report_path
    assert automation.calls == 2


def test_profile_python_benchmark_falls_back_to_clean_helper_retry(tmp_path: Path) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"
    config = BenchmarkConfig(launch_via=LaunchVia.TORCHRUN, nproc_per_node=1, nnodes="1")
    report_path = output_dir / "demo__baseline.nsys-rep"

    class _Bench:
        def get_profile_torchrun_spec(self, *, profiler, config=None, output_path=None):
            assert profiler == "nsys"
            return TorchrunLaunchSpec(script_path=bench_path, script_args=["--skip-preflight"])

    class _Automation:
        def __init__(self, _output_dir):
            self.calls = 0
            self.last_error = None

        def profile_nsys(self, **_kwargs):
            self.calls += 1
            self.last_error = (
                "Nsight Systems exited successfully but no report artifact was produced "
                f"at {report_path}"
            )
            return None

    def _fake_helper_run(args, **kwargs):
        result_idx = args.index("--result") + 1
        result_file = Path(args[result_idx])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("rep", encoding="utf-8")
        result_file.write_text(
            json.dumps({"report": str(report_path), "last_error": None}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    automation = _Automation(output_dir)

    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", return_value=automation),
        patch.object(run_benchmarks.subprocess, "run", side_effect=_fake_helper_run),
        patch.object(run_benchmarks.time, "sleep", return_value=None),
    ):
        report = profile_python_benchmark(
            _Bench(),
            bench_path,
            bench_path.parent,
            output_dir,
            config=config,
            variant="baseline",
            output_stem="demo",
        )

    assert report == report_path
    assert automation.calls == 2


def test_run_profile_subprocess_captures_output_and_writes_logs(tmp_path: Path) -> None:
    log_base = tmp_path / "captured"

    result = _run_profile_subprocess(
        command=[
            sys.executable,
            "-c",
            "import sys; print('stdout-line'); sys.stderr.write('stderr-line\\n')",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5.0,
        log_base=log_base,
        terminate_reason="captured",
        capture_output=True,
        timeout_collect_error_message="timeout follow-up failed",
        wait_error_message="communicate failed",
    )

    assert result.process.returncode == 0
    assert result.timed_out is False
    assert result.failure_warning is None
    assert result.stdout_log.read_text() == "stdout-line\n"
    assert result.stderr_log.read_text() == "stderr-line\n"
    assert json.loads(log_base.with_suffix(".command.json").read_text())["command"][0] == sys.executable


def test_run_profile_subprocess_streams_output_to_logs(tmp_path: Path) -> None:
    log_base = tmp_path / "streamed"

    result = _run_profile_subprocess(
        command=[
            sys.executable,
            "-c",
            "import sys; print('stream-stdout'); sys.stderr.write('stream-stderr\\n')",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5.0,
        log_base=log_base,
        terminate_reason="streamed",
        capture_output=False,
        timeout_collect_error_message="timeout follow-up failed",
        wait_error_message="wait failed",
    )

    assert result.process.returncode == 0
    assert result.timed_out is False
    assert result.failure_warning is None
    assert result.stdout_log.read_text() == "stream-stdout\n"
    assert result.stderr_log.read_text() == "stream-stderr\n"
    assert json.loads(log_base.with_suffix(".command.json").read_text())["command"][0] == sys.executable


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="requires Linux /proc process-group enumeration",
)
def test_run_profile_subprocess_reaps_lingering_process_group_members(tmp_path: Path) -> None:
    log_base = tmp_path / "lingering"
    child_pid_path = tmp_path / "lingering_child.pid"
    launcher = (
        "import pathlib, subprocess, sys, time; "
        f"pid_path = pathlib.Path({str(child_pid_path)!r}); "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        "); "
        "pid_path.write_text(str(child.pid), encoding='utf-8'); "
        "time.sleep(0.2)"
    )

    result = _run_profile_subprocess(
        command=[sys.executable, "-c", launcher],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=5.0,
        log_base=log_base,
        terminate_reason="lingering",
        capture_output=True,
        timeout_collect_error_message="timeout follow-up failed",
        wait_error_message="communicate failed",
    )

    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
    try:
        deadline = time.time() + 5.0
        while deadline > time.time():
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)

        assert result.process.returncode == 0
        assert result.timed_out is False
        assert result.failure_warning is None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass


def test_profile_cuda_executable_ncu_uses_shared_subprocess_runner(tmp_path: Path) -> None:
    executable = tmp_path / "demo_exec"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    class _ProfilerConfig:
        nvtx_includes: list[str] = []

        @staticmethod
        def get_ncu_command_for_target(
            _output_prefix: str,
            _target: list[str],
            metrics: list[str] | None = None,
            nvtx_includes: list[str] | None = None,
        ) -> list[str]:
            return ["ncu", "--set", "basic", str(executable)]

    fake_result = SimpleNamespace(
        process=SimpleNamespace(returncode=0),
        stdout_log=tmp_path / "stdout.log",
        stderr_log=tmp_path / "stderr.log",
        timed_out=False,
        failure_warning=None,
    )

    with (
        patch.object(run_benchmarks, "check_ncu_available", return_value=True),
        patch.object(run_benchmarks, "build_profiler_config_from_benchmark", return_value=_ProfilerConfig()),
        patch.object(run_benchmarks, "_run_profile_subprocess", return_value=fake_result) as run_mock,
    ):
        result = run_benchmarks.profile_cuda_executable_ncu(
            executable,
            chapter_dir=tmp_path,
            output_dir=tmp_path / "profiles",
            config=BenchmarkConfig(),
        )

    assert result is None
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["command"] == ["ncu", "--force-overwrite", "--set", "basic", str(executable)]
    assert run_mock.call_args.kwargs["capture_output"] is True


def test_retry_nsys_in_clean_helper_passes_owner_markers(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "helper_report.nsys-rep"
    report_path.write_text("report\n", encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setenv("AISP_BENCHMARK_OWNER_RUN_ID", "owner-run-xyz")
    monkeypatch.setenv("AISP_BENCHMARK_OWNER_PID", "31337")
    monkeypatch.setattr(run_benchmarks, "build_repo_python_env", lambda repo_root, base_env=None: dict(base_env or {}))

    def _fake_subprocess_run(command, **kwargs):
        observed["command"] = list(command)
        observed["env"] = dict(kwargs["env"])
        result_arg = command[command.index("--result") + 1]
        Path(result_arg).write_text(
            json.dumps({"report": str(report_path), "last_error": None}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_benchmarks.subprocess, "run", _fake_subprocess_run)

    result = run_benchmarks._retry_nsys_in_clean_helper(
        output_dir=tmp_path / "profiles",
        output_name="demo__baseline",
        target_command=[sys.executable, "-c", "print('demo')"],
        trace_forks=False,
        profile_preset="light",
        full_timeline=False,
        timeout=5.0,
        wait_mode="primary",
        env={"PYTHONPATH": str(tmp_path)},
    )

    assert result == report_path
    command = observed["command"]
    assert "--aisp-owner-run-id" in command
    assert "--aisp-owner-pid" in command
    assert command[command.index("--aisp-owner-run-id") + 1] == "owner-run-xyz"
    assert command[command.index("--aisp-owner-pid") + 1] == "31337"


def test_profile_cuda_executable_retries_missing_artifact_with_clean_helper(tmp_path: Path) -> None:
    executable = tmp_path / "demo_exec"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    report_path = tmp_path / "profiles" / "demo_exec__optimized.nsys-rep"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("report\n", encoding="utf-8")

    class _FakeAutomation:
        profile_calls = 0

        def __init__(self, _output_dir: Path) -> None:
            self.last_error = None

        def profile_nsys(self, **_kwargs):
            type(self).profile_calls += 1
            self.last_error = "No report artifact was produced"
            return None

    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", _FakeAutomation),
        patch.object(run_benchmarks, "_retry_nsys_in_clean_helper", return_value=report_path) as helper_mock,
    ):
        result = run_benchmarks.profile_cuda_executable(
            executable,
            chapter_dir=tmp_path,
            output_dir=tmp_path / "profiles",
            variant="optimized",
            timeout_seconds=5,
        )

    assert result == report_path
    assert _FakeAutomation.profile_calls == 2
    helper_mock.assert_called_once()
