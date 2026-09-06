import sys
import os
import json
import subprocess
import time
from contextlib import contextmanager
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
from core.profiling.nsight_automation import NsightAutomation
from core.profiling.profiler_wrapper import (
    render_ncu_python_profile_wrapper,
    render_nsys_python_profile_wrapper,
)


def _load_nsys_wrapper_test_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_profile_call: bool = False,
    start_status: int = 0,
    stop_status: int = 0,
):
    benchmark_path = tmp_path / "profile_fixture.py"
    benchmark_path.write_text(
        """
EVENTS = []
FAIL_PROFILE_CALL = False

class _Benchmark:
    profile_require_teardown = False

    def setup(self):
        EVENTS.append("setup")

    def benchmark_fn(self):
        EVENTS.append("benchmark")
        if FAIL_PROFILE_CALL and EVENTS.count("benchmark") == 2:
            raise ValueError("primary profile failure")

    def teardown(self):
        EVENTS.append("teardown")

def get_benchmark():
    return _Benchmark()
""",
        encoding="utf-8",
    )
    wrapper = render_nsys_python_profile_wrapper(
        benchmark_path=benchmark_path,
        nvtx_includes=["compute_kernel:profile"],
        target_label=None,
        target_override_argv=None,
        validity_profile="strict",
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
    )
    scope = {"__name__": "rendered_nsys_wrapper_test"}
    exec(compile(wrapper, str(tmp_path / "rendered_nsys_wrapper.py"), "exec"), scope)
    events = scope["_BENCHMARK_MODULE"].EVENTS
    scope["_BENCHMARK_MODULE"].FAIL_PROFILE_CALL = fail_profile_call

    class _Cudart:
        def cudaProfilerStart(self):
            events.append("profiler_start")
            return start_status

        def cudaProfilerStop(self):
            events.append("profiler_stop")
            return stop_status

    @contextmanager
    def _nvtx_range(*_args, **_kwargs):
        events.append("nvtx_enter")
        try:
            yield
        finally:
            events.append("nvtx_exit")

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _Cudart(), raising=False)
    monkeypatch.setattr(os, "_exit", lambda code: events.append(f"exit:{code}"))
    scope["ramp_gpu_clocks"] = lambda **_kwargs: None
    scope["nvtx_range"] = _nvtx_range
    return scope, events


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


def test_nsys_wrapper_flushes_profile_range_before_success_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, events = _load_nsys_wrapper_test_seam(tmp_path, monkeypatch)

    scope["_run_profile"]()

    assert events == [
        "setup",
        "benchmark",
        "synchronize",
        "profiler_start",
        "nvtx_enter",
        "benchmark",
        "nvtx_exit",
        "synchronize",
        "profiler_stop",
        "exit:0",
    ]


def test_nsys_wrapper_stops_capture_without_masking_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scope, events = _load_nsys_wrapper_test_seam(
        tmp_path,
        monkeypatch,
        fail_profile_call=True,
        stop_status=9,
    )

    with pytest.raises(ValueError, match="primary profile failure"):
        scope["_run_profile"]()

    assert "profiler_stop" in events
    assert "exit:0" not in events
    assert "cudaProfilerStop() failed with status 9" in capsys.readouterr().err


@pytest.mark.parametrize("start_status, stop_status, expected", [(7, 0, "Start"), (0, 8, "Stop")])
def test_nsys_wrapper_fails_for_cuda_profiler_api_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_status: int,
    stop_status: int,
    expected: str,
) -> None:
    scope, events = _load_nsys_wrapper_test_seam(
        tmp_path,
        monkeypatch,
        start_status=start_status,
        stop_status=stop_status,
    )

    with pytest.raises(RuntimeError, match=rf"cudaProfiler{expected}\(\) failed"):
        scope["_run_profile"]()

    assert "exit:0" not in events


def test_nsys_capture_command_uses_cuda_profiler_api_range_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    class _Process:
        returncode = 0

        def __init__(self, command, **_kwargs):
            command = list(command)
            commands.append(command)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("report", encoding="utf-8")

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(NsightAutomation, "_check_command", lambda *_args: True)
    monkeypatch.setattr("core.profiling.nsight_automation.subprocess.Popen", _Process)
    automation = NsightAutomation(tmp_path)

    assert automation.profile_nsys(
        command=[sys.executable, "workload.py"],
        output_name="ranged",
        capture_range_cuda_profiler_api=True,
        cuda_graph_trace="node",
    ) is not None
    assert "--capture-range=cudaProfilerApi" in commands[-1]
    assert "--capture-range-end=stop" in commands[-1]
    assert "--cuda-graph-trace=node" in commands[-1]

    assert automation.profile_nsys(
        command=[sys.executable, "workload.py"],
        output_name="unranged",
    ) is not None
    assert "--capture-range=cudaProfilerApi" not in commands[-1]
    assert "--capture-range-end=stop" not in commands[-1]
    assert "--cuda-graph-trace=node" not in commands[-1]


@pytest.mark.parametrize(
    "preset, first_error",
    [
        ("light", "Error in sitecustomize: incompatible startup module"),
        ("full", "unsupported full timeline category"),
    ],
)
def test_nsys_capture_retries_preserve_api_and_graph_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    first_error: str,
) -> None:
    commands: list[list[str]] = []

    class _Process:
        def __init__(self, command, **_kwargs):
            command = list(command)
            commands.append(command)
            self.returncode = 7 if len(commands) == 1 else 0
            if self.returncode == 0:
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("report", encoding="utf-8")

        def communicate(self, timeout=None):
            return "", first_error if self.returncode else ""

    monkeypatch.setattr(NsightAutomation, "_check_command", lambda *_args: True)
    monkeypatch.setattr("core.profiling.nsight_automation.subprocess.Popen", _Process)
    automation = NsightAutomation(tmp_path)
    monkeypatch.setattr(
        automation,
        "_wait_for_output_artifact",
        lambda output_path, **_kwargs: output_path.exists(),
    )

    report = automation.profile_nsys(
        command=[sys.executable, "workload.py"],
        output_name=f"retry_{preset}",
        preset=preset,
        capture_range_cuda_profiler_api=True,
        cuda_graph_trace="node",
    )

    assert report is not None
    assert len(commands) == 2
    for command in commands:
        assert "--capture-range=cudaProfilerApi" in command
        assert "--capture-range-end=stop" in command
        assert "--cuda-graph-trace=node" in command


def test_nsys_kernel_report_validation_rejects_success_shaped_empty_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_text("report", encoding="utf-8")
    empty = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "Processing [capture.sqlite] with [cuda_gpu_kern_sum.py]...\n"
            "SKIPPED: capture.sqlite does not contain CUDA kernel data.\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(run_benchmarks.subprocess, "run", lambda *_args, **_kwargs: empty)

    valid, detail = run_benchmarks._validate_nsys_cuda_kernel_report(report)

    assert valid is False
    assert detail is not None and "contains no CUDA kernel data" in detail


def test_nsys_kernel_report_validation_accepts_real_kernel_summary_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_text("report", encoding="utf-8")
    captured = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "Processing [capture.sqlite] with [cuda_gpu_kern_sum.py]...\n"
            "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
            "100.0,35872,1,35872.0,35872.0,35872,35872,0.0,real_cuda_kernel\n"
        ),
        stderr="",
    )
    nvtx = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "Processing [capture.sqlite] with [nvtx_sum.py]...\n"
            "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Style,Range\n"
            "100.0,35872,1,35872.0,35872.0,35872,35872,0.0,PushPop,:compute_kernel:profile\n"
        ),
        stderr="",
    )
    calls = []

    def _fake_stats(args, **_kwargs):
        calls.append(list(args))
        return captured if len(calls) == 1 else nvtx

    stale_sqlite = report.with_suffix(".sqlite")
    stale_sqlite.write_text("historical", encoding="utf-8")
    monkeypatch.setattr(run_benchmarks.subprocess, "run", _fake_stats)

    valid, detail = run_benchmarks._validate_nsys_cuda_kernel_report(
        report,
        expected_nvtx_label="compute_kernel:profile",
    )

    assert valid is True
    assert detail is None
    assert len(calls) == 2
    validation_sqlite = Path(calls[0][calls[0].index("--sqlite") + 1])
    assert validation_sqlite != stale_sqlite
    assert calls[1][calls[1].index("--sqlite") + 1] == str(validation_sqlite)
    assert "--force-export=true" in calls[0]
    assert "--force-export=true" not in calls[1]
    assert stale_sqlite.read_text(encoding="utf-8") == "historical"


def test_nsys_kernel_report_validation_rejects_wrong_profile_nvtx_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_text("report", encoding="utf-8")
    kernel = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
            "100.0,35872,1,35872.0,real_cuda_kernel\n"
        ),
        stderr="",
    )
    wrong_nvtx = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "Time (%),Total Time (ns),Instances,Style,Range\n"
            "100.0,35872,1,PushPop,:setup_only\n"
        ),
        stderr="",
    )
    responses = iter((kernel, wrong_nvtx))
    monkeypatch.setattr(
        run_benchmarks.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )

    valid, detail = run_benchmarks._validate_nsys_cuda_kernel_report(
        report,
        expected_nvtx_label="compute_kernel:profile",
    )

    assert valid is False
    assert detail is not None and "expected profiled workload NVTX range" in detail


def test_ncu_wrapper_remains_independent_of_cuda_profiler_api(tmp_path: Path) -> None:
    wrapper = render_ncu_python_profile_wrapper(
        benchmark_path=tmp_path / "benchmark.py",
        configured_nvtx_includes=None,
        target_label=None,
        target_override_argv=None,
        profile_type="minimal",
        ncu_metric_set="minimal",
        pm_sampling_interval=None,
        ncu_replay_mode="application",
        validity_profile="strict",
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
        profile_nvtx_label="compute_kernel:profile",
    )

    assert "cudaProfilerStart" not in wrapper
    assert "cudaProfilerStop" not in wrapper


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
            self.kwargs = []
            self.last_error = None

        def profile_nsys(self, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
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
        patch.object(run_benchmarks, "_validate_nsys_cuda_kernel_report", return_value=(True, None)),
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
    assert all(call["capture_range_cuda_profiler_api"] is False for call in automation.kwargs)
    assert all(call["cuda_graph_trace"] is None for call in automation.kwargs)


def test_profile_python_benchmark_retries_direct_python_wrapper_once(tmp_path: Path) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"
    class _Automation:
        def __init__(self, _output_dir):
            self.calls = 0
            self.kwargs = []
            self.last_error = None

        def profile_nsys(self, **kwargs):
            self.calls += 1
            self.kwargs.append(kwargs)
            report_path = output_dir / f"{kwargs['output_name']}.nsys-rep"
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
        patch.object(run_benchmarks, "_validate_nsys_cuda_kernel_report", return_value=(True, None)),
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

    assert report is not None
    assert report.name.endswith("_attempt2.nsys-rep")
    assert automation.calls == 2
    assert all(call["capture_range_cuda_profiler_api"] is True for call in automation.kwargs)
    assert all(call["cuda_graph_trace"] == "node" for call in automation.kwargs)
    assert automation.kwargs[0]["output_name"] != automation.kwargs[1]["output_name"]
    assert not (output_dir / "demo__baseline.nsys-rep").exists()


def test_profile_python_benchmark_rejects_empty_wrapper_capture(tmp_path: Path) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"
    class _Automation:
        last_error = None

        def profile_nsys(self, **kwargs):
            assert kwargs["capture_range_cuda_profiler_api"] is True
            assert kwargs["cuda_graph_trace"] == "node"
            report_path = output_dir / f"{kwargs['output_name']}.nsys-rep"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("report", encoding="utf-8")
            return report_path

    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", return_value=_Automation()),
        patch.object(
            run_benchmarks,
            "_validate_nsys_cuda_kernel_report",
            return_value=(False, "Nsight Systems report contains no CUDA kernel data"),
        ),
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

    assert report is None
    assert run_benchmarks._get_profile_failure_detail("nsys") == (
        "Nsight Systems report contains no CUDA kernel data"
    )


def test_profile_python_benchmark_clean_helper_preserves_api_capture_contract(
    tmp_path: Path,
) -> None:
    bench_path = tmp_path / "demo.py"
    bench_path.write_text("print('ok')\n", encoding="utf-8")
    output_dir = tmp_path / "profiles"

    class _Automation:
        def __init__(self):
            self.calls = 0
            self.last_error = None

        def profile_nsys(self, **kwargs):
            self.calls += 1
            self.last_error = (
                "Nsight Systems exited successfully but no report artifact was produced "
                f"at {output_dir / (kwargs['output_name'] + '.nsys-rep')}"
            )
            return None

    def _fake_helper_run(args, **_kwargs):
        payload_path = Path(args[args.index("--payload") + 1])
        result_path = Path(args[args.index("--result") + 1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["capture_range_cuda_profiler_api"] is True
        assert payload["cuda_graph_trace"] == "node"
        report_path = output_dir / f"{payload['output_name']}.nsys-rep"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("report", encoding="utf-8")
        result_path.write_text(
            json.dumps({"report": str(report_path), "last_error": None}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    automation = _Automation()
    with (
        patch.object(run_benchmarks, "check_nsys_available", return_value=True),
        patch("core.profiling.nsight_automation.NsightAutomation", return_value=automation),
        patch.object(run_benchmarks, "_validate_nsys_cuda_kernel_report", return_value=(True, None)),
        patch.object(run_benchmarks.subprocess, "run", side_effect=_fake_helper_run),
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

    assert report is not None
    assert report.name.endswith("_attempt3.nsys-rep")
    assert automation.calls == 2


def test_nsys_capture_helper_forwards_api_and_graph_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.profiling import nsys_capture_helper

    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "profiles"
    payload_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "output_name": "helper_capture",
                "command": [sys.executable, "workload.py"],
                "capture_range_cuda_profiler_api": True,
                "cuda_graph_trace": "node",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class _Automation:
        last_error = None

        def __init__(self, actual_output_dir):
            assert actual_output_dir == output_dir

        def profile_nsys(self, **kwargs):
            captured.update(kwargs)
            report = output_dir / "helper_capture.nsys-rep"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("report", encoding="utf-8")
            return report

    monkeypatch.setattr(nsys_capture_helper, "NsightAutomation", _Automation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nsys_capture_helper",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
        ],
    )

    assert nsys_capture_helper.main() == 0
    assert captured["capture_range_cuda_profiler_api"] is True
    assert captured["cuda_graph_trace"] == "node"
    assert json.loads(result_path.read_text(encoding="utf-8"))["report"].endswith(
        "helper_capture.nsys-rep"
    )


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
        patch.object(run_benchmarks, "_validate_nsys_cuda_kernel_report", return_value=(True, None)),
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
        nvtx_includes = ["compute_kernel:fixture"]

        @staticmethod
        def get_ncu_command_for_target(
            output_prefix: str,
            target: list[str],
            metrics: list[str] | None = None,
            nvtx_includes: list[str] | None = None,
        ) -> list[str]:
            assert nvtx_includes == ["compute_kernel:fixture"]
            return [
                "ncu",
                "--set",
                "basic",
                "--nvtx",
                "--nvtx-include",
                nvtx_includes[0],
                "-o",
                output_prefix,
                *target,
            ]

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
    command = run_mock.call_args.kwargs["command"]
    assert command[:4] == ["ncu", "--force-overwrite", "--set", "basic"]
    assert command[command.index("--nvtx-include") + 1] == "compute_kernel:fixture"
    assert command[-1] == str(executable)
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
