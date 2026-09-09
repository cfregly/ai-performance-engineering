"""Build/launch control-plane tests; real NCU capture is validated on B200."""
from pathlib import Path
import subprocess

import pytest

from core.benchmark import cuda_binary_benchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness import run_benchmarks
from core.harness.benchmark_harness import BenchmarkConfig


def _benchmark(tmp_path: Path) -> CudaBinaryBenchmark:
    benchmark = object.__new__(CudaBinaryBenchmark)
    benchmark.chapter_dir = tmp_path
    benchmark.binary_name = "example"
    benchmark.run_args = ["--iters", "10", "--input", "path with spaces"]
    benchmark.exec_path = tmp_path / "example_sm100"
    benchmark._verify_exec_path = None
    benchmark._profile_exec_path = None
    benchmark.require_tma_instructions = False
    return benchmark


@pytest.mark.parametrize("selection", [None, "", " "])
def test_child_profile_requires_a_range_before_build(tmp_path, monkeypatch, selection):
    benchmark = _benchmark(tmp_path)
    benchmark.ncu_profile_nvtx_include = selection
    monkeypatch.setattr(benchmark, "_build_binary", lambda **kw: pytest.fail("must reject before build"))
    with pytest.raises(RuntimeError, match="explicit ncu_profile_nvtx_include"):
        benchmark.get_ncu_profile_command()


def test_profile_build_preserves_the_ordinary_timing_binary(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path)
    benchmark.ncu_profile_nvtx_include = "compute_kernel:profile"
    original = benchmark.exec_path
    original.write_text("ordinary timing build")
    seen = []
    monkeypatch.setattr(cuda_binary_benchmark, "detect_supported_arch", lambda: "sm_100")

    def build(argv, **kwargs):
        seen.append(argv)
        # Build seam only: this fixture file is never executed or treated as GPU evidence.
        (tmp_path / argv[2]).write_text("profile build control fixture")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(cuda_binary_benchmark, "_run_subprocess_capture", build)
    command, selection = benchmark.get_ncu_profile_command()
    assert seen == [["make", "ARCH=sm_100", "example_profile_sm100", "NVTX_ENABLED=1"]]
    assert command == [str(tmp_path / "example_profile_sm100"), *benchmark.run_args]
    assert selection == "compute_kernel:profile"
    assert benchmark.exec_path == original
    assert original.read_text() == "ordinary timing build"
    assert benchmark._profile_exec_path == tmp_path / "example_profile_sm100"


def test_profile_build_failure_is_not_hidden(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path)
    benchmark.ncu_profile_nvtx_include = "compute_kernel:profile"
    monkeypatch.setattr(cuda_binary_benchmark, "detect_supported_arch", lambda: "sm_100")
    monkeypatch.setattr(cuda_binary_benchmark, "_run_subprocess_capture", lambda argv, **kw: subprocess.CompletedProcess(argv, 2, "", "compiler diagnostic"))
    with pytest.raises(RuntimeError, match="compiler diagnostic"):
        benchmark.get_ncu_profile_command()
    assert benchmark._profile_exec_path is None


def test_ncu_targets_the_marked_child_without_creating_a_python_wrapper(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path)
    target = [str(tmp_path / "example_profile_sm100"), *benchmark.run_args]
    monkeypatch.setattr(benchmark, "get_ncu_profile_command", lambda: (target, "compute_kernel:profile"))
    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(run_benchmarks, "render_ncu_python_profile_wrapper", lambda **kw: pytest.fail("compiled child must be launched directly"))
    seen = []

    class LaunchObserved(BaseException):
        pass

    def capture_launch(**kwargs):
        seen.append(kwargs["command"])
        # Stop before launching NCU; this test asserts no successful GPU capture.
        raise LaunchObserved

    monkeypatch.setattr(run_benchmarks, "_run_profile_subprocess", capture_launch)
    with pytest.raises(LaunchObserved):
        run_benchmarks.profile_python_benchmark_ncu(
            benchmark, tmp_path / "baseline_example.py", tmp_path, tmp_path / "reports",
            BenchmarkConfig(validity_profile="portable", lock_gpu_clocks=False),
        )
    assert len(seen) == 1
    command = seen[0]
    assert command[-len(target):] == target
    assert command[command.index("--nvtx-include") + 1] == "compute_kernel:profile"
    assert "--launch-count" not in command
    assert "--kernel-name" not in command


def test_torch_does_not_emit_a_parent_only_trace_for_a_compiled_child(tmp_path, monkeypatch):
    benchmark = _benchmark(tmp_path)
    monkeypatch.setattr(run_benchmarks, "TORCH_PROFILER_AVAILABLE", True)
    monkeypatch.setattr(run_benchmarks, "render_torch_python_profile_wrapper", lambda **kw: pytest.fail("parent-only trace must not be launched"))
    output = tmp_path / "reports"
    assert run_benchmarks.profile_python_benchmark_torch(
        benchmark, tmp_path / "baseline_example.py", tmp_path, output,
    ) is None
    assert "compiled child process" in run_benchmarks._get_profile_failure_detail("torch")
    assert not output.exists()


def test_python_benchmark_keeps_torch_profiler_eligibility():
    assert run_benchmarks._torch_profiler_not_applicable_reason(object()) is None
