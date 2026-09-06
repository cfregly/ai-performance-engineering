"""Focused regressions for benchmark evidence and worker lifecycle handling."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from core.benchmark.verify_runner import VerifyRunner
from core.benchmark.cuda_binary_benchmark import _run_subprocess_capture
from core.harness import benchmark_harness as harness_module
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness


class _CpuTransportBenchmark(BaseBenchmark):
    """Small real benchmark used to exercise the isolated-runner protocol."""

    allow_cpu = True

    def setup(self) -> None:
        self.device = torch.device("cpu")
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = self.input * 2.0

    def get_verify_inputs(self):
        return {"input": self.input}

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("CPU work did not execute")
        return self.output

    def get_input_signature(self):
        return {"shape": (8,), "dtype": "torch.float32"}

    def get_output_tolerance(self):
        return (0.0, 0.0)

    def validate_result(self):
        torch.testing.assert_close(self.output, self.input * 2.0)


class _TermResistantDescendantBenchmark(_CpuTransportBenchmark):
    """Spawn a same-group descendant that ignores SIGTERM, then block."""

    pid_path = ""

    def benchmark_fn(self) -> None:
        child_code = (
            "import os,pathlib,signal,sys,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8');"
            "time.sleep(60)"
        )
        subprocess.Popen(
            [sys.executable, "-c", child_code, self.pid_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(60)


class _InvalidReportedTimingBenchmark(_CpuTransportBenchmark):
    """Exercise the production reported-time path with a corrupt observation."""

    use_reported_time = True

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        self.last_time_ms = math.nan


def _cpu_harness(**overrides) -> tuple[BenchmarkHarness, BenchmarkConfig]:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        enforce_environment_validation=False,
        **overrides,
    )
    harness = BenchmarkHarness(config=config)
    harness._ensure_runtime_initialized()
    return harness, config


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_cpu_memory_tracking_does_not_query_cuda_allocator(monkeypatch) -> None:
    harness, config = _cpu_harness()
    config.enable_memory_tracking = True

    def unexpected_cuda_call(*args, **kwargs):
        raise AssertionError("CPU memory tracking must not query the CUDA allocator")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda_call)
    with harness._memory_tracking(config) as memory:
        output = torch.arange(8, dtype=torch.float32).square()
    assert memory is None
    torch.testing.assert_close(output, torch.tensor([0., 1., 4., 9., 16., 25., 36., 49.]))


def _wait_for_pid_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def test_isolated_subprocess_removes_verify_output_tempfile(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_module.tempfile, "tempdir", str(tmp_path))
    harness, config = _cpu_harness(measurement_timeout_seconds=30)

    result = harness._benchmark_with_subprocess(_CpuTransportBenchmark(), config)

    assert not result.errors, result.errors
    assert list(tmp_path.glob("aisp_verify_output_*.pt")) == []


@pytest.mark.skipif(os.name != "posix", reason="process-group lifecycle requires POSIX")
def test_subprocess_timeout_reaps_term_resistant_descendant(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    benchmark = _TermResistantDescendantBenchmark()
    benchmark.pid_path = str(pid_path)
    harness, config = _cpu_harness(measurement_timeout_seconds=5)
    child_pid = None

    try:
        result = harness._benchmark_with_subprocess(benchmark, config)
        assert result.timeout_stage == "measurement"
        assert pid_path.exists(), "benchmark descendant did not start before timeout"
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        assert _wait_for_pid_exit(child_pid), f"timed-out descendant {child_pid} survived"
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None and _pid_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="process-group lifecycle requires POSIX")
def test_cuda_binary_capture_timeout_reaps_compiler_descendants(tmp_path):
    pid_path = tmp_path / "compiler-descendant.pid"
    child_code = (
        "import os,pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8');"
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);"
        "time.sleep(60)"
    )
    child_pid = None

    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_subprocess_capture(
                [sys.executable, "-c", parent_code, child_code, str(pid_path)],
                cwd=tmp_path,
                timeout=3,
            )
        assert pid_path.exists(), "compiler descendant did not start before timeout"
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        assert _wait_for_pid_exit(child_pid), f"timed-out compiler descendant {child_pid} survived"
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None and _pid_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize("invalid_sample", [math.nan, math.inf, -0.001])
def test_timing_statistics_reject_invalid_samples(invalid_sample):
    harness, config = _cpu_harness()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        harness._compute_stats([1.0, invalid_sample], config)


def test_full_benchmark_loop_does_not_credit_nan_reported_time():
    harness, _ = _cpu_harness()

    with pytest.raises(ValueError, match="finite and nonnegative"):
        harness.benchmark(_InvalidReportedTimingBenchmark())


def test_output_comparison_rejects_unbounded_tolerance():
    expected = torch.tensor([0.0])
    different = torch.tensor([1_000_000.0])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        VerifyRunner().compare_perf_outputs(
            expected,
            different,
            tolerance=(math.inf, math.inf),
        )
