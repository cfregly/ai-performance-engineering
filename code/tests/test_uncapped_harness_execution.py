"""Exercise uncapped ownership with real CPU work and real isolated children."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest
import torch

from core.harness import benchmark_harness as harness_module
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness
from core.utils.cooperative_subprocess import CooperativeProcessInterrupted


EXECUTION_FIELDS = (
    "setup_timeout_seconds", "warmup_timeout_seconds",
    "measurement_timeout_seconds", "timeout_seconds",
)
OWNED_GROUP_WAIT_SUPPORTED = os.name == "posix" and all(
    hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")
)
requires_owned_worker = pytest.mark.skipif(
    not OWNED_GROUP_WAIT_SUPPORTED,
    reason="owned isolated execution requires POSIX waitid with WNOWAIT",
)


class _ObservedCpuBenchmark(BaseBenchmark):
    """Compute actual tensor outputs and retain the configuration that executed."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.observation_path = ""
        self.cleanup_path = ""
        self.progress_path = ""
        self.interrupt_kind = ""
        self.interrupt_after_calls = 1
        self.interrupted = False
        self.failure_stage = ""
        self.setup_delay_seconds = 0.0
        self.calls = 0
        self.validated = False
        self.input = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        return _config(
            setup_timeout_seconds=13, warmup_timeout_seconds=14,
            measurement_timeout_seconds=15, timeout_seconds=16,
        )

    def setup(self) -> None:
        self.device = torch.device("cpu")
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None
        if self.observation_path:
            Path(self.observation_path).write_text(json.dumps({
                name: getattr(self._config, name)
                for name in EXECUTION_FIELDS + ("execution_timeout_seconds",)
            }))
        if self.setup_delay_seconds:
            time.sleep(self.setup_delay_seconds)
        if self.failure_stage == "setup":
            raise RuntimeError("Real CPU setup failure")
        if self.interrupt_kind == "setup":
            os.kill(os.getpid(), signal.SIGINT)

    def benchmark_fn(self) -> None:
        self.output = self.input.square()
        self.calls += 1
        if self.failure_stage == "warmup":
            raise RuntimeError("Real CPU warmup failure")
        if self.failure_stage == "measurement" and self.calls > self._config.warmup:
            raise TimeoutError("Real CPU measurement operation failed")
        if self.interrupt_kind and not self.interrupted and self.calls >= self.interrupt_after_calls:
            self.interrupted = True
            if self.interrupt_kind == "raised":
                raise KeyboardInterrupt("Real worker interruption")
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.05)

    def get_verify_inputs(self):
        return {"input": self.input}

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("CPU work has not executed")
        return self.output

    def get_input_signature(self) -> dict:
        return {"shape": (8,), "dtype": "torch.float32"}

    def get_output_tolerance(self):
        return (0.0, 0.0)

    def validate_result(self):
        torch.testing.assert_close(self.output, self.input.square(), rtol=0, atol=0)
        self.validated = True

    def teardown(self) -> None:
        if self.interrupt_kind:
            time.sleep(0.05)
        if self.cleanup_path:
            Path(self.cleanup_path).write_text("CPU benchmark teardown completed\n")
        if self.progress_path:
            Path(self.progress_path).write_text(json.dumps({
                "calls": self.calls, "validated": self.validated,
            }))


class _CoordinatorFailureCpuBenchmark(_ObservedCpuBenchmark):
    """Keep real setup active until the coordinator cancels its owned worker."""

    def __init__(self) -> None:
        super().__init__()
        self.ready_path = ""

    def setup(self) -> None:
        super().setup()
        # The isolated runner redirects Python prints, so exercise its native pipe.
        os.write(1, b"Real CPU worker stdout before coordinator failure\n")
        print("Real CPU worker stderr before coordinator failure", file=sys.stderr, flush=True)
        Path(self.ready_path).write_text("Real CPU setup is active\n")
        signal.pause()


def _config(**overrides) -> BenchmarkConfig:
    values = dict(
        device=torch.device("cpu"), iterations=1, warmup=5,
        enable_profiling=False, enable_nsys=False, enable_ncu=False,
        profile_type="none", enable_memory_tracking=False,
        enforce_environment_validation=False, adaptive_iterations=False,
        cross_validate_timing=False,
    )
    values.update(overrides)
    return BenchmarkConfig(**values)


@pytest.mark.parametrize("default_field", EXECUTION_FIELDS)
def test_inner_merge_retains_exact_operator_value_even_when_it_equals_a_default(tmp_path: Path, default_field: str) -> None:
    selected = getattr(BenchmarkConfig(), default_field)
    benchmark = _ObservedCpuBenchmark()
    benchmark.observation_path = str(tmp_path / "executed-config.json")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(execution_timeout_seconds=selected, use_subprocess=False)
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert not result.errors, result.errors
    observed = json.loads(Path(benchmark.observation_path).read_text())
    assert observed == {name: selected for name in EXECUTION_FIELDS + ("execution_timeout_seconds",)}
    assert Path(benchmark.cleanup_path).read_text() == "CPU benchmark teardown completed\n"


@requires_owned_worker
def test_uncapped_real_isolated_worker_completes_cpu_work_and_preserves_zero(tmp_path: Path) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.observation_path = str(tmp_path / "executed-config.json")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    result = BenchmarkHarness(config=_config(execution_timeout_seconds=0)).benchmark(benchmark)
    assert not result.errors, result.errors
    observed = json.loads(Path(benchmark.observation_path).read_text())
    assert observed == {name: 0 for name in EXECUTION_FIELDS + ("execution_timeout_seconds",)}
    assert Path(benchmark.cleanup_path).is_file()
    assert result.timing.iterations > 0


@pytest.mark.parametrize("overrides", [
    {"execution_timeout_seconds": 0},
    {"measurement_timeout_seconds": 0},
    {"measurement_timeout_seconds": None, "timeout_seconds": 0},
])
def test_direct_thread_method_rejects_zero_even_if_config_declares_subprocess(tmp_path: Path, overrides: dict) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.observation_path = str(tmp_path / "must-not-execute.json")
    config = _config(**overrides)
    with pytest.raises(ValueError, match="owned subprocess"):
        BenchmarkHarness(config=config)._benchmark_with_threading(benchmark, config)
    assert not Path(benchmark.observation_path).exists()


@requires_owned_worker
@pytest.mark.parametrize("kind", ["main", "raised"])
def test_owned_worker_interrupt_completes_real_teardown(tmp_path: Path, kind: str) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.interrupt_kind = kind
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert result.errors
    assert Path(benchmark.cleanup_path).read_text() == "CPU benchmark teardown completed\n"


@requires_owned_worker
@pytest.mark.parametrize("stage, expected_calls", [("setup", 0), ("warmup", 1), ("measurement", 6)])
def test_uncapped_interrupt_stops_before_more_work_or_validation(tmp_path: Path, stage: str, expected_calls: int) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.interrupt_kind = "setup" if stage == "setup" else "main"
    benchmark.interrupt_after_calls = expected_calls
    benchmark.progress_path = str(tmp_path / "progress.json")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(iterations=20, execution_timeout_seconds=0)
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert result.errors
    assert Path(benchmark.cleanup_path).is_file()
    assert json.loads(Path(benchmark.progress_path).read_text()) == {
        "calls": expected_calls, "validated": False,
    }


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
@pytest.mark.parametrize("stage", ["setup", "warmup", "measurement"])
@pytest.mark.parametrize("execution_timeout", [0, 30])
def test_real_worker_failure_keeps_cause_without_timeout_fields(tmp_path: Path, stage: str, execution_timeout: int) -> None:
    if execution_timeout == 0 and not OWNED_GROUP_WAIT_SUPPORTED:
        pytest.skip("owned isolated execution requires POSIX waitid with WNOWAIT")
    benchmark = _ObservedCpuBenchmark()
    benchmark.failure_stage = stage
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(
        execution_timeout_seconds=execution_timeout,
        subprocess_stderr_dir=str(tmp_path),
    )
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert any(f"Real CPU {stage}" in error for error in result.errors), result.errors
    assert not any("Executed child device" in error for error in result.errors)
    assert result.timeout_stage is None
    assert result.timeout_duration_seconds is None
    assert result.timeout_limit_seconds is None
    assert result.timing.iterations == 0
    assert Path(benchmark.cleanup_path).is_file()
    record = json.loads((tmp_path / "ObservedCpuBenchmark_subprocess.stdout.log").read_text())
    child = json.loads(record["result_json"])
    assert child["timeout_stage"] is None
    assert child["timeout_duration_seconds"] is None
    assert child["timeout_limit_seconds"] is None


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
def test_real_setup_deadline_keeps_exact_child_timeout_receipt(tmp_path: Path) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.setup_delay_seconds = 1.1
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(
        setup_timeout_seconds=1, measurement_timeout_seconds=30,
        subprocess_stderr_dir=str(tmp_path),
    )
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert result.errors
    assert Path(benchmark.cleanup_path).is_file()
    record = json.loads((tmp_path / "ObservedCpuBenchmark_subprocess.stdout.log").read_text())
    child = json.loads(record["result_json"])
    assert result.timeout_stage == child["timeout_stage"] == "setup"
    assert result.timeout_duration_seconds == child["timeout_duration_seconds"]
    assert result.timeout_duration_seconds > 1
    assert result.timeout_limit_seconds == child["timeout_limit_seconds"] == 1
    assert result.validation_message == child["validation_message"]


@requires_owned_worker
def test_lifecycle_timeout_error_is_not_an_uncapped_execution_deadline(tmp_path: Path) -> None:
    module = tmp_path / "cpu_lifecycle_failure.py"
    module.write_text(
        "from contextlib import ExitStack\n"
        "from core.harness import benchmark_harness\n"
        "from tests.test_uncapped_harness_execution import _ObservedCpuBenchmark\n"
        "class FailingClose(ExitStack):\n"
        "    def close(self):\n"
        "        super().close()\n"
        "        raise TimeoutError('CPU lifecycle close failure')\n"
        "benchmark_harness.ExitStack = FailingClose\n"
        "def get_benchmark():\n"
        "    return _ObservedCpuBenchmark()\n"
    )
    benchmark = _ObservedCpuBenchmark()
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    benchmark._module_file_override = str(module)
    benchmark._factory_name_override = "get_benchmark"
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert result.errors
    assert Path(benchmark.cleanup_path).is_file()
    record = json.loads((tmp_path / "get_benchmark_subprocess.stdout.log").read_text())
    assert record["success"] is False
    failure = "\n".join(record["errors"])
    assert "TimeoutError: CPU lifecycle close failure" in failure
    assert "AssertionError" not in failure
    assert "TypeError" not in failure
    assert not any("Executed child device" in error for error in result.errors)
    assert json.loads(record["result_json"])["timeout_stage"] is None
    assert result.timeout_stage is None


@requires_owned_worker
@pytest.mark.parametrize("failure", [KeyboardInterrupt, BrokenPipeError])
def test_real_created_worker_uses_cooperative_followup_after_handoff_failure(tmp_path: Path, monkeypatch, failure) -> None:
    actual_communicate = harness_module.communicate_cooperatively
    calls = []

    def interrupted_handoff(process, **kwargs):
        calls.append((process, kwargs.copy()))
        if len(calls) == 1:
            raise failure("Before cooperative communication starts")
        return actual_communicate(process, **kwargs)

    def unexpected_forced_cleanup(*args, **kwargs):
        raise AssertionError("Uncapped handoff must not use forced termination")

    monkeypatch.setattr(harness_module, "communicate_cooperatively", interrupted_handoff)
    monkeypatch.setattr(harness_module, "_terminate_process_group", unexpected_forced_cleanup)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    harness = BenchmarkHarness(config=config)
    if failure is KeyboardInterrupt:
        with pytest.raises(CooperativeProcessInterrupted) as interrupted:
            harness._benchmark_with_subprocess(_ObservedCpuBenchmark(), config)
        assert interrupted.value.result["cancelled"] is True
        assert calls[1][1]["cancellation_requested"] is True
    else:
        result = harness._benchmark_with_subprocess(_ObservedCpuBenchmark(), config)
        assert any("stdin closed early" in error for error in result.errors)
        assert "input" not in calls[1][1]
    assert len(calls) == 2
    process = calls[0][0]
    assert process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(process.pid, 0)


@requires_owned_worker
@pytest.mark.parametrize("failure, incomplete_receipt", [
    (RuntimeError, False),
    (TimeoutError, False),
    pytest.param(RuntimeError, True, id="incomplete-receipt"),
])
def test_coordinator_failure_retains_original_cause_output_and_worker_cleanup(tmp_path: Path, monkeypatch, failure, incomplete_receipt: bool) -> None:
    benchmark = _CoordinatorFailureCpuBenchmark()
    benchmark.ready_path = str(tmp_path / "ready.txt")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    benchmark.progress_path = str(tmp_path / "progress.json")
    original_error = failure(f"Original {failure.__name__} during coordinator handoff")
    actual_communicate = harness_module.communicate_cooperatively
    calls = []

    def failed_handoff(process, **kwargs):
        calls.append((process, kwargs.copy()))
        if len(calls) == 1:
            process.stdin.write(kwargs["input"])
            process.stdin.close()
            while not Path(benchmark.ready_path).exists():
                status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
                if status is not None:
                    raise AssertionError("Real CPU worker exited before setup became active")
                time.sleep(0.01)
            raise original_error
        try:
            return actual_communicate(process, **kwargs)
        except BaseException as error:
            if incomplete_receipt:
                # Change only the receipt to exercise the parent's diagnostic.
                execution = error.cooperative_process_result["execution"]
                execution["completion_scope"] = "cleanup_incomplete"
                execution["capture_errors"] = ["Injected observation failure"]
            raise

    def unexpected_forced_cleanup(*args, **kwargs):
        raise AssertionError("A coordinator error must preserve cooperative worker cleanup")

    monkeypatch.setattr(harness_module, "communicate_cooperatively", failed_handoff)
    monkeypatch.setattr(harness_module, "_terminate_process_group", unexpected_forced_cleanup)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config)._benchmark_with_subprocess(benchmark, config)

    assert len(calls) == 2
    assert calls[1][1]["abort_error"] is original_error
    assert any(str(original_error) in error for error in result.errors), result.errors
    assert result.timeout_stage is None
    assert result.timeout_duration_seconds is None
    assert result.timeout_limit_seconds is None
    receipt = original_error.cooperative_process_result
    assert receipt["cancelled"] is True
    assert receipt["cancellation"]["reason"] == "coordinator_error"
    assert receipt["execution"]["live_process_group_members"] == []
    diagnostic = next((error for error in result.errors if "cleanup is incomplete" in error), None)
    if incomplete_receipt:
        assert diagnostic is not None
        assert str(calls[0][0].pid) in diagnostic
        assert "Injected observation failure" in diagnostic
    else:
        assert diagnostic is None
    assert "Real CPU worker stdout before coordinator failure" in receipt["stdout"]
    assert "Real CPU worker stderr before coordinator failure" in receipt["stderr"]
    for stream in ("stdout", "stderr"):
        saved = tmp_path / f"CoordinatorFailureCpuBenchmark_subprocess.{stream}.log"
        assert saved.read_text() == receipt[stream]
    assert Path(benchmark.cleanup_path).read_text() == "CPU benchmark teardown completed\n"
    assert json.loads(Path(benchmark.progress_path).read_text()) == {"calls": 0, "validated": False}
    process = calls[0][0]
    assert process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(process.pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
