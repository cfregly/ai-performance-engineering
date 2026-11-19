"""Exercise uncapped ownership with real CPU work and real isolated children."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from core.harness import benchmark_harness as harness_module
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness
from core.utils import cooperative_subprocess as cooperative_owner
from core.utils.cooperative_subprocess import CooperativeCleanupIncomplete, CooperativeProcessInterrupted
from core.utils.cooperative_subprocess import parse_cleanup_incomplete_event
from core.utils.execution_timeouts import MAX_EXECUTION_TIMEOUT_SECONDS, MAX_SUBPROCESS_WAIT_SECONDS


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
        self.config_overrides = {}
        self.input = None
        self.output = None

    def get_config(self) -> BenchmarkConfig:
        values = dict(
            setup_timeout_seconds=13, warmup_timeout_seconds=14,
            measurement_timeout_seconds=15, timeout_seconds=16,
        )
        values.update(self.config_overrides)
        return _config(**values)

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


class _CoordinatorSignalCpuBenchmark(_ObservedCpuBenchmark):
    """Expose real measurement and teardown boundaries to the coordinator."""

    def __init__(self) -> None:
        super().__init__()
        self.ready_path = ""
        self.pause_during = "measurement"
        self.cleanup_ready_path = ""
        self.cleanup_release_path = ""

    def _wait_for_coordinator(self) -> None:
        os.write(1, b"Real CPU worker waiting for coordinator SIGINT\n")
        print("Real CPU worker stderr before coordinator SIGINT", file=sys.stderr, flush=True)
        Path(self.ready_path).write_text("Real CPU work is active\n")
        signal.pause()

    def setup(self) -> None:
        super().setup()
        if self.pause_during == "setup":
            self._wait_for_coordinator()

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        if self.pause_during == "measurement" and self.calls > self._config.warmup:
            self._wait_for_coordinator()

    def teardown(self) -> None:
        if self.cleanup_ready_path:
            Path(self.cleanup_ready_path).write_text("Real CPU teardown is active\n")
            while not Path(self.cleanup_release_path).exists():
                time.sleep(0.01)
        super().teardown()


class _NestedIncompleteCpuBenchmark(_ObservedCpuBenchmark):
    """Run real inner CPU work and reject an unavailable cleanup observation."""

    def __init__(self) -> None:
        super().__init__()
        self.inner_prefix = ""
        self.interrupt_inner = False

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        from core.utils import cooperative_subprocess
        from core.utils.cooperative_process_artifacts import run_recorded_cooperative_process

        actual_group_query = cooperative_subprocess._live_process_group_members
        interrupted = False

        def unavailable_group_query(group):
            nonlocal interrupted
            members = actual_group_query(group)
            if not members:
                if self.interrupt_inner and not interrupted:
                    interrupted = True
                    os.kill(os.getpid(), signal.SIGINT)
                raise PermissionError("Nested CPU cleanup observation unavailable")
            return members

        cooperative_subprocess._live_process_group_members = unavailable_group_query
        try:
            run_recorded_cooperative_process(
                [sys.executable, "-u", "-c", "import sys\nprint('nested CPU stdout')\nprint('nested CPU stderr', file=sys.stderr)\n"],
                output_prefix=Path(self.inner_prefix),
            )
        finally:
            cooperative_subprocess._live_process_group_members = actual_group_query


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


@pytest.mark.parametrize("overrides, benchmark_overrides", [
    ({"measurement_timeout_seconds": 0}, {}),
    ({"measurement_timeout_seconds": None, "timeout_seconds": 0}, {}),
    ({"timeout_multiplier": 0.0001}, {}),
    ({}, {"measurement_timeout_seconds": 0}),
    ({}, {"execution_timeout_seconds": 0}),
])
def test_implicit_uncapped_execution_is_rejected_before_child_creation(tmp_path: Path, monkeypatch, overrides: dict, benchmark_overrides: dict) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.config_overrides = benchmark_overrides
    benchmark.observation_path = str(tmp_path / "must-not-execute.json")
    config = _config(**overrides)

    def unexpected_child(*args, **kwargs):
        raise AssertionError("Implicit uncapped execution must fail before child creation")

    monkeypatch.setattr(harness_module.subprocess, "Popen", unexpected_child)
    with pytest.raises(ValueError, match="requires explicit execution_timeout_seconds=0"):
        BenchmarkHarness(config=config).benchmark(benchmark)
    assert not Path(benchmark.observation_path).exists()


@pytest.mark.parametrize("measurement", [MAX_SUBPROCESS_WAIT_SECONDS + 1, MAX_EXECUTION_TIMEOUT_SECONDS + 1])
def test_oversized_resolved_measurement_is_rejected_before_child_creation(tmp_path: Path, monkeypatch, measurement) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.observation_path = str(tmp_path / "must-not-execute.json")
    config = _config(measurement_timeout_seconds=measurement)
    assert config.execution_timeout_seconds is None

    def unexpected_child(*args, **kwargs):
        raise AssertionError("An unsupported resolved wait must fail before child creation")

    monkeypatch.setattr(harness_module.subprocess, "Popen", unexpected_child)
    with pytest.raises(ValueError, match="supported wait"):
        BenchmarkHarness(config=config).benchmark(benchmark)
    assert not Path(benchmark.observation_path).exists()


def test_oversized_direct_thread_wait_is_rejected_before_benchmark_work(tmp_path: Path) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.observation_path = str(tmp_path / "must-not-execute.json")
    config = _config(use_subprocess=False, measurement_timeout_seconds=10 ** 100)
    with pytest.raises(ValueError, match="Thread timeout.*supported wait"):
        BenchmarkHarness(config=config)._benchmark_with_threading(benchmark, config)
    assert not Path(benchmark.observation_path).exists()


@requires_owned_worker
def test_isolated_worker_rejects_oversized_thread_wait_before_import(
    tmp_path: Path, monkeypatch,
) -> None:
    module = tmp_path / "must_not_import.py"
    imported = tmp_path / "imported.txt"
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('Imported unexpectedly')\n"
    )
    benchmark = _ObservedCpuBenchmark()
    benchmark._module_file_override = str(module)
    benchmark._factory_name_override = "get_benchmark"
    actual_communicate = cooperative_owner.communicate_cooperatively

    def oversized_wait(process, **kwargs):
        payload = json.loads(kwargs["input"])
        payload["config_dict"]["measurement_timeout_seconds"] = 10 ** 100
        kwargs["input"] = json.dumps(payload)
        return actual_communicate(process, **kwargs)

    monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", oversized_wait)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert any("Thread timeout" in error for error in result.errors), result.errors
    assert not imported.exists()


@requires_owned_worker
@pytest.mark.parametrize("selector", [None, False, 0.0, -0.0])
@pytest.mark.parametrize("deadline", ["measurement", "legacy"])
def test_isolated_worker_rejects_implicit_zero_before_import(tmp_path: Path, monkeypatch, selector, deadline: str) -> None:
    module = tmp_path / "must_not_import.py"
    imported = tmp_path / "imported.txt"
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('Imported unexpectedly')\n"
    )
    benchmark = _ObservedCpuBenchmark()
    benchmark._module_file_override = str(module)
    benchmark._factory_name_override = "get_benchmark"
    actual_communicate = cooperative_owner.communicate_cooperatively

    def implicit_zero(process, **kwargs):
        payload = json.loads(kwargs["input"])
        payload["config_dict"]["execution_timeout_seconds"] = selector
        if deadline == "legacy":
            payload["config_dict"].pop("measurement_timeout_seconds", None)
            payload["config_dict"]["timeout_seconds"] = 0
        kwargs["input"] = json.dumps(payload)
        return actual_communicate(process, **kwargs)

    monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", implicit_zero)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    expected = (
        "requires explicit execution_timeout_seconds=0"
        if selector is None
        else "execution_timeout_seconds must be a nonnegative integer or None"
    )
    assert any(expected in error for error in result.errors), result.errors
    assert not any("Executed child device" in error for error in result.errors)
    assert not imported.exists()
    assert result.timeout_stage is None


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
    actual_communicate = cooperative_owner.communicate_cooperatively
    calls = []

    def interrupted_handoff(process, **kwargs):
        calls.append((process, kwargs.copy()))
        if len(calls) == 1:
            raise failure("Before cooperative communication starts")
        return actual_communicate(process, **kwargs)

    def unexpected_forced_cleanup(*args, **kwargs):
        raise AssertionError("Uncapped handoff must not use forced termination")

    monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", interrupted_handoff)
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
        assert any(
            "Before cooperative communication starts" in error
            for error in result.errors
        )
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
    actual_communicate = cooperative_owner.communicate_cooperatively
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
                execution["live_process_group_members"] = None
                execution["capture_errors"] = ["Injected observation failure"]
            raise

    def unexpected_forced_cleanup(*args, **kwargs):
        raise AssertionError("A coordinator error must preserve cooperative worker cleanup")

    monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", failed_handoff)
    monkeypatch.setattr(harness_module, "_terminate_process_group", unexpected_forced_cleanup)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    next_work = tmp_path / "next-work.txt"
    if incomplete_receipt:
        with pytest.raises(CooperativeCleanupIncomplete) as stopped:
            BenchmarkHarness(config=config).benchmark(benchmark)
            next_work.write_text("The sweep incorrectly continued\n")
        assert stopped.value.__cause__ is original_error
        assert not isinstance(stopped.value, Exception)
        assert not next_work.exists()
        assert stopped.value.result is original_error.cooperative_process_result
    else:
        result = BenchmarkHarness(config=config).benchmark(benchmark)

    assert len(calls) == 2
    assert calls[1][1]["abort_error"] is original_error
    if not incomplete_receipt:
        assert any(str(original_error) in error for error in result.errors), result.errors
        assert result.timeout_stage is None
        assert result.timeout_duration_seconds is None
        assert result.timeout_limit_seconds is None
    receipt = original_error.cooperative_process_result
    assert receipt["cancelled"] is True
    assert receipt["cancellation"]["reason"] == "coordinator_error"
    if incomplete_receipt:
        assert receipt["execution"]["live_process_group_members"] is None
        assert str(calls[0][0].pid) in str(stopped.value)
        assert "Injected observation failure" in receipt["execution"]["capture_errors"]
    else:
        assert receipt["execution"]["live_process_group_members"] == []
        diagnostic = next((error for error in result.errors if "cleanup is incomplete" in error), None)
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


@requires_owned_worker
@pytest.mark.parametrize("stage", ["measurement", "abort_wait"])
def test_real_coordinator_sigint_aborts_sweep_after_worker_cleanup(tmp_path: Path, monkeypatch, stage: str) -> None:
    benchmark = _CoordinatorSignalCpuBenchmark()
    benchmark.ready_path = str(tmp_path / "ready.txt")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    benchmark.progress_path = str(tmp_path / "progress.json")
    next_work = tmp_path / "next-work.txt"
    finished = threading.Event()
    interrupt_seen = threading.Event()
    original_error = RuntimeError("Real coordinator failure before operator SIGINT")
    if stage == "abort_wait":
        benchmark.pause_during = "setup"
        benchmark.cleanup_ready_path = str(tmp_path / "cleanup-ready.txt")
        benchmark.cleanup_release_path = str(tmp_path / "cleanup-release.txt")
        actual_communicate = cooperative_owner.communicate_cooperatively

        def failed_handoff(process, **kwargs):
            if "abort_error" not in kwargs:
                process.stdin.write(kwargs["input"])
                process.stdin.close()
                while not Path(benchmark.ready_path).exists():
                    status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
                    if status is not None:
                        raise AssertionError("Real CPU worker exited before setup became active")
                    time.sleep(0.01)
                raise original_error
            return actual_communicate(process, **kwargs)

        monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", failed_handoff)

    def receive_operator_interrupt(signum, frame):
        interrupt_seen.set()
        raise KeyboardInterrupt("Actual coordinator SIGINT")

    def interrupt_coordinator():
        ready = Path(benchmark.cleanup_ready_path or benchmark.ready_path)
        while not ready.exists():
            if finished.wait(0.01):
                return
        os.kill(os.getpid(), signal.SIGINT)
        if stage == "abort_wait":
            interrupt_seen.wait()
            Path(benchmark.cleanup_release_path).write_text("Operator interrupt observed\n")

    previous_handler = signal.signal(signal.SIGINT, receive_operator_interrupt)
    sender = threading.Thread(target=interrupt_coordinator)
    sender.start()
    try:
        config = _config(iterations=20, execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
        with pytest.raises(KeyboardInterrupt) as interrupted:
            BenchmarkHarness(config=config).benchmark(benchmark)
            next_work.write_text("The sweep incorrectly continued\n")
    finally:
        finished.set()
        if stage == "abort_wait":
            Path(benchmark.cleanup_release_path).write_text("Release test-owned teardown\n")
        sender.join()
        signal.signal(signal.SIGINT, previous_handler)

    assert interrupt_seen.is_set()
    assert not next_work.exists()
    assert Path(benchmark.cleanup_path).read_text() == "CPU benchmark teardown completed\n"
    assert json.loads(Path(benchmark.progress_path).read_text()) == {
        "calls": 6 if stage == "measurement" else 0, "validated": False,
    }
    receipt = getattr(interrupted.value, "result", None)
    assert receipt is not None
    assert receipt["cancelled"] is True
    assert receipt["execution"]["live_process_group_members"] == []
    assert receipt["cancellation"]["operator_interrupt_observed"] is True
    assert receipt["cancellation"]["interrupt_count"] == 1
    if stage == "abort_wait":
        assert receipt["cancellation"]["repeated_interrupts"] == 0
        assert interrupted.value.__cause__ is original_error
    for stream in ("stdout", "stderr"):
        saved = tmp_path / f"CoordinatorSignalCpuBenchmark_subprocess.{stream}.log"
        assert saved.read_text() == receipt[stream]
    with pytest.raises(ProcessLookupError):
        os.kill(receipt["execution"]["process_id"], 0)


@requires_owned_worker
@pytest.mark.parametrize("interrupt_inner", [False, True])
def test_nested_incomplete_cleanup_survives_worker_protocol_and_stops_next_work(tmp_path: Path, monkeypatch, interrupt_inner: bool) -> None:
    benchmark = _NestedIncompleteCpuBenchmark()
    benchmark.inner_prefix = str(tmp_path / "inner")
    benchmark.interrupt_inner = interrupt_inner
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    benchmark.progress_path = str(tmp_path / "progress.json")
    actual_popen = cooperative_owner.subprocess.Popen
    workers = []
    worker_nonces = []

    def observed_worker(*args, **kwargs):
        worker_nonces.append(kwargs["env"]["AISP_CLEANUP_EVENT_NONCE"])
        process = actual_popen(*args, **kwargs)
        workers.append(process)
        return process

    monkeypatch.setattr(cooperative_owner.subprocess, "Popen", observed_worker)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    expected = CooperativeProcessInterrupted if interrupt_inner else CooperativeCleanupIncomplete
    next_work = tmp_path / "next-work.txt"
    with pytest.raises(expected) as stopped:
        BenchmarkHarness(config=config).benchmark(benchmark)
        next_work.write_text("The sweep incorrectly continued\n")

    receipt = stopped.value.result
    assert receipt["execution"]["completion_scope"] == "cleanup_incomplete"
    assert receipt["execution"]["coordinator_error"] == {
        "type": "PermissionError", "message": "Nested CPU cleanup observation unavailable",
    }
    assert receipt["stdout"] == "nested CPU stdout\n"
    assert receipt["stderr"] == "nested CPU stderr\n"
    assert json.loads((tmp_path / "inner.execution.json").read_text()) == receipt
    assert not next_work.exists()
    assert Path(benchmark.cleanup_path).is_file()
    assert json.loads(Path(benchmark.progress_path).read_text()) == {"calls": 1, "validated": False}
    worker_stdout = (tmp_path / "NestedIncompleteCpuBenchmark_subprocess.stdout.log").read_text()
    worker_stderr = (tmp_path / "NestedIncompleteCpuBenchmark_subprocess.stderr.log").read_text()
    response = json.loads(worker_stdout)
    assert response["success"] is False
    assert response["cleanup_incomplete"] == receipt
    assert response["operator_interrupted"] is interrupt_inner
    assert parse_cleanup_incomplete_event(
        worker_stderr,
        expected_nonce=worker_nonces[0],
    ) == receipt
    assert len(workers) == 1
    assert workers[0].returncode == 0
    assert receipt["execution"]["process_group_id"] != workers[0].pid
    with pytest.raises(ProcessLookupError):
        os.kill(workers[0].pid, 0)


@requires_owned_worker
def test_nested_cleanup_stays_terminal_after_outer_coordinator_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    benchmark = _NestedIncompleteCpuBenchmark()
    benchmark.inner_prefix = str(tmp_path / "inner")
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    benchmark.progress_path = str(tmp_path / "progress.json")
    actual_communicate = cooperative_owner.communicate_cooperatively
    original = RuntimeError("Coordinator failed after receiving the nested terminal event")

    def fail_after_nested_event(process, **kwargs):
        receipt = actual_communicate(process, **kwargs)
        assert parse_cleanup_incomplete_event(
            receipt["stderr"],
            expected_nonce=receipt["execution"]["cleanup_event_nonce"],
        ) is not None
        original.cooperative_process_result = receipt
        raise original

    monkeypatch.setattr(cooperative_owner, "communicate_cooperatively", fail_after_nested_event)
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    next_work = tmp_path / "next-work.txt"
    with pytest.raises(CooperativeCleanupIncomplete) as stopped:
        BenchmarkHarness(config=config).benchmark(benchmark)
        next_work.write_text("The sweep incorrectly continued\n")

    assert stopped.value.__cause__ is original
    assert stopped.value.result["execution"]["completion_scope"] == "cleanup_incomplete"
    assert stopped.value.result["execution"]["process_group_id"] != original.cooperative_process_result["execution"]["process_group_id"]
    assert not next_work.exists()


@pytest.mark.parametrize(
    ("execution_timeout_seconds", "runner_name"),
    [(0, "run_cooperative_process"), (1, "run_owned_process")],
)
def test_authenticated_malformed_nested_cleanup_event_fails_closed(
    tmp_path: Path,
    monkeypatch,
    execution_timeout_seconds: int,
    runner_name: str,
) -> None:
    nonce = "matching-cleanup-event-nonce"
    process_id = 43210
    malformed_event = json.dumps({
        "event": "aisp_cleanup_incomplete",
        "nonce": nonce,
        "receipt": None,
    })

    def malformed_worker_result(*args, input_factory, **kwargs):
        input_factory(SimpleNamespace(pid=process_id))
        return {
            "returncode": 0,
            "stdout": "worker stdout before malformed cleanup event\n",
            "stderr": malformed_event + "\n",
            "timeout_hit": False,
            "execution": {
                "process_id": process_id,
                "process_group_id": process_id,
                "completion_scope": "process_tree_quiescent",
                "live_process_group_members": [],
                "cleanup_event_nonce": nonce,
            },
        }

    monkeypatch.setattr(
        harness_module,
        "require_cooperative_process_support",
        lambda: None,
    )
    monkeypatch.setattr(harness_module, runner_name, malformed_worker_result)
    benchmark = _ObservedCpuBenchmark()
    next_work = tmp_path / "next-work.txt"
    with pytest.raises(CooperativeCleanupIncomplete) as stopped:
        BenchmarkHarness(config=_config(
            execution_timeout_seconds=execution_timeout_seconds,
            subprocess_stderr_dir=str(tmp_path),
        )).benchmark(benchmark)
        next_work.write_text("The sweep incorrectly continued\n")

    receipt = stopped.value.result
    assert receipt["execution"]["completion_scope"] == "cleanup_incomplete"
    assert receipt["execution"]["live_process_group_members"] is None
    assert receipt["execution"]["cleanup_protocol_scope"] == "authenticated_nested_event"
    assert receipt["execution"]["outer_completion_scope"] == "process_tree_quiescent"
    assert receipt["outer_execution"]["completion_scope"] == "process_tree_quiescent"
    assert "requires a receipt object" in receipt["cleanup_protocol_error"]
    assert stopped.value.outer_process_result["execution"]["completion_scope"] == "process_tree_quiescent"
    assert stopped.value.cleanup_protocol_error == receipt["cleanup_protocol_error"]
    assert not next_work.exists()


def test_authenticated_malformed_nested_cleanup_event_preserves_interrupt_priority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    nonce = "matching-interrupted-cleanup-event-nonce"
    process_id = 43211
    outer_receipt = {
        "returncode": 130,
        "stdout": "worker stdout before interrupted malformed event\n",
        "stderr": json.dumps({
            "event": "aisp_cleanup_incomplete",
            "nonce": nonce,
            "receipt": None,
        }) + "\n",
        "timeout_hit": False,
        "cancelled": True,
        "cancellation": {
            "operator_interrupt_observed": True,
            "interrupt_count": 1,
        },
        "execution": {
            "process_id": process_id,
            "process_group_id": process_id,
            "completion_scope": "process_tree_quiescent",
            "live_process_group_members": [],
            "cleanup_event_nonce": nonce,
        },
    }

    def interrupted_worker(*args, input_factory, **kwargs):
        input_factory(SimpleNamespace(pid=process_id))
        raise CooperativeProcessInterrupted(outer_receipt)

    monkeypatch.setattr(
        harness_module,
        "require_cooperative_process_support",
        lambda: None,
    )
    monkeypatch.setattr(
        harness_module,
        "run_cooperative_process",
        interrupted_worker,
    )

    with pytest.raises(CooperativeProcessInterrupted) as stopped:
        BenchmarkHarness(config=_config(
            execution_timeout_seconds=0,
            subprocess_stderr_dir=str(tmp_path),
        )).benchmark(_ObservedCpuBenchmark())

    assert stopped.value.result is outer_receipt
    cleanup_receipt = stopped.value.cleanup_incomplete_receipt
    assert cleanup_receipt["execution"]["completion_scope"] == "cleanup_incomplete"
    assert cleanup_receipt["execution"]["cleanup_protocol_scope"] == "authenticated_nested_event"
    assert stopped.value.cleanup_protocol_error == cleanup_receipt["cleanup_protocol_error"]
    assert stopped.value.outer_process_result is outer_receipt
