"""Exercise uncapped ownership with real CPU work and real isolated children."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
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


class _ObservedCpuBenchmark(BaseBenchmark):
    """Compute actual tensor outputs and retain the configuration that executed."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.observation_path = ""
        self.cleanup_path = ""
        self.interrupt_kind = ""
        self.interrupted = False
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

    def benchmark_fn(self) -> None:
        self.output = self.input.square()
        if self.interrupt_kind and not self.interrupted:
            self.interrupted = True
            if self.interrupt_kind == "stored":
                raise KeyboardInterrupt("Real worker Future interruption")
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

    def teardown(self) -> None:
        if self.interrupt_kind:
            time.sleep(0.05)
        if self.cleanup_path:
            Path(self.cleanup_path).write_text("CPU benchmark teardown completed\n")


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


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
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


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
@pytest.mark.parametrize("kind", ["main", "stored"])
def test_owned_thread_interrupt_waits_for_real_teardown_without_result_exception_loop(tmp_path: Path, kind: str) -> None:
    benchmark = _ObservedCpuBenchmark()
    benchmark.interrupt_kind = kind
    benchmark.cleanup_path = str(tmp_path / "cleanup.txt")
    config = _config(execution_timeout_seconds=0, subprocess_stderr_dir=str(tmp_path))
    result = BenchmarkHarness(config=config).benchmark(benchmark)
    assert result.errors
    assert Path(benchmark.cleanup_path).read_text() == "CPU benchmark teardown completed\n"


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
def test_stored_future_timeout_error_is_not_an_uncapped_wait_deadline(tmp_path: Path) -> None:
    module = tmp_path / "cpu_future_failure.py"
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
    assert json.loads(record["result_json"])["timeout_stage"] is None
    assert result.timeout_stage is None


@pytest.mark.skipif(os.name != "posix", reason="owned isolated execution requires POSIX")
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
