"""Behavior checks for audit findings W1-061, W1-062 and W1-063.

Timing fixtures test report transport, not GPU measurement or qualification.
The detector cases execute actual CPU work with visible environment diagnostics.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from core.benchmark.models import BenchmarkResult, TimingStats
from core.harness import benchmark_harness as harness_module
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness


class _CPUWork:
    def __init__(self) -> None:
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = self.input + self.input


class _LoggingWork(_CPUWork):
    def benchmark_fn(self) -> None:
        print("audit detector regression")
        self.output = self.input + self.input


def _cpu_harness(**kwargs) -> tuple[BenchmarkHarness, BenchmarkConfig]:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=2,
        warmup=5,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        # These are detector/control-flow tests, not accepted benchmark results.
        # Real local environment warnings remain visible; no CUDA is emulated.
        enforce_environment_validation=False,
        **kwargs,
    )
    harness = BenchmarkHarness(config=config)
    harness._ensure_runtime_initialized()
    return harness, config


@pytest.mark.parametrize("sync_detection", [False, True])
@pytest.mark.parametrize("antipattern_detection", [False, True])
def test_detector_switches_are_independent(sync_detection, antipattern_detection):
    harness, config = _cpu_harness(
        detect_benchmark_fn_sync=sync_detection,
        detect_benchmark_fn_antipatterns=antipattern_detection,
    )
    work = _CPUWork()
    samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
    assert len(samples) == 2
    torch.testing.assert_close(work.output, work.input * 2)


def test_antipattern_detection_still_rejects_logging_when_sync_detection_is_off():
    harness, config = _cpu_harness(
        detect_benchmark_fn_sync=False,
        detect_benchmark_fn_antipatterns=True,
        benchmark_fn_antipattern_policy="error",
    )
    work = _LoggingWork()
    with pytest.raises(RuntimeError, match="ANTI-PATTERN"):
        harness._benchmark_custom(work.benchmark_fn, config)
    assert work.output is None


def test_summary_timing_preserves_only_supplied_statistics():
    harness, config = _cpu_harness()
    # Deliberately asymmetric summary: inventing Gaussian samples would alter it.
    child_timing = TimingStats(
        mean_ms=4.0, median_ms=1.0, std_ms=6.0, min_ms=0.1, max_ms=25.0,
        iterations=100, warmup_iterations=7, p99_ms=24.0,
        percentiles={99.0: 24.0}, raw_times_ms=None,
    )
    transported = TimingStats.model_validate_json(child_timing.model_dump_json())
    result = harness._result_from_timing(transported, config)
    assert result.timing.model_dump() == child_timing.model_dump()
    assert result.timing.raw_times_ms is None
    assert result.timing.p90_ms is None
    assert result.timing.p95_ms is None
    assert result.timing.iterations != config.iterations


def test_measured_samples_still_drive_statistics():
    harness, config = _cpu_harness()
    result = harness._compute_stats([1.0, 1.0, 10.0], config)
    assert result.timing.raw_times_ms == [1.0, 1.0, 10.0]
    assert result.timing.mean_ms == 4.0
    assert result.timing.median_ms == 1.0
    assert result.timing.iterations == 3


@pytest.mark.parametrize(
    ("line", "expected"),
    [("throughput: 1,234.5 tok/s", 1234.5),
     ("1234.5tok/s", 1234.5),
     ("1000 tokens/s", 1000.0),
     ("1 toks/s", 1.0),
     ("0 TOKENS/S", 0.0),
     ("no throughput available", None)],
)
def test_torchrun_throughput_log_parser(line, expected):
    assert BenchmarkHarness._extract_tokens_per_s([line]) == expected


def test_torchrun_throughput_selects_highest_valid_observation():
    assert BenchmarkHarness._extract_tokens_per_s(
        ["warmup 2 tok/s", "broken 1.2.3 tok/s", "measured 3,000 tokens/s"]
    ) == 3000.0


class _CPUTransportBenchmark(BaseBenchmark):
    """Real CPU work for subprocess result/output transport, not a GPU proxy."""

    allow_cpu = True

    def setup(self) -> None:
        self.device = torch.device("cpu")
        self.input = torch.arange(8, dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> None:
        self.output = self.input + self.input

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
        torch.testing.assert_close(self.output, self.input * 2)


def test_real_cpu_subprocess_preserves_samples_and_computed_output():
    harness, config = _cpu_harness()
    work = _CPUTransportBenchmark()
    result = harness._benchmark_with_subprocess(work, config)
    assert not result.errors, result.errors
    assert result.timing.raw_times_ms is not None
    assert len(result.timing.raw_times_ms) == config.iterations
    torch.testing.assert_close(work._subprocess_verify_output, torch.arange(8) * 2.0)


def _summary_payload():
    timing = TimingStats(
        mean_ms=4.0, median_ms=1.0, std_ms=6.0, min_ms=0.1, max_ms=25.0,
        iterations=100, warmup_iterations=7, p99_ms=24.0,
        percentiles={99.0: 24.0}, raw_times_ms=None,
    )
    return {"success": True, "result_json": BenchmarkResult(timing=timing).model_dump_json()}


def _transport_fixture(monkeypatch, payload, tmp_path):
    """Supply a protocol fixture through real pipes; do not simulate GPU work.

    The patched command seam tests deserialization and error propagation only.
    The separate CPU subprocess test above executes the real isolated runner.
    """
    fixture_file = tmp_path / "child-protocol-fixture.json"
    fixture_file.write_text(json.dumps(payload))
    original_popen = subprocess.Popen

    def run_fixture(command, **kwargs):
        assert command[1:3] == ["-m", "core.harness.isolated_runner"]
        return original_popen(
            [sys.executable, "-c",
             "import pathlib,sys; sys.stdin.read(); print(pathlib.Path(sys.argv[1]).read_text())",
             str(fixture_file)],
            **kwargs,
        )

    monkeypatch.setattr(harness_module.subprocess, "Popen", run_fixture)
    harness, config = _cpu_harness()
    return harness._benchmark_with_subprocess(_CPUTransportBenchmark(), config)


def test_summary_only_child_transport_preserves_unavailable_percentiles(monkeypatch, tmp_path):
    payload = _summary_payload()
    result = _transport_fixture(monkeypatch, payload, tmp_path)
    expected = BenchmarkResult.model_validate_json(payload["result_json"])
    assert not result.errors
    assert result.timing == expected.timing
    assert result.timing.raw_times_ms is None
    assert result.timing.p90_ms is None
    assert result.timing.p95_ms is None


def test_block_mean_child_transport_preserves_measurement_scope(monkeypatch, tmp_path):
    timing = TimingStats(
        mean_ms=3.5, median_ms=3.5, std_ms=2.12, min_ms=2.0, max_ms=5.0,
        iterations=2, warmup_iterations=7, raw_times_ms=[2.0, 5.0],
        sample_scope="block_mean", iterations_per_sample=4,
    )
    payload = {"success": True, "result_json": BenchmarkResult(timing=timing).model_dump_json()}
    result = _transport_fixture(monkeypatch, payload, tmp_path)
    assert not result.errors, result.errors
    assert result.timing.raw_times_ms == [2.0, 5.0]
    assert result.timing.sample_scope == "block_mean"
    assert result.timing.iterations_per_sample == 4
    assert result.timing.warmup_iterations == 7


def test_real_cpu_timer_subprocess_preserves_observation_units():
    harness, config = _cpu_harness()
    harness.mode = harness_module.BenchmarkMode.PYTORCH
    config.min_run_time_ms = 10
    work = _CPUTransportBenchmark()
    result = harness._benchmark_with_subprocess(work, config)
    assert not result.errors, result.errors
    assert result.timing.sample_scope == "block_mean"
    assert result.timing.iterations_per_sample >= 1
    assert result.timing.iterations == len(result.timing.raw_times_ms) > 0
    torch.testing.assert_close(work._subprocess_verify_output, torch.arange(8) * 2.0)


@pytest.mark.parametrize("malformed_field", ["verify_output", "output_tolerance"])
def test_child_reconstruction_error_does_not_credit_summary(monkeypatch, tmp_path, malformed_field):
    payload = _summary_payload()
    payload[malformed_field] = {"kind": "unknown"} if malformed_field == "verify_output" else {"rtol": "invalid"}
    result = _transport_fixture(monkeypatch, payload, tmp_path)
    assert any("Error processing subprocess result" in error for error in result.errors)
    assert result.timing.iterations == 0
    assert not result.timing.raw_times_ms
    assert result.timing.p99_ms is None


@pytest.mark.parametrize("field,value", [
    ("std_ms", -1.0), ("mean_ms", float("nan")), ("max_ms", float("inf")),
    ("mean_ms", 30.0), ("median_ms", 30.0), ("iterations", 0), ("p99_ms", 30.0),
])
def test_malformed_child_summary_is_rejected(monkeypatch, tmp_path, field, value):
    payload = _summary_payload()
    child = json.loads(payload["result_json"])
    child["timing"][field] = value
    payload["result_json"] = json.dumps(child)
    result = _transport_fixture(monkeypatch, payload, tmp_path)
    assert any("Invalid timing summary" in error for error in result.errors)
    assert result.timing.iterations == 0
    assert not result.timing.raw_times_ms
