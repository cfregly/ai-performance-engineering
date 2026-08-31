"""Observed timing provenance, with real CPU work/process and Timer payload tests."""

from pathlib import Path
import socket

import pytest
import torch
from torch.utils.benchmark import Measurement, TaskSpec, Timer

from core.harness.benchmark_harness import (
    BenchmarkConfig, BenchmarkHarness, BenchmarkMode, TorchrunLaunchSpec,
)


def cpu_harness(*, iterations=10, mode=BenchmarkMode.PYTORCH):
    config = BenchmarkConfig(
        device=torch.device("cpu"), iterations=iterations, warmup=5,
        use_subprocess=False, enable_profiling=False, enable_memory_tracking=False,
        enforce_environment_validation=False, min_run_time_ms=10,
        measurement_timeout_seconds=40, nproc_per_node=1,
        rdzv_backend="static", rdzv_endpoint="127.0.0.1:29500",
    )
    harness = BenchmarkHarness(mode=mode, config=config)
    harness._ensure_runtime_initialized()
    return harness, config


@pytest.mark.parametrize("iterations", [1, 10])
def test_timer_does_not_pad_or_discard_observed_blocks(monkeypatch, iterations):
    # A real PyTorch Measurement supplies block means. This is a timing-payload
    # fixture, not a measured performance result; real CPU Timer is checked below.
    measurement = Measurement(
        number_per_run=4, raw_times=[0.008, 0.020], task_spec=TaskSpec(stmt="fn()", setup="pass"),
    )
    monkeypatch.setattr(Timer, "blocked_autorange", lambda *args, **kwargs: measurement)
    harness, config = cpu_harness(iterations=iterations)
    samples = harness._benchmark_pytorch(lambda: None, config)
    assert samples == [2.0, 5.0]
    result = harness._compute_stats(samples, config)
    assert result.timing.iterations == 2
    assert result.timing.sample_scope == "block_mean"
    assert result.timing.iterations_per_sample == 4
    assert result.timing.raw_times_ms == [2.0, 5.0]


def test_real_cpu_timer_preserves_observed_block_count_and_output():
    harness, config = cpu_harness(iterations=100000)
    inputs = torch.arange(32, dtype=torch.float32)
    output = torch.empty_like(inputs)
    calls = 0

    def work():
        nonlocal calls
        calls += 1
        torch.add(inputs, 3, out=output)

    samples = harness._benchmark_pytorch(work, config)
    result = harness._compute_stats(samples, config)
    assert 0 < result.timing.iterations == len(result.timing.raw_times_ms) < config.iterations
    assert result.timing.sample_scope == "block_mean"
    assert calls >= result.timing.iterations * result.timing.iterations_per_sample
    assert all(value > 0 for value in samples)
    torch.testing.assert_close(output, inputs + 3)
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.timing == result.timing


class CPUProcess:
    name = "audit_cpu_process"

    def __init__(self, path: Path):
        self.path = path

    def get_torchrun_spec(self, config):
        return TorchrunLaunchSpec(
            script_path=self.path, config_arg_map={"iterations": "--steps"}, name=self.name,
        )


def test_real_torchrun_reports_one_observed_process_interval(tmp_path):
    script = tmp_path / "cpu_work.py"
    output = tmp_path / "result.txt"
    script.write_text(
        "import argparse\nfrom pathlib import Path\nimport torch\n"
        "p=argparse.ArgumentParser(); p.add_argument('--steps', type=int); a=p.parse_args()\n"
        "x=torch.arange(8); y=x.clone()\n"
        "for _ in range(a.steps): y.add_(x)\n"
        f"Path({str(output)!r}).write_text(str(y.tolist()))\n"
        "print('123 tokens/s')\n"
    )
    harness, config = cpu_harness(iterations=7, mode=BenchmarkMode.CUSTOM)
    # Static loopback avoids this Mac's unresolvable reverse-DNS hostname in
    # elastic c10d advertisements. This remains a real local torchrun process.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    result = harness._benchmark_with_torchrun(CPUProcess(script), config)
    assert not result.errors, result.errors
    assert output.read_text() == str((torch.arange(8) * 8).tolist())
    assert result.timing.iterations == 1
    assert len(result.timing.raw_times_ms) == 1
    assert result.timing.sample_scope == "process_wall"
    assert result.timing.iterations_per_sample == 1
    assert result.timing.mean_ms == result.timing.raw_times_ms[0] > 0
    assert result.custom_metrics["torchrun.requested_iterations"] == 7
    assert result.custom_metrics["torchrun.amortized_ms_per_requested_iteration"] == pytest.approx(result.timing.mean_ms / 7)
    assert result.throughput.tokens_per_s == 123
