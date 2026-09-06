from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch

from core.benchmark.models import BenchmarkResult
from core.harness.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    TorchrunLaunchSpec,
)


class _CpuTorchrunTarget:
    name = "reported_timing_cpu_control"

    def __init__(
        self,
        script_path: Path,
        *,
        timing_iterations_per_sample: int,
    ) -> None:
        self.script_path = script_path
        self.timing_iterations_per_sample = timing_iterations_per_sample

    def get_torchrun_spec(self, config: BenchmarkConfig) -> TorchrunLaunchSpec:
        return TorchrunLaunchSpec(
            script_path=self.script_path,
            script_args=[str(self.script_path.with_suffix(".timing"))],
            config_arg_map={
                "iterations": "--iterations",
                "warmup": "--warmup",
            },
            name=self.name,
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=self.timing_iterations_per_sample,
        )


def _cpu_harness(*, iterations: int = 6) -> tuple[BenchmarkHarness, BenchmarkConfig]:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=iterations,
        warmup=5,
        nproc_per_node=1,
        measurement_timeout_seconds=60,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        enforce_environment_validation=False,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config.rdzv_endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config), config


def test_real_torchrun_uses_declared_worker_iteration_mean(tmp_path: Path) -> None:
    script_path = tmp_path / "cpu_timing_worker.py"
    timing_path = script_path.with_suffix(".timing")
    script_path.write_text(
        """import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("timing_path", type=Path)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--warmup", type=int, required=True)
args = parser.parse_args()

def work():
    return sum(value * value for value in range(20_000))

for _ in range(args.warmup):
    work()
samples_ms = []
for _ in range(args.iterations):
    start_ns = time.perf_counter_ns()
    result = work()
    samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
if result <= 0:
    raise RuntimeError("CPU timing control did not execute")
mean_ms = sum(samples_ms) / len(samples_ms)
encoded = f"{mean_ms:.9f}"
args.timing_path.write_text(encoded, encoding="utf-8")
print(f"rank0 time_per_iter_ms: {encoded}", flush=True)
""",
        encoding="utf-8",
    )
    harness, config = _cpu_harness(iterations=6)

    result = harness._benchmark_with_torchrun(
        _CpuTorchrunTarget(
            script_path,
            timing_iterations_per_sample=config.iterations,
        ),
        config,
    )

    reported_ms = float(timing_path.read_text(encoding="utf-8"))
    assert not result.errors, result.errors
    assert result.timing.mean_ms == pytest.approx(reported_ms)
    assert result.timing.raw_times_ms == pytest.approx([reported_ms])
    assert result.timing.sample_scope == "rank0_iteration_mean"
    assert result.timing.iterations == 1
    assert result.timing.iterations_per_sample == config.iterations
    assert result.timing.percentiles == {}
    assert result.timing.p50_ms is None
    assert result.timing.p99_ms is None
    assert result.custom_metrics["torchrun.process_wall_ms"] > 0
    restored = BenchmarkResult.model_validate_json(result.model_dump_json())
    assert restored.timing.sample_scope == "rank0_iteration_mean"
    assert restored.timing.iterations_per_sample == config.iterations


def test_real_torchrun_rejects_missing_declared_worker_timing(tmp_path: Path) -> None:
    script_path = tmp_path / "cpu_worker_without_timing.py"
    script_path.write_text(
        """import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("timing_path", type=Path)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--warmup", type=int, required=True)
args = parser.parse_args()
value = sum(item * item for item in range(args.iterations + args.warmup))
args.timing_path.write_text(str(value), encoding="utf-8")
""",
        encoding="utf-8",
    )
    harness, config = _cpu_harness(iterations=2)

    with pytest.raises(RuntimeError, match="requires exactly one"):
        harness._benchmark_with_torchrun(
            _CpuTorchrunTarget(
                script_path,
                timing_iterations_per_sample=config.iterations,
            ),
            config,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"timing_source": "rank0_time_per_iter_ms"},
            "requires a positive timing_iterations_per_sample",
        ),
        (
            {"timing_source": "rank0_time_per_iter_ms", "timing_iterations_per_sample": 0},
            "requires a positive timing_iterations_per_sample",
        ),
        (
            {"timing_source": "process_wall", "timing_iterations_per_sample": 2},
            "process_wall timing cannot declare",
        ),
        ({"timing_source": "unknown"}, "Unsupported torchrun timing_source"),
    ),
)
def test_torchrun_timing_spec_rejects_ambiguous_contracts(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TorchrunLaunchSpec(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("encoded", ("nan", "inf", "0", "-2", "not-a-number"))
def test_rank0_worker_timing_rejects_invalid_values(encoded: str) -> None:
    with pytest.raises(RuntimeError, match="numeric|finite and positive"):
        BenchmarkHarness._extract_rank0_time_per_iter_ms(
            [f"rank0 time_per_iter_ms: {encoded}"]
        )
