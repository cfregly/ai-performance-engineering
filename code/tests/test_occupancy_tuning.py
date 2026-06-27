"""Unit tests for labs/occupancy_tuning helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from labs.occupancy_tuning import sweep_schedules
from labs.occupancy_tuning.triton_matmul_schedules import SCHEDULES

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_schedules_handles_unknown_names() -> None:
    known = SCHEDULES[0].name
    resolved = sweep_schedules.resolve_schedules([known])
    assert len(resolved) == 1 and resolved[0].name == known

    with pytest.raises(ValueError):
        sweep_schedules.resolve_schedules(["does_not_exist"])


def test_benchmark_schedule_uses_reused_cuda_events_for_samples() -> None:
    source = (REPO_ROOT / "labs" / "occupancy_tuning" / "sweep_schedules.py").read_text(
        encoding="utf-8"
    )
    benchmark_section = source.split("def benchmark_schedule", maxsplit=1)[1].split(
        "def run_sweep",
        maxsplit=1,
    )[0]
    sample_loop = benchmark_section.split("for _ in range(max(1, iterations)):", maxsplit=1)[
        1
    ].split("mean_ms = statistics.mean(times_ms)", maxsplit=1)[0]

    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "start_event.record()" in sample_loop
    assert "end_event.record()" in sample_loop
    assert "end_event.synchronize()" in sample_loop
    assert "times_ms.append(start_event.elapsed_time(end_event))" in sample_loop
    assert "time.perf_counter()" not in benchmark_section


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_run_sweep_executes_single_schedule() -> None:
    schedule = SCHEDULES[0]
    results = sweep_schedules.run_sweep(
        [schedule],
        size=64,
        iterations=1,
        warmup=0,
        dtype=torch.float16,
        use_compile=False,
    )
    assert len(results) == 1
    result = results[0]
    assert result.name == schedule.name
    assert result.mean_ms >= 0.0
    assert result.tflops >= 0.0
