"""Unit tests for labs/occupancy_tuning helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

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


def test_schedule_metadata_import_leaves_kernel_unloaded() -> None:
    script = """
import sys
from labs.occupancy_tuning import sweep_schedules
from labs.occupancy_tuning.optimized_proton_matmul_bm64_bn256_bk32 import get_benchmark
from labs.occupancy_tuning.triton_matmul_schedules import SCHEDULES
assert 'labs.occupancy_tuning.triton_matmul' not in sys.modules
assert sweep_schedules.resolve_schedules([SCHEDULES[0].name]) == [SCHEDULES[0]]
benchmark = get_benchmark()
assert benchmark.schedule.block_n == 256
assert 'labs.occupancy_tuning.triton_matmul' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                   check=True, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("name", ["matmul_kernel", "run_one", "describe_schedule", "triton_matmul"])
def test_public_kernel_api_requires_actual_triton_when_unavailable(name: str) -> None:
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("This negative control requires an actually absent Triton installation")
    import labs.occupancy_tuning as lab

    assert name in lab.__all__
    with pytest.raises(ModuleNotFoundError) as failure:
        getattr(lab, name)
    assert failure.value.name == "triton"


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
    ].split("sample_count = len(times_ms)", maxsplit=1)[0]

    assert benchmark_section.count("torch.cuda.Event(enable_timing=True)") == 2
    assert "current_stream = torch.cuda.current_stream()" in benchmark_section
    assert "start_event.record(current_stream)" in sample_loop
    assert "end_event.record(current_stream)" in sample_loop
    assert "start_event.record()" not in sample_loop
    assert "end_event.record()" not in sample_loop
    assert "end_event.synchronize()" in sample_loop
    assert "times_ms.append(start_event.elapsed_time(end_event))" in sample_loop
    assert "times_ms.sort()" in benchmark_section
    assert "times_sorted = sorted(times_ms)" not in benchmark_section
    assert "statistics.mean" not in benchmark_section
    assert "statistics.median" not in benchmark_section
    assert "min(times_ms)" not in benchmark_section
    assert "max(times_ms)" not in benchmark_section
    assert "time.perf_counter()" not in benchmark_section


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_run_sweep_executes_single_schedule() -> None:
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Real Triton is required to execute the CUDA schedule sweep")
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
