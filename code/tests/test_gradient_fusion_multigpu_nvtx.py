from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.baseline_gradient_fusion_multigpu import BaselineGradientFusionMultiGPU
from ch04.gradient_fusion_multigpu import (
    BASELINE_PROFILE_NVTX_RANGE,
    OPTIMIZED_PROFILE_NVTX_RANGE,
    _run_collectives,
)
from ch04.optimized_gradient_fusion_multigpu import OptimizedGradientFusionMultiGPU
from core.profiling.profiler_config import build_profiler_config_from_benchmark


@pytest.mark.parametrize(
    ("benchmark_type", "expected_range"),
    [
        (BaselineGradientFusionMultiGPU, BASELINE_PROFILE_NVTX_RANGE),
        (OptimizedGradientFusionMultiGPU, OPTIMIZED_PROFILE_NVTX_RANGE),
    ],
)
def test_direct_torchrun_profile_declares_its_real_nvtx_range(
    benchmark_type: type,
    expected_range: str,
) -> None:
    config = benchmark_type().get_config()
    assert config.profiling.nsys_nvtx_include == (expected_range,)

    profiler = build_profiler_config_from_benchmark(config)
    command = profiler.get_ncu_command_for_target(
        "/tmp/gradient-fusion-profile",
        ["python", "gradient_fusion_multigpu.py"],
    )
    include_index = command.index("--nvtx-include")
    assert command[include_index + 1] == expected_range


def test_real_target_limits_nvtx_range_to_timed_collectives() -> None:
    source = (Path(__file__).resolve().parents[1] / "ch04/gradient_fusion_multigpu.py").read_text(
        encoding="utf-8"
    )
    run_body = source.split("def run_benchmark", 1)[1].split("def parse_args", 1)[0]
    warmup_call = run_body.index("_run_collectives(mode, tensors, fused, iterations=5)")
    start_event = run_body.index("start.record()")
    range_start = run_body.index("with nvtx_range(profile_range, enable=True):")
    timed_call = run_body.index("_run_collectives(mode, tensors, fused, iterations=iterations)")
    end_event = run_body.index("end.record()")

    assert warmup_call < start_event < range_start < timed_call < end_event
    assert 'BASELINE_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_many_allreduces"' in source
    assert 'OPTIMIZED_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_fused_allreduce"' in source
    assert 'if mode == "baseline"\n        else OPTIMIZED_PROFILE_NVTX_RANGE' in run_body


def _gloo_worker(rank: int, world_size: int, init_method: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        tensors = [torch.tensor([rank + 1.0 + index]) for index in range(3)]
        _run_collectives("baseline", tensors, None, iterations=1)
        for index, tensor in enumerate(tensors):
            torch.testing.assert_close(tensor, torch.tensor([3.0 + 2.0 * index]))

        fused = torch.tensor([rank + 1.0, 2.0 * (rank + 1.0)])
        _run_collectives("optimized", [], fused, iterations=1)
        torch.testing.assert_close(fused, torch.tensor([3.0, 6.0]))
        (Path(result_dir) / f"rank-{rank}.passed").write_text("passed\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU collective control requires the Gloo backend",
)
def test_gradient_fusion_collective_paths_run_on_two_real_ranks(tmp_path: Path) -> None:
    world_size = 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _gloo_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), str(result_dir)),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "rank-0.passed",
        "rank-1.passed",
    ]
