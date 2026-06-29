from __future__ import annotations

"""Demonstrate how monitored_barrier surfaces stragglers."""


import datetime

from core.common.device_utils import resolve_local_rank
import time

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env

from core.benchmark.gpu_requirements import require_min_gpus


def init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices required for straggler demo.")

    if not dist.is_initialized():
        setup_single_gpu_env("barrier_straggler_multigpu", min_world_size=2)
        local_rank = resolve_local_rank()
        torch.cuda.set_device(local_rank)
        # Use GLOO backend since monitored_barrier is only implemented for GLOO
        dist.init_process_group("gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = resolve_local_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def main() -> None:
    # Check GPU requirements early - barrier testing needs multiple GPUs
    require_min_gpus(2, "barrier_straggler_multigpu.py")
    
    rank, world_size, device = init_distributed()

    if rank == 1 and world_size > 1:
        time.sleep(2.0)  # Inject artificial delay

    dummy_tensor = torch.randn(1000, 1000, device=device)
    _ = torch.mm(dummy_tensor, dummy_tensor.t())

    try:
        start = time.perf_counter()
        dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))
        elapsed = time.perf_counter() - start
        print(f"Rank {rank} completed barrier in {elapsed:.2f}s", flush=True)
    except RuntimeError as exc:
        print(f"Rank {rank} barrier timeout: {exc}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
