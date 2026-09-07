"""Two-rank NCCL worker for communicator reinitialization versus reuse."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status
from ch04.reinit_comm_multigpu_result import WORLD_SIZE, write_reinit_comm_child_result
from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from core.profiling.nvtx_helper import nvtx_range


@dataclass
class _RendezvousState:
    store: dist.Store
    rank: int
    world_size: int
    generation: int = 0


def _initialize_process_group(local_rank: int, rendezvous: _RendezvousState) -> None:
    if dist.is_initialized():
        raise RuntimeError("Reinit-comm process group is already initialized")
    # A recreated default process group resets its internal store-key sequence.
    # Isolate generations so a faster rank cannot consume an old NCCL ID whose
    # communicator has already closed its bootstrap socket.
    store = dist.PrefixStore(f"reinit_comm/{rendezvous.generation}", rendezvous.store)
    rendezvous.generation += 1
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=rendezvous.rank,
        world_size=rendezvous.world_size,
        timeout=datetime.timedelta(seconds=120),
        device_id=local_rank,
    )


def _run_iteration(
    variant: str,
    *,
    local_rank: int,
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    rendezvous: _RendezvousState,
) -> None:
    if variant == "baseline":
        if dist.is_initialized():
            torch.cuda.synchronize(local_rank)
            dist.destroy_process_group()
        torch.cuda.set_device(local_rank)
        _initialize_process_group(local_rank, rendezvous)
    elif variant != "optimized":
        raise ValueError(f"Unsupported reinit-comm variant: {variant!r}")
    if not dist.is_initialized():
        raise RuntimeError("Reinit-comm optimized iteration requires its setup group")
    output.copy_(input_tensor)
    dist.all_reduce(output, op=dist.ReduceOp.SUM)


def _init_worker() -> tuple[int, int, int, torch.device]:
    require_min_gpus(WORLD_SIZE)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if name not in os.environ:
            raise RuntimeError("SKIPPED: reinit-comm worker requires torchrun rank context")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"SKIPPED: reinit-comm requires exactly {WORLD_SIZE} ranks, got {world_size}"
        )
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def run_worker(variant: str, *, iterations: int, warmup: int) -> None:
    if variant not in {"baseline", "optimized"}:
        raise ValueError(f"Unsupported reinit-comm variant: {variant!r}")
    if iterations <= 0 or warmup < 0:
        raise ValueError("Iterations must be positive and warmup non-negative")
    rank, world_size, local_rank, device = _init_worker()
    store_iterator = dist.rendezvous("env://", rank=rank, world_size=world_size)
    store, _, _ = next(store_iterator)
    rendezvous = _RendezvousState(store, rank, world_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(torch.initial_seed())
    input_tensor = torch.randn((1, 1), device=device, dtype=torch.float32, generator=generator)
    output = torch.empty_like(input_tensor)

    try:
        if variant == "optimized":
            _initialize_process_group(local_rank, rendezvous)
        for _ in range(warmup):
            _run_iteration(
                variant,
                local_rank=local_rank,
                input_tensor=input_tensor,
                output=output,
                rendezvous=rendezvous,
            )
        if not dist.is_initialized():
            # Establish an untimed rendezvous solely to align rank-local clocks.
            _initialize_process_group(local_rank, rendezvous)
        dist.barrier()
        torch.cuda.synchronize(device)

        start_ns = time.perf_counter_ns()
        with nvtx_range("reinit_comm", enable=True):
            for _ in range(iterations):
                _run_iteration(
                    variant,
                    local_rank=local_rank,
                    input_tensor=input_tensor,
                    output=output,
                    rendezvous=rendezvous,
                )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        # Report the slowest rank's complete host interval. Communicator setup and
        # destruction are CPU costs and therefore must not use CUDA-event timing.
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        time_per_iter_ms = float(elapsed.item()) / iterations
        if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
            raise RuntimeError("Reinit-comm worker timing must be finite and positive")

        write_reinit_comm_child_result(
            variant=variant,
            rank=rank,
            world_size=world_size,
            iterations=iterations,
            warmup=warmup,
            input_tensor=input_tensor,
            output=output,
            time_per_iter_ms=time_per_iter_ms,
        )
        if rank == 0:
            print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    run_worker(args.variant, iterations=args.iterations, warmup=args.warmup)


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
