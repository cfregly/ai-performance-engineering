#!/usr/bin/env python3
"""Explicit two-rank worker for Chapter 4 phase-separated inference.

Every rank performs the complete prefill and decode MLPs and both WORLD
all-reduces; the optimized variant separates model/storage paths per phase.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional

from ch04.disaggregated_multigpu_result import (
    DisaggregatedResultContract,
    make_result_contract,
    write_disaggregated_child_result,
)
from ch04.distributed_helper import run_main_with_skip_status
from ch04.reduction_common import ReusableReductionMlp
from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from core.profiling.nvtx_helper import nvtx_range

WORLD_SIZE = 2
BATCH_SIZE = 2
PREFILL_LEN = 512
HIDDEN_DIM = 256
SEED = 42
BASELINE_NVTX_RANGE = "baseline_disaggregated_multigpu"
OPTIMIZED_NVTX_RANGE = "optimized_disaggregated_multigpu"


def _make_models(
    variant: str,
    *,
    hidden_dim: int,
    device: torch.device,
    seed: int,
    wrap_ddp: bool,
) -> tuple[nn.Module, nn.Module]:
    if variant not in {"baseline", "optimized"}:
        raise ValueError(f"Unsupported disaggregated variant: {variant!r}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if variant == "baseline":
        model: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        ).to(device).eval()
        if wrap_ddp:
            model = nn.parallel.DistributedDataParallel(model)
        return model, model

    prefill: nn.Module = ReusableReductionMlp(hidden_dim, hidden_dim * 2).to(device).eval()
    decode: nn.Module = copy.deepcopy(prefill)
    if wrap_ddp:
        prefill = nn.parallel.DistributedDataParallel(prefill)
        decode = nn.parallel.DistributedDataParallel(decode)
    return prefill, decode


def _make_inputs(
    contract: DisaggregatedResultContract, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Model construction consumes the same RNG sequence in both variants; these
    # calls preserve the original seed-42 input construction exactly.
    prefill = torch.randn(
        contract.batch_size,
        contract.prefill_len,
        contract.hidden_dim,
        device=device,
    )
    decode = torch.randn(
        contract.batch_size,
        1,
        contract.hidden_dim,
        device=device,
    )
    return prefill, decode


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def _functional_mlp(model: nn.Module, value: torch.Tensor) -> torch.Tensor:
    """Independent functional reference for either model representation."""
    module = _unwrap(model)
    if isinstance(module, ReusableReductionMlp):
        first, second = module.fc1, module.fc2
    elif isinstance(module, nn.Sequential) and len(module) == 3:
        first, second = module[0], module[2]
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise TypeError("Baseline disaggregated model has unexpected layers")
    else:
        raise TypeError(f"Unsupported disaggregated model type: {type(module).__name__}")
    hidden = functional.linear(value, first.weight, first.bias)
    hidden = functional.relu(hidden, inplace=False)
    return functional.linear(hidden, second.weight, second.bias)


def _reference_output(
    prefill_model: nn.Module,
    decode_model: nn.Module,
    prefill_input: torch.Tensor,
    decode_input: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        return torch.cat(
            (
                _functional_mlp(prefill_model, prefill_input),
                _functional_mlp(decode_model, decode_input),
            ),
            dim=1,
        )


def _run_iteration(
    variant: str,
    prefill_model: nn.Module,
    decode_model: nn.Module,
    prefill_input: torch.Tensor,
    decode_input: torch.Tensor,
    *,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete original MLP and reduction work for one iteration."""
    with torch.inference_mode():
        prefill_output = prefill_model(prefill_input)
        dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)
        if variant == "baseline":
            prefill_output = prefill_output / world_size
        else:
            prefill_output.div_(world_size)

        decode_output = decode_model(decode_input)
        dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)
        if variant == "baseline":
            decode_output = decode_output / world_size
        else:
            decode_output.div_(world_size)
    return prefill_output, decode_output


def _init_distributed() -> tuple[int, int, torch.device]:
    require_min_gpus(WORLD_SIZE)
    if not dist.is_available():
        raise RuntimeError("SKIPPED: torch.distributed is unavailable")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if name not in os.environ:
            raise RuntimeError("SKIPPED: disaggregated worker requires torchrun rank context")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=120),
            device_id=local_rank,
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != WORLD_SIZE:
        raise RuntimeError(
            f"SKIPPED: disaggregated worker requires exactly {WORLD_SIZE} ranks, "
            f"got {world_size}"
        )
    return rank, world_size, torch.device("cuda", local_rank)


def run_worker(variant: str, *, iterations: int, warmup: int) -> None:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    rank, world_size, device = _init_distributed()
    contract = make_result_contract(
        variant=variant,
        world_size=world_size,
        batch_size=BATCH_SIZE,
        prefill_len=PREFILL_LEN,
        hidden_dim=HIDDEN_DIM,
        iterations=iterations,
        warmup=warmup,
        seed=SEED,
    )
    try:
        prefill_model, decode_model = _make_models(
            variant,
            hidden_dim=contract.hidden_dim,
            device=device,
            seed=contract.seed,
            wrap_ddp=True,
        )
        prefill_input, decode_input = _make_inputs(contract, device)
        reference_output = _reference_output(
            prefill_model,
            decode_model,
            prefill_input,
            decode_input,
        )

        timed_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        for _ in range(warmup):
            timed_outputs = _run_iteration(
                variant,
                prefill_model,
                decode_model,
                prefill_input,
                decode_input,
                world_size=world_size,
            )
        torch.cuda.synchronize(device)
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        profile_range = (
            BASELINE_NVTX_RANGE if variant == "baseline" else OPTIMIZED_NVTX_RANGE
        )
        with nvtx_range(profile_range, enable=True):
            for _ in range(iterations):
                timed_outputs = _run_iteration(
                    variant,
                    prefill_model,
                    decode_model,
                    prefill_input,
                    decode_input,
                    world_size=world_size,
                )
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        time_per_iter_ms = float(elapsed.item()) / iterations
        if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
            raise RuntimeError("Disaggregated worker timing must be finite and positive")
        if timed_outputs is None:
            raise RuntimeError("Disaggregated worker produced no timed output")
        # Materialize the verification view after timing, matching the single-GPU
        # pair's capture path while preserving both actual final phase outputs.
        timed_output = torch.cat(timed_outputs, dim=1)

        write_disaggregated_child_result(
            contract=contract,
            rank=rank,
            prefill_input=prefill_input,
            decode_input=decode_input,
            reference_output=reference_output,
            timed_output=timed_output,
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
    parser.add_argument("--iterations", type=int, default=10)
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
