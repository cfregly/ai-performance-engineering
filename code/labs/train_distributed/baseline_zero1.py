"""Baseline ZeRO-1: manually shard optimizer state and broadcast parameters."""

from __future__ import annotations

import argparse
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.utils import get
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


class OptimizerStateSharder:
    """Only keep a slice of optimizer states per rank (ZeRO-1)."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.params: list[torch.nn.Parameter] = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self._partition_parameters()
        self.communication_time = 0.0
        self.step_time = 0.0

    def _partition_parameters(self):
        world_size = get("ws")
        rank = get("rank")
        shard_size = len(self.params) // world_size
        remainder = len(self.params) % world_size

        start = rank * shard_size + min(rank, remainder)
        end = start + shard_size + (1 if rank < remainder else 0)

        self.local_indices = list(range(start, end))
        self.local_params = {self.params[i] for i in self.local_indices}

        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p in self.local_params]

    def step(self, closure=None):
        step_start = time.perf_counter()
        comm_start = step_start

        # Gradient all-reduce across all ranks.
        for param in self.params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= get("ws")

        torch.cuda.synchronize()
        self.communication_time += time.perf_counter() - comm_start

        # Optimizer step on local shard only.
        self.optimizer.step(closure)

        # Broadcast updated params to keep replicas in sync.
        shard_size = len(self.params) // get("ws")
        remainder = len(self.params) % get("ws")
        for idx, param in enumerate(self.params):
            if idx < (shard_size + 1) * remainder:
                owner = idx // (shard_size + 1)
            else:
                owner = (idx - remainder) // shard_size
            dist.broadcast(param.data, src=owner)

        torch.cuda.synchronize()
        self.step_time += time.perf_counter() - step_start

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


def _build_model(hidden_size: int, device):
    layers: Iterable[nn.Module] = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(hidden_size, hidden_size))
    model = nn.Sequential(*layers).to(device)
    return model


def _build_adamw(params) -> AdamW:
    return AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)


def run_training(model, optimizer, batch_size: int, device, steps: int, label: str):
    rank = get("rank")
    x = torch.empty(batch_size, model[0].in_features, device=device)
    y = torch.empty_like(x)

    # Warmup step to avoid counting setup overhead.
    optimizer.zero_grad()
    x.normal_()
    y.normal_()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    if rank == 0:
        print_memory_stats(f"{label} - after warmup", model, optimizer, rank, device)
    dist.barrier()

    peak_memories = []
    for step in range(steps):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()

        x.normal_()
        y.normal_()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_memories.append(peak)
        if rank == 0 and step == 0:
            print(f"[{label}] peak memory first step: {peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"[{label}] max peak memory over {steps} steps: {max(peak_memories):.2f} MB")
    return max(peak_memories)


def main():
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: baseline ZeRO-1 demo is running on a single GPU; sharding benefits require world_size>=2.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")
    # Baseline: full optimizer state on every rank.
    base_model = _build_model(args.hidden_size, device)
    baseline_opt = _build_adamw(base_model.parameters())
    mem_baseline = run_training(
        base_model, baseline_opt, args.batch_size, device, args.steps, label="baseline-adam"
    )

    # ZeRO-1 style optimizer state partitioning.
    sharded_model = _build_model(args.hidden_size, device)
    sharded_opt = OptimizerStateSharder(_build_adamw(sharded_model.parameters()))
    mem_zero1 = run_training(
        sharded_model, sharded_opt, args.batch_size, device, args.steps, label="zero1"
    )

    if rank == 0:
        saved = mem_baseline - mem_zero1
        pct = (saved / mem_baseline * 100) if mem_baseline > 0 else 0
        print(f"[summary] baseline peak: {mem_baseline:.2f} MB | zero1 peak: {mem_zero1:.2f} MB "
              f"({saved:.2f} MB saved, {pct:.1f}% reduction)")

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero1.py",
        base_args=["--mode", "baseline", "--variant", "single", "--batch-size", "16", "--hidden-size", "10000"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:zero1",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="baseline_zero1",
    )
