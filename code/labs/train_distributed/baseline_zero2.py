"""Matched dense-DDP/AdamW baseline for the ZeRO optimizer comparison.

Legacy GradientSharder/train educational helpers remain available but are not
executed by main() or the harness comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.zero2_torchrun_benchmark import Zero2TorchrunBenchmark
from labs.train_distributed.training_utils.utils import get


def parse_args():
    from labs.train_distributed.zero2_common import parse_args as common_args
    return common_args()


class GradientSharder:
    """ZeRO-2 style gradient+optimizer sharding."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        self._shard_parameters()
        self.local_index_set = set(self.local_indices)
        self._reduce_inputs: dict[int, torch.Tensor] = {}
        self._shard_grads: dict[int, torch.Tensor] = {}
        world_size = get("ws")
        for idx, param in enumerate(self.params):
            self._reduce_inputs[idx] = torch.empty(
                param.numel() * world_size,
                device=param.device,
                dtype=param.dtype,
            )
            self._shard_grads[idx] = torch.empty(
                param.numel(),
                device=param.device,
                dtype=param.dtype,
            )
        self.communication_time = 0.0
        self.step_time = 0.0

    def _shard_parameters(self):
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
        world_size = get("ws")

        for idx, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue

            flattened = grad.data.contiguous().view(-1)
            in_tensor = self._reduce_inputs[idx]
            in_tensor.view(world_size, -1).copy_(flattened.unsqueeze(0))

            shard_grad = self._shard_grads[idx]
            dist.reduce_scatter_tensor(shard_grad, in_tensor, op=dist.ReduceOp.SUM)

            if idx in self.local_index_set:
                shard_grad.div_(world_size)
                param.grad = shard_grad.view_as(grad.data)
            else:
                param.grad = None

        torch.cuda.synchronize()
        self.communication_time += time.perf_counter() - comm_start

        self.optimizer.step(closure)

        shard_size = len(self.params) // world_size
        remainder = len(self.params) % world_size
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
    from labs.train_distributed.zero2_common import build_model
    return build_model(hidden_size, device)


def _build_adamw(params) -> AdamW:
    return AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)


def train(model, optimizer, batch_size, device, steps, label):
    rank = get("rank")
    input_dim = model[0].in_features
    x = torch.empty(batch_size, input_dim, device=device)
    y = torch.empty_like(x)

    optimizer.zero_grad()
    x.normal_()
    y.normal_()
    nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()
    torch.cuda.synchronize()

    if rank == 0:
        print_memory_stats(f"{label} warmup", model, optimizer, rank, device)
    dist.barrier()

    peaks = []
    for step in range(steps):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()

        x.normal_()
        y.normal_()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peaks.append(peak)
        if rank == 0 and step == 0:
            print(f"[{label}] peak memory first step: {peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"[{label}] max peak memory over {steps} steps: {max(peaks):.2f} MB")
    return max(peaks)


def main():
    from labs.train_distributed.zero2_common import run_training
    run_training(parse_args(), optimized=False, multi_gpu=False)


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return Zero2TorchrunBenchmark(
        mode="baseline",
        variant="single",
        script_path=Path(__file__).parent / "zero2.py",
        base_args=["--mode", "baseline", "--variant", "single", "--batch-size", "16", "--hidden-size", "10000", "--grad-accum", "1"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:zero2",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="baseline_zero2",
    )
