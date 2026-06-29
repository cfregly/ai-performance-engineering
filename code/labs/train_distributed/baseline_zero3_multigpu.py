"""Baseline ZeRO-3: shard params, grads, and optimizer state."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from core.benchmark.gpu_requirements import require_min_gpus
from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.utils import get
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


class ParamShard:
    """Track per-parameter shards and materialize full weights as needed."""

    def __init__(self, param: torch.nn.Parameter):
        self.param = param
        self.shard_dim = 0
        self.rank = get("rank")
        self.world_size = get("ws")
        self.full_data = torch.empty(
            param.data.shape,
            device=param.data.device,
            dtype=param.data.dtype,
        )

        shards = param.data.chunk(self.world_size, dim=self.shard_dim)
        local_shard = shards[self.rank].contiguous()
        self.local_shard = local_shard
        self.local_grad = torch.empty_like(local_shard)
        self.param.data = self.local_shard

    def all_gather(self):
        dist.all_gather_into_tensor(self.full_data, self.local_shard)
        self.param.data = self.full_data

    def drop_full(self):
        self.param.data = self.local_shard
        if self.param.grad is not None and self.param.grad.shape != self.local_shard.shape:
            grad_shards = self.param.grad.data.chunk(self.world_size, dim=self.shard_dim)
            self.local_grad.copy_(grad_shards[self.rank])
            self.param.grad.data = self.local_grad


def attach_zero3_hooks(model, shard_map):
    """Materialize full params for forward/backward then release."""

    def pre_hook(module, inputs):
        for _, param in module.named_parameters(recurse=False):
            manager = shard_map.get(param)
            if manager:
                manager.all_gather()

    def post_hook(module, inputs, outputs):
        for _, param in module.named_parameters(recurse=False):
            manager = shard_map.get(param)
            if manager:
                manager.drop_full()

    for module in model.modules():
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)
        module.register_full_backward_pre_hook(pre_hook)
        module.register_full_backward_hook(post_hook)


class Zero3Optimizer:
    """Optimizer that keeps only shards of params/grads/state on each rank."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        self.shards = {p: ParamShard(p) for p in self.params}
        self._filter_param_groups()
        self.communication_time = 0.0
        self.step_time = 0.0

    def _filter_param_groups(self):
        local_params = {p for p, shard in self.shards.items() if shard.rank == get("rank")}
        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p in local_params]

    def step(self, closure=None):
        step_start = time.perf_counter()
        comm_start = step_start

        # Reduce gradients shard-by-shard.
        for param, shard in self.shards.items():
            grad = param.grad
            if grad is None:
                continue
            if grad.shape != param.data.shape:
                grad_shards = grad.data.chunk(shard.world_size, dim=shard.shard_dim)
                param.grad = grad_shards[shard.rank].contiguous()
                grad = param.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= get("ws")

        torch.cuda.synchronize()
        self.communication_time += time.perf_counter() - comm_start

        self.optimizer.step(closure)
        torch.cuda.synchronize()
        self.step_time += time.perf_counter() - step_start

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


def _build_model(hidden_size: int, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def _build_adamw(params) -> AdamW:
    return AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05, foreach=False)


def train(model, optimizer, batch_size, device, steps, label, shard_map=None):
    rank = get("rank")
    input_dim = model[0].in_features
    x = torch.empty(batch_size, input_dim, device=device)
    y = torch.empty_like(x)

    if shard_map:
        attach_zero3_hooks(model, shard_map)

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
    require_min_gpus(2, script_name="baseline_zero3_multigpu.py")
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: baseline ZeRO-3 demo is running on a single GPU; sharding benefits require world_size>=2.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")

    baseline_model = _build_model(args.hidden_size, device)
    baseline_opt = _build_adamw(baseline_model.parameters())
    baseline_peak = train(baseline_model, baseline_opt, args.batch_size, device, args.steps, "baseline-adam")

    zero3_model = _build_model(args.hidden_size, device)
    zero3_opt = Zero3Optimizer(_build_adamw(zero3_model.parameters()))
    peak_zero3 = train(
        zero3_model, zero3_opt, args.batch_size, device, args.steps, "zero3", shard_map=zero3_opt.shards
    )

    if rank == 0:
        saved = baseline_peak - peak_zero3
        pct = (saved / baseline_peak * 100) if baseline_peak > 0 else 0
        print(f"[summary] baseline peak: {baseline_peak:.2f} MB | zero3 peak: {peak_zero3:.2f} MB "
              f"({saved:.2f} MB saved, {pct:.1f}% reduction)")

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero3.py",
        base_args=["--mode", "baseline", "--variant", "multigpu", "--batch-size", "16", "--hidden-size", "10000"],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:zero3_multigpu",
        default_nproc_per_node=None,
        name="baseline_zero3_multigpu",
    )
