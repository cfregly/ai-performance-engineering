"""Optimized ZeRO-2: overlap reduce-scatter gradients with sharded optimizer states."""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from core.benchmark.gpu_requirements import require_min_gpus
from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.training_utils.utils import get


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--extra-grad-mb",
        type=int,
        default=0,
        help="Extra gradient payload size (MB) to amplify communication.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the DDP module.")
    return parser.parse_args()


def _build_model(hidden_size: int, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def _optimizer_cfg(params, lr):
    kwargs = dict(
        optimizer_class=torch.optim.AdamW,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    try:
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            kwargs["fused"] = True
    except (TypeError, ValueError):
        pass
    return kwargs


def _completed_future(tensor: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


def _reduce_scatter_allgather_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Reduce-scatter + all-gather to keep DDP bucket shape intact."""
    group = process_group if process_group is not None else dist.group.WORLD
    world_size = group.size()
    buffer = bucket.buffer()
    if world_size <= 1:
        return _completed_future(buffer)

    # Average before collectives to match DDP allreduce semantics.
    buffer.div_(world_size)
    flat = buffer
    numel = flat.numel()
    shard_size = (numel + world_size - 1) // world_size
    padded = None
    if numel % world_size != 0:
        padded = flat.new_zeros(shard_size * world_size)
        padded[:numel].copy_(flat)
        flat = padded

    input_2d = flat.view(world_size, shard_size)
    output = flat.new_empty(shard_size)
    work = dist.reduce_scatter_tensor(output, input_2d, group=group, async_op=True)

    def _allgather(fut: torch.futures.Future) -> torch.Tensor:
        shard = fut.value()[0]
        gathered = shard.new_empty(shard_size * world_size)
        dist.all_gather_into_tensor(gathered, shard, group=group)
        if padded is not None:
            gathered = gathered[:numel]
        buffer.copy_(gathered)
        return buffer

    return work.get_future().then(_allgather)


_reduce_scatter_allgather_hook.__annotations__["bucket"] = dist.GradBucket
_reduce_scatter_allgather_hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]


def main():
    require_min_gpus(2, script_name="optimized_zero2_multigpu.py")
    args = parse_args()
    try:
        local_rank = get("lrank")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=local_rank)
        if get("ws") < 2:
            print("Warning: Optimized ZeRO-2 running with world_size < 2; no sharding benefit will be observed.")
        rank = get("rank")
        device = torch.device(f"cuda:{local_rank}")

        model = _build_model(args.hidden_size, device).to(torch.bfloat16)
        extra_param = None
        if args.extra_grad_mb > 0:
            elem_bytes = torch.finfo(torch.bfloat16).bits // 8
            numel = (args.extra_grad_mb * 1024 * 1024) // elem_bytes
            extra_param = torch.nn.Parameter(torch.zeros(numel, device=device, dtype=torch.bfloat16))
            model.register_parameter("extra_grad_payload", extra_param)
        ddp_model = DDP(
            model,
            device_ids=[device],
            static_graph=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
        )
        if rank == 0:
            print("Using custom reduce-scatter hook for ZeRO-2.")
        ddp_model.register_comm_hook(state=dist.group.WORLD, hook=_reduce_scatter_allgather_hook)

        if args.compile:
            ddp_model = torch.compile(ddp_model, mode="reduce-overhead")

        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            overlap_with_ddp=True,
            parameters_as_bucket_view=True,
            **_optimizer_cfg(ddp_model.parameters(), args.learning_rate),
        )

        grad_clip = 1.0
        x = torch.empty(args.batch_size, args.hidden_size, device=device)
        y = torch.empty_like(x)

        # Warmup
        x.normal_()
        y.normal_()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            warm_loss = nn.functional.mse_loss(ddp_model(x), y)
        if extra_param is not None:
            warm_loss = warm_loss + extra_param.sum() * 0.0
        warm_loss.backward()
        optimizer.step()

        if rank == 0:
            print_memory_stats("optimized-zero2 warmup", ddp_model, optimizer, rank, device)
        dist.barrier()
        torch.cuda.synchronize(device)

        total_tokens = 0
        start = perf_counter()
        loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(args.grad_accum):
                x.normal_()
                y.normal_()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = nn.functional.mse_loss(ddp_model(x), y) / args.grad_accum
                if extra_param is not None:
                    loss = loss + extra_param.sum() * 0.0
                loss.backward()
                total_tokens += x.numel()

            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                elapsed = perf_counter() - start
                toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                loss_value_buffer[0].copy_(loss.detach())
                loss_value = float(loss_value_buffer.detach().cpu()[0])
                print(
                    f"[optimized-zero2] step {step}/{args.steps} "
                    f"loss={loss_value:.4f} "
                    f"tokens/s per rank={toks_per_sec:,.0f}"
                )

        torch.cuda.synchronize(device)
        total_time = perf_counter() - start
        if rank == 0:
            toks_per_sec = total_tokens / total_time
            print(f"[optimized-zero2] finished {args.steps} steps | {toks_per_sec:,.0f} toks/s per rank")

        dist.destroy_process_group()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero2.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "multigpu",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--grad-accum",
            "1",
            "--extra-grad-mb",
            "12288",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:zero2_multigpu",
        default_nproc_per_node=None,
        name="optimized_zero2_multigpu",
    )
