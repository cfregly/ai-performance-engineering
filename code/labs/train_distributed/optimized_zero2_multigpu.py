"""ZeRO-2 comparison: RS/AG gradient communication with sharded AdamW state.

DDP restores full gradients after reduce-scatter/all-gather. The sharded optimizer
runs explicitly after backward; this path does not overlap optimizer computation
with DDP and does not keep gradients sharded between steps.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from labs.train_distributed.training_utils.zero2_torchrun_benchmark import Zero2TorchrunBenchmark


def parse_args():
    from labs.train_distributed.zero2_common import parse_args as common_args
    return common_args()


def _build_model(hidden_size: int, device):
    from labs.train_distributed.zero2_common import build_model
    return build_model(hidden_size, device)


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


def _build_optimizer(params, lr):
    """Construct the sharded optimizer used by the explicit training step."""
    params = list(params)
    return ZeroRedundancyOptimizer(
        params,
        # The custom communication hook does not run a ZeRO optimizer step.
        # Enabling overlap here would turn every explicit step() into a no-op.
        overlap_with_ddp=False,
        parameters_as_bucket_view=True,
        **_optimizer_cfg(params, lr),
    )


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

    output = flat.new_empty(shard_size)
    # Use the concatenation form accepted by both Gloo and NCCL.  Passing the
    # stack form ``[world_size, shard_size]`` aborts in ProcessGroupGloo because
    # its first dimension must be ``output.size(0) * world_size``.
    if dist.get_backend(group) == "gloo":
        # Gloo's Work object does not implement get_future().  This branch is
        # exercised only by the explicitly labeled CPU correctness profile;
        # the performance benchmark remains on the asynchronous NCCL path.
        dist.reduce_scatter_tensor(output, flat, group=group)
        gathered = output.new_empty(shard_size * world_size)
        dist.all_gather_into_tensor(gathered, output, group=group)
        if padded is not None:
            gathered = gathered[:numel]
        buffer.copy_(gathered)
        return _completed_future(buffer)
    work = dist.reduce_scatter_tensor(output, flat, group=group, async_op=True)

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
    from labs.train_distributed.zero2_common import run_training
    run_training(parse_args(), optimized=True, multi_gpu=True)


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return Zero2TorchrunBenchmark(
        mode="optimized",
        variant="multigpu",
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
