"""Launch context for the communicator reinitialization benchmark pair."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ch04.distributed_helper import setup_single_gpu_env
from core.common.device_utils import resolve_local_rank


@dataclass(frozen=True)
class ReinitCommLaunchContext:
    """Describe either a real launcher context or the standalone rank-zero case."""

    rank: int
    world_size: int
    local_rank: int
    standalone: bool


def resolve_reinit_comm_launch(example_name: str) -> ReinitCommLaunchContext:
    """Resolve torchrun when present; otherwise require one local CUDA rank."""
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        rank, world_size, local_rank = setup_single_gpu_env(
            example_name,
            min_world_size=1,
        )
        return ReinitCommLaunchContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            standalone=False,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(f"SKIPPED: {example_name} requires CUDA")
    local_rank = resolve_local_rank()
    available = torch.cuda.device_count()
    if local_rank not in range(available):
        raise RuntimeError(
            f"SKIPPED: {example_name} requested local_rank={local_rank} "
            f"but only {available} GPU(s) are visible"
        )
    return ReinitCommLaunchContext(
        rank=0,
        world_size=1,
        local_rank=local_rank,
        standalone=True,
    )


def initialize_reinit_process_group(
    context: ReinitCommLaunchContext,
    *,
    backend: str,
    device_id: torch.device | int | None,
) -> dist.Store | None:
    """Initialize the declared group and retain a standalone HashStore if used."""
    if dist.is_initialized():
        raise RuntimeError("Reinit communicator process group is already initialized")

    common: dict[str, object] = {
        "backend": backend,
        "rank": context.rank,
        "world_size": context.world_size,
        "timeout": datetime.timedelta(seconds=60),
    }
    if device_id is not None:
        common["device_id"] = device_id

    if context.standalone:
        if context.rank != 0 or context.world_size != 1:
            raise ValueError("Standalone reinit communicator must be rank 0 of world size 1")
        store = dist.HashStore()
        dist.init_process_group(store=store, **common)
        return store

    dist.init_process_group(init_method="env://", **common)
    return None
