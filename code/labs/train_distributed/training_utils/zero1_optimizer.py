"""ZeRO-1 optimizer for explicit, accumulated and clipped training steps."""

from collections.abc import Iterable

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


def make_zero1_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    learning_rate: float,
    *,
    fused: bool = True,
) -> ZeroRedundancyOptimizer:
    # Hook-driven overlap performs updates from DDP's backward hook and makes
    # explicit step() calls no-ops. These examples accumulate and clip gradients
    # before calling step(), so updates must use the explicit-step API.
    return ZeroRedundancyOptimizer(
        parameters,
        optimizer_class=torch.optim.AdamW,
        overlap_with_ddp=False,
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=fused,
    )
