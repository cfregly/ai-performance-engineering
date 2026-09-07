"""Shared gradient-accumulation scheduling for the optimized DDP examples."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradientAccumulationStep:
    """Scaling and synchronization policy for one consumed microbatch."""

    group_size: int
    should_step: bool


def validate_gradient_accumulation(total_microbatches: int, grad_accum: int) -> None:
    """Reject nonpositive microbatch and accumulation counts."""

    if total_microbatches <= 0:
        raise ValueError("steps must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")


def ddp_static_graph_enabled(grad_accum: int) -> bool:
    """Use DDP's static reducer only when every backward synchronizes.

    In the pinned PyTorch 2.9.1 runtime, entering ``no_sync`` on the first
    static-graph backward leaves the reducer's autograd-hook state inconsistent
    (pytorch/pytorch#143580). The default no-accumulation path retains the
    optimization. Re-enable it for accumulation after an upgraded runtime passes
    the first-no-sync, partial-group, and dense-reference regression cases.
    """

    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    return grad_accum == 1


def build_gradient_accumulation_plan(
    total_microbatches: int,
    grad_accum: int,
) -> tuple[GradientAccumulationStep, ...]:
    """Build full and trailing partial groups with their actual divisors."""

    validate_gradient_accumulation(total_microbatches, grad_accum)
    plan: list[GradientAccumulationStep] = []
    for group_start in range(0, total_microbatches, grad_accum):
        group_size = min(grad_accum, total_microbatches - group_start)
        plan.extend(
            GradientAccumulationStep(
                group_size=group_size,
                should_step=offset == group_size - 1,
            )
            for offset in range(group_size)
        )
    return tuple(plan)


def gradient_sync_context(
    model: Any,
    accumulation: GradientAccumulationStep,
    *,
    distributed: bool,
) -> AbstractContextManager[None]:
    """Suppress DDP reduction until the optimizer-step microbatch."""

    if distributed and not accumulation.should_step:
        return model.no_sync()
    return nullcontext()


__all__ = [
    "GradientAccumulationStep",
    "build_gradient_accumulation_plan",
    "ddp_static_graph_enabled",
    "gradient_sync_context",
    "validate_gradient_accumulation",
]
