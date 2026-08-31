"""Schedule registry for the Blackwell grouped-GEMM optimization lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import triton


@dataclass(frozen=True)
class KernelSchedule:
    name: str
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int
    fused_weights: bool
    persistent: bool
    note: str


VARIANT_SCHEDULES: Dict[str, KernelSchedule] = {
    "baseline": KernelSchedule(
        name="baseline",
        block_m=64,
        block_n=64,
        block_k=32,
        group_m=1,
        num_warps=4,
        num_stages=2,
        fused_weights=False,
        persistent=False,
        note="Step 0: small tiles with a one-block-per-tile launch model.",
    ),
    "large_tiles": KernelSchedule(
        name="large_tiles",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=1,
        num_warps=8,
        num_stages=3,
        fused_weights=False,
        persistent=False,
        note="Step 1: larger tiles keep the tensor cores fed.",
    ),
    "full_stack": KernelSchedule(
        name="full_stack",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=4,
        num_warps=8,
        num_stages=4,
        fused_weights=True,
        persistent=False,
        note="Step 3: autotune + fused route weights + GROUP_SIZE_M swizzle.",
    ),
    "persistent": KernelSchedule(
        name="persistent",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=4,
        num_warps=8,
        num_stages=4,
        fused_weights=True,
        persistent=True,
        note="Step 4: persistent tile residency with a strided tile loop.",
    ),
}


EXPERIMENTAL_SCHEDULES: Dict[str, KernelSchedule] = {
    "fast_math_control": KernelSchedule(
        name="fast_math_control",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=1,
        num_warps=8,
        num_stages=3,
        fused_weights=False,
        persistent=False,
        note="Control probe: mirrors Step 1 because the Triton path has no separate fast-math toggle.",
    ),
    "latency10": KernelSchedule(
        name="latency10",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=4,
        num_warps=8,
        num_stages=6,
        fused_weights=True,
        persistent=False,
        note="Negative control: deeper buffering can consume SMEM without hiding more latency.",
    ),
    "two_cta": KernelSchedule(
        name="two_cta",
        block_m=128,
        block_n=128,
        block_k=64,
        group_m=4,
        num_warps=16,
        num_stages=4,
        fused_weights=True,
        persistent=False,
        note="Negative control: more resident warps are not a reliable win for this workload.",
    ),
    "tile_n256": KernelSchedule(
        name="tile_n256",
        block_m=128,
        block_n=256,
        block_k=64,
        group_m=4,
        num_warps=8,
        num_stages=4,
        fused_weights=True,
        persistent=False,
        note="Negative control: larger N-tiles increase working-set pressure.",
    ),
}


FULL_STACK_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8,
        num_stages=6,
    ),
]


def resolve_schedule(name: str, *, allow_experimental: bool = True) -> KernelSchedule:
    if name in VARIANT_SCHEDULES:
        return VARIANT_SCHEDULES[name]
    if allow_experimental and name in EXPERIMENTAL_SCHEDULES:
        return EXPERIMENTAL_SCHEDULES[name]
    raise KeyError(f"Unknown grouped GEMM schedule '{name}'")


def public_variant_names() -> Iterable[str]:
    return VARIANT_SCHEDULES.keys()


def experimental_variant_names() -> Iterable[str]:
    return EXPERIMENTAL_SCHEDULES.keys()
