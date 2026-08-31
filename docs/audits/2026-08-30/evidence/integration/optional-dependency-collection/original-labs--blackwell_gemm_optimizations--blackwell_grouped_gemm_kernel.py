"""Triton kernels for the Blackwell grouped-GEMM optimization lab."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_autotune import (
    FULL_STACK_AUTOTUNE_CONFIGS,
    KernelSchedule,
)


@triton.jit
def grouped_gemm_batched_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    route_w_ptr,
    counts_ptr,
    num_experts,
    max_rows,
    n_dim,
    k_dim,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_wb,
    stride_wm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FUSE_WEIGHTS: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_id = tl.program_id(1)

    num_tiles_m = tl.cdiv(max_rows, BLOCK_M)
    num_tiles_n = tl.cdiv(n_dim, BLOCK_N)

    group_id = pid // (GROUP_M * num_tiles_n)
    first_tile_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_tiles_m - first_tile_m, GROUP_M)
    tile_in_group = pid - group_id * (GROUP_M * num_tiles_n)
    pid_m = first_tile_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    valid_rows = tl.load(counts_ptr + expert_id)
    # Skip fully-invalid tiles (see grouped_gemm_autotune_kernel): every row past the
    # expert's token count is padding, so the whole MMA + store is wasted work. Saves
    # the all-padding tiles a skewed histogram leaves in the max_rows-sized grid.
    if pid_m * BLOCK_M >= valid_rows:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + expert_id * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_dim, BLOCK_K):
        mask_a = (offs_m[:, None] < valid_rows) & (offs_k[None, :] < k_dim)
        mask_b = (offs_k[:, None] < k_dim) & (offs_n[None, :] < n_dim)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if FUSE_WEIGHTS:
        route_weights = tl.load(
            route_w_ptr + expert_id * stride_wb + offs_m * stride_wm,
            mask=offs_m < valid_rows,
            other=0.0,
        ).to(tl.float32)
        acc *= route_weights[:, None]

    c_ptrs = (
        c_ptr
        + expert_id * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    mask_c = (offs_m[:, None] < valid_rows) & (offs_n[None, :] < n_dim)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.autotune(
    configs=FULL_STACK_AUTOTUNE_CONFIGS,
    key=["max_rows", "n_dim", "k_dim"],
)
@triton.jit
def grouped_gemm_autotune_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    route_w_ptr,
    counts_ptr,
    num_experts,
    max_rows,
    n_dim,
    k_dim,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_wb,
    stride_wm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_id = tl.program_id(1)

    num_tiles_m = tl.cdiv(max_rows, BLOCK_M)
    num_tiles_n = tl.cdiv(n_dim, BLOCK_N)

    group_id = pid // (GROUP_M * num_tiles_n)
    first_tile_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_tiles_m - first_tile_m, GROUP_M)
    tile_in_group = pid - group_id * (GROUP_M * num_tiles_n)
    pid_m = first_tile_m + (tile_in_group % group_size_m)
    pid_n = tile_in_group // group_size_m

    valid_rows = tl.load(counts_ptr + expert_id)
    # Skip fully-invalid tiles: when this tile's first row is already past the
    # expert's token count, every row is padding, so the whole MMA + store is wasted
    # work. Early-return saves the entire tile's compute. This is the grouped-GEMM's
    # core win over a naive padded batched GEMM (which cannot skip padding rows): the
    # launched grid is sized to the busiest expert (max_rows), so a skewed histogram
    # leaves many all-padding tiles that the old kernel still ran the full GEMM on.
    # No-op for the balanced histogram (every expert fills max_rows, no padding tile).
    if pid_m * BLOCK_M >= valid_rows:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + expert_id * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Full-tile fast path: when every row (M) and column (N) of this output tile is
    # valid, skip the per-iteration 2D boundary masks (Triton predicated loads + the
    # mask compute), the dominant overhead vs cuBLAS-batched on the common full
    # (balanced) tiles. Partial edge tiles keep the masked path. Same idea as the
    # ch07 TMA-copy full-tile fast path.
    tile_m_full = (pid_m + 1) * BLOCK_M <= valid_rows
    tile_n_full = (pid_n + 1) * BLOCK_N <= n_dim
    if tile_m_full and tile_n_full:
        k_main = (k_dim // BLOCK_K) * BLOCK_K
        for _ in range(0, k_main, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        if k_main < k_dim:
            k_tail = (k_main + tl.arange(0, BLOCK_K)) < k_dim
            a = tl.load(a_ptrs, mask=k_tail[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_tail[:, None], other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)
    else:
        for _ in range(0, k_dim, BLOCK_K):
            mask_a = (offs_m[:, None] < valid_rows) & (offs_k[None, :] < k_dim)
            mask_b = (offs_k[:, None] < k_dim) & (offs_n[None, :] < n_dim)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32)
            offs_k += BLOCK_K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    route_weights = tl.load(
        route_w_ptr + expert_id * stride_wb + offs_m * stride_wm,
        mask=offs_m < valid_rows,
        other=0.0,
    ).to(tl.float32)
    acc *= route_weights[:, None]

    c_ptrs = (
        c_ptr
        + expert_id * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    mask_c = (offs_m[:, None] < valid_rows) & (offs_n[None, :] < n_dim)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def grouped_gemm_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    route_w_ptr,
    counts_ptr,
    num_experts,
    max_rows,
    n_dim,
    k_dim,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_wb,
    stride_wm,
    num_tiles_m,
    num_tiles_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_expert = num_tiles_m * num_tiles_n
    total_tiles = num_experts * tiles_per_expert

    for tile_id in range(pid, total_tiles, NUM_SMS):
        expert_id = tile_id // tiles_per_expert
        tile_in_expert = tile_id - expert_id * tiles_per_expert

        group_id = tile_in_expert // (GROUP_M * num_tiles_n)
        first_tile_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_tiles_m - first_tile_m, GROUP_M)
        tile_in_group = tile_in_expert - group_id * (GROUP_M * num_tiles_n)
        pid_m = first_tile_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m

        valid_rows = tl.load(counts_ptr + expert_id)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = (
            a_ptr
            + expert_id * stride_ab
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + expert_id * stride_bb
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Skip the GEMM on fully-invalid (all-padding) tiles (see the autotune kernel).
        # This Triton rejects `continue` in the persistent tile loop, so guard the
        # K-loop instead: on a skipped tile acc stays zero and the masked route-load +
        # masked store below are no-ops, so only the wasted MMA is elided.
        if pid_m * BLOCK_M < valid_rows:
            for _ in range(0, k_dim, BLOCK_K):
                mask_a = (offs_m[:, None] < valid_rows) & (offs_k[None, :] < k_dim)
                mask_b = (offs_k[:, None] < k_dim) & (offs_n[None, :] < n_dim)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                acc += tl.dot(a, b, out_dtype=tl.float32)
                offs_k += BLOCK_K
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk

        route_weights = tl.load(
            route_w_ptr + expert_id * stride_wb + offs_m * stride_wm,
            mask=offs_m < valid_rows,
            other=0.0,
        ).to(tl.float32)
        acc *= route_weights[:, None]

        c_ptrs = (
            c_ptr
            + expert_id * stride_cb
            + offs_m[:, None] * stride_cm
            + offs_n[None, :] * stride_cn
        )
        mask_c = (offs_m[:, None] < valid_rows) & (offs_n[None, :] < n_dim)
        tl.store(c_ptrs, acc, mask=mask_c)


def _validate_grouped_inputs(
    packed_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    route_weights: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if packed_tokens.ndim != 3 or expert_weights.ndim != 3:
        raise ValueError("Grouped GEMM kernels expect 3D batched tensors")
    if route_weights.ndim != 2:
        raise ValueError("route_weights must be 2D (experts, max_rows)")
    if counts.ndim != 1:
        raise ValueError("counts must be 1D")
    if packed_tokens.shape[0] != expert_weights.shape[0]:
        raise ValueError("Expert batch dimension mismatch")
    if packed_tokens.shape[0] != route_weights.shape[0]:
        raise ValueError("Route-weight batch dimension mismatch")
    if packed_tokens.shape[0] != counts.shape[0]:
        raise ValueError("Counts batch dimension mismatch")
    if packed_tokens.shape[2] != expert_weights.shape[1]:
        raise ValueError("Inner K dimension mismatch")
    if out.shape != (packed_tokens.shape[0], packed_tokens.shape[1], expert_weights.shape[2]):
        raise ValueError("Output buffer shape mismatch")
    if not packed_tokens.is_contiguous():
        raise ValueError("packed_tokens must be contiguous")
    if not expert_weights.is_contiguous():
        raise ValueError("expert_weights must be contiguous")
    if not route_weights.is_contiguous():
        raise ValueError("route_weights must be contiguous")
    if not counts.is_contiguous():
        raise ValueError("counts must be contiguous")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")


def launch_grouped_gemm_standard(
    packed_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    route_weights: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
    schedule: KernelSchedule,
    route_weight_factors: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_grouped_inputs(packed_tokens, expert_weights, route_weights, counts, out)
    out.zero_()
    num_experts, max_rows, k_dim = packed_tokens.shape
    n_dim = expert_weights.shape[2]
    grid = (
        triton.cdiv(max_rows, schedule.block_m) * triton.cdiv(n_dim, schedule.block_n),
        num_experts,
    )
    grouped_gemm_batched_kernel[grid](
        packed_tokens,
        expert_weights,
        out,
        route_weights,
        counts,
        num_experts,
        max_rows,
        n_dim,
        k_dim,
        packed_tokens.stride(0),
        packed_tokens.stride(1),
        packed_tokens.stride(2),
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        route_weights.stride(0),
        route_weights.stride(1),
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_K=schedule.block_k,
        GROUP_M=schedule.group_m,
        FUSE_WEIGHTS=schedule.fused_weights,
        num_warps=schedule.num_warps,
        num_stages=schedule.num_stages,
    )
    if not schedule.fused_weights:
        if route_weight_factors is None:
            route_weight_factors = route_weights.unsqueeze(-1).to(out.dtype)
        out.mul_(route_weight_factors)
    return out


def launch_grouped_gemm_autotuned(
    packed_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    route_weights: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _validate_grouped_inputs(packed_tokens, expert_weights, route_weights, counts, out)
    out.zero_()
    num_experts, max_rows, k_dim = packed_tokens.shape
    n_dim = expert_weights.shape[2]
    grid = lambda meta: (
        triton.cdiv(max_rows, meta["BLOCK_M"]) * triton.cdiv(n_dim, meta["BLOCK_N"]),
        num_experts,
    )
    grouped_gemm_autotune_kernel[grid](
        packed_tokens,
        expert_weights,
        out,
        route_weights,
        counts,
        num_experts,
        max_rows,
        n_dim,
        k_dim,
        packed_tokens.stride(0),
        packed_tokens.stride(1),
        packed_tokens.stride(2),
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        route_weights.stride(0),
        route_weights.stride(1),
    )
    return out


def launch_grouped_gemm_persistent(
    packed_tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    route_weights: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
    schedule: KernelSchedule,
    *,
    num_sms: int | None = None,
) -> torch.Tensor:
    _validate_grouped_inputs(packed_tokens, expert_weights, route_weights, counts, out)
    out.zero_()
    num_experts, max_rows, k_dim = packed_tokens.shape
    n_dim = expert_weights.shape[2]
    if num_sms is None:
        props = torch.cuda.get_device_properties(packed_tokens.device)
        num_sms = int(props.multi_processor_count)
    grid = (int(num_sms),)
    grouped_gemm_persistent_kernel[grid](
        packed_tokens,
        expert_weights,
        out,
        route_weights,
        counts,
        num_experts,
        max_rows,
        n_dim,
        k_dim,
        packed_tokens.stride(0),
        packed_tokens.stride(1),
        packed_tokens.stride(2),
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        route_weights.stride(0),
        route_weights.stride(1),
        triton.cdiv(max_rows, schedule.block_m),
        triton.cdiv(n_dim, schedule.block_n),
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_K=schedule.block_k,
        GROUP_M=schedule.group_m,
        NUM_SMS=int(num_sms),
        num_warps=schedule.num_warps,
        num_stages=schedule.num_stages,
    )
    return out
