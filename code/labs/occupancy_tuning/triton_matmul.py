"""Minimal Triton matmul used to capture Proton + Nsight deltas."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
from core.utils.compile_utils import enable_tf32

# arch_config patches Triton's SM-architecture handling for GB10 (sm_121a -> sm_120).
# The patch used to de-suffix every other arch too, which on GB300 rewrote
# sm_103a -> sm_103 and made the Triton 3.7 JIT abort (uncatchable "LLVM ERROR:
# Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st") — this import was
# capability-gated to GB10 as a workaround. triton_compat.py now preserves the 'a'
# suffix for all arches and clamps only sm_121, so the import is safe everywhere.
import arch_config  # noqa: F401

enable_tf32()


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # num_warps is Triton's launch meta-parameter (passed at the call site), NOT a kernel
    # parameter; it is the launch kwarg in run_one instead. (A dead `num_warps:
    # tl.constexpr` param here was once suspected as a tcgen05.wait.st abort trigger;
    # probe B in code/upstream/triton-tcgen05-wait-st/STATUS.md refuted that — the
    # abort was the sm_103a de-suffix in triton_compat.py, fixed 2026-06-11.)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


def run_one(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    *,
    bm: int = 128,
    bn: int = 128,
    bk: int = 64,
    nw: int = 4,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cuda",
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    grid: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Run the Triton matmul kernel once.

    Args:
        M/N/K: Problem sizes.
        bm/bn/bk: Tile dimensions (BLOCK_M/N/K).
        nw: Number of warps per program.
        dtype: Tensor dtype.
        device: Target device (defaults to CUDA).
        a/b/c: Optional preallocated tensors to reuse between runs.
        grid: Optional launch grid override.

    Returns:
        Output tensor `c` with shape (M, N).
    """

    dev = torch.device(device)
    if not dev.type == "cuda":  # pragma: no cover - lab expects CUDA but guard for clarity
        raise RuntimeError("Triton matmul requires a CUDA device.")

    if a is None:
        a = torch.randn((M, K), dtype=dtype, device=dev)
    if b is None:
        b = torch.randn((K, N), dtype=dtype, device=dev)
    if c is None:
        c = torch.empty((M, N), dtype=dtype, device=dev)

    launch = grid if grid is not None else (triton.cdiv(M, bm), triton.cdiv(N, bn))

    matmul_kernel[launch](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=nw,
    )
    return c


def describe_schedule(bm: int, bn: int, bk: int, nw: int) -> str:
    """Return a short label useful for Proton metadata tables."""

    return f"bm={bm},bn={bn},bk={bk},nw={nw}"


if __name__ == "__main__":
    run_one()
    torch.cuda.synchronize()
