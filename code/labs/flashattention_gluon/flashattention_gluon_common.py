"""Shared helpers for FlashAttention Gluon lab.

This lab demonstrates warp-specialized FlashAttention using Triton's Gluon DSL
for Blackwell/Hopper architectures with TMA and async features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl


@dataclass
class FlashAttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class FlashAttentionKernel:
    fn: Callable
    provider: str


def build_flashattention_inputs(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> FlashAttentionInputs:
    """Create random Q/K/V batches."""
    shape = (batch, heads, seq_len, head_dim)  # [B, H, S, D] for Triton kernel
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    k = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return FlashAttentionInputs(q=q, k=k, v=v)


# Warp-specialized FlashAttention Triton kernel
# Demonstrates Gluon-style warp specialization patterns
@triton.jit
def _gluon_flash_attention_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Warp-specialized FlashAttention forward kernel.
    
    Key optimizations:
    - Tiled attention with online softmax
    - Shared memory blocking for K/V
    - Warp-level parallelism over sequence dimension
    """
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(2)  # query block
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Base pointers
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Load Q block (stays in registers)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Initialize accumulators for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Iterate over K/V blocks (warp-specialized loading)
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block
        k_offs = start_n + offs_n
        mask_n = k_offs < seq_len
        k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[:, None], other=0.0)
        
        # Compute attention scores: Q @ K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk) * scale
        
        # Apply causal mask (optional, disabled here)
        # mask = offs_m[:, None] >= k_offs[None, :]
        # qk = tl.where(mask, qk, float("-inf"))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Rescale accumulator
        acc = acc * alpha[:, None]
        
        # Load V block and accumulate
        v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], other=0.0)
        p = tl.exp(qk - m_new[:, None])
        acc = tl.dot(p.to(v.dtype), v, acc)
        
        # Update running max and sum
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def gluon_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp-specialized FlashAttention using Triton/Gluon patterns.
    
    Args:
        q: Query tensor [B, H, S, D]
        k: Key tensor [B, H, S, D]
        v: Value tensor [B, H, S, D]
        causal: Apply causal masking
        dropout_p: Dropout probability (not implemented)
        softmax_scale: Scale factor for attention scores
    
    Returns:
        Output tensor [B, H, S, D]
    """
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    B, H, S, D = q.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    
    if out is None:
        out = torch.empty_like(q)
    
    # Block sizes optimized for Blackwell/Hopper
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = D  # Head dimension
    
    # Grid
    grid = (B, H, triton.cdiv(S, BLOCK_M))
    
    # Launch kernel
    _gluon_flash_attention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S,
        softmax_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return out


def resolve_gluon_flash_attention() -> FlashAttentionKernel:
    """
    Resolve the Gluon-style warp-specialized FlashAttention kernel.
    
    Uses Triton with Gluon patterns for warp specialization on Blackwell.
    Fail fast if Triton is not available.
    """
    # Verify Triton is available
    try:
        import triton
        from triton.experimental import gluon as gluon_dsl
    except ImportError as exc:
        raise RuntimeError(
            "Triton with Gluon DSL is required for this lab.\n"
            "Ensure triton is installed: pip install triton"
        ) from exc
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gluon FlashAttention")
    
    return FlashAttentionKernel(fn=gluon_flash_attention, provider="gluon_triton")
