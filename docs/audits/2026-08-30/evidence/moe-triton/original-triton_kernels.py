#!/usr/bin/env python3
"""Triton kernels for MoE optimization journey.

These kernels demonstrate manual optimizations that torch.compile does automatically.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_silu_mul_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU(gate) * up kernel.
    
    Instead of:
        gate = silu(gate)  # kernel 1
        out = gate * up    # kernel 2
    
    We do both in one kernel, eliminating memory round-trip.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load gate and up
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    
    # Fused SiLU * up: silu(x) = x * sigmoid(x)
    gate_sigmoid = tl.sigmoid(gate.to(tl.float32))
    silu_gate = gate * gate_sigmoid.to(gate.dtype)
    out = silu_gate * up
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up operation.
    
    Saves one kernel launch and one memory round-trip.
    """
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_silu_mul_kernel[grid](
        gate, up, out,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit  
def fused_expert_ffn_kernel(
    x_ptr, w1_ptr, w2_ptr, w3_ptr, out_ptr,
    M, K, N,  # M=tokens, K=hidden, N=intermediate
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2n, stride_w2k,
    stride_w3k, stride_w3n,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused expert FFN: out = (SiLU(x @ W1) * (x @ W3)) @ W2
    
    This fuses 3 matmuls + activation into fewer memory operations.
    Each block handles BLOCK_M tokens.
    """
    pid_m = tl.program_id(0)
    
    # Compute ranges
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)
    
    # Pointers
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    
    # Load x block [BLOCK_M, K]
    mask_m = rm < M
    mask_k = rk < K
    x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Accumulate gate = x @ W1 and up = x @ W3
    gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        
        # Load x slice
        x_slice = tl.load(
            x_ptr + rm[:, None] * stride_xm + k_range[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :], other=0.0
        )
        
        # Load W1 and W3 slices
        w1_slice = tl.load(
            w1_ptr + k_range[:, None] * stride_w1k + rn[None, :] * stride_w1n,
            mask=k_mask[:, None] & (rn < N)[None, :], other=0.0
        )
        w3_slice = tl.load(
            w3_ptr + k_range[:, None] * stride_w3k + rn[None, :] * stride_w3n,
            mask=k_mask[:, None] & (rn < N)[None, :], other=0.0
        )
        
        # Accumulate
        gate_acc += tl.dot(x_slice.to(tl.float32), w1_slice.to(tl.float32))
        up_acc += tl.dot(x_slice.to(tl.float32), w3_slice.to(tl.float32))
    
    # Fused SiLU * up
    gate_sigmoid = tl.sigmoid(gate_acc)
    hidden = (gate_acc * gate_sigmoid) * up_acc  # SiLU(gate) * up
    
    # Second matmul: hidden @ W2
    out_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_range = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_range < N
        
        w2_slice = tl.load(
            w2_ptr + n_range[:, None] * stride_w2n + rk[None, :] * stride_w2k,
            mask=n_mask[:, None] & mask_k[None, :], other=0.0
        )
        
        hidden_slice = hidden if n_start == 0 else tl.load(
            # Would need to store/reload hidden - simplified here
            w2_ptr, mask=False, other=0.0  # Placeholder
        )
        out_acc += tl.dot(hidden.to(tl.float32), w2_slice.to(tl.float32))
    
    # Store output
    out_ptrs = out_ptr + rm[:, None] * stride_om + rk[None, :] * stride_ok
    tl.store(out_ptrs, out_acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def grouped_gemm_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Problem sizes per group
    m_ptr, n_ptr, k_ptr,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Group info
    num_groups,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Grouped GEMM kernel - computes multiple small GEMMs efficiently.
    
    This is the pattern used by CUTLASS grouped GEMM for MoE.
    Each group is one expert's computation.
    """
    pid = tl.program_id(0)
    
    # Find which group this block belongs to
    # (Simplified - real impl would use prefix sums)
    group_id = pid % num_groups
    
    # Get problem size for this group
    M = tl.load(m_ptr + group_id)
    N = tl.load(n_ptr + group_id)
    K = tl.load(k_ptr + group_id)
    
    # Compute block indices within group
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    block_id = pid // num_groups
    block_m = block_id % num_blocks_m
    block_n = block_id // num_blocks_m
    
    # Compute output tile
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Main loop over K
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + rk
        
        # Load A and B tiles
        a = tl.load(a_ptr + rm[:, None] * stride_am + k_range[None, :] * stride_ak,
                   mask=(rm[:, None] < M) & (k_range[None, :] < K), other=0.0)
        b = tl.load(b_ptr + k_range[:, None] * stride_bk + rn[None, :] * stride_bn,
                   mask=(k_range[:, None] < K) & (rn[None, :] < N), other=0.0)
        
        acc += tl.dot(a, b)
    
    # Store result
    c = acc.to(tl.bfloat16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            c, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Simple fused operations for the journey
def fused_gate_up_proj(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Fused gate and up projection with SiLU.
    
    Equivalent to: SiLU(x @ W1) * (x @ W3)
    But computes both matmuls, then fuses activation.
    """
    gate = x @ w1
    up = x @ w3
    return fused_silu_mul(gate, up)


__all__ = [
    'fused_silu_mul',
    'fused_gate_up_proj',
    'fused_silu_mul_kernel',
]




