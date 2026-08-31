#!/usr/bin/env python3
"""Triton kernels for MoE optimization journey.

Only the elementwise SiLU/multiply kernel is used by the shared MoE model.
The former fused_expert_ffn_kernel and grouped_gemm_kernel experiments were
removed: they omitted intermediate/output tiles and per-group storage offsets.
They are not complete FFN or grouped-GEMM implementations. Gate/up projection
below uses ordinary PyTorch matmuls followed by the selected activation path.
CPU and autograd-enabled inputs explicitly use PyTorch rather than Triton.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
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
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        # Fused SiLU * up: silu(x) = x * sigmoid(x)
        gate_sigmoid = tl.sigmoid(gate.to(tl.float32))
        silu_gate = (gate.to(tl.float32) * gate_sigmoid).to(gate.dtype)
        out = silu_gate.to(tl.float32) * up.to(tl.float32)

        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)
else:
    fused_silu_mul_kernel = None


def fused_silu_mul_backend(gate: torch.Tensor, up: torch.Tensor) -> str:
    """Validate the pair and identify the selected implementation, not a speed claim."""
    if gate.shape != up.shape:
        raise ValueError("gate and up must have matching shapes")
    if gate.device != up.device:
        raise ValueError("gate and up must use the same device")
    if gate.dtype != up.dtype:
        raise TypeError("gate and up must have matching dtypes")
    if gate.layout != torch.strided or up.layout != torch.strided:
        raise ValueError("gate and up must have strided tensor layouts")
    if not gate.is_floating_point():
        raise TypeError("gate and up must have floating point dtype")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        return "pytorch_autograd"
    if gate.device.type == "cpu":
        return "pytorch_cpu"
    if not gate.is_cuda:
        raise ValueError("supported devices are CPU and CUDA")
    if not TRITON_AVAILABLE:
        return "pytorch_no_triton"
    if gate.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Triton inference supports float32, float16 and bfloat16")
    return "triton_inference"


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute SiLU(gate) * up without detaching trainable inputs.

    CPU, missing-Triton, and autograd-enabled inputs use differentiable PyTorch
    operations, never a Triton-execution claim. CUDA inference uses the Triton
    kernel. Contiguous copies for strided views occur inside this call and are
    part of its work, not an excluded setup step. Empty inputs launch no kernel.
    """
    backend = fused_silu_mul_backend(gate, up)
    if backend != "triton_inference":
        return F.silu(gate) * up
    with torch.cuda.device(gate.device):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty_like(gate)
        n_elements = gate.numel()
        if n_elements:
            grid = (triton.cdiv(n_elements, 1024),)
            fused_silu_mul_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=1024)
    return out


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
    'TRITON_AVAILABLE',
    'fused_silu_mul_backend',
    'fused_silu_mul',
    'fused_gate_up_proj',
    'fused_silu_mul_kernel',
]
