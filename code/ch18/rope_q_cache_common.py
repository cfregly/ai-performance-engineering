"""Shared helpers for RoPE + Q + KV cache fusion benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class RopeQCacheConfig:
    batch_size: int = 8
    heads: int = 32
    head_dim: int = 128
    max_seq_len: int = 256
    steps: int = 64
    dtype: torch.dtype = torch.bfloat16

    @property
    def hidden_size(self) -> int:
        return self.heads * self.head_dim

    @property
    def tokens_per_iter(self) -> int:
        return self.batch_size * self.steps


def build_rope_tables(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    half = head_dim // 2
    cos = torch.empty(max_seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.empty_like(cos)
    cos[:, :half].copy_(cos_half)
    cos[:, half:].copy_(cos_half)
    sin[:, :half].copy_(sin_half)
    sin[:, half:].copy_(sin_half)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and (q.requires_grad or cos.requires_grad or sin.requires_grad):
        return (q * cos) + (rotate_half(q) * sin)

    out = torch.empty_like(q)
    return apply_rope_out(q, cos, sin, out)


def apply_rope_out(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    half = q.shape[-1] // 2
    q1 = q[..., :half]
    q2 = q[..., half:]
    out1 = out[..., :half]
    out2 = out[..., half:]
    torch.mul(q1, cos[..., :half], out=out1)
    out1.addcmul_(q2, sin[..., :half], value=-1)
    torch.mul(q2, cos[..., half:], out=out2)
    out2.addcmul_(q1, sin[..., half:])
    return out


def apply_rope_inplace(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scratch: torch.Tensor,
) -> torch.Tensor:
    half = q.shape[-1] // 2
    q1 = q[..., :half]
    q2 = q[..., half:]
    cos1 = cos[..., :half]
    cos2 = cos[..., half:]
    sin1 = sin[..., :half]
    sin2 = sin[..., half:]
    scratch.copy_(q1)
    q1.mul_(cos1).addcmul_(q2, sin1, value=-1)
    q2.mul_(cos2).addcmul_(scratch, sin2)
    return q
