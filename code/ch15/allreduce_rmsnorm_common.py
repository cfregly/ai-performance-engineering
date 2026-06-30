"""Shared helpers for AR + RMSNorm fusion benchmarks (ch15)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AllReduceRMSNormConfig:
    tp_size: int = 8
    batch_size: int = 16
    hidden_size: int = 4096
    eps: float = 1e-5
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_iter(self) -> int:
        return self.batch_size


def build_shards(device: torch.device, cfg: AllReduceRMSNormConfig) -> torch.Tensor:
    """Create synthetic tensor-parallel shards for all-reduce."""
    return torch.randn(
        cfg.tp_size,
        cfg.batch_size,
        cfg.hidden_size,
        device=device,
        dtype=cfg.dtype,
    )


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm (no weight) used by both baseline and optimized."""
    variance = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def naive_allreduce(shards: torch.Tensor) -> torch.Tensor:
    """Naive all-reduce: sequential accumulation (models unfused AR)."""
    out = shards[0].clone()
    for idx in range(1, shards.shape[0]):
        out.add_(shards[idx])
    return out


def vectorized_allreduce(shards: torch.Tensor) -> torch.Tensor:
    """Vectorized all-reduce: single reduction (models optimized AR)."""
    return torch.sum(shards, dim=0)


def fused_allreduce_rmsnorm(shards: torch.Tensor, eps: float) -> torch.Tensor:
    """All-reduce + RMSNorm fused in a single graph."""
    return rms_norm(vectorized_allreduce(shards), eps)


def fused_allreduce_rmsnorm_out(
    shards: torch.Tensor,
    eps: float,
    *,
    reduced: torch.Tensor,
    squares: torch.Tensor,
    variance: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """All-reduce + RMSNorm using caller-owned buffers."""
    torch.sum(shards, dim=0, out=reduced)
    torch.mul(reduced, reduced, out=squares)
    torch.mean(squares, dim=-1, keepdim=True, out=variance)
    variance.add_(eps)
    torch.rsqrt(variance, out=variance)
    torch.mul(reduced, variance, out=out)
    return out
