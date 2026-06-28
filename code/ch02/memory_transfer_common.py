"""Shared helpers for Ch02 host-device transfer benchmarks."""

from __future__ import annotations

from typing import Optional

import torch


def compute_transfer_digest(
    device_data: torch.Tensor,
    digest_buffer: Optional[torch.Tensor],
    *,
    block_elems: int = 1_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute an int64 block-sum digest, reusing ``digest_buffer`` when possible."""
    if block_elems <= 0:
        raise RuntimeError("block_elems must be positive")
    data_bits = device_data.view(torch.int32)
    numel = int(data_bits.numel())
    if numel <= 0:
        raise RuntimeError("device_data must be non-empty for verification")

    digest_blocks = (numel + block_elems - 1) // block_elems
    if (
        digest_buffer is None
        or digest_buffer.shape != (digest_blocks,)
        or digest_buffer.device != data_bits.device
        or digest_buffer.dtype != torch.int64
    ):
        digest_buffer = torch.empty(digest_blocks, dtype=torch.int64, device=data_bits.device)

    full_blocks = numel // block_elems
    if full_blocks:
        full_view = data_bits[: full_blocks * block_elems].view(full_blocks, block_elems)
        torch.sum(full_view, dim=1, dtype=torch.int64, out=digest_buffer[:full_blocks])

    tail_start = full_blocks * block_elems
    if tail_start < numel:
        torch.sum(data_bits[tail_start:numel], dim=0, dtype=torch.int64, out=digest_buffer[full_blocks])

    digest = digest_buffer
    return digest, digest_buffer
