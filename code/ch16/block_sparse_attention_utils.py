"""Utilities for block-sparse attention masks (FlashInfer + dense baseline)."""

from __future__ import annotations

from typing import Tuple

import torch


def build_block_sparse_pattern(
    *,
    seq_len: int,
    block_size: int,
    window_blocks: int,
) -> torch.Tensor:
    if seq_len % block_size != 0:
        raise ValueError("seq_len must be divisible by block_size")
    blocks = seq_len // block_size
    row_ids = torch.arange(blocks).unsqueeze(1)
    col_ids = torch.arange(blocks).unsqueeze(0)
    return (col_ids >= row_ids - window_blocks) & (col_ids <= row_ids + window_blocks)


def build_dense_attention_mask(
    block_mask: torch.Tensor,
    *,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.full(block_mask.shape, float("-inf"), device=device, dtype=dtype)
    values.masked_fill_(block_mask.to(device=device, dtype=torch.bool), 0.0)
    blocks = values.shape[0]
    return values[:, None, :, None].expand(blocks, block_size, blocks, block_size).reshape(
        blocks * block_size,
        blocks * block_size,
    )


def build_bsr_from_block_mask(
    block_mask: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    blocks = block_mask.shape[0]
    mask = block_mask.to(dtype=torch.bool)
    row_counts = mask.sum(dim=1, dtype=torch.int32)
    indptr_src = torch.empty(blocks + 1, dtype=torch.int32, device=mask.device)
    indptr_src[0] = 0
    torch.cumsum(row_counts, dim=0, out=indptr_src[1:])
    indices_src = torch.nonzero(mask, as_tuple=False)[:, 1].to(torch.int32)
    indptr = indptr_src.to(device=device)
    indices = indices_src.to(device=device)
    total_blocks = float(blocks * blocks)
    allowed_blocks = float(indices_src.numel())
    sparsity_ratio = 1.0 - (allowed_blocks / total_blocks)
    return indptr, indices, sparsity_ratio
