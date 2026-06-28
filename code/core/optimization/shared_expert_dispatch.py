"""Shared dispatch helpers for benchmarks that use identical/shared experts.

These utilities support benchmark pairs where routing/placement changes should
not change the final tensor semantics (e.g., identical experts), while still
paying realistic dispatch costs (grouping, gather/scatter, kernel launch count).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_dispatch_inputs(
    flat_tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if flat_tokens.dim() != 2:
        raise ValueError(f"flat_tokens must be 2D [T, H], got shape {tuple(flat_tokens.shape)}")
    if expert_ids.dim() != 1:
        raise ValueError(f"expert_ids must be 1D [T], got shape {tuple(expert_ids.shape)}")
    if flat_tokens.shape[0] != expert_ids.shape[0]:
        raise ValueError(
            f"flat_tokens and expert_ids must agree on T, got {flat_tokens.shape[0]} vs {expert_ids.shape[0]}"
        )
    if out.shape != flat_tokens.shape:
        raise ValueError(f"out must match flat_tokens shape {tuple(flat_tokens.shape)}, got {tuple(out.shape)}")


def dispatch_shared_expert_mask_scan(
    flat_tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    expert: nn.Module,
    *,
    num_experts: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dispatch by scanning all experts and masking tokens per expert.

    This baseline intentionally uses `mask.any()` which can trigger host sync
    in eager mode. Use only as a baseline dispatch implementation.
    """
    _validate_dispatch_inputs(flat_tokens, expert_ids, out)
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    out.zero_()
    for expert_id in range(int(num_experts)):
        mask = expert_ids == expert_id
        if mask.any():
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            out.index_copy_(0, indices, expert(flat_tokens.index_select(0, indices)))
    return out


def dispatch_shared_expert_active_experts(
    flat_tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    expert: nn.Module,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dispatch by iterating only active experts (unique expert ids)."""
    _validate_dispatch_inputs(flat_tokens, expert_ids, out)

    active = torch.unique(expert_ids)
    for idx in range(active.numel()):
        expert_id = active[idx]
        indices = (expert_ids == expert_id).nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        out.index_copy_(0, indices, expert(flat_tokens.index_select(0, indices)))
    return out


def dispatch_shared_expert_precomputed_indices(
    flat_tokens: torch.Tensor,
    expert: nn.Module,
    *,
    out: torch.Tensor,
    index_groups: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Dispatch with token index groups precomputed for static routing."""
    if flat_tokens.dim() != 2:
        raise ValueError(f"flat_tokens must be 2D [T, H], got shape {tuple(flat_tokens.shape)}")
    if out.shape != flat_tokens.shape:
        raise ValueError(f"out must match flat_tokens shape {tuple(flat_tokens.shape)}, got {tuple(out.shape)}")
    for indices in index_groups:
        if indices.dim() != 1:
            raise ValueError(f"each index group must be 1D, got shape {tuple(indices.shape)}")
        if indices.numel() == 0:
            continue
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        out.index_copy_(0, indices, expert(flat_tokens.index_select(0, indices)))
    return out


def dispatch_shared_expert_sort_scatter(
    flat_tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    expert: nn.Module,
    *,
    out: torch.Tensor,
    sort_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch by sorting tokens by expert id, then scattering back.

    This avoids Python loops and host synchronization while still paying a
    realistic pack (gather) + unpack (scatter) cost.
    """
    _validate_dispatch_inputs(flat_tokens, expert_ids, out)
    if sort_idx is None:
        sort_idx = torch.argsort(expert_ids)
    if sort_idx.dim() != 1 or sort_idx.shape[0] != expert_ids.shape[0]:
        raise ValueError(f"sort_idx must be 1D [T], got shape {tuple(sort_idx.shape)}")
    if sort_idx.dtype != torch.int64:
        sort_idx = sort_idx.to(torch.int64)
    packed = flat_tokens.index_select(0, sort_idx)
    packed_out = expert(packed)
    out.index_copy_(0, sort_idx, packed_out)
    return out


def dispatch_shared_expert_packed_scatter(
    packed_tokens: torch.Tensor,
    expert: nn.Module,
    *,
    out: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    """Dispatch when tokens are already packed in route order.

    This variant avoids the per-iteration gather when the caller can prepare the
    transfer buffer in setup or during a communication packing phase.
    """
    if packed_tokens.dim() != 2:
        raise ValueError(f"packed_tokens must be 2D [T, H], got shape {tuple(packed_tokens.shape)}")
    if out.shape != packed_tokens.shape:
        raise ValueError(f"out must match packed_tokens shape {tuple(packed_tokens.shape)}, got {tuple(out.shape)}")
    if sort_idx.dim() != 1 or sort_idx.shape[0] != packed_tokens.shape[0]:
        raise ValueError(f"sort_idx must be 1D [T], got shape {tuple(sort_idx.shape)}")
    if sort_idx.dtype != torch.int64:
        sort_idx = sort_idx.to(torch.int64)
    packed_out = expert(packed_tokens)
    out.index_copy_(0, sort_idx, packed_out)
    return out
