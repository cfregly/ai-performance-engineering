from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from core.optimization.shared_expert_dispatch import (
    dispatch_shared_expert_active_experts,
    dispatch_shared_expert_packed_scatter,
    dispatch_shared_expert_precomputed_indices,
    dispatch_shared_expert_sort_scatter,
)


def _build_expert() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.SiLU(),
        nn.Linear(8, 4, bias=False),
    )


def test_packed_scatter_matches_sort_scatter() -> None:
    expert = _build_expert()
    flat_tokens = torch.randn(12, 4)
    expert_ids = torch.tensor([2, 0, 1, 2, 1, 0, 3, 2, 3, 1, 0, 3], dtype=torch.int64)
    sort_idx = torch.argsort(expert_ids)
    packed_tokens = flat_tokens.index_select(0, sort_idx)

    expected = torch.empty_like(flat_tokens)
    actual = torch.empty_like(flat_tokens)

    dispatch_shared_expert_sort_scatter(
        flat_tokens,
        expert_ids,
        expert,
        out=expected,
        sort_idx=sort_idx,
    )
    dispatch_shared_expert_packed_scatter(
        packed_tokens,
        expert,
        out=actual,
        sort_idx=sort_idx,
    )

    assert torch.allclose(actual, expected)


def test_active_experts_overwrites_output_without_zero_fill() -> None:
    source = inspect.getsource(dispatch_shared_expert_active_experts)
    assert "out.zero_()" not in source

    expert = _build_expert()
    flat_tokens = torch.randn(8, 4)
    expert_ids = torch.tensor([1, 0, 1, 2, 0, 2, 1, 0], dtype=torch.int64)
    expected = torch.empty_like(flat_tokens)
    actual = torch.full_like(flat_tokens, float("nan"))

    dispatch_shared_expert_sort_scatter(
        flat_tokens,
        expert_ids,
        expert,
        out=expected,
        sort_idx=torch.argsort(expert_ids),
    )
    dispatch_shared_expert_active_experts(
        flat_tokens,
        expert_ids,
        expert,
        out=actual,
    )

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected)


def test_precomputed_indices_match_active_expert_dispatch() -> None:
    source = inspect.getsource(dispatch_shared_expert_precomputed_indices)
    assert "torch.unique(" not in source
    assert "expert_ids ==" not in source
    assert "out.zero_()" not in source

    expert = _build_expert()
    flat_tokens = torch.randn(8, 4)
    expert_ids = torch.tensor([1, 0, 1, 2, 0, 2, 1, 0], dtype=torch.int64)
    index_groups = [
        (expert_ids == expert_id).nonzero(as_tuple=False).squeeze(-1)
        for expert_id in torch.unique(expert_ids).tolist()
    ]
    expected = torch.empty_like(flat_tokens)
    actual = torch.full_like(flat_tokens, float("nan"))

    dispatch_shared_expert_active_experts(
        flat_tokens,
        expert_ids,
        expert,
        out=expected,
    )
    dispatch_shared_expert_precomputed_indices(
        flat_tokens,
        expert,
        out=actual,
        index_groups=index_groups,
    )

    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected)


def test_packed_scatter_rejects_mismatched_sort_index_length() -> None:
    expert = _build_expert()
    packed_tokens = torch.randn(4, 4)
    out = torch.empty_like(packed_tokens)

    with pytest.raises(ValueError, match="sort_idx must be 1D"):
        dispatch_shared_expert_packed_scatter(
            packed_tokens,
            expert,
            out=out,
            sort_idx=torch.tensor([0, 1, 2], dtype=torch.int64),
        )
