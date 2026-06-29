"""Shared helpers for MXFP8 MoE microbenchmarks on Blackwell GPUs."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported

MX_BLOCK_SIZE = 32  # Microscaling block granularity for MXFP8/NVFP4 paths.
BucketByExpertResult = Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor, torch.Tensor]
BucketByExpertWithHostOrder = Tuple[
    torch.Tensor,
    List[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
]


def require_blackwell(example_name: str) -> None:
    """Fail fast when the GPU is not a Blackwell/GB-series device."""
    ensure_blackwell_tma_supported(example_name)


def balanced_assignments(num_tokens: int, num_experts: int, device: torch.device) -> torch.Tensor:
    """Deterministically map each token to an expert to avoid empty buckets."""
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    return mapping % num_experts


def bucket_by_expert(
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    num_experts: int,
    token_ids: Optional[torch.Tensor] = None,
    return_expert_order_list: bool = False,
) -> Union[BucketByExpertResult, BucketByExpertWithHostOrder]:
    """Reorder tokens by expert so grouped kernels can consume contiguous ranges.

    Returns:
        bucketed: Tokens concatenated per expert (M x K).
        m_splits: Number of tokens per expert in bucketed order.
        bucket_indices: Indices that map bucketed rows back to the original order.
        expert_order: Tensor of expert ids aligned with m_splits.
        bucket_token_ids: Original token ids for each row in bucketed.
        expert_order_list: Optional host-side expert ids aligned with m_splits
            when return_expert_order_list is True.
    """
    flat_assignments = assignments.reshape(-1)
    if token_ids is None:
        if flat_assignments.numel() != tokens.shape[0]:
            raise ValueError("assignments must contain one expert id per token row")
        token_ids = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64)
    elif token_ids.numel() != flat_assignments.numel():
        raise ValueError("token_ids must contain one source token id per assignment")
    gather_index = torch.argsort(flat_assignments)
    counts_tensor = torch.bincount(flat_assignments, minlength=num_experts)
    counts_host = counts_tensor.detach().cpu()
    m_splits: List[int] = []
    expert_order_list: List[int] = []
    for expert in range(num_experts):
        count = int(counts_host[expert])
        if count:
            m_splits.append(count)
            expert_order_list.append(expert)
    if not m_splits:
        raise RuntimeError("No expert received tokens; assignment mapping is empty.")

    bucket_token_ids_tensor = token_ids.index_select(0, gather_index)
    if flat_assignments.numel() == tokens.shape[0]:
        bucketed = tokens.index_select(0, gather_index)
    else:
        bucketed = tokens.index_select(0, bucket_token_ids_tensor)
    expert_range = torch.arange(num_experts, device=tokens.device, dtype=torch.int64)
    expert_order_tensor = expert_range[counts_tensor[:num_experts] > 0]
    if return_expert_order_list:
        return bucketed, m_splits, gather_index, expert_order_tensor, bucket_token_ids_tensor, expert_order_list
    return bucketed, m_splits, gather_index, expert_order_tensor, bucket_token_ids_tensor


def restore_bucketed(
    output: torch.Tensor,
    bucket_indices: torch.Tensor,
    num_tokens: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Scatter bucketed outputs back to the original token order."""
    if out.shape != (num_tokens, output.shape[-1]):
        raise ValueError("restore_bucketed() requires a preallocated output buffer with matching shape")
    out[bucket_indices] = output
    return out


def restore_bucketed_reduce(
    output: torch.Tensor,
    bucket_token_ids: torch.Tensor,
    num_tokens: int,
    weights: torch.Tensor,
    out: torch.Tensor,
    weight_out: torch.Tensor,
    weighted_out: Optional[torch.Tensor] = None,
    bucket_token_ids_expanded: Optional[torch.Tensor] = None,
    weights_expanded: Optional[torch.Tensor] = None,
    weight_out_expanded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter-accumulate bucketed outputs to tokens, handling duplicate assignments."""
    if out.shape != (num_tokens, output.shape[-1]):
        raise ValueError("restore_bucketed_reduce() requires a preallocated output buffer with matching shape")
    if weight_out.shape != (num_tokens,):
        raise ValueError("restore_bucketed_reduce() requires a preallocated weight buffer with matching shape")
    if weighted_out is not None:
        if weighted_out.shape != output.shape:
            raise ValueError("restore_bucketed_reduce() weighted_out must match output shape")
        if weighted_out.dtype != out.dtype:
            raise ValueError("restore_bucketed_reduce() weighted_out must match output dtype")
        if weighted_out.device != output.device:
            raise ValueError("restore_bucketed_reduce() weighted_out must live on the output device")
    out.zero_()
    weight_out.zero_()
    weights = weights.to(dtype=out.dtype, copy=False)
    if bucket_token_ids_expanded is None:
        bucket_token_ids_expanded = bucket_token_ids.unsqueeze(-1).expand_as(output)
    elif bucket_token_ids_expanded.shape != output.shape:
        raise ValueError("restore_bucketed_reduce() expanded bucket ids must match output shape")
    if weights_expanded is None:
        weights_expanded = weights.unsqueeze(-1)
    else:
        if weights_expanded.shape != (output.shape[0], 1):
            raise ValueError("restore_bucketed_reduce() expanded weights must have shape [rows, 1]")
        if weights_expanded.dtype != out.dtype:
            raise ValueError("restore_bucketed_reduce() expanded weights must match output dtype")
        if weights_expanded.device != output.device:
            raise ValueError("restore_bucketed_reduce() expanded weights must live on the output device")
    if weight_out_expanded is None:
        weight_out_expanded = weight_out.unsqueeze(-1)
    else:
        if weight_out_expanded.shape != (num_tokens, 1):
            raise ValueError("restore_bucketed_reduce() expanded output weights must have shape [num_tokens, 1]")
        if weight_out_expanded.dtype != out.dtype:
            raise ValueError("restore_bucketed_reduce() expanded output weights must match output dtype")
        if weight_out_expanded.device != output.device:
            raise ValueError("restore_bucketed_reduce() expanded output weights must live on the output device")
    if weighted_out is None:
        weighted_output = output.to(dtype=out.dtype, copy=True)
        weighted_output.mul_(weights_expanded)
    else:
        weighted_output = weighted_out
        torch.mul(output, weights_expanded, out=weighted_output)
    out.scatter_add_(0, bucket_token_ids_expanded, weighted_output)
    weight_out.scatter_add_(0, bucket_token_ids, weights)
    weight_out.clamp_(min=torch.finfo(out.dtype).eps)
    out.div_(weight_out_expanded)
    return out


def block_quantize_mxfp8(
    tensor: torch.Tensor, block_size: int = MX_BLOCK_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dimension using MXFP8-style block scaling.

    Returns the quantized tensor and the per-block scales (E8M0 equivalent).
    """
    last_dim = tensor.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(f"Last dimension ({last_dim}) must be divisible by block_size={block_size}")
    finfo = torch.finfo(torch.float8_e4m3fn)
    reshaped = tensor.reshape(-1, last_dim // block_size, block_size)
    max_abs = reshaped.abs().amax(dim=-1)
    max_abs = torch.clamp(max_abs, min=torch.finfo(torch.float32).eps)
    scale = (max_abs / finfo.max).to(torch.float32)
    quantized = (reshaped / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    quantized = quantized.reshape_as(tensor)
    scale = scale.reshape(*tensor.shape[:-1], last_dim // block_size)
    return quantized, scale


def block_dequantize_mxfp8(
    quantized: torch.Tensor, scale: torch.Tensor, block_size: int = MX_BLOCK_SIZE, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Dequantize MXFP8 blocks back to ``dtype``."""
    last_dim = quantized.shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(f"Last dimension ({last_dim}) must be divisible by block_size={block_size}")
    reshaped = quantized.reshape(-1, last_dim // block_size, block_size)
    scale = scale.reshape(-1, last_dim // block_size, 1).to(dtype)
    dequant = (reshaped.to(dtype) * scale).reshape_as(quantized)
    return dequant


__all__ = [
    "MX_BLOCK_SIZE",
    "require_blackwell",
    "balanced_assignments",
    "bucket_by_expert",
    "restore_bucketed",
    "restore_bucketed_reduce",
    "block_quantize_mxfp8",
    "block_dequantize_mxfp8",
]
