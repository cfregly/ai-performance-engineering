"""Shared helpers for FlexAttention lab benchmarks.

Extracts the flex_attention DSL pieces from the Colfax FlexAttention guide
into utilities that both baseline and optimized lab variants can reuse.

Key knobs exposed by the lab:
 - block_size: tile size passed into create_block_mask (how coarse sparse blocks are)
 - doc_span: tokens per “document” that define boundaries for the mask
 - seq_len/head_dim/heads: shape of the synthetic qkv tensors
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from core.common.device_utils import require_cuda_device


@dataclass
class FlexAttentionInputs:
    """Container for tensors needed by the lab benchmarks."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_mask: BlockMask
    rel_bias: torch.Tensor

resolve_device = partial(require_cuda_device, "FlexAttention labs require CUDA")


def _make_doc_ids(batch: int, heads: int, seq_len: int, doc_span: int, device: torch.device) -> torch.Tensor:
    """Assign a document id per token to gate cross-document attention."""
    doc_ids_1d = (torch.arange(seq_len, device=device) // doc_span).to(torch.int32)
    doc_ids = doc_ids_1d.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1).contiguous()
    return doc_ids


def build_flex_attention_inputs(
    *,
    batch: int = 2,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    doc_span: int = 256,
    block_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
    use_vmap: bool | None = None,
) -> FlexAttentionInputs:
    """Create qkv tensors, a block mask, and a relative bias table."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    doc_ids = _make_doc_ids(batch, heads, seq_len, doc_span, device=device)

    def document_mask(b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        # Return a tensor predicate to keep vmap happy (avoid Tensor->bool control flow).
        return torch.eq(doc_ids[b, h, q_idx], doc_ids[b, h, kv_idx])

    # FlexAttention API surface changed: newer builds expose a use_vmap flag while older
    # (e.g., packaged with this environment) do not. Only pass it when supported to avoid
    # TypeError on older builds.
    mask_kwargs = {"device": device, "BLOCK_SIZE": block_size}
    if "use_vmap" in inspect.signature(create_block_mask).parameters:
        mask_kwargs["use_vmap"] = True if use_vmap is None else use_vmap

    block_mask = create_block_mask(
        document_mask,
        batch,
        heads,
        seq_len,
        seq_len,
        **mask_kwargs,
    )

    rel_positions = torch.arange(seq_len, device=device)
    # Gently decaying bias highlights the score_mod hook from the blog.
    rel_bias_1d = torch.exp(-rel_positions / float(doc_span))
    rel_bias = rel_bias_1d.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1).contiguous()

    return FlexAttentionInputs(q=q, k=k, v=v, block_mask=block_mask, rel_bias=rel_bias)


def build_qkv_inputs(
    *,
    batch: int = 2,
    heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create only qkv tensors for cases where custom masks are not needed."""
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def make_relative_bias_score_mod(rel_bias: torch.Tensor) -> Callable:
    """Return a score_mod closure that adds a distance-based bias."""

    max_rel = rel_bias.size(-1) - 1

    def score_mod(
        score: torch.Tensor,
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        distance = (q_idx - kv_idx).abs().clamp_max(max_rel)
        bias = rel_bias[batch_idx, head_idx, distance]
        return score + bias

    return score_mod


def compute_attention_workload_metrics(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    doc_span: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Return attention-specific workload metrics for the FlexAttention lab."""
    bytes_per_element = float(torch.finfo(dtype).bits // 8)
    span = max(doc_span, 1)
    num_docs = max(1, math.ceil(seq_len / span))
    active_pairs_per_head = 0.0
    for doc_idx in range(num_docs):
        start = doc_idx * span
        stop = min(seq_len, start + span)
        tokens_in_doc = max(stop - start, 0)
        active_pairs_per_head += float(tokens_in_doc * tokens_in_doc)
    total_active_pairs = float(batch * heads) * active_pairs_per_head
    total_flops = 4.0 * total_active_pairs * float(head_dim)
    total_bytes = float(batch * heads * seq_len * head_dim) * bytes_per_element * 4.0
    return {
        "flex_attention.batch": float(batch),
        "flex_attention.heads": float(heads),
        "flex_attention.seq_len": float(seq_len),
        "flex_attention.head_dim": float(head_dim),
        "flex_attention.doc_span": float(doc_span),
        "flex_attention.active_score_pairs": total_active_pairs,
        "flex_attention.total_flops": float(total_flops),
        "flex_attention.total_bytes": float(total_bytes),
        "flex_attention.arithmetic_intensity": float(total_flops / max(total_bytes, 1.0)),
    }
