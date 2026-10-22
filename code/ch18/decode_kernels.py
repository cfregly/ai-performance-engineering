"""Decode kernel builders shared by the bucketed decode demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from core.harness.serving_stack import load_vllm_native_extensions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPILE_MODE = "reduce-overhead" if torch.cuda.is_available() else "default"


@dataclass
class DecodeKernel:
    """Light wrapper so callers can introspect the backend type."""

    fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    backend: str

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.fn(tokens, kv, mask)


class VLLMDecodeKernel:
    """Small vLLM unified-attention decode step.

    Uses vLLM's current Triton unified-attention op with its block-table KV-cache
    layout. This is intentionally a minimal wrapper to keep the benchmark
    self-contained while still exercising a fused decode kernel.
    """

    def __init__(self, hidden: int, max_batch: int = 32, device: str = DEVICE) -> None:
        try:
            load_vllm_native_extensions()
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"FAIL FAST: vLLM unified attention unavailable ({exc}). "
                "Install the pinned serving stack and verify both stable-libTorch "
                "native extensions."
            ) from exc

        self._unified_attention = unified_attention

        self.device = device
        self.hidden = hidden
        self.num_heads = 1
        self.head_size = hidden
        self.scale = 1.0 / math.sqrt(float(self.head_size))

        self.max_batch = max_batch
        # Match the pinned vLLM unified-attention block-table layout.
        self.block_size = 16
        self.num_blocks = 4

        # Unified attention consumes [num_blocks, block_size, num_heads, head_size].
        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_size,
        )
        self.key_cache = torch.randn(
            cache_shape, device=self.device, dtype=torch.float16
        )
        self.value_cache = torch.randn(
            cache_shape, device=self.device, dtype=torch.float16
        )

        # Block tables: each sequence gets assigned blocks.
        self.block_tables = (
            torch.arange(self.num_blocks, dtype=torch.int32, device=self.device)
            .unsqueeze(0)
            .expand(self.max_batch, -1)
            .contiguous()
        )

        # Sequence lengths (tokens per sequence). Keep this small so the benchmark
        # cost is dominated by kernel launch/graph churn rather than very long context.
        self.seq_lens = torch.full(
            (self.max_batch,), self.block_size, dtype=torch.int32, device=self.device
        )
        self.query_start_loc = torch.arange(
            self.max_batch + 1, dtype=torch.int32, device=self.device
        )

    def ensure_capacity(self, batch: int) -> None:
        if batch <= self.max_batch:
            return

        self.block_tables = (
            torch.arange(self.num_blocks, dtype=torch.int32, device=self.device)
            .unsqueeze(0)
            .expand(batch, -1)
            .contiguous()
        )
        self.seq_lens = torch.full(
            (batch,), self.block_size, dtype=torch.int32, device=self.device
        )
        self.query_start_loc = torch.arange(
            batch + 1, dtype=torch.int32, device=self.device
        )
        self.max_batch = batch

    def __call__(
        self, tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch = tokens.size(0)
        self.ensure_capacity(batch)

        # Query shape: [batch, num_heads, head_size]
        query = (
            tokens.view(batch, self.num_heads, self.head_size)
            .to(torch.float16)
            .contiguous()
        )
        out = torch.empty(
            (batch, self.num_heads, self.head_size),
            device=self.device,
            dtype=torch.float16,
        )

        self._unified_attention(
            q=query,
            k=self.key_cache,
            v=self.value_cache,
            out=out,
            cu_seqlens_q=self.query_start_loc[: batch + 1],
            max_seqlen_q=1,
            seqused_k=self.seq_lens[:batch],
            max_seqlen_k=self.block_size,
            softmax_scale=self.scale,
            causal=True,
            window_size=(-1, -1),
            block_table=self.block_tables[:batch],
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=None,
        )

        flat = out.view(batch, self.hidden)
        if mask is not None:
            flat.masked_fill_(~mask[:, None], float("-inf"))
        return flat

    @property
    def bytes(self) -> int:
        return sum(
            cache.numel() * cache.element_size()
            for cache in (self.key_cache, self.value_cache)
        )



def _torch_decode(hidden: int) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """Explicit torch-based decode path used only when prefer_vllm=False."""
    def _decode(tokens: torch.Tensor, kv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = torch.tanh(tokens + kv)
        if mask is not None:
            attn_scores.masked_fill_(~mask[:, None], float("-inf"))
        return attn_scores

    try:
        return torch.compile(_decode, mode=COMPILE_MODE, fullgraph=False, dynamic=False)
    except Exception:
        return _decode


def build_decode_kernel(
    hidden: int,
    *,
    max_batch: int = 32,
    prefer_vllm: bool = True,
    device: str = DEVICE,
) -> DecodeKernel:
    """
    Build a vLLM-backed decode kernel. Raises if vLLM unavailable.
    """
    if prefer_vllm:
        try:
            kernel = VLLMDecodeKernel(hidden=hidden, max_batch=max_batch, device=device)
        except Exception as exc:
            raise RuntimeError(f"FAIL FAST: vLLM decode kernel unavailable ({exc})") from exc
        return DecodeKernel(fn=kernel, backend="vllm")

    # Only use torch if explicitly requested (prefer_vllm=False)
    torch_kernel = _torch_decode(hidden)
    return DecodeKernel(fn=torch_kernel, backend="torch")
