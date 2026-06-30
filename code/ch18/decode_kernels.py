"""Decode kernel builders shared by the bucketed decode demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

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
    """Small PagedAttention-backed decode step.

    Uses vLLM's CUDA custom op (paged_attention_v1) with the vLLM v1 KV-cache
    layout. This is intentionally a minimal wrapper to keep the benchmark
    self-contained while still exercising the fused decode kernel.
    """

    def __init__(self, hidden: int, max_batch: int = 32, device: str = DEVICE) -> None:
        try:
            from vllm import _custom_ops as vllm_ops
            from vllm.v1.attention.ops.paged_attn import PagedAttention
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"FAIL FAST: vLLM PagedAttention unavailable ({exc}). "
                "Install the pinned serving stack and verify vllm._C import."
            ) from exc

        self._ops = vllm_ops
        self._paged_attention = vllm_ops.paged_attention_v1

        self.device = device
        self.hidden = hidden
        self.num_heads = 1
        self.head_size = hidden
        self.kv_cache_dtype = "auto"
        self.scale = 1.0 / math.sqrt(float(self.head_size))

        self.max_batch = max_batch
        # vLLM paged attention requires block_size >= 8 (typically 8, 16, or 32)
        self.block_size = 16
        self.num_blocks = 4
        self.max_seq_len = self.num_blocks * self.block_size

        # Allocate the v1 KV cache layout: [2, num_blocks, num_heads * head_size * block_size].
        kv_elems = self.num_heads * self.head_size * self.block_size
        self.kv_cache = torch.randn(
            (2, self.num_blocks, kv_elems), device=self.device, dtype=torch.float16
        )
        self.key_cache, self.value_cache = PagedAttention.split_kv_cache(
            self.kv_cache, num_kv_heads=self.num_heads, head_size=self.head_size
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

        # K/V scale factors (1.0 = no scaling) - required by vLLM API.
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

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

        self._paged_attention(
            out=out,
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            block_tables=self.block_tables[:batch],
            seq_lens=self.seq_lens[:batch],
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            alibi_slopes=None,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )

        flat = out.view(batch, self.hidden)
        if mask is not None:
            flat.masked_fill_(~mask[:, None], float("-inf"))
        return flat

    @property
    def bytes(self) -> int:
        return self.kv_cache.numel() * self.kv_cache.element_size()



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
