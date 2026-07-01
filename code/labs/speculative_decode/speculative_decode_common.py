"""Shared helpers for the speculative decoding lab benchmarks.

This lab focuses on the speculative decoding control-flow rather than attention
or KV-cache mechanics. We use a simple *token-local* model (TokenMLP) so that:

- The target model is GEMM-heavy and easy to scale (large hidden size).
- The draft model is a smaller approximation of the target (sliced weights).
- The optimized path can batch verification of K proposed tokens into one
  target-model call, demonstrating the core speculative-decoding speedup.

To simulate a well-distilled draft model (without running training in setup),
we scale the target model's "tail" hidden dimensions so that a sliced draft
matches the target's greedy predictions with a high acceptance rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.triton_compat import ensure_triton_compat

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - Triton is optional at import time
    triton = None
    tl = None
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class SpecDecodeWorkload:
    vocab_size: int
    target_hidden: int
    target_layers: int
    draft_hidden: int
    speculative_k: int
    total_tokens: int
    tail_scale: float
    dtype: torch.dtype


def default_workload(*, dtype: torch.dtype = torch.bfloat16) -> SpecDecodeWorkload:
    return SpecDecodeWorkload(
        vocab_size=32000,
        target_hidden=8192,
        target_layers=2,
        draft_hidden=1024,
        speculative_k=64,
        total_tokens=256,
        tail_scale=0.01,
        dtype=dtype,
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _accept_prefix_length_kernel(
        matches_ptr,
        out_ptr,
        stride_t,
        K: tl.constexpr,  # noqa: N803
    ):
        accepted = tl.full((), 0, tl.int64)
        prefix_alive = True
        for idx in range(0, K):
            match = tl.load(matches_ptr + idx * stride_t)
            prefix_alive = prefix_alive & match
            accepted += prefix_alive.to(tl.int64)
        tl.store(out_ptr, accepted)


def accept_prefix_length(matches: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Write the contiguous accepted-token prefix length for a [1, K] match row."""

    if matches.numel() == 0:
        out.zero_()
        return out
    if TRITON_AVAILABLE and matches.is_cuda:
        ensure_triton_compat()
        _accept_prefix_length_kernel[(1,)](
            matches,
            out,
            matches.stride(-1),
            K=matches.shape[-1],
        )
        return out
    prefix = torch.cumprod(matches, dim=-1, dtype=torch.int64)
    torch.sum(prefix.reshape(-1), dim=0, out=out)
    return out


class TokenMLP(nn.Module):
    """Position-independent toy LM: next-token logits from token ids."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size, device=device, dtype=dtype)

        layers: List[nn.Module] = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype))
            layers.append(nn.GELU(approximate="tanh"))
        self.mlp = nn.Sequential(*layers)
        self._linear_layers: tuple[nn.Linear, ...] = tuple(
            module for module in self.mlp if isinstance(module, nn.Linear)
        )
        self.out = nn.Linear(self.hidden_size, self.vocab_size, device=device, dtype=dtype)
        self._linear_weight_t_views: tuple[torch.Tensor, ...] = ()
        self._out_weight_t: torch.Tensor | None = None
        self._forward_buffers: dict[
            tuple[int, torch.device, torch.dtype],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self.cache_weight_views()

    def cache_weight_views(self) -> None:
        self._linear_weight_t_views = tuple(layer.weight.t() for layer in self._linear_layers)
        self._out_weight_t = self.out.weight.t()

    def _ensure_forward_buffers(
        self,
        num_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (int(num_tokens), device, dtype)
        buffers = self._forward_buffers.get(cache_key)
        if buffers is None:
            buffer_shape = (num_tokens, self.hidden_size)
            buffers = (
                torch.empty(buffer_shape, device=device, dtype=dtype),
                torch.empty(buffer_shape, device=device, dtype=dtype),
            )
            self._forward_buffers[cache_key] = buffers
        return buffers

    def prepare_forward_buffers(
        self,
        num_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._ensure_forward_buffers(num_tokens, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, seq]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be int32 or int64")
        batch, seq = token_ids.shape
        embedded = self.embed(token_ids)
        hidden = embedded.reshape(batch * seq, self.hidden_size)
        hidden = self.mlp(hidden)
        logits = self.out(hidden).reshape(batch, seq, self.vocab_size)
        return logits

    def forward_into(self, token_ids: torch.Tensor, logits_out: torch.Tensor) -> torch.Tensor:
        """Run inference while writing logits into a caller-owned output buffer."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, seq]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be int32 or int64")
        batch, seq = token_ids.shape
        expected_shape = (batch, seq, self.vocab_size)
        if tuple(logits_out.shape) != expected_shape:
            raise ValueError(f"logits_out must have shape {expected_shape}")

        if torch.is_grad_enabled():
            logits_out.copy_(self(token_ids))
            return logits_out

        dtype = self.embed.weight.dtype
        if logits_out.dtype != dtype:
            raise ValueError("logits_out dtype must match model parameters")

        num_tokens = batch * seq
        hidden, scratch = self._ensure_forward_buffers(num_tokens, device=token_ids.device, dtype=dtype)
        flat_ids = token_ids.reshape(num_tokens)
        torch.index_select(self.embed.weight, 0, flat_ids, out=hidden)

        return self._forward_logits_into(logits_out, hidden, scratch, batch, seq)

    def forward_into_prepared(
        self,
        token_ids: torch.Tensor,
        logits_out: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run inference using caller-prepared hidden/scratch buffers."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape [batch, seq]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be int32 or int64")
        batch, seq = token_ids.shape
        expected_shape = (batch, seq, self.vocab_size)
        if tuple(logits_out.shape) != expected_shape:
            raise ValueError(f"logits_out must have shape {expected_shape}")

        if torch.is_grad_enabled():
            logits_out.copy_(self(token_ids))
            return logits_out

        dtype = self.embed.weight.dtype
        if logits_out.dtype != dtype:
            raise ValueError("logits_out dtype must match model parameters")
        hidden, scratch = buffers
        num_tokens = batch * seq
        expected_hidden_shape = (num_tokens, self.hidden_size)
        if tuple(hidden.shape) != expected_hidden_shape or tuple(scratch.shape) != expected_hidden_shape:
            raise ValueError(f"prepared buffers must have shape {expected_hidden_shape}")
        if hidden.device != token_ids.device or scratch.device != token_ids.device:
            raise ValueError("prepared buffers must match token_ids device")
        if hidden.dtype != dtype or scratch.dtype != dtype:
            raise ValueError("prepared buffers must match model parameter dtype")

        flat_ids = token_ids.reshape(num_tokens)
        torch.index_select(self.embed.weight, 0, flat_ids, out=hidden)
        return self._forward_logits_into(logits_out, hidden, scratch, batch, seq)

    def forward_into_prepared_unchecked(
        self,
        token_ids: torch.Tensor,
        logits_out: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Fast inference path for callers that already validated static buffers."""
        batch, seq = token_ids.shape
        num_tokens = batch * seq
        hidden, scratch = buffers
        flat_ids = token_ids.reshape(num_tokens)
        torch.index_select(self.embed.weight, 0, flat_ids, out=hidden)
        return self._forward_logits_into(logits_out, hidden, scratch, batch, seq)

    def _forward_logits_into(
        self,
        logits_out: torch.Tensor,
        hidden: torch.Tensor,
        scratch: torch.Tensor,
        batch: int,
        seq: int,
    ) -> torch.Tensor:
        num_tokens = batch * seq
        current = hidden
        alternate = scratch
        if len(self._linear_weight_t_views) != len(self._linear_layers) or self._out_weight_t is None:
            self.cache_weight_views()
        for layer, weight_t in zip(self._linear_layers, self._linear_weight_t_views, strict=True):
            torch.matmul(current, weight_t, out=alternate)
            if layer.bias is not None:
                alternate.add_(layer.bias)
            F.gelu(alternate, approximate="tanh", out=alternate)
            current, alternate = alternate, current

        flat_logits = logits_out.view(num_tokens, self.vocab_size)
        torch.matmul(current, self._out_weight_t, out=flat_logits)
        if self.out.bias is not None:
            flat_logits.add_(self.out.bias)
        return logits_out


def scale_tail_dims_(target: TokenMLP, draft_hidden: int, tail_scale: float) -> None:
    """Scale hidden dims >= draft_hidden to make the draft approximation accurate."""
    if not isinstance(target, TokenMLP):
        raise TypeError("target must be TokenMLP")
    if draft_hidden <= 0:
        raise ValueError("draft_hidden must be > 0")
    if draft_hidden >= target.hidden_size:
        raise ValueError("draft_hidden must be < target.hidden_size")
    if not (0.0 < float(tail_scale) <= 1.0):
        raise ValueError("tail_scale must be in (0, 1]")

    cutoff = int(draft_hidden)
    scale = float(tail_scale)

    with torch.inference_mode():
        target.embed.weight[:, cutoff:].mul_(scale)

        for layer in target._linear_layers:
            layer.weight[cutoff:, :].mul_(scale)
            layer.weight[:, cutoff:].mul_(scale)
            if layer.bias is not None:
                layer.bias[cutoff:].mul_(scale)

        target.out.weight[:, cutoff:].mul_(scale)


def build_draft_from_target(target: TokenMLP, draft_hidden: int) -> TokenMLP:
    """Create a smaller draft model by slicing a target model's weights."""
    if not isinstance(target, TokenMLP):
        raise TypeError("target must be TokenMLP")
    if draft_hidden <= 0:
        raise ValueError("draft_hidden must be > 0")
    if draft_hidden >= target.hidden_size:
        raise ValueError("draft_hidden must be < target.hidden_size")

    device = target.embed.weight.device
    dtype = target.embed.weight.dtype
    draft = TokenMLP(
        vocab_size=target.vocab_size,
        hidden_size=int(draft_hidden),
        num_layers=target.num_layers,
        device=device,
        dtype=dtype,
    ).eval()

    target_linears = target._linear_layers
    draft_linears = draft._linear_layers
    if len(target_linears) != len(draft_linears):
        raise RuntimeError("Target/draft layer mismatch (unexpected)")

    cutoff = int(draft_hidden)
    with torch.inference_mode():
        draft.embed.weight.copy_(target.embed.weight[:, :cutoff])
        for t_layer, d_layer in zip(target_linears, draft_linears, strict=True):
            d_layer.weight.copy_(t_layer.weight[:cutoff, :cutoff])
            if t_layer.bias is not None and d_layer.bias is not None:
                d_layer.bias.copy_(t_layer.bias[:cutoff])

        draft.out.weight.copy_(target.out.weight[:, :cutoff])
        if target.out.bias is not None and draft.out.bias is not None:
            draft.out.bias.copy_(target.out.bias)

    return draft
