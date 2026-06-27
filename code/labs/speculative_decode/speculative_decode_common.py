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
        self.out = nn.Linear(self.hidden_size, self.vocab_size, device=device, dtype=dtype)

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

        for module in target.mlp:
            if isinstance(module, nn.Linear):
                module.weight[cutoff:, :].mul_(scale)
                module.weight[:, cutoff:].mul_(scale)
                if module.bias is not None:
                    module.bias[cutoff:].mul_(scale)

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

    target_linears = [m for m in target.mlp if isinstance(m, nn.Linear)]
    draft_linears = [m for m in draft.mlp if isinstance(m, nn.Linear)]
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
