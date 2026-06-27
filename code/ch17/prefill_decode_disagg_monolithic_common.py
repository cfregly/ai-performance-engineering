"""Shared model definitions for Chapter 17 monolithic prefill/decode benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLLM(nn.Module):
    """Simplified LLM used for monolithic prefill+decode."""

    def __init__(self, hidden_dim: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefill over the full prompt (compute-bound path)."""
        x = torch.randn(
            prompt_tokens.size(0),
            prompt_tokens.size(1),
            self.hidden_dim,
            device=prompt_tokens.device,
            dtype=torch.bfloat16,
        )
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return x[:, -1:, :]

    def decode(self, kv_cache: torch.Tensor, num_tokens: int = 16) -> torch.Tensor:
        """Decode a small number of tokens (memory-bound path)."""
        token_count = int(num_tokens)
        if token_count <= 0:
            return kv_cache.new_empty(kv_cache.shape[0], 0, kv_cache.shape[-1])
        x = kv_cache
        if token_count == 1:
            for layer in self.layers:
                x = torch.relu_(layer(x))
            return x

        output = kv_cache.new_empty(kv_cache.shape[0], token_count, kv_cache.shape[-1])
        for token_idx in range(token_count):
            for layer in self.layers:
                x = torch.relu_(layer(x))
            output[:, token_idx : token_idx + 1, :].copy_(x)
        return output
