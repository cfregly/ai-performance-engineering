"""Shared model definitions for Chapter 15 monolithic inference benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLLM(nn.Module):
    """Simplified LLM for inference simulation."""

    def __init__(self, *, vocab_size: int = 10000, hidden_dim: int = 512, num_layers: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefill: process the full prompt (compute-bound path)."""
        x = self.embed(prompt_tokens)
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return x[:, -1:, :]

    def decode_step(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Advance the decode state by one token-equivalent step."""
        x = kv_cache
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return x

    def decode_autoregressive(
        self,
        kv_cache: torch.Tensor,
        *,
        num_tokens: int = 16,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate tokens one step at a time while optionally reusing an output buffer."""
        token_count = int(num_tokens)
        expected_shape = (kv_cache.shape[0], token_count, kv_cache.shape[-1])
        if output_buffer is None:
            output_buffer = kv_cache.new_empty(expected_shape)
        elif tuple(output_buffer.shape) != expected_shape:
            raise ValueError(
                f"output_buffer shape {tuple(output_buffer.shape)} does not match expected {expected_shape}"
            )

        current = kv_cache
        for token_idx in range(token_count):
            current = self.decode_step(current)
            output_buffer[:, token_idx : token_idx + 1, :] = current
        return output_buffer

    def decode(self, kv_cache: torch.Tensor, *, num_tokens: int = 16) -> torch.Tensor:
        """Decode: generate tokens from the cached state (memory-bound path)."""
        return self.decode_autoregressive(kv_cache, num_tokens=num_tokens)
