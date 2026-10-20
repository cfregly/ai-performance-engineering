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
        # The default Linear init shrinks activations at every layer until the
        # biases dominate. Every prompt then decays to the same output, and
        # verification cannot separate a right answer from a wrong one. He init
        # with zero bias preserves the expected activation scale at each layer,
        # so the decoded state depends on the prompt and on the number of
        # decode steps.
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        self.register_buffer("_prefill_input_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_prefill_angle_buffer", torch.empty(0), persistent=False)
        # Per-feature angular frequencies for the sinusoidal token features.
        # _prefill_input builds them in float32 on first use.
        self.register_buffer("_prefill_feature_scale", torch.empty(0), persistent=False)

    def _prefill_input(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = int(prompt_tokens.size(0))
        seq_len = int(prompt_tokens.size(1))
        shape = (batch_size, seq_len, self.hidden_dim)
        numel = batch_size * seq_len * self.hidden_dim
        if (
            self._prefill_input_buffer.numel() < numel
            or self._prefill_input_buffer.device != prompt_tokens.device
            or self._prefill_input_buffer.dtype != torch.bfloat16
        ):
            self._prefill_input_buffer = torch.empty(
                numel,
                device=prompt_tokens.device,
                dtype=torch.bfloat16,
            )
        if (
            self._prefill_angle_buffer.numel() < numel
            or self._prefill_angle_buffer.device != prompt_tokens.device
            or self._prefill_angle_buffer.dtype != torch.float32
        ):
            self._prefill_angle_buffer = torch.empty(
                numel,
                device=prompt_tokens.device,
                dtype=torch.float32,
            )
        # Module.to(torch.bfloat16) casts every floating buffer, including the
        # frequencies. In bf16 only 512 of 1024 frequencies stay distinct, so
        # rebuild them in float32 after any cast or move.
        if (
            self._prefill_feature_scale.numel() != self.hidden_dim
            or self._prefill_feature_scale.device != prompt_tokens.device
            or self._prefill_feature_scale.dtype != torch.float32
        ):
            self._prefill_feature_scale = torch.linspace(
                1.0 / max(self.hidden_dim, 1),
                1.0,
                self.hidden_dim,
                device=prompt_tokens.device,
                dtype=torch.float32,
            )
        # bf16 holds integers exactly only up to 256, so a bf16 product would
        # merge nearby token ids. Build the angles in float32 and round only
        # the sine features to bf16.
        angles = self._prefill_angle_buffer[:numel].view(shape)
        angles.copy_(prompt_tokens.unsqueeze(-1).expand(shape))
        angles.mul_(self._prefill_feature_scale)
        prefill_input = self._prefill_input_buffer[:numel].view(shape)
        torch.sin(angles, out=prefill_input)
        return prefill_input

    def prefill(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """Prefill over the full prompt (compute-bound path)."""
        x = self._prefill_input(prompt_tokens)
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return x.mean(dim=1, keepdim=True)

    def decode_step(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Advance the decode state by one token-equivalent step."""
        x = kv_cache
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return x

    def decode(
        self,
        kv_cache: torch.Tensor,
        num_tokens: int = 16,
        *,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a small number of tokens (memory-bound path)."""
        token_count = int(num_tokens)
        if token_count <= 0:
            return kv_cache[:, :0, :]
        if token_count == 1:
            return self.decode_step(kv_cache)

        x = kv_cache
        output_shape = (kv_cache.shape[0], token_count, kv_cache.shape[-1])
        if output_buffer is None:
            output = torch.empty(output_shape, device=kv_cache.device, dtype=kv_cache.dtype)
        else:
            if (
                output_buffer.shape != output_shape
                or output_buffer.device != kv_cache.device
                or output_buffer.dtype != kv_cache.dtype
            ):
                raise ValueError("output_buffer does not match decode shape/device/dtype")
            output = output_buffer
        for token_idx in range(token_count):
            x = self.decode_step(x)
            output[:, token_idx : token_idx + 1, :].copy_(x)
        return output
