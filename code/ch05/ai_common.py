"""Shared helpers for Chapter 5 AI orchestration benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyBlock(nn.Module):
    """Shared tiny MLP block used by the Chapter 5 AI benchmarks."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BufferedTinyBlock(nn.Module):
    """TinyBlock variant that reuses inference buffers for repeated calls."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self._hidden_buffer: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self._linear1_weight_t: torch.Tensor | None = None
        self._linear2_weight_t: torch.Tensor | None = None

    def cache_weight_views(self) -> None:
        self._linear1_weight_t = self.linear1.weight.t()
        self._linear2_weight_t = self.linear2.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        numel = 1
        for dim in shape:
            numel *= dim
        cached = getattr(self, name)
        if (
            not isinstance(cached, torch.Tensor)
            or cached.device != device
            or cached.dtype != dtype
            or cached.numel() < numel
        ):
            cached = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, cached)
        return cached[:numel].view(shape)

    def _ensure_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = tuple(x.shape[:-1])
        hidden_shape = (*prefix, self.linear1.out_features)
        output_shape = (*prefix, self.linear2.out_features)
        hidden = self._workspace("_hidden_buffer", hidden_shape, device=x.device, dtype=x.dtype)
        output = self._workspace("_output_buffer", output_shape, device=x.device, dtype=x.dtype)
        return hidden, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.linear2(self.relu(self.linear1(x)))

        if self._linear1_weight_t is None or self._linear2_weight_t is None:
            self.cache_weight_views()
        hidden, output = self._ensure_buffers(x)
        torch.matmul(x, self._linear1_weight_t, out=hidden)
        if self.linear1.bias is not None:
            hidden.add_(self.linear1.bias)
        torch.relu_(hidden)
        torch.matmul(hidden, self._linear2_weight_t, out=output)
        if self.linear2.bias is not None:
            output.add_(self.linear2.bias)
        return output


def compute_ai_workload_metrics(
    *,
    batch_size: int,
    hidden_dim: int,
    num_blocks: int,
    parameter_count: int,
) -> dict:
    """Return metrics derived from the actual benchmark workload."""
    return {
        "ai.batch_size": float(batch_size),
        "ai.hidden_dim": float(hidden_dim),
        "ai.num_blocks": float(num_blocks),
        "ai.parameters_millions": float(parameter_count) / 1_000_000.0,
        "ai.activation_elements_per_iteration": float(batch_size * hidden_dim * num_blocks),
    }
