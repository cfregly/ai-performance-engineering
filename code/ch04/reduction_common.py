"""Shared helpers for ch04 inference-style MLP benchmarks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ReusableReductionMlp(nn.Module):
    """Two-layer inference MLP with scratch buffers for the no-grad fast path."""

    def __init__(self, hidden_dim: int, inner_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = tuple(x.shape[:-1])
        fc1_shape = (*prefix, self.fc1.out_features)
        fc2_shape = (*prefix, self.fc2.out_features)
        if (
            self._fc1_buffer is None
            or self._fc1_buffer.shape != fc1_shape
            or self._fc1_buffer.device != x.device
            or self._fc1_buffer.dtype != x.dtype
        ):
            self._fc1_buffer = torch.empty(fc1_shape, device=x.device, dtype=x.dtype)
        if (
            self._fc2_buffer is None
            or self._fc2_buffer.shape != fc2_shape
            or self._fc2_buffer.device != x.device
            or self._fc2_buffer.dtype != x.dtype
        ):
            self._fc2_buffer = torch.empty(fc2_shape, device=x.device, dtype=x.dtype)
        return self._fc1_buffer, self._fc2_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        torch.matmul(x, self.fc1.weight.t(), out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.matmul(fc1_out, self.fc2.weight.t(), out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out
