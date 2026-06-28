"""Shared model and setup constants for the Chapter 20 autotuning pair."""

from __future__ import annotations

import torch
import torch.nn as nn

AUTOTUNING_SETUP_PREWARM_ITERS = 10


class AutotuneModel(nn.Module):
    """Pointwise-heavy block that benefits from compiler fusion."""

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 0.01
        y.mul_(self.scale)
        y.add_(self.bias)
        torch.nn.functional.silu(y, inplace=True)
        y.mul_(1.0001).add_(0.0001)
        y.mul_(0.999).add_(0.001)
        torch.nn.functional.silu(y, inplace=True)
        y.mul_(1.0001).add_(0.0001)
        y.mul_(0.999).add_(0.001)
        torch.nn.functional.silu(y, inplace=True)
        y.mul_(1.0001).add_(0.0001)
        y.mul_(0.999).add_(0.001)
        torch.nn.functional.silu(y, inplace=True)
        y.mul_(1.0001).add_(0.0001)
        y.mul_(0.999).add_(0.001)
        return y
