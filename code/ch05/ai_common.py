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
