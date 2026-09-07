"""Seed a launched benchmark worker from its parent-provided seed."""

from __future__ import annotations

import torch


def apply_worker_seed(seed: int) -> None:
    """Apply the transported harness seed without deriving a rank-local stream."""
    worker_seed = int(seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
