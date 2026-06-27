"""Shared helpers for Chapter 1 training-loop performance benchmarks."""

from __future__ import annotations

from typing import Tuple

import torch
from core.benchmark.metrics import compute_environment_metrics


def seed_chapter1(seed: int = 42) -> None:
    """Seed CPU and CUDA deterministically for benchmark reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_training_mlp(hidden_dim: int) -> torch.nn.Sequential:
    """Return the small MLP used by the Chapter 1 goodput benchmarks."""
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_dim, 10),
    )


def capture_tf32_state() -> Tuple[bool, bool | None]:
    """Snapshot the current TF32 backend settings so callers can restore them."""
    cudnn_state = None
    if torch.backends.cudnn.is_available():
        cudnn_state = bool(torch.backends.cudnn.allow_tf32)
    return bool(torch.backends.cuda.matmul.allow_tf32), cudnn_state


def set_tf32_state(enabled: bool) -> None:
    """Enable or disable TF32 across CUDA matmul/cuDNN backends."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = enabled


def restore_tf32_state(state: Tuple[bool, bool | None] | None) -> None:
    """Restore a snapshot returned by capture_tf32_state()."""
    if state is None:
        return
    matmul_state, cudnn_state = state
    set_tf32_state(matmul_state)
    if cudnn_state is None or not torch.backends.cudnn.is_available():
        return
    torch.backends.cudnn.allow_tf32 = cudnn_state


def get_environment_custom_metrics() -> dict:
    """Return real runtime environment metrics for Chapter 1 benchmarks."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_memory_gb = 0.0
    if gpu_count > 0:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)
    return compute_environment_metrics(
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        cuda_version=torch.version.cuda or "",
        pytorch_version=torch.__version__,
    )
