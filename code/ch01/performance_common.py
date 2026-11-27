"""Shared helpers for Chapter 1 training-loop performance benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from core.benchmark.metrics import compute_environment_metrics
from core.utils.compile_utils import configure_tf32, restore_tf32


TF32State = tuple[tuple[str | None, str | None], str]


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


def preallocate_fused_microbatches(
    microbatches: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    fusion: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if fusion <= 0:
        raise ValueError("fusion must be positive")
    if len(microbatches) != len(targets):
        raise ValueError("microbatches and targets must have the same length")

    fused_batches: list[torch.Tensor] = []
    fused_targets: list[torch.Tensor] = []
    for start in range(0, len(microbatches), fusion):
        batch_group = microbatches[start : start + fusion]
        target_group = targets[start : start + fusion]
        rows = sum(int(batch.shape[0]) for batch in batch_group)
        target_rows = sum(int(target.shape[0]) for target in target_group)
        fused_batch = batch_group[0].new_empty((rows, *batch_group[0].shape[1:]))
        fused_target = target_group[0].new_empty((target_rows, *target_group[0].shape[1:]))

        row_offset = 0
        for batch in batch_group:
            next_row_offset = row_offset + int(batch.shape[0])
            fused_batch[row_offset:next_row_offset].copy_(batch)
            row_offset = next_row_offset

        target_offset = 0
        for target in target_group:
            next_target_offset = target_offset + int(target.shape[0])
            fused_target[target_offset:next_target_offset].copy_(target)
            target_offset = next_target_offset

        fused_batches.append(fused_batch)
        fused_targets.append(fused_target)
    return fused_batches, fused_targets


def set_tf32_state(enabled: bool) -> TF32State:
    """Set matmul precision and CUDA backend policy, then return their prior settings."""
    matmul_precision = torch.get_float32_matmul_precision()
    backend_state = configure_tf32(enable_matmul=enabled, enable_cudnn=enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return backend_state, matmul_precision


def restore_tf32_state(state: TF32State | None) -> None:
    """Restore a snapshot returned by set_tf32_state()."""
    if state is None:
        return
    backend_state, matmul_precision = state
    restore_tf32(backend_state)
    torch.set_float32_matmul_precision(matmul_precision)


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
