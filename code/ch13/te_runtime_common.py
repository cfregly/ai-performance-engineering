"""Shared runtime initialization helpers for Transformer Engine benchmarks."""

from __future__ import annotations

import argparse
import ctypes
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import torch

from core.env import apply_env_defaults

_TORCH_CUDA_LIBS = (
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libtorch_nvshmem.so",
    "libc10_cuda.so",
)

_TE_PRECISION_OUTPUT_TOLERANCES: dict[str, tuple[float, float]] = {
    "prediction": (0.4, 1.0),
    "parameter.fc1.weight": (0.001, 0.00075),
    "parameter.fc1.bias": (0.001, 0.00005),
    "parameter.fc2.weight": (0.001, 0.00075),
    "parameter.fc2.bias": (0.001, 0.00005),
}

TE_PRECISION_DEFAULT_BATCH_SIZE = 256


def parse_te_precision_batch_size(
    argv: Iterable[str],
    *,
    default: int = TE_PRECISION_DEFAULT_BATCH_SIZE,
) -> int:
    """Parse the shared batch override used by both TE precision arms."""
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        exit_on_error=False,
    )
    parser.add_argument("--batch-size", type=int, default=default)
    try:
        args, _ = parser.parse_known_args(list(argv))
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError("--batch-size must be a positive integer") from exc
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    return int(args.batch_size)


def get_te_precision_output_tolerances() -> dict[str, tuple[float, float]]:
    """Return the calibrated full-output policy for the TE2.18 precision pair."""
    return dict(_TE_PRECISION_OUTPUT_TOLERANCES)


@lru_cache(maxsize=1)
def ensure_te_runtime_initialized() -> None:
    """Apply runtime defaults only when a TE-backed benchmark actually runs."""
    apply_env_defaults()
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in _TORCH_CUDA_LIBS:
        candidate = torch_lib_dir / name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
