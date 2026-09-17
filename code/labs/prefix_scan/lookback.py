"""CUDA extension adapter for single-pass decoupled look-back."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA decoupled look-back requires an NVIDIA GPU")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("SKIPPED: CUDA toolkit is required to build decoupled look-back")
    return load(
        name="prefix_scan_decoupled_lookback",
        sources=[str(Path(__file__).with_name("scan_lookback.cu"))],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class LookbackPlan:
    def __init__(self, x):
        if x.device.type != "cuda":
            raise RuntimeError("SKIPPED: CUDA decoupled look-back requires CUDA input")
        if (
            x.dtype != torch.int32
            or x.ndim != 1
            or not x.is_contiguous()
            or not 0 < x.numel() <= 2**31 - 1
        ):
            raise ValueError(
                "look-back requires a nonempty contiguous int32 vector with at most INT_MAX elements"
            )
        self.extension = load_extension()
        self.x = x
        self.output = torch.empty_like(x)
        tiles = (x.numel() + 255) // 256
        self.next_tile = torch.empty(1, device=x.device, dtype=torch.int32)
        self.aggregates = torch.empty(tiles, device=x.device, dtype=torch.int32)
        self.prefixes = torch.empty_like(self.aggregates)
        self.states = torch.empty_like(self.aggregates)

    def run(self):
        # Per-invocation state reset is part of the measured workload.
        self.next_tile.zero_()
        self.states.zero_()
        self.extension.scan(
            self.x, self.output, self.next_tile, self.aggregates, self.prefixes, self.states
        )
        return self.output
