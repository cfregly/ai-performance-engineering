"""Harness wrapper for the optimized strict `tma_copy` path.

`tma_copy` now means a real tensor-map/TMA-capable run only. Unsupported
hosts skip instead of publishing an async-pipeline fallback under the same
benchmark name.

The binary times descriptor_tma_2d_copy_kernel on the same 4096x4096 float32
matrix and 64x64 tiles as the scalar baseline. Each block copies its tile into
shared memory with one descriptor-backed TMA load, blends every element with
the elements 1 and 64 places later in the tile, and writes the tile back with
a TMA store. Before timing, the binary checks a non-square shape with partial
tiles. After timing, it checks its full output exactly against the same CPU
reference as the baseline and exits nonzero on any mismatch. It prints
`TIME_MS` for the average of its 20 timed launches. In a VERIFY build, it
prints a checksum of the full timed output.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

import torch

from ch07.baseline_tma_copy import TMA_COPY_BYTES_PER_ITERATION, TMA_COPY_WORKLOAD
from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.hardware_capabilities import ensure_tma_box_supported


class OptimizedTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the descriptor-backed 2D TMA tiled-neighbor binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_copy",
            friendly_name="Descriptor-Backed 2D TMA Tiled-Neighbor Transform",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params=dict(TMA_COPY_WORKLOAD),
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )
        self.register_workload_metadata(
            bytes_per_iteration=TMA_COPY_BYTES_PER_ITERATION,
        )

    def setup(self) -> None:
        ensure_tma_box_supported((64, 64), description="tma_copy")
        super().setup()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return copy-path metrics for the timed 2D TMA kernel."""
        cuda_version = torch.version.cuda or "0.0"
        cuda_major = int(cuda_version.split(".", maxsplit=1)[0])
        return {
            "copy.tensor_map_2d_requested": 1.0,
            "copy.tensor_map_2d_runtime_candidate": 1.0 if cuda_major >= 13 else 0.0,
            "copy.tensor_map_2d_required": 1.0,
        }

def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedTMACopyBenchmark()
