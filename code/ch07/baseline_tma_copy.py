"""Harness wrapper for the scalar baseline in the strict `tma_copy` benchmark.

Both arms time the same transform on a 4096x4096 float32 matrix split into
64x64 tiles. Each output blends its element with the elements 1 and 64 places
later in the tile's row-major order, clamped to the tile's last element, so
every tap stays inside the tile. This baseline reads all three taps with
scalar global loads. The optimized arm launches the same grid and copies each
tile into shared memory with one descriptor-backed TMA load.

Each binary checks its full output exactly against the same CPU reference and
exits nonzero on any mismatch. Each prints `TIME_MS` for the average of its 20
timed launches. In a VERIFY build, each prints a checksum of the full timed
output for the harness to compare. The pair runs only on hosts with real TMA
support.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.hardware_capabilities import ensure_tma_box_supported

# The shape both CUDA binaries hard-code. test_audit_fix_o13_w2011.py checks
# these values against both sources.
TMA_COPY_WORKLOAD = {
    "N": 4096 * 4096,
    "rows": 4096,
    "cols": 4096,
    "tile_rows": 64,
    "tile_cols": 64,
    "lookahead": 64,
    "source_reads_per_output": 3,
    "dtype": "float32",
}
# Bytes count compulsory traffic, one 4-byte read and one 4-byte write per
# element. The two neighbor taps reread elements of the same tile, so they are
# not counted. The optimized binary's bandwidth line uses the same
# 2 * matrix_bytes.
TMA_COPY_BYTES_PER_ITERATION = float(TMA_COPY_WORKLOAD["N"] * 2 * 4)


class BaselineTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the scalar baseline for the strict `tma_copy` pair."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_copy",
            friendly_name="Scalar Neighbor Gather Copy",
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
        """Return copy-path metrics for the timed scalar kernel."""
        return {
            "copy.tensor_map_2d_requested": 0.0,
            "copy.tensor_map_2d_runtime_candidate": 0.0,
        }

def get_benchmark() -> BaselineTMACopyBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineTMACopyBenchmark()
