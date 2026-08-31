"""Harness wrapper for the scalar baseline in the strict `tma_copy` benchmark.

The pair now requires real TMA capability on the host and compares a scalar
baseline against the descriptor-backed TMA path under the same name.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.hardware_capabilities import ensure_tma_box_supported


class BaselineTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the scalar baseline for the strict `tma_copy` pair."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 25
        lookahead = 64
        source_reads_per_output = 3
        bytes_per_element = (source_reads_per_output + 1) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_tma_copy",
            friendly_name="Scalar Neighbor Gather Copy",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "N": n_elems,
                "lookahead": lookahead,
                "source_reads_per_output": source_reads_per_output,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_element),
        )

    def setup(self) -> None:
        ensure_tma_box_supported((64, 64), description="tma_copy")
        super().setup()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return {
            "copy.async_pipeline_1d": 0.0,
            "copy.tensor_map_2d_requested": 0.0,
            "copy.tensor_map_2d_runtime_candidate": 0.0,
        }

def get_benchmark() -> BaselineTMACopyBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineTMACopyBenchmark()
