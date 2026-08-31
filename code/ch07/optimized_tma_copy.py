"""Harness wrapper for the optimized strict `tma_copy` path.

`tma_copy` now means a real tensor-map/TMA-capable run only. Unsupported
hosts skip instead of publishing an async-pipeline fallback under the same
benchmark name.
"""

from __future__ import annotations
from typing import Optional

from pathlib import Path

import torch

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.hardware_capabilities import ensure_tma_box_supported


class OptimizedTMACopyBenchmark(CudaBinaryBenchmark):
    """Wrap the descriptor-backed TMA neighbor-copy binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 25
        lookahead = 64
        source_reads_per_output = 3
        bytes_per_element = (source_reads_per_output + 1) * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_copy",
            friendly_name="Pipeline + Tensor-Map Neighbor Copy",
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
        cuda_version = torch.version.cuda or "0.0"
        cuda_major = int(cuda_version.split(".", maxsplit=1)[0])
        return {
            "copy.async_pipeline_1d": 1.0,
            "copy.tensor_map_2d_requested": 1.0,
            "copy.tensor_map_2d_runtime_candidate": 1.0 if cuda_major >= 13 else 0.0,
            "copy.tensor_map_2d_required": 1.0,
        }

def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedTMACopyBenchmark()
