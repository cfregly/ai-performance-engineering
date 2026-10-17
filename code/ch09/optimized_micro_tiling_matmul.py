"""Python harness wrapper for optimized_micro_tiling_matmul.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedMicroTilingMatmulBenchmark(CudaBinaryBenchmark):
    """Wraps the shared-memory matmul in which each thread accumulates a register micro-tile."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_micro_tiling_matmul",
            friendly_name="Optimized Micro Tiling Matmul",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for micro_tiling_matmul."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp32",
        )
    # get_verify_output inherited from CudaBinaryBenchmark - uses checksum from -DVERIFY=1 build

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return simple_signature(batch_size=1, dtype="float32", workload=1).to_dict()


def get_benchmark() -> OptimizedMicroTilingMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedMicroTilingMatmulBenchmark()


