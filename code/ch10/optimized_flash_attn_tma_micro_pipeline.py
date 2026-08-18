"""Optimized FlashAttention-style micro-pipeline using cuda::pipeline/TMA."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedFlashAttnTmaMicroPipelineBenchmark(CudaBinaryBenchmark):
    """Runs the async three-stage flash-attn micro-pipeline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_flash_attn_tma_micro_pipeline",
            friendly_name="Optimized Flash Attn Tma Micro Pipeline",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
            run_args=(),
            requires_pipeline_api=True,
            require_tma_instructions=True,
            workload_params={
                "batch_size": 2048,
                "dtype": "float32",
                "seq_len": 4096,
                "d_head": 64,
                "tile_kv": 32,
                "threads": 128,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the async-copy workload without fake stage timing."""
        from ch10.benchmark_metrics_common import compute_pipeline_variant_metrics

        return compute_pipeline_variant_metrics(
            self._workload_params,
            num_stages=3,
            uses_tma=True,
        )

    def get_input_signature(self) -> dict:
        """Signature for optimized FlashAttn micro-pipeline."""
        return simple_signature(
            batch_size=2048,
            dtype="float32",
            seq_len=4096,
            d_head=64,
            tile_kv=32,
            threads=128,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttnTmaMicroPipelineBenchmark()
