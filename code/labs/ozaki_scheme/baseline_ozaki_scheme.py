"""Baseline wrapper for the native FP64 Ozaki scheme anchor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark


class BaselineOzakiSchemeBenchmark(CudaBinaryBenchmark):
    """Native FP64 accuracy anchor for the Ozaki scheme lab."""

    def __init__(self) -> None:
        self._shape = (4096, 4096, 4096)
        self._run_args = [
            "--m", str(self._shape[0]),
            "--n", str(self._shape[1]),
            "--k", str(self._shape[2]),
            "--warmup", "3",
            "--iters", "10",
            "--seed", "2026",
            "--input-scale", "0.001",
        ]
        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name="baseline_ozaki_scheme",
            friendly_name="Baseline Ozaki Scheme",
            iterations=1,
            warmup=5,
            timeout_seconds=240,
            run_args=self._run_args,
            workload_params={
                "M": self._shape[0],
                "N": self._shape[1],
                "K": self._shape[2],
                "dtype": "float64",
                "input_scale": 0.001,
                "matmul_iters": 10,
            },
        )
        bytes_per_iteration = float(
            (self._shape[0] * self._shape[1] +
             self._shape[0] * self._shape[2] +
             self._shape[1] * self._shape[2]) * 8
        )
        flops_per_iteration = float(2 * self._shape[0] * self._shape[1] * self._shape[2])
        self.register_workload_metadata(
            bytes_per_iteration=bytes_per_iteration,
            custom_units_per_iteration=flops_per_iteration,
            custom_unit_name="FLOPs",
        )

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=1,
            dtype="float64",
            m=self._shape[0],
            n=self._shape[1],
            k=self._shape[2],
            input_scale=0.001,
            iters=10,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineOzakiSchemeBenchmark()
