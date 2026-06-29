"""Fixed retained-bit Ozaki emulation variant for the cuBLAS floating-point emulation lab."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark

_FLOAT_PATTERNS = {
    "tflops": re.compile(r"TFLOPS:\s*([0-9.eE+-]+)"),
    "max_abs_error": re.compile(r"MAX_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "mean_abs_error": re.compile(r"MEAN_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "retained_bits": re.compile(r"RETAINED_BITS:\s*(-?\d+)"),
    "emulation_used": re.compile(r"EMULATION_USED:\s*(\d+)"),
}


class OptimizedOzakiSchemeFixedBenchmark(CudaBinaryBenchmark):
    """Fixed retained-bit variant using cuBLAS floating-point emulation."""

    def __init__(self) -> None:
        self._shape = (4096, 4096, 4096)
        self._parsed_metrics: Optional[dict[str, float]] = None
        self._run_args = [
            "--m", str(self._shape[0]),
            "--n", str(self._shape[1]),
            "--k", str(self._shape[2]),
            "--warmup", "3",
            "--iters", "10",
            "--seed", "2026",
            "--input-scale", "0.001",
            "--emulation-strategy", "eager",
            "--fixed-bits", "12",
        ]
        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name="optimized_ozaki_scheme_fixed",
            friendly_name="Optimized Ozaki Scheme Fixed",
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
                "emulation_strategy": "eager",
                "fixed_bits": 12,
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
        self.story_metadata = {
            "scheme": "Ozaki",
            "variant": "fixed",
            "source_library": "cuBLAS",
            "retained_bits_mode": "fixed mantissa budget",
            "emulation_strategy": "eager",
            "adaptive_behavior": False,
        }

    def benchmark_fn(self) -> None:
        super().benchmark_fn()
        self._parsed_metrics = None

    def _parse_last_metrics(self) -> dict[str, float]:
        stdout = self.last_stdout or ""
        metrics: dict[str, float] = {}
        for key, pattern in _FLOAT_PATTERNS.items():
            match = pattern.search(stdout)
            if not match:
                continue
            value = float(match.group(1))
            metrics[key] = value
        return metrics

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
        if self._parsed_metrics is None and self.last_stdout is not None:
            self._parsed_metrics = self._parse_last_metrics()
        return self._parsed_metrics

    def get_output_tolerance(self) -> tuple[float, float]:
        return (1e-2, 1e-2)


def get_benchmark() -> BaseBenchmark:
    return OptimizedOzakiSchemeFixedBenchmark()
