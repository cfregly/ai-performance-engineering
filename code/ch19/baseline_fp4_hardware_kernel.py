"""Python harness wrapper for baseline_fp4_hardware_kernel.cu (Chapter 19).

This benchmark participates in the harness as a *comparable* baseline for the
FP4 hardware-kernel example.

Both wrappers verify the full FP16 output matrix C, not a checksum. Each run
passes --dump-output to the binary, which writes C after its timed loop.

Pairs with: optimized_fp4_hardware_kernel.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import torch

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark

_M = 1024
_N = 1024
_K = 1024
_OUTPUT_BYTES = _M * _N * torch.float16.itemsize


class _FP4HardwareKernelBinaryBenchmark(CudaBinaryBenchmark):
    """Runs one FP4 GEMM binary and keeps the full C it wrote after timing."""

    def __init__(self, *, binary_name: str, friendly_name: str, variant: str) -> None:
        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name=binary_name,
            friendly_name=friendly_name,
            iterations=1,
            warmup=1,
            timeout_seconds=300,
            workload_params={
                "batch_size": 1,
                "dtype": "nvfp4_e2m1",
                "M": _M,
                "N": _N,
                "K": _K,
            },
        )
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._variant = variant
        self.output: torch.Tensor | None = None
        self._output_path: Path | None = None
        self._output_dir: tempfile.TemporaryDirectory[str] | None = None

    def setup(self) -> None:
        super().setup()
        self._output_dir = tempfile.TemporaryDirectory(prefix="aisp_fp4_hardware_kernel_")
        self._output_path = Path(self._output_dir.name) / "timed_output.f16"
        self.run_args = ["--dump-output", str(self._output_path)]

    def benchmark_fn(self) -> None:
        if self._output_path is None:
            raise RuntimeError("setup() must create the timed output path")
        self.output = None
        # The binaries print no dump marker. Remove the previous dump so a run
        # that does not write C fails here instead of passing with stale data.
        self._output_path.unlink(missing_ok=True)
        super().benchmark_fn()
        if not self._output_path.is_file():
            raise RuntimeError(
                f"{self.binary_name} did not write --dump-output {self._output_path}"
            )
        actual_bytes = self._output_path.stat().st_size
        if actual_bytes != _OUTPUT_BYTES:
            raise RuntimeError(
                f"Timed output dump has {actual_bytes} bytes, expected {_OUTPUT_BYTES}"
            )
        # The binaries write C column-major, so element (row, col) is at row + col * M.
        column_major = torch.from_file(
            str(self._output_path),
            shared=False,
            size=_M * _N,
            dtype=torch.float16,
        )
        self.output = column_major.reshape(_N, _M).t().contiguous()

    def get_verify_output(self) -> torch.Tensor:
        if self._last_result is None or self.output is None:
            raise RuntimeError("get_verify_output() requires a completed timed execution")
        return self.output.detach().clone()

    def validate_result(self) -> Optional[str]:
        error = super().validate_result()
        if error is not None:
            return error
        if self.output is None or tuple(self.output.shape) != (_M, _N):
            return "Timed binary did not retain the complete output"
        return None

    def teardown(self) -> None:
        self.output = None
        self._output_path = None
        self.run_args = []
        if self._output_dir is not None:
            self._output_dir.cleanup()
            self._output_dir = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        if self.last_time_ms is None:
            raise RuntimeError("Benchmark did not capture TIME_MS output")
        flops = 2.0 * _M * _N * _K
        tflops = flops / (float(self.last_time_ms) * 1e9)
        return {"tflops": float(tflops), "variant": self._variant}

    def get_input_signature(self) -> dict:
        return simple_signature(batch_size=1, dtype="nvfp4_e2m1", M=_M, N=_N, K=_K).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        # Verification compares every element of the FP16 C matrix. Both variants
        # multiply the same dequantized NVFP4 values but accumulate in a different
        # FP32 order, so C can differ by a few FP16 rounding steps.
        return (1e-2, 1e-2)


class BaselineFP4HardwareKernelBenchmark(_FP4HardwareKernelBinaryBenchmark):
    """Runs baseline_fp4_hardware_kernel.cu (manual NVFP4 GEMM)."""

    def __init__(self) -> None:
        super().__init__(
            binary_name="baseline_fp4_hardware_kernel",
            friendly_name="Baseline Fp4 Hardware Kernel",
            variant="baseline_manual_fp4",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineFP4HardwareKernelBenchmark()
