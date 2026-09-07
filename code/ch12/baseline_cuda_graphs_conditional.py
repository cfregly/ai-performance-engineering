"""Harness wrapper and shared verification contract for conditional CUDA graphs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark

CONDITIONAL_ELEMENTS = 1 << 16
CONDITIONAL_KERNEL_ITERS, CONDITIONAL_ITERATIONS = 1024, 5000
_OUTPUT_MARKER = f"OUTPUT_DUMPED: {CONDITIONAL_ELEMENTS}"


class _CudaGraphsConditionalBinaryBenchmark(CudaBinaryBenchmark):
    def __init__(self, *, binary_name: str, friendly_name: str) -> None:
        super().__init__(
            chapter_dir=Path(__file__).parent,
            binary_name=binary_name,
            friendly_name=friendly_name,
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "N": CONDITIONAL_ELEMENTS,
                "KERNEL_ITERS": CONDITIONAL_KERNEL_ITERS,
                "ITERS": CONDITIONAL_ITERATIONS,
                "dtype": "float32",
                "batch_size": 1,
            },
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            custom_units_per_iteration=float(CONDITIONAL_ELEMENTS * CONDITIONAL_ITERATIONS),
            custom_unit_name="element_updates",
        )
        self.output: torch.Tensor | None = None
        self._output_path: Path | None = None
        self._output_dir: tempfile.TemporaryDirectory[str] | None = None

    def setup(self) -> None:
        super().setup()
        self._output_dir = tempfile.TemporaryDirectory(prefix="aisp_cuda_graphs_conditional_")
        self._output_path = Path(self._output_dir.name) / "timed_output.f32"
        self.run_args = ["--dump-output", str(self._output_path)]

    def benchmark_fn(self) -> None:
        if self._output_path is None:
            raise RuntimeError("setup() must create the timed output path")
        self.output = None
        super().benchmark_fn()
        if self._last_result is None or _OUTPUT_MARKER not in self._last_result.raw_stdout:
            raise RuntimeError(
                f"CUDA binary did not confirm a complete output dump ({_OUTPUT_MARKER})"
            )
        expected_bytes = CONDITIONAL_ELEMENTS * torch.float32.itemsize
        actual_bytes = self._output_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Timed output dump has {actual_bytes} bytes; expected {expected_bytes}"
            )
        self.output = torch.from_file(
            str(self._output_path),
            shared=False,
            size=CONDITIONAL_ELEMENTS,
            dtype=torch.float32,
        )

    def get_verify_output(self) -> torch.Tensor:
        if self._last_result is None or self.output is None:
            raise RuntimeError("get_verify_output() requires a completed timed execution")
        return self.output.detach().clone()

    def validate_result(self) -> str | None:
        error = super().validate_result()
        if error is not None:
            return error
        if self.output is None or self.output.numel() != CONDITIONAL_ELEMENTS:
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


class BaselineCudaGraphsConditionalBenchmark(_CudaGraphsConditionalBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(
            binary_name="baseline_cuda_graphs_conditional",
            friendly_name="Baseline Cuda Graphs Conditional",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineCudaGraphsConditionalBenchmark()
