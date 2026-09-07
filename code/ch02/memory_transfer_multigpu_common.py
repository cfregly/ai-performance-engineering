"""Shared harness support for the Chapter 2 multi-GPU transfer pair."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BenchmarkConfig

TRANSFER_ELEMENTS = 100 * 1024 * 1024
TRANSFER_BYTES = TRANSFER_ELEMENTS * 4
TRANSFER_INNER_ITERATIONS = 100
TRANSFER_PATTERN_PERIOD = 4093


def load_complete_transfer_output(path: Path, *, expected_elements: int) -> torch.Tensor:
    """Map one complete raw float32 destination dump, rejecting truncation."""
    expected_bytes = expected_elements * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Timed destination dump has {actual_bytes} bytes; expected {expected_bytes} "
            f"for {expected_elements} float32 elements"
        )
    return torch.from_file(
        str(path),
        shared=False,
        size=expected_elements,
        dtype=torch.float32,
    )


def validate_transfer_pattern(output: torch.Tensor) -> None:
    """Validate every element against the deterministic CUDA input pattern."""
    if output.device.type != "cpu" or output.dtype != torch.float32 or output.ndim != 1:
        raise RuntimeError("Timed destination must be a one-dimensional CPU float32 tensor")
    period = torch.arange(TRANSFER_PATTERN_PERIOD, dtype=torch.float32)
    period.sub_(2046).mul_(1.0 / 256.0)
    full_periods, remainder = divmod(output.numel(), TRANSFER_PATTERN_PERIOD)
    covered = full_periods * TRANSFER_PATTERN_PERIOD
    valid = True
    if full_periods:
        valid = torch.equal(
            output[:covered].view(full_periods, TRANSFER_PATTERN_PERIOD),
            period.expand(full_periods, -1),
        )
    if valid and remainder:
        valid = torch.equal(output[covered:], period[:remainder])
    if valid:
        return

    chunk_elements = 1 << 20
    for start in range(0, output.numel(), chunk_elements):
        end = min(start + chunk_elements, output.numel())
        offsets = torch.arange(start, end, dtype=torch.int64)
        expected = period[offsets.remainder(TRANSFER_PATTERN_PERIOD)]
        mismatches = torch.nonzero(output[start:end] != expected, as_tuple=False)
        if mismatches.numel():
            index = start + int(mismatches[0, 0])
            raise RuntimeError(
                "Timed destination data mismatch at element "
                f"{index}: expected {float(expected[index - start])}, got {float(output[index])}"
            )
    raise RuntimeError("Timed destination data mismatch")


class MemoryTransferMultiGPUBenchmark(CudaBinaryBenchmark):
    """CUDA-binary wrapper that retains the complete timed destination."""

    multi_gpu_required = True

    def __init__(self, *, chapter_dir: Path, binary_name: str, friendly_name: str) -> None:
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name=binary_name,
            friendly_name=friendly_name,
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "elements": TRANSFER_ELEMENTS,
                "bytes": TRANSFER_BYTES,
                "inner_iterations": TRANSFER_INNER_ITERATIONS,
                "dtype": "float32",
                "batch_size": 1,
            },
        )
        self.output: torch.Tensor | None = None
        self._output_path: Path | None = None
        self._output_dir: tempfile.TemporaryDirectory[str] | None = None
        self._output_pattern_validated = False

    def setup(self) -> None:
        super().setup()
        self._output_dir = tempfile.TemporaryDirectory(prefix="aisp_multigpu_transfer_")
        self._output_path = Path(self._output_dir.name) / "timed_destination.f32"
        self.run_args = ["--dump-output", str(self._output_path)]

    def benchmark_fn(self) -> None:
        if self._output_path is None:
            raise RuntimeError("setup() must create the timed destination path")
        self.output = None
        self._output_pattern_validated = False
        super().benchmark_fn()
        if self._last_result is None:
            raise RuntimeError("CUDA binary execution did not retain its reported timing")
        marker = f"OUTPUT_VALIDATED: {TRANSFER_ELEMENTS}"
        if marker not in self._last_result.raw_stdout:
            raise RuntimeError(f"CUDA binary did not confirm complete destination validation ({marker})")
        self.output = load_complete_transfer_output(
            self._output_path,
            expected_elements=TRANSFER_ELEMENTS,
        )

    def get_verify_output(self) -> torch.Tensor:
        if self._last_result is None or self.output is None:
            raise RuntimeError("get_verify_output() requires a completed timed binary execution")
        if not self._output_pattern_validated:
            validate_transfer_pattern(self.output)
            self._output_pattern_validated = True
        return self.output.detach().clone()

    def validate_result(self) -> str | None:
        error = super().validate_result()
        if error is not None:
            return error
        if self.output is None or self.output.numel() != TRANSFER_ELEMENTS:
            return "Timed binary did not retain the complete destination output"
        return None

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        config.multi_gpu_required = True
        return config

    def teardown(self) -> None:
        self.output = None
        self._output_path = None
        self.run_args = []
        if self._output_dir is not None:
            self._output_dir.cleanup()
            self._output_dir = None
        super().teardown()
