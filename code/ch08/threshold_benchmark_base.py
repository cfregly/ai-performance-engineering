"""Shared threshold benchmark utilities for Chapter 8."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension

THRESHOLD_SECONDARY_SCALE = 1.5
THRESHOLD_INNER_SCALE = 0.85
THRESHOLD_OUTER_SCALE = 1.25


class ThresholdBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    rows: int = 1 << 26  # 64M elements to better amortize TMA setup on Blackwell bridge comparisons
    threshold: float = 2.5
    nvtx_label: str = "threshold"
    output_tolerance = (0.1, 1.0)
    inner_iterations: int = 8

    def __init__(self) -> None:
        super().__init__()
        self.inputs: Optional[torch.Tensor] = None
        self.outputs: Optional[torch.Tensor] = None
        self.host_inputs: Optional[torch.Tensor] = None
        self.extension = None
        self._inner_iteration_range = range(self.inner_iterations)
        self.register_workload_metadata(requests_per_iteration=float(self.inner_iterations))

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA required for Chapter 8 threshold benchmarks")

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_threshold_kernels",
            cuda_source_file=str(Path(__file__).with_name("threshold_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(42)
        device_inputs = self._generate_inputs()
        self.inputs = device_inputs.contiguous()
        self.host_inputs = self.inputs.cpu().pin_memory()
        self.outputs = torch.empty_like(self.inputs)

        self._invoke_kernel()
        torch.cuda.synchronize()
        self._validate_correctness()
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            for _ in self._inner_iteration_range:
                self._invoke_kernel()
        if self.inputs is None or self.outputs is None:
            raise RuntimeError("benchmark_fn() must run after setup() initializes tensors")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.outputs.detach(),
            batch_size=self.rows,
            parameter_count=0,
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=self.output_tolerance,
        )

    def teardown(self) -> None:
        self.inputs = None
        self.outputs = None
        self.host_inputs = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _generate_inputs(self) -> torch.Tensor:
        return torch.randn(self.rows, device=self.device, dtype=torch.float32) * 0.25

    def _validate_correctness(self) -> None:
        assert self.inputs is not None
        assert self.outputs is not None

        sine = torch.sin(self.inputs)
        cosine = torch.cos(self.inputs)
        magnitude = self.inputs.abs() + sine * cosine * 0.0001
        abs_inputs = self.inputs.abs()
        active = abs_inputs > self.threshold
        outer = abs_inputs > (self.threshold * THRESHOLD_SECONDARY_SCALE)

        scale = torch.where(outer, THRESHOLD_OUTER_SCALE, THRESHOLD_INNER_SCALE)
        reference = magnitude * scale
        reference.copysign_(self.inputs)
        reference.masked_fill_(active.logical_not_(), 0.0)
        torch.cuda.synchronize()
        max_error = scalar_tensor_to_float(torch.max(torch.abs(reference - self.outputs)))
        if max_error > 5e-3:
            raise RuntimeError(
                f"Threshold kernel validation failed (max error={max_error:.4f})"
            )

    def get_config(self) -> BenchmarkConfig:
        # Reduce runtime so full benchmark runs avoid harness watchdog on large data sets.
        return BenchmarkConfig(iterations=10, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.inputs is None or self.outputs is None:
            return "Buffers not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return threshold kernel optimization metrics."""
        bytes_transferred = float(self.rows * 4 * 2 * self.inner_iterations)  # float32 read + write
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.threshold": self.threshold,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
        }
