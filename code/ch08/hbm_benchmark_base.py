"""Shared utilities for Chapter 8 HBM benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension


class HBMBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    rows: int = 8192
    cols: int = 4096
    inner_iterations: int = 8
    nvtx_label: str = "hbm"
    output_tolerance = (1e-2, 5e-3)

    def __init__(self) -> None:
        super().__init__()
        # HBM benchmark - fixed dimensions to measure memory access patterns
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for HBM benchmarks")
        self.device = torch.device("cuda")
        bytes_per_iteration = float(self.rows * self.cols * 4 + self.rows * 4)
        self.register_workload_metadata(
            bytes_per_iteration=bytes_per_iteration * self.inner_iterations,
            requests_per_iteration=float(self.inner_iterations),
        )
        self.extension = None
        self.matrix_row: Optional[torch.Tensor] = None
        self.matrix_col: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.host_col: Optional[torch.Tensor] = None
        self._inner_iteration_range = range(self.inner_iterations)

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_hbm_kernels",
            cuda_source_file=str(Path(__file__).with_name("hbm_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        torch.manual_seed(42)
        host_row = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        )
        host_col = host_row.transpose(0, 1).contiguous()

        self.host_col = host_col.pin_memory()
        self.matrix_row = host_row.to(self.device, non_blocking=False).contiguous()
        self.matrix_col = self.host_col.to(self.device, non_blocking=False).contiguous()
        self.output = None
        self._output_buffer = torch.empty(self.rows, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            if self._output_buffer is None:
                raise RuntimeError("setup() must initialize the output buffer")
            self.output = self._output_buffer
            for _ in self._inner_iteration_range:
                self._invoke_kernel()
        if self.matrix_row is None or self.matrix_col is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run after setup() initializes tensors")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "matrix_row": self.matrix_row,
                "matrix_col": self.matrix_col,
            },
            output=self.output.detach(),
            batch_size=self.rows,
            parameter_count=int(self.rows * self.cols),
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=self.output_tolerance,
        )

    def teardown(self) -> None:
        self.matrix_row = None
        self.matrix_col = None
        self.output = None
        self._output_buffer = None
        self.host_col = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _mix_reference(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 1.00005 + torch.sin(tensor) * 0.00095 + torch.cos(tensor) * 0.00073

    def _validate_correctness(self) -> None:
        assert self.matrix_row is not None
        assert self.output is not None

        reference = self._mix_reference(self.matrix_row).sum(dim=1)
        torch.cuda.synchronize()
        max_error = scalar_tensor_to_float(torch.max(torch.abs(reference - self.output)))
        if max_error > 5e-3:
            raise RuntimeError(f"HBM kernel validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.output is None:
            return "Output buffer not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison - HBM kernel has numerical drift."""
        return super().get_output_tolerance()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return HBM optimization metrics for memory bandwidth analysis."""
        elements = self.rows * self.cols
        bytes_per_element = 4  # float32
        bytes_transferred = float(elements * bytes_per_element * self.inner_iterations)
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.cols": float(self.cols),
            f"{self.nvtx_label}.elements": float(elements),
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
        }
