"""Shared base for Chapter 8 AI optimization benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension


class AiOptimizationBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    rows: int = 1 << 15  # 32,768 samples
    cols: int = 512
    inner_iterations: int = 8
    nvtx_label: str = "ai_optimization"
    output_tolerance = (0.1, 1.0)

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Chapter 8 AI optimization benchmarks")
        self.device = torch.device("cuda")
        self.extension = None
        self.inputs: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._inner_iteration_range = range(self.inner_iterations)
        self.register_workload_metadata(requests_per_iteration=float(self.inner_iterations))

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name="ch08_ai_optimization_kernels",
            cuda_source_file=str(Path(__file__).with_name("ai_optimization_kernels.cu")),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.inputs = torch.randn(
            self.rows,
            self.cols,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        self.weights = torch.randn(self.cols, device=self.device, dtype=torch.float32).contiguous()
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

    def capture_verification_payload(self) -> None:
        """Register verification payload once after timing."""
        if self.inputs is None or self.weights is None or self.output is None:
            raise RuntimeError("capture_verification_payload() requires initialized tensors")
        self._set_verification_payload(
            inputs={"inputs": self.inputs, "weights": self.weights},
            output=self.output.detach(),
            batch_size=self.rows,
            parameter_count=int(self.cols),
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=self.output_tolerance,
        )

    def teardown(self) -> None:
        self.inputs = None
        self.weights = None
        self.output = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    def _invoke_kernel(self) -> None:
        raise NotImplementedError

    def _validate_correctness(self) -> None:
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None

        reference = torch.tanh(torch.matmul(self.inputs, self.weights))
        torch.cuda.synchronize()
        max_error = scalar_tensor_to_float(torch.max(torch.abs(reference - self.output)))
        if max_error > 1e-3:
            raise RuntimeError(f"AI optimization kernel validation failed (max error={max_error:.4f})")

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            timing_method="wall_clock",
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output buffer not initialized"
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
        """Return AI optimization kernel metrics for roofline analysis."""
        flops = float(self.rows * self.cols * 2 * self.inner_iterations)  # matmul + tanh approx
        bytes_transferred = float((self.rows * self.cols + self.cols + self.rows) * 4 * self.inner_iterations)
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.cols": float(self.cols),
            f"{self.nvtx_label}.flops": flops,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
            f"{self.nvtx_label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
        }
