"""Shared utilities for loop unrolling benchmarks."""

from __future__ import annotations

from functools import partial
import math
from pathlib import Path
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.extension_loader_template import load_cuda_extension

_KERNEL_SOURCE = Path(__file__).with_name("loop_unrolling_kernels.cu")
_EXTENSION_NAME = "ch08_loop_unrolling_kernels"


resolve_device = partial(
    require_cuda_device,
    "CUDA required for Chapter 8 loop-unrolling benchmarks",
)


class LoopUnrollingBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Base class that manages CUDA extension loading and tensor setup."""

    rows: int = 1 << 17  # 131,072 rows
    elements_per_row: int = 512
    weight_period: int = 8
    nvtx_label: str = "loop_unrolling"
    output_tolerance = (5e-3, 5e-3)
    inner_iterations: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        bytes_per_iteration = float(
            self.rows * self.elements_per_row * 4 + self.weight_period * 4 + self.rows * 4
        )
        self.register_workload_metadata(
            bytes_per_iteration=bytes_per_iteration * self.inner_iterations,
            requests_per_iteration=float(self.inner_iterations),
        )
        self.extension = None
        self.inputs: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._inner_iteration_range = range(self.inner_iterations)
        # Loop unrolling benchmark: fixed dimensions for measurement

    def setup(self) -> None:
        self.extension = load_cuda_extension(
            extension_name=_EXTENSION_NAME,
            cuda_source_file=str(_KERNEL_SOURCE),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.inputs = torch.randn(
            self.rows,
            self.elements_per_row,
            device=self.device,
            dtype=torch.float32,
        )
        self.weights = torch.randn(
            self.weight_period,
            device=self.device,
            dtype=torch.float32,
        )
        self.output = None
        self._output_buffer = torch.empty(
            self.rows,
            device=self.device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            if self._output_buffer is None:
                raise RuntimeError("setup() must initialize the output buffer")
            self.output = self._output_buffer
            for _ in self._inner_iteration_range:
                self._invoke_kernel()
        if self.inputs is None or self.weights is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run after setup() initializes tensors")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "inputs": self.inputs,
                "weights": self.weights,
            },
            output=self.output.detach(),
            batch_size=self.rows,
            parameter_count=int(self.weight_period),
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

        repeats = math.ceil(self.elements_per_row / self.weight_period)
        tiled_weights = self.weights.repeat(repeats)[: self.elements_per_row]
        reference = (self.inputs * tiled_weights).sum(dim=1)
        torch.cuda.synchronize()
        max_error = scalar_tensor_to_float(torch.max(torch.abs(reference - self.output)))
        if max_error > 5e-3:
            raise RuntimeError(
                f"Loop unrolling kernel validation failed (max error={max_error:.4f})"
            )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
        )

    def validate_result(self) -> Optional[str]:
        if self.extension is None:
            return "CUDA extension not loaded"
        if self.inputs is None or self.weights is None:
            return "Inputs not initialized"
        if self.output is None:
            return "Output buffer not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return super().get_output_tolerance()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return loop unrolling optimization metrics."""
        total_elements = self.rows * self.elements_per_row
        flops = float(total_elements * 2 * self.inner_iterations)  # mul + add per element
        bytes_transferred = float((total_elements * 4 + self.weight_period * 4) * self.inner_iterations)
        return {
            f"{self.nvtx_label}.rows": float(self.rows),
            f"{self.nvtx_label}.elements_per_row": float(self.elements_per_row),
            f"{self.nvtx_label}.total_elements": float(total_elements),
            f"{self.nvtx_label}.weight_period": float(self.weight_period),
            f"{self.nvtx_label}.flops": flops,
            f"{self.nvtx_label}.bytes_transferred": bytes_transferred,
            f"{self.nvtx_label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
        }
