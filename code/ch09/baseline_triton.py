"""baseline_triton.py - Baseline Triton matmul wrapper."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def baseline_elementwise(input_tensor: torch.Tensor, output: torch.Tensor, **_: int) -> None:
    """
    Baseline element-wise operation using standard PyTorch.
    This is compared against Triton's optimized kernel in the optimized version.
    """
    # Standard PyTorch operation: output = input * 2.0 + 1.0
    output.copy_(input_tensor * 2.0 + 1.0)


class BaselineTritonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline element-wise operation using standard PyTorch (compared to Triton kernel)."""

    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.N = 1_000_000
        # Triton benchmark - fixed N for kernel comparison
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._output_buffer = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range("baseline_triton"):
            if self._output_buffer is None:
                raise RuntimeError("setup() must initialize _output_buffer")
            baseline_elementwise(self.input, self._output_buffer)
            self.output = self._output_buffer
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
            batch_size=self.input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            timing_method="wall_clock",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(self.N * 2),
            total_bytes=float(self.N * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp32",
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTritonBenchmark()
