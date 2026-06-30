"""baseline_warp_specialization_training.py - Eager fusion baseline.

Baseline: a "training-style epilogue" chain of elementwise ops executed in
eager mode. Each op materializes intermediates, stressing bandwidth and
kernel-launch overhead.

Optimized variant uses torch.compile to fuse the chain into fewer kernels,
reducing memory traffic and enabling warp-specialized schedules in Inductor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


def _epilogue_chain(
    x: torch.Tensor,
    scale0: torch.Tensor,
    bias0: torch.Tensor,
    scale1: torch.Tensor,
    bias1: torch.Tensor,
    scale2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    y = x
    y = y * scale0 + bias0
    y = F.silu(y)
    y = y * scale1 + bias1
    y = F.gelu(y)
    y = y * 0.5
    y = torch.sigmoid(y) * y
    y = y + bias2
    y = y * scale2
    y = torch.relu_(y)
    y = F.silu(y)
    y = y * 1.1 + 0.1
    y = F.gelu(y)
    return y


class BaselineWarpSpecializationTrainingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: eager elementwise chain (many kernels, many intermediates)."""

    def __init__(self) -> None:
        super().__init__()
        self.x: Optional[torch.Tensor] = None
        self.scale0: Optional[torch.Tensor] = None
        self.bias0: Optional[torch.Tensor] = None
        self.scale1: Optional[torch.Tensor] = None
        self.bias1: Optional[torch.Tensor] = None
        self.scale2: Optional[torch.Tensor] = None
        self.bias2: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

        self.rows = 4096
        self.cols = 4096
        self.dtype = torch.bfloat16
        tokens = self.rows * self.cols
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.x = torch.randn(self.rows, self.cols, device=self.device, dtype=self.dtype)
        # Keep parameters small (broadcast-friendly) but resident on GPU.
        self.scale0 = torch.tensor(0.9, device=self.device, dtype=self.dtype)
        self.bias0 = torch.tensor(0.1, device=self.device, dtype=self.dtype)
        self.scale1 = torch.tensor(1.05, device=self.device, dtype=self.dtype)
        self.bias1 = torch.tensor(-0.05, device=self.device, dtype=self.dtype)
        self.scale2 = torch.tensor(0.97, device=self.device, dtype=self.dtype)
        self.bias2 = torch.tensor(0.2, device=self.device, dtype=self.dtype)
        self.output = None
        self._verify_output_buffer = torch.empty(
            (min(128, self.rows), min(256, self.cols)),
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.x is None
            or self.scale0 is None
            or self.bias0 is None
            or self.scale1 is None
            or self.bias1 is None
            or self.scale2 is None
            or self.bias2 is None
        ):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_warp_specialization_training"):
            with torch.inference_mode():
                self.output = _epilogue_chain(
                    self.x,
                    self.scale0,
                    self.bias0,
                    self.scale1,
                    self.bias1,
                    self.scale2,
                    self.bias2,
                )
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.x is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("capture_verification_payload() requires a completed benchmark run")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"x": self.x},
            output=self._verify_output_buffer,
            batch_size=int(self.rows),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": False,
            },
            # BF16 elementwise chains can diverge slightly after fusion.
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.x = None
        self.scale0 = None
        self.bias0 = None
        self.scale1 = None
        self.bias1 = None
        self.scale2 = None
        self.bias2 = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.x is None:
            return "Input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializationTrainingBenchmark()
