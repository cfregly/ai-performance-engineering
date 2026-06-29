"""baseline_fp8_static.py - Dynamic-scale overhead baseline for static FP8 quantization (Ch13).

This baseline models the *overhead* of dynamic FP8 scaling (per-forward amax
reductions and scale derivation) while keeping the *numerical output invariant*
versus the optimized static-quantization path.

WHY THIS DESIGN:
- Dynamic scaling overhead is real (extra reductions + scale math).
- Using dynamic scales would change outputs vs static scales, forcing loose tolerances.
- For a clean comparable benchmark pair, we compute the dynamic amax/scale work but
  still apply the frozen calibration scales for quantization so baseline/optimized
  outputs match exactly.

Paired with: optimized_fp8_static.py
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from ch13.optimized_fp8_static import StaticFP8Linear


class BaselineFP8StaticBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: pays per-forward amax/scale overhead, uses frozen scales for compute."""

    def __init__(self) -> None:
        super().__init__()
        self.static_linear: Optional[StaticFP8Linear] = None
        self.x: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0

        self.batch_size = 32
        self.seq_len = 512
        self.dim = 4096

        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.static_linear = StaticFP8Linear(self.dim, self.dim, device=self.device)
        self.parameter_count = sum(p.numel() for p in self.static_linear.parameters())

        # Calibration pass: identical to optimized_fp8_static so frozen scales match.
        with self.static_linear.calibration_mode():
            for _ in range(50):
                x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
                _ = self.static_linear(x)
        self.static_linear.freeze_scales()

        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        self._verify_input = self.x.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            self.dim,
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(3):
            with torch.inference_mode():
                _ = self.static_linear(self.x)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.static_linear is None or self.x is None:
            raise RuntimeError("Benchmark not configured")

        with torch.inference_mode():
            # Dynamic scaling overhead (not applied to quantization for output parity).
            input_amax = self.x.abs().amax(dim=-1)
            weight_amax = self.static_linear.weight.abs().amax(dim=1)
            _ = torch.clamp(input_amax / self.static_linear.fp8_max, min=1e-12)
            _ = torch.clamp(weight_amax / self.static_linear.fp8_max, min=1e-12)
            self.output = self.static_linear(self.x)

        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("capture_verification_payload() requires benchmark_fn() output")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            :,
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.static_linear = None
        self.x = None
        self._verify_input = None
        self.output = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.static_linear is None:
            return "Linear layer not initialized"
        if self.output is None:
            return "Output not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFP8StaticBenchmark()
