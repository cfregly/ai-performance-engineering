"""baseline_compute_bound.py - Compute-bound kernel baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineComputeBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compute-heavy kernel to illustrate high arithmetic intensity."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 4096
        self._repeat_range = range(self.repeats)
        self._payload_parameter_count = 0
        # Compute-bound benchmark - fixed dimensions for roofline analysis
        tokens = self.N * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.N, self.N * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.N * 2, self.N),
        ).to(self.device, dtype=torch.float16).eval()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty_like(self.input)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), self._nvtx_range("baseline_compute_bound"):
            out = self.input
            for _ in self._repeat_range:
                out = self.model(out)
            self.output = out
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
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-1, 1e-1),
        )

    def teardown(self) -> None:
        self.model = None
        self.input = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return compute-bound analysis metrics using the centralized helper.
        
        These metrics help understand WHY the kernel is compute-bound
        and HOW to improve tensor core utilization.
        """
        from core.benchmark.metrics import compute_roofline_metrics
        
        # Estimate FLOPs for the model (2 linear layers: 2*M*N*K per layer)
        # Layer 1: N -> N*2, Layer 2: N*2 -> N
        layer1_flops = 2 * self.N * (self.N * 2) * self.N
        layer2_flops = 2 * self.N * self.N * (self.N * 2)
        total_flops = (layer1_flops + layer2_flops) * self.repeats
        
        # Estimate bytes moved (simplified: input + output)
        element_size = 2  # FP16
        total_bytes = (self.N + self.N) * element_size * self.repeats
        
        # Use elapsed time from last run if available
        elapsed_ms = getattr(self, '_last_elapsed_ms', None)
        
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=elapsed_ms,
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.model is None:
            return "Model/input not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return BaselineComputeBoundBenchmark()
