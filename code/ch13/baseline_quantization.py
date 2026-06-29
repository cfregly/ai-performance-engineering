"""baseline_quantization.py - Baseline FP32 precision without quantization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP32 precision without quantization (full precision)."""

    signature_equivalence_group = "ch13_quantization_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.batch_size = 8192
        self.in_features = 4096
        self.hidden_features = 4096
        self.out_features = 4096
        tokens = self.batch_size * self.in_features
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model in FP32."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_features, self.out_features),
        ).to(self.device).to(torch.float32)
        
        self.model.eval()
        self.data = torch.randn(
            self.batch_size,
            self.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.out_features),
            device=self.device,
            dtype=torch.float32,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP32 inference without quantization."""
        if self.model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("baseline_quantization"):
            with torch.inference_mode():
                self.output = self.model(self.data)
        if self.data is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.data is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self.data},
            output=self._verify_output_buffer,
            batch_size=self.data.shape[0],
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.data = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="int8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaselineQuantizationBenchmark:
    """Factory function for harness discovery."""
    return BaselineQuantizationBenchmark()
