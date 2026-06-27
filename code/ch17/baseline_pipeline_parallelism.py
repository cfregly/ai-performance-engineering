"""baseline_pipeline_parallel_multigpuism.py - Baseline sequential processing without pipeline parallelism.

Demonstrates sequential processing of model layers without pipeline parallelism.
Pipeline parallelism: This baseline processes all layers sequentially on a single GPU.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification import ToleranceSpec
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselinePipelineParallelismBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Sequential processing without pipeline parallelism (single GPU)."""

    _PIPELINE_STAGE_COUNT = 4
    _PIPELINE_STAGE_BOUNDARIES = [(0, 1), (2, 3), (4, 5), (6, 6)]
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_data = None
        self.batch_size = 256
        self.hidden_size = 1024
        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            samples_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            samples_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with all layers on single GPU."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Baseline: Sequential processing on single GPU
        # Pipeline parallelism splits model layers across multiple GPUs
        # This baseline processes all layers sequentially on one GPU
        dtype = torch.bfloat16
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        ).to(self.device, dtype=dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Input data for inference
        self.input_data = torch.randn(self.batch_size, self.hidden_size, device=self.device, dtype=dtype)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential processing of all layers."""
        with self._nvtx_range("baseline_pipeline_parallel_multigpuism"):
            with torch.inference_mode():
                activations = self.input_data
                for layer in self.model:
                    activations = layer(activations)
                self.output = activations
        if self.output is None or self.input_data is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        if self.output is None or self.input_data is None:
            raise RuntimeError("benchmark_fn() must be called before capture_verification_payload()")
        dtype = self.output.dtype
        signature_overrides = {
            "pipeline_stages": self._PIPELINE_STAGE_COUNT,
            "pipeline_stage_boundaries": self._PIPELINE_STAGE_BOUNDARIES,
        }
        self._set_verification_payload(
            inputs={"input": self.input_data},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            output_tolerance=ToleranceSpec(
                rtol=1e-3,
                atol=1e-3,
                justification="torch.compile fusion can change bf16 rounding vs eager execution",
            ),
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            signature_overrides=signature_overrides,
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_data = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark() -> BaselinePipelineParallelismBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePipelineParallelismBenchmark()

