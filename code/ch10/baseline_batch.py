"""baseline_batch.py - Baseline small batch size in GEMM context."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch10.workload_config import WORKLOAD


class BaselineBatchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: small batch size, limited GPU utilization.
    
    Processes the same data as optimized but in small micro-batches,
    demonstrating the overhead of multiple small kernel launches.
    """
    
    def __init__(self):
        super().__init__()
        self.model: nn.Sequential | None = None
        self.input_flat: torch.Tensor | None = None
        self.inputs_chunked: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self._microbatch_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.output: torch.Tensor | None = None
        self.workload = WORKLOAD
        self.micro_batch_size = self.workload.baseline_micro_batch_size
        self.micro_batches = self.workload.baseline_micro_batches
        self.total_batch_size = self.micro_batch_size * self.micro_batches  # 512
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        tokens = self.total_batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model with small batch size."""
        # Harness provides seeding - model and input creation order must match optimized
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ffn_dim, self.hidden_dim),
        ).to(self.device).eval()
        
        # Generate flat input (same shape/order as optimized for verification)
        self.input_flat = torch.randn(
            self.total_batch_size,
            self.hidden_dim,
            device=self.device,
        )
        # Reshape for micro-batch processing
        self.inputs_chunked = self.input_flat.view(
            self.micro_batches,
            self.micro_batch_size,
            self.hidden_dim,
        )
        # Pre-allocate output buffer (output itself is set during benchmark_fn for harness validity)
        self._output_buffer = torch.empty(self.total_batch_size, self.hidden_dim, device=self.device)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        output_chunked = self._output_buffer.view(
            self.micro_batches,
            self.micro_batch_size,
            self.hidden_dim,
        )
        self._microbatch_pairs = list(
            zip(self.inputs_chunked.unbind(0), output_chunked.unbind(0), strict=True)
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with small batch size (multiple kernel launches)."""
        if self.model is None or self.inputs_chunked is None or self._output_buffer is None:
            raise RuntimeError("Benchmark not configured")
        if not self._microbatch_pairs:
            raise RuntimeError("setup() must initialize microbatch views")
        output = self._output_buffer
        with self._nvtx_range("batch_baseline"):
            with torch.inference_mode():
                # Process each micro-batch separately (many small kernel launches)
                for input_chunk, output_chunk in self._microbatch_pairs:
                    output_chunk.copy_(self.model(input_chunk))
        self.output = output
        if self.output is None or self.input_flat is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input_flat},
            output=self._verify_output_buffer,
            batch_size=self.total_batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.05, 0.05),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input_flat = None
        self.inputs_chunked = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self._microbatch_pairs = []
        self.output = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            timing_method="wall_clock",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual micro-batching structure."""
        from ch10.benchmark_metrics_common import compute_batch_workload_metrics

        return compute_batch_workload_metrics(
            total_batch_size=self.total_batch_size,
            micro_batch_size=self.micro_batch_size,
            micro_batches=self.micro_batches,
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs_chunked is None:
            return "Inputs not initialized"
        if self.output is None:
            return "Output not initialized"
        return None


def get_benchmark() -> BaselineBatchBenchmark:
    """Factory function for harness discovery."""
    return BaselineBatchBenchmark()
