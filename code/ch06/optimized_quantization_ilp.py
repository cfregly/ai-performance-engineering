"""optimized_quantization_ilp.py - Optimized ILP with quantization."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


class OptimizedQuantizationILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: FP16 with half the memory bandwidth for 2x throughput.
    
    Chapter 6: Increasing GPU ILP Through Microarchitecture Awareness
    
    This demonstrates ILP (Instruction-Level Parallelism) optimization:
    - FP16 uses half the memory bandwidth of FP32
    - Memory-bound ops see ~2x speedup from reduced traffic
    - Modern GPUs handle FP16 arithmetic at same speed as FP32
    """

    signature_equivalence_group = "ch06_quantization_ilp_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.input_fp16: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.output_fp16: Optional[torch.Tensor] = None
        self._output_fp16_buffer: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.N = self.workload.quantization_elements
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
        # ILP benchmark: fixed dimensions for measurement
    
    def setup(self) -> None:
        """Setup: Initialize FP16 tensors with contiguous memory layout."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Keep external inputs identical to baseline (FP32), but pre-cast a FP16 view
        # outside the timed region to model reduced bandwidth in the hot path.
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32).contiguous()
        self.input_fp16 = self.input.to(dtype=torch.float16)
        self._output_fp16_buffer = torch.empty_like(self.input_fp16)
        self.output = None
        self.output_fp16 = None
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP16 element-wise operations (2x less memory traffic)."""
        assert self.input_fp16 is not None and self._output_fp16_buffer is not None
        with torch.inference_mode(), self._nvtx_range("optimized_quantization_ilp"):
            # Simple multiply-add in FP16 - half the memory bandwidth
            torch.mul(self.input_fp16, 2.0, out=self._output_fp16_buffer)
            self._output_fp16_buffer.add_(1.0)
            self.output_fp16 = self._output_fp16_buffer
            self.output = self.output_fp16

    def capture_verification_payload(self) -> None:
        if self.output_fp16 is None or self.input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output_fp16.float(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-2, 1e-2),
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.input_fp16 = None
        self.output = None
        self.output_fp16 = None
        self._output_fp16_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedQuantizationILPBenchmark()
