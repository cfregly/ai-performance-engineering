"""optimized_nccl_quantization.py - GPU-side quantization with fused collectives."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedNcclQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Quantization with NCCL collective operations."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.quantized = None
        self.dequantized = None
        self._abs_buffer = None
        self._max_abs = None
        self._scales = None
        self._dequant_scales = None
        self._quant_float = None
        self.stream = torch.cuda.Stream()
        self.num_chunks = 16
        self.chunk_len = 1 << 14
        self._last = 0.0
        tokens = self.num_chunks * self.chunk_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._enable_nvtx = False
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize quantized model for NCCL."""

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        self._abs_buffer = torch.empty_like(self.tensor)
        self._max_abs = torch.empty(self.num_chunks, 1, device=self.device, dtype=torch.float32)
        self._scales = torch.empty_like(self._max_abs)
        self._dequant_scales = torch.empty_like(self._max_abs)
        self._quant_float = torch.empty_like(self.tensor)
        self.quantized = torch.empty_like(self.tensor, dtype=torch.int8)
        self.dequantized = torch.empty_like(self.tensor)
        self._verify_output_buffer = torch.empty_like(self.tensor)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantization operations with NCCL."""
        with nvtx_range("optimized_nccl_quantization", enable=self._enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            if (
                self._abs_buffer is None
                or self._max_abs is None
                or self._scales is None
                or self._dequant_scales is None
                or self._quant_float is None
                or self.quantized is None
                or self.dequantized is None
            ):
                raise RuntimeError("Quantization buffers not initialized")
            with torch.cuda.stream(self.stream):
                torch.abs(self.tensor, out=self._abs_buffer)
                torch.amax(self._abs_buffer, dim=1, keepdim=True, out=self._max_abs)
                self._max_abs.clamp_(min=1e-6)
                torch.div(127.0, self._max_abs, out=self._scales)
                torch.mul(self.tensor, self._scales, out=self._quant_float)
                torch.round(self._quant_float, out=self._quant_float)
                torch.clamp(self._quant_float, -127, 127, out=self._quant_float)
                self.quantized.copy_(self._quant_float)
                torch.div(self._max_abs, 127.0, out=self._dequant_scales)
                torch.mul(self.quantized, self._dequant_scales, out=self.dequantized)
                self.output = self.dequantized
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        if self.output is None or self.tensor is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.tensor},
            output=self._verify_output_buffer,
            batch_size=self.num_chunks,
            parameter_count=0,
            precision_flags={
                "fp16": self.tensor.dtype == torch.float16,
                "bf16": self.tensor.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        self.quantized = None
        self.dequantized = None
        self._abs_buffer = None
        self._max_abs = None
        self._scales = None
        self._dequant_scales = None
        self._quant_float = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="int8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedNcclQuantizationBenchmark()
