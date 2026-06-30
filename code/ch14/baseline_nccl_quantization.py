"""Baseline NCCL quantization – quantize on CPU and serialize transfers."""

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


class BaselineNCCLQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Simulate per-rank CPU-side quantization with serialized copies."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.tensor = None
        self.num_chunks = 16
        self._chunk_range = range(self.num_chunks)
        self.chunk_len = 1 << 14
        self._last = 0.0
        self._host_chunk: Optional[torch.Tensor] = None
        self._host_abs: Optional[torch.Tensor] = None
        self._host_quant_float: Optional[torch.Tensor] = None
        self._host_quantized: Optional[torch.Tensor] = None
        self._host_dequant: Optional[torch.Tensor] = None
        self._host_max_abs: Optional[torch.Tensor] = None
        self._host_scale: Optional[torch.Tensor] = None
        self._host_dequant_scale: Optional[torch.Tensor] = None
        self._host_sum: Optional[torch.Tensor] = None
        tokens = self.num_chunks * self.chunk_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_input_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._enable_nvtx = False
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: initialize synthetic gradients."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        self._chunk_range = range(self.num_chunks)
        use_pinned_host = torch.cuda.is_available()
        self._host_chunk = torch.empty(self.chunk_len, dtype=torch.float32, pin_memory=use_pinned_host)
        self._host_abs = torch.empty_like(self._host_chunk)
        self._host_quant_float = torch.empty_like(self._host_chunk)
        self._host_quantized = torch.empty(self.chunk_len, dtype=torch.int8, pin_memory=use_pinned_host)
        self._host_dequant = torch.empty_like(self._host_chunk)
        self._host_max_abs = torch.empty((), dtype=torch.float32)
        self._host_scale = torch.empty((), dtype=torch.float32)
        self._host_dequant_scale = torch.empty((), dtype=torch.float32)
        self._host_sum = torch.empty((), dtype=torch.float32)
        self._verify_input_buffer = torch.empty_like(self.tensor)
        self._verify_output_buffer = torch.empty_like(self.tensor)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: CPU quantization + host/device transfers."""
        with nvtx_range("baseline_nccl_quantization", enable=self._enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            if (
                self._host_chunk is None
                or self._host_abs is None
                or self._host_quant_float is None
                or self._host_quantized is None
                or self._host_dequant is None
                or self._host_max_abs is None
                or self._host_scale is None
                or self._host_dequant_scale is None
                or self._host_sum is None
            ):
                raise RuntimeError("Host quantization buffers not initialized")
            total = 0.0
            for idx in self._chunk_range:
                self._host_chunk.copy_(self.tensor[idx], non_blocking=False)
                torch.abs(self._host_chunk, out=self._host_abs)
                torch.amax(self._host_abs, dim=0, out=self._host_max_abs)
                self._host_max_abs.clamp_(min=1e-6)
                torch.div(127.0, self._host_max_abs, out=self._host_scale)
                torch.mul(self._host_chunk, self._host_scale, out=self._host_quant_float)
                torch.round(self._host_quant_float, out=self._host_quant_float)
                torch.clamp(self._host_quant_float, -127, 127, out=self._host_quant_float)
                self._host_quantized.copy_(self._host_quant_float)
                torch.div(self._host_max_abs, 127.0, out=self._host_dequant_scale)
                self._host_dequant.copy_(self._host_quantized)
                self._host_dequant.mul_(self._host_dequant_scale)
                torch.sum(self._host_dequant, dim=0, out=self._host_sum)
                total += float(self._host_sum)
                self.tensor[idx].copy_(self._host_dequant, non_blocking=False)
            self._last = total
            self.output = self.tensor
        if self.output is None or self.tensor is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if (
            self.tensor is None
            or self.output is None
            or self._verify_input_buffer is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_input_buffer.copy_(self.tensor)
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input_buffer},
            output=self._verify_output_buffer,
            batch_size=self.num_chunks,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        self._host_chunk = None
        self._host_abs = None
        self._host_quant_float = None
        self._host_quantized = None
        self._host_dequant = None
        self._host_max_abs = None
        self._host_scale = None
        self._host_dequant_scale = None
        self._host_sum = None
        self.output = None
        self._verify_input_buffer = None
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
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="int8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineNCCLQuantizationBenchmark()
