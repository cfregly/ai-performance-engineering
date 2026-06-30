"""FP8 per-channel quantization benchmark for better accuracy.

Per-channel (per-output-channel) quantization uses a separate scale factor
for each output channel, preserving more accuracy than per-tensor scaling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class FP8PerChannelLinear(nn.Module):
    """Linear layer with simulated per-output-channel FP8 quantization.
    
    Each row of the weight matrix gets its own scale factor, providing
    better accuracy than per-tensor scaling especially when weight
    magnitudes vary significantly across output channels.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_max = 448.0  # E4M3 max value
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer("_weight_q", torch.empty(0), persistent=False)
        self.register_buffer("_weight_scale", torch.empty(0), persistent=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_amax = self.weight.abs().amax(dim=1)  # [out_features]
        weight_scale = torch.clamp(weight_amax / self.fp8_max, min=1e-12)  # [out_features]
        weight_q = torch.clamp(
            self.weight / weight_scale.unsqueeze(1),
            -self.fp8_max,
            self.fp8_max,
        ).round()
        return weight_q, weight_scale

    def prepare_fp8_weights(self) -> None:
        """Precompute static per-channel weight quantization for inference."""
        with torch.inference_mode():
            weight_q, weight_scale = self._quantize_weight()
            self._weight_q = weight_q.contiguous()
            self._weight_scale = weight_scale.contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel FP8 quantized forward pass.
        
        Uses per-tensor scaling for input (standard) and per-output-channel
        scaling for weights (the key improvement).
        """
        # Per-tensor quantization of input
        input_amax = x.abs().max()
        input_scale = torch.clamp(input_amax / self.fp8_max, min=1e-12)
        x_q = torch.clamp(x / input_scale, -self.fp8_max, self.fp8_max).round()

        if (
            tuple(self._weight_q.shape) == tuple(self.weight.shape)
            and self._weight_q.device == self.weight.device
            and self._weight_q.dtype == self.weight.dtype
            and self._weight_scale.numel() == self.weight.size(0)
        ):
            weight_q = self._weight_q
            weight_scale = self._weight_scale
        else:
            weight_q, weight_scale = self._quantize_weight()
        
        # Simulated FP8 GEMM
        output_q = torch.nn.functional.linear(x_q, weight_q, bias=None)
        
        # Dequantize with per-channel weight scales
        output_q.mul_(input_scale)
        output_q.mul_(weight_scale)
        output = output_q.to(x.dtype)
        
        if self.bias is not None:
            output.add_(self.bias)
        
        return output


class OptimizedFP8PerChannelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Per-channel FP8 quantization for better accuracy."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 32
        self.seq_len = 512
        self.in_features = 4096
        self.out_features = 4096
        self.dtype = torch.float32
        self._last = 0.0
        self._error_sum = 0.0
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        
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
        """Setup: Initialize per-channel FP8 model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create model with per-channel FP8
        self.model = FP8PerChannelLinear(
            self.in_features, self.out_features
        ).to(self.device, self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.in_features,
            device=self.device, dtype=self.dtype
        )
        self._verify_input = self.x.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            self.out_features,
            device=self.device,
            dtype=torch.float32,
        )
        self.model.prepare_fp8_weights()
        
        # Warmup
        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Per-channel FP8 forward pass."""
        with torch.inference_mode():
            self.output = self.model(self.x)
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        dtype = self._verify_input.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            :,
        ]
        self._verify_output_buffer.copy_(output_slice)
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.x = None
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None or self.x is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedFP8PerChannelBenchmark()
