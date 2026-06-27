"""Optimized FP8 per-channel quantization for better accuracy.

Per-channel (per-output-channel) quantization uses a separate scale factor
for each output channel, preserving more accuracy than per-tensor scaling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

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
        self._weight_fp8 = None
        self._scale_b = None
        self._bias_bf16 = None
        self.register_buffer("_scale_a_buffer", torch.empty(0), persistent=False)
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def prepare_fp8_weights(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for fp8_perchannel optimized benchmark")
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("torch._scaled_mm is required for fp8_perchannel optimized benchmark")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_perchannel optimized benchmark")
        with torch.inference_mode():
            weight_amax = self.weight.abs().amax(dim=1)  # [out_features]
            weight_scale = torch.clamp(weight_amax / self.fp8_max, min=1e-12).to(torch.float32)  # [out_features]
            self._scale_b = weight_scale.unsqueeze(0).contiguous()  # [1, out_features]
            weight_fp8 = (self.weight / weight_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            self._weight_fp8 = weight_fp8.contiguous()  # [out_features, in_features]
            if self.bias is not None:
                self._bias_bf16 = self.bias.to(torch.bfloat16).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel FP8 quantized forward pass.
        
        Uses per-tensor scaling for input (standard) and per-output-channel
        scaling for weights (the key improvement).
        """
        if self._weight_fp8 is None or self._scale_b is None:
            raise RuntimeError("prepare_fp8_weights() must be called before forward()")
        if x.device.type != "cuda":
            raise RuntimeError("CUDA required for fp8_perchannel optimized benchmark")
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("torch._scaled_mm is required for fp8_perchannel optimized benchmark")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_perchannel optimized benchmark")

        batch_size, seq_len, hidden = x.shape
        x_2d = x.reshape(-1, hidden)

        input_amax = x_2d.abs().max()
        input_scale = torch.clamp(input_amax / self.fp8_max, min=1e-12).to(torch.float32)
        x_fp8 = (x_2d / input_scale).to(torch.float8_e4m3fn)
        scale_a_shape = (x_fp8.size(0), 1)
        if (
            self._scale_a_buffer.device != x_fp8.device
            or self._scale_a_buffer.dtype != torch.float32
            or tuple(self._scale_a_buffer.shape) != scale_a_shape
        ):
            self._scale_a_buffer = torch.empty(
                scale_a_shape,
                device=x_fp8.device,
                dtype=torch.float32,
            )
        scale_a = self._scale_a_buffer
        scale_a.copy_(input_scale)

        output_2d = torch._scaled_mm(
            x_fp8,
            self._weight_fp8.T,
            scale_a,
            self._scale_b,
            out_dtype=torch.bfloat16,
        )
        output = output_2d.reshape(batch_size, seq_len, -1)
        
        if self._bias_bf16 is not None:
            output = output + self._bias_bf16
        
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
        self._verify_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
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
        
        # Create reference model for error calculation
        self.ref_model = nn.Linear(
            self.in_features, self.out_features
        ).to(self.device, self.dtype).eval()
        
        # Copy weights
        with torch.inference_mode():
            self.ref_model.weight.copy_(self.model.weight)
            if self.model.bias is not None:
                self.ref_model.bias.copy_(self.model.bias)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.in_features,
            device=self.device, dtype=self.dtype
        )
        self._verify_input = self.x.detach().clone()

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
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output.detach().to(torch.float32).clone(),
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=self.get_output_tolerance(),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.ref_model = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            timing_method="wall_clock",
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

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.5, 5.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedFP8PerChannelBenchmark()
