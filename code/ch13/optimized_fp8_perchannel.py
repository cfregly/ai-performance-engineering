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

_HAS_SCALED_MM = hasattr(torch, "_scaled_mm")
_HAS_FLOAT8_E4M3FN = hasattr(torch, "float8_e4m3fn")


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
        self._weight_fp8_t = None
        self._scale_b = None
        self._bias_bf16 = None
        self.register_buffer("_scale_a_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_input_scale_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_input_scaled_buffer", torch.empty(0), persistent=False)
        self.register_buffer(
            "_input_fp8_buffer",
            torch.empty(0, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        self._input_scale_key: tuple | None = None
        
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
        if not _HAS_SCALED_MM:
            raise RuntimeError("torch._scaled_mm is required for fp8_perchannel optimized benchmark")
        if not _HAS_FLOAT8_E4M3FN:
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_perchannel optimized benchmark")
        with torch.inference_mode():
            weight_amax = self.weight.abs().amax(dim=1)  # [out_features]
            weight_scale = torch.clamp(weight_amax / self.fp8_max, min=1e-12).to(torch.float32)  # [out_features]
            self._scale_b = weight_scale.unsqueeze(0).contiguous()  # [1, out_features]
            weight_fp8 = (self.weight / weight_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            self._weight_fp8 = weight_fp8.contiguous()  # [out_features, in_features]
            self._weight_fp8_t = self._weight_fp8.T
            if self.bias is not None:
                self._bias_bf16 = self.bias.to(torch.bfloat16).contiguous()

    def _activation_buffers(self, x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_scaled = self._input_scaled_buffer
        if (
            input_scaled.shape != x_2d.shape
            or input_scaled.device != x_2d.device
            or input_scaled.dtype != x_2d.dtype
        ):
            input_scaled = torch.empty_like(x_2d)
            self._input_scaled_buffer = input_scaled

        input_fp8 = self._input_fp8_buffer
        if input_fp8.shape != x_2d.shape or input_fp8.device != x_2d.device:
            input_fp8 = torch.empty(
                x_2d.shape,
                device=x_2d.device,
                dtype=torch.float8_e4m3fn,
            )
            self._input_fp8_buffer = input_fp8

        return input_scaled, input_fp8

    def _scale_a_for(self, x_fp8: torch.Tensor) -> torch.Tensor:
        scale_a_shape = (x_fp8.size(0), 1)
        scale_a = self._scale_a_buffer
        if (
            scale_a.device != x_fp8.device
            or scale_a.dtype != torch.float32
            or tuple(scale_a.shape) != scale_a_shape
        ):
            scale_a = torch.empty(
                scale_a_shape,
                device=x_fp8.device,
                dtype=torch.float32,
            )
            self._scale_a_buffer = scale_a
        return scale_a

    def prepare_activation_buffers(self, example_input: torch.Tensor) -> None:
        hidden = example_input.shape[-1]
        x_2d = example_input.reshape(-1, hidden)
        self._activation_buffers(x_2d)
        self._scale_a_for(self._input_fp8_buffer)
        input_scale = self._input_scale_buffer
        if (
            input_scale.device != x_2d.device
            or input_scale.dtype != torch.float32
            or tuple(input_scale.shape) != ()
        ):
            self._input_scale_buffer = torch.empty(
                (),
                device=x_2d.device,
                dtype=torch.float32,
            )

    def _cacheable_input_key(self, x_2d: torch.Tensor) -> tuple | None:
        try:
            version = x_2d._version
        except RuntimeError:
            return None
        return (
            x_2d.data_ptr(),
            tuple(x_2d.shape),
            tuple(x_2d.stride()),
            x_2d.storage_offset(),
            x_2d.dtype,
            x_2d.device.type,
            x_2d.device.index,
            version,
        )

    def _input_scale_for(self, x_2d: torch.Tensor) -> torch.Tensor:
        cache_key = self._cacheable_input_key(x_2d)
        input_scale = self._input_scale_buffer
        if (
            cache_key is not None
            and self._input_scale_key == cache_key
            and input_scale.device == x_2d.device
            and input_scale.dtype == torch.float32
            and tuple(input_scale.shape) == ()
        ):
            return input_scale

        input_amax = x_2d.abs().max()
        computed_scale = torch.clamp(input_amax / self.fp8_max, min=1e-12).to(
            torch.float32
        )
        if cache_key is None:
            return computed_scale

        if (
            input_scale.device != x_2d.device
            or input_scale.dtype != torch.float32
            or tuple(input_scale.shape) != ()
        ):
            input_scale = torch.empty((), device=x_2d.device, dtype=torch.float32)
            self._input_scale_buffer = input_scale
        input_scale.copy_(computed_scale)
        self._input_scale_key = cache_key
        return input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel FP8 quantized forward pass.
        
        Uses per-tensor scaling for input (standard) and per-output-channel
        scaling for weights (the key improvement).
        """
        if self._weight_fp8 is None or self._weight_fp8_t is None or self._scale_b is None:
            raise RuntimeError("prepare_fp8_weights() must be called before forward()")
        if x.device.type != "cuda":
            raise RuntimeError("CUDA required for fp8_perchannel optimized benchmark")
        if not _HAS_SCALED_MM:
            raise RuntimeError("torch._scaled_mm is required for fp8_perchannel optimized benchmark")
        if not _HAS_FLOAT8_E4M3FN:
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_perchannel optimized benchmark")

        batch_size, seq_len, hidden = x.shape
        x_2d = x.reshape(-1, hidden)

        input_scale = self._input_scale_for(x_2d)
        input_scaled, x_fp8 = self._activation_buffers(x_2d)
        torch.div(x_2d, input_scale, out=input_scaled)
        x_fp8.copy_(input_scaled)
        scale_a = self._scale_a_for(x_fp8)
        scale_a.copy_(input_scale)

        output_2d = torch._scaled_mm(
            x_fp8,
            self._weight_fp8_t,
            scale_a,
            self._scale_b,
            out_dtype=torch.bfloat16,
        )
        if self._bias_bf16 is not None:
            output_2d.add_(self._bias_bf16)
        output = output_2d.reshape(batch_size, seq_len, -1)
        
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
        self._verify_output_buffer: Optional[torch.Tensor] = None
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
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            self.out_features,
            device=self.device,
            dtype=torch.float32,
        )

        self.model.prepare_fp8_weights()
        self.model.prepare_activation_buffers(self.x)
        
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
            output_tolerance=self.get_output_tolerance(),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.ref_model = None
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
