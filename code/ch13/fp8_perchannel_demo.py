"""
fp8_perchannel_demo.py - Per-Channel FP8 Scaling (Ch13)

WHAT: Per-channel (per-column) scaling computes a separate scale factor for
each output channel, rather than one scale for the entire tensor.

WHY: Activation ranges vary significantly across channels:
  - Per-tensor: One scale fits all → outliers cause clipping
  - Per-channel: Each channel optimal → better quantization accuracy

ACCURACY IMPACT:
  - Per-tensor: 0.5-2% accuracy loss typical
  - Per-channel: <0.1% accuracy loss typical (matches FP16 closely)

WHEN TO USE:
  - Training with FP8 where accuracy matters
  - Models with high activation variance across channels
  - When per-tensor causes visible quality degradation

TRADE-OFF:
  - More scales to store (one per channel vs one per tensor)
  - Slightly more compute for scale application
  - Better accuracy, especially for fine-tuning

REQUIREMENTS:
  - PyTorch 2.1+ with FP8 support
  - SM 89+ (Ada/Hopper/Blackwell) for hardware FP8
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class ScalingMode(Enum):
    """FP8 scaling granularity."""
    PER_TENSOR = "per_tensor"      # One scale for entire tensor
    PER_CHANNEL = "per_channel"    # One scale per output channel
    PER_BLOCK = "per_block"        # One scale per block (for large tensors)


@dataclass
class FP8Config:
    """Configuration for FP8 operations."""
    scaling_mode: ScalingMode = ScalingMode.PER_CHANNEL
    fp8_format: str = "e4m3"  # e4m3 for forward, e5m2 for backward
    margin: float = 0.0       # Safety margin to prevent overflow
    amax_history_len: int = 16  # History for delayed scaling
    

class FP8PerChannelLinear(nn.Module):
    """Linear layer with per-channel FP8 quantization.
    
    During forward:
    1. Compute per-channel amax of input activations
    2. Quantize input to FP8 with per-channel scales
    3. Quantize weights to FP8 (can be per-tensor for weights)
    4. Compute FP8 GEMM
    5. Dequantize output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[FP8Config] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or FP8Config()
        
        # Standard weight and bias
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        # FP8 format ranges
        self.fp8_max = 448.0 if self.config.fp8_format == "e4m3" else 57344.0
        
        # Amax history for delayed scaling
        self.register_buffer(
            'input_amax_history',
            torch.zeros(
                self.config.amax_history_len,
                in_features,
                device=device,
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            'weight_amax_history',
            torch.zeros(
                self.config.amax_history_len,
                device=device,
                dtype=torch.float32,
            )
        )
        self.register_buffer('amax_counter', torch.tensor(0, device=device))
        self.register_buffer(
            '_stats_buffer',
            torch.empty(4, device=device, dtype=torch.float32),
            persistent=False,
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _compute_per_channel_scale(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Compute per-channel scale factors for weight tensor.
        
        For weights [out_features, in_features], compute one scale per output channel (row).
        This is the standard per-channel quantization approach for linear layers.
        
        Args:
            tensor: Weight tensor [out_features, in_features]
            dim: Dimension representing output channels (typically 0 for weights)
            
        Returns:
            scales: [out_features] scale factors, one per output channel
        """
        # For weight tensor [out_features, in_features], compute amax per row (output channel)
        # Each row gets its own scale factor
        if dim == 0 or dim == -2:
            # Per-row (per-output-channel) scaling - compute max over in_features
            amax = tensor.abs().amax(dim=1)  # [out_features]
        else:
            # Per-column scaling
            amax = tensor.abs().amax(dim=0)  # [in_features]
        
        # Apply margin
        amax = amax * (1.0 + self.config.margin)
        
        # Compute scale: scale = amax / fp8_max
        scale = amax / self.fp8_max
        
        # Prevent division by zero
        scale = torch.clamp(scale, min=1e-12)
        
        return scale
    
    def _quantize_per_channel_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight tensor with per-output-channel scales.
        
        Args:
            weight: Weight tensor [out_features, in_features]
            scale: Per-channel scale factors [out_features]
            
        Returns:
            quantized: FP8 weight tensor (simulated as FP32)
            scale: Scale factors for dequantization [out_features]
        """
        # Expand scale for broadcasting: [out_features] -> [out_features, 1]
        scale_expanded = scale.unsqueeze(1)
        
        # Scale and clamp to FP8 range
        scaled = weight / scale_expanded
        clamped = torch.clamp(scaled, -self.fp8_max, self.fp8_max)
        
        # Round to simulate FP8 precision
        quantized = clamped.round()
        
        return quantized, scale
    
    def _dequantize_per_channel(
        self,
        output_q: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize output from FP8 GEMM with per-output-channel weight scales.
        
        For per-tensor input quantization and per-output-channel weight quantization:
          output[..., o] = sum_i(input_q[..., i] * weight_q[o, i]) * input_scale * weight_scale[o]
        
        Args:
            output_q: Quantized output [..., out_features]
            input_scale: Per-tensor input scale (scalar)
            weight_scale: Per-output-channel weight scales [out_features]
            output_dtype: Target dtype for output
            
        Returns:
            Dequantized output tensor
        """
        output_q.mul_(input_scale * weight_scale)

        return output_q.to(output_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with per-channel FP8 quantization.
        
        Per-channel scaling applies per-output-channel scales to weights,
        which preserves more accuracy than per-tensor scaling.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            output: [..., out_features]
        """
        original_dtype = x.dtype
        
        if self.config.scaling_mode == ScalingMode.PER_CHANNEL:
            # Per-tensor quantization of input (standard approach)
            input_amax = x.abs().max()
            input_scale = torch.clamp(input_amax / self.fp8_max, min=1e-12)
            x_q = torch.clamp(x / input_scale, -self.fp8_max, self.fp8_max).round()
            
            # Per-output-channel quantization of weights (the key improvement)
            # Each row of the weight matrix gets its own scale
            weight_scale = self._compute_per_channel_scale(self.weight, dim=0)  # [out_features]
            weight_q, weight_scale = self._quantize_per_channel_weight(self.weight, weight_scale)
            
            # FP8 GEMM (simulated)
            output_q = torch.nn.functional.linear(x_q, weight_q, bias=None)
            
            # Dequantize with per-channel weight scales
            output = self._dequantize_per_channel(
                output_q, input_scale, weight_scale, original_dtype
            )
            
        elif self.config.scaling_mode == ScalingMode.PER_TENSOR:
            # Per-tensor (simpler but less accurate for varied weight distributions)
            input_amax = x.abs().max()
            input_scale = torch.clamp(input_amax / self.fp8_max, min=1e-12)
            x_q = torch.clamp(x / input_scale, -self.fp8_max, self.fp8_max).round()
            
            weight_amax = self.weight.abs().max()
            weight_scale = torch.clamp(weight_amax / self.fp8_max, min=1e-12)
            weight_q = torch.clamp(self.weight / weight_scale, -self.fp8_max, self.fp8_max).round()
            
            output_q = torch.nn.functional.linear(x_q, weight_q, bias=None)
            output = output_q
            output.mul_(input_scale)
            output.mul_(weight_scale)
            output = output.to(original_dtype)
            
        else:
            # No quantization fallback
            output = torch.nn.functional.linear(x, self.weight, bias=None)
        
        # Add bias if present
        if self.bias is not None:
            output.add_(self.bias)
        
        # Update amax history for delayed scaling
        if self.training:
            idx = self.amax_counter % self.config.amax_history_len
            with torch.inference_mode():
                self.input_amax_history[idx] = x.abs().amax(dim=list(range(x.ndim - 1)))
                self.weight_amax_history[idx] = self.weight.abs().max()
                self.amax_counter += 1
        
        return output
    
    def get_quantization_stats(self) -> dict:
        """Get statistics about the quantization."""
        stats_buffer = self._stats_buffer
        stats_buffer[0].copy_(self.input_amax_history.mean())
        stats_buffer[1].copy_(self.input_amax_history.std())
        stats_buffer[2].copy_(self.weight_amax_history.mean())
        stats_buffer[3].copy_(self.amax_counter.to(dtype=torch.float32))
        stats_host = stats_buffer.detach().cpu()
        input_amax_mean = float(stats_host[0])
        input_amax_std = float(stats_host[1])
        weight_amax_mean = float(stats_host[2])
        amax_counter = float(stats_host[3])
        return {
            "scaling_mode": self.config.scaling_mode.value,
            "fp8_format": self.config.fp8_format,
            "fp8_max": self.fp8_max,
            "input_amax_mean": input_amax_mean,
            "input_amax_std": input_amax_std,
            "weight_amax_mean": weight_amax_mean,
            "amax_counter": amax_counter,
        }


#============================================================================
# Benchmark
#============================================================================

class FP8PerChannelBenchmark:
    """Compare per-tensor vs per-channel FP8 scaling."""
    
    def __init__(
        self,
        batch_size: int = 32,
        seq_len: int = 512,
        in_features: int = 4096,
        out_features: int = 4096,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        
        # Create layers with different scaling modes
        self.fp32_linear = nn.Linear(in_features, out_features).to(device, dtype)
        
        self.per_tensor_linear = FP8PerChannelLinear(
            in_features, out_features,
            config=FP8Config(scaling_mode=ScalingMode.PER_TENSOR),
            device=device, dtype=dtype
        )
        
        self.per_channel_linear = FP8PerChannelLinear(
            in_features, out_features,
            config=FP8Config(scaling_mode=ScalingMode.PER_CHANNEL),
            device=device, dtype=dtype
        )
        
        # Copy weights for fair comparison
        with torch.inference_mode():
            self.per_tensor_linear.weight.copy_(self.fp32_linear.weight)
            self.per_channel_linear.weight.copy_(self.fp32_linear.weight)
            if self.fp32_linear.bias is not None:
                self.per_tensor_linear.bias.copy_(self.fp32_linear.bias)
                self.per_channel_linear.bias.copy_(self.fp32_linear.bias)
    
    def measure_accuracy(
        self,
        num_samples: int = 100,
        input_variance: float = 1.0,
    ) -> dict:
        """Measure quantization error for different scaling modes.
        
        Uses weights with varying magnitudes across output channels to highlight
        per-channel benefits. Per-channel weight scaling adapts to each row's range.
        """
        error_sums = torch.zeros(2, device=self.device, dtype=torch.float32)
        
        # Create weights with varying magnitudes per output channel
        # This stresses per-tensor scaling (one scale for all) vs per-channel (one per row)
        with torch.inference_mode():
            # Scale each output channel differently: some large, some small
            output_scales = torch.logspace(-1, 1, self.out_features, device=self.device)
            output_scales = output_scales[torch.randperm(self.out_features)]
            
            scaled_weight = self.fp32_linear.weight * output_scales.unsqueeze(1)
            self.per_tensor_linear.weight.copy_(scaled_weight)
            self.per_channel_linear.weight.copy_(scaled_weight)
        
        for _ in range(num_samples):
            # Standard input distribution
            x = torch.randn(
                self.batch_size, self.seq_len, self.in_features,
                device=self.device, dtype=self.dtype
            )
            
            with torch.inference_mode():
                # Reference: FP32 with scaled weights
                ref_output = torch.nn.functional.linear(x, scaled_weight, self.fp32_linear.bias)
                pt_output = self.per_tensor_linear(x)
                pc_output = self.per_channel_linear(x)
            
            # Compute relative error
            ref_norm = ref_output.abs().mean()
            pt_error = (pt_output - ref_output).abs().mean() / ref_norm
            pc_error = (pc_output - ref_output).abs().mean() / ref_norm

            error_sums[0].add_(pt_error.detach())
            error_sums[1].add_(pc_error.detach())

        error_totals_host = error_sums.detach().cpu()
        pt_error_total = float(error_totals_host[0])
        pc_error_total = float(error_totals_host[1])
        
        return {
            "per_tensor_error_pct": 100 * pt_error_total / num_samples,
            "per_channel_error_pct": 100 * pc_error_total / num_samples,
        }
    
    def measure_throughput(
        self,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> dict:
        """Measure throughput for different modes."""
        x = torch.randn(
            self.batch_size, self.seq_len, self.in_features,
            device=self.device, dtype=self.dtype
        )
        
        results = {}
        
        for name, layer in [
            ("fp32", self.fp32_linear),
            ("per_tensor", self.per_tensor_linear),
            ("per_channel", self.per_channel_linear),
        ]:
            # Warmup
            for _ in range(num_warmup):
                _ = layer(x)
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(num_iterations):
                _ = layer(x)
            end.record()
            end.synchronize()
            
            elapsed_ms = start.elapsed_time(end) / num_iterations
            
            # Calculate TFLOPS
            flops = 2 * self.batch_size * self.seq_len * self.in_features * self.out_features
            tflops = flops / (elapsed_ms / 1000) / 1e12
            
            results[name] = {
                "elapsed_ms": elapsed_ms,
                "tflops": tflops,
            }
        
        return results


#============================================================================
# Main
#============================================================================

#============================================================================
# Benchmark Harness Integration
#============================================================================

class FP8PerChannelDemoBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for FP8 per-channel demo."""

    def __init__(self):
        super().__init__()
        self.demo_benchmark = None
        self.batch_size = 32
        self.seq_len = 512
        self.in_features = 4096
        self.out_features = 4096
        self._last = 0.0
        self.output = None
        self._verify_input = None
        self._payload_verify_input = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._per_channel_linear: Optional[FP8PerChannelLinear] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize per-channel FP8 benchmark."""
        torch.manual_seed(42)
        self.demo_benchmark = FP8PerChannelBenchmark(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=torch.float32,
            device=str(self.device),
        )
        self._per_channel_linear = self.demo_benchmark.per_channel_linear
        self._verify_input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        self._payload_verify_input = self._verify_input.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            self.out_features,
            device=self.device,
            dtype=torch.float32,
        )
        # Warmup
        _ = self.demo_benchmark.measure_throughput(num_warmup=5, num_iterations=3)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Per-channel FP8 forward pass."""
        per_channel_linear = self._per_channel_linear
        if per_channel_linear is None:
            raise RuntimeError("Demo benchmark not initialized")
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        with torch.inference_mode():
            output = per_channel_linear(self._verify_input)
            self.output = output

    def capture_verification_payload(self) -> None:
        verify_input = self._payload_verify_input
        if self.output is None or verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            :,
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": verify_input},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self._per_channel_linear = None
        self.demo_benchmark = None
        self._verify_input = None
        self._payload_verify_input = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)
    
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
        if self.demo_benchmark is None:
            return "Benchmark not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return FP8PerChannelDemoBenchmark()
