"""Baseline: FP4 with per-forward dequantization (slow).

Chapter 19: Blackwell Native Precision Operations

This baseline shows naive FP4 inference that dequantizes weights
on EVERY forward pass. This is what happens without proper caching.

The optimized version keeps FP4 as the source-of-truth format and, on
Blackwell, caches an FP8 execution bridge so it does not rebuild the bridge
on every forward pass.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# FP4 E2M1 representable values
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_SIGNED_VALUES = torch.cat((FP4_VALUES, -FP4_VALUES))
FP4_MAX = 6.0
_FP4_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}
_FP4_SIGNED_VALUES_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_values_for(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return FP4_VALUES
    cached = _FP4_VALUES_CACHE.get(device)
    if cached is None:
        cached = FP4_VALUES.to(device=device)
        _FP4_VALUES_CACHE[device] = cached
    return cached


def _fp4_signed_values_for(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return FP4_SIGNED_VALUES
    cached = _FP4_SIGNED_VALUES_CACHE.get(device)
    if cached is None:
        cached = FP4_SIGNED_VALUES.to(device=device)
        _FP4_SIGNED_VALUES_CACHE[device] = cached
    return cached


def _unpack_fp4_codes(packed_data: torch.Tensor) -> torch.Tensor:
    unpacked = torch.empty(packed_data.numel() * 2, device=packed_data.device, dtype=torch.uint8)
    torch.bitwise_right_shift(packed_data, 4, out=unpacked[0::2])
    torch.bitwise_and(packed_data, 0x0F, out=unpacked[1::2])
    return unpacked.long()


def quantize_fp4_baseline(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Baseline FP4 quantization with per-tensor scaling.
    
    No per-channel scaling, no block scaling - simple per-tensor approach.
    """
    device = tensor.device
    dtype = tensor.dtype
    
    # Flatten for quantization
    flat = tensor.flatten().float()
    
    # Per-tensor scale (baseline: no per-channel/block)
    absmax = flat.abs().max()
    scale = (absmax / FP4_MAX).clamp(min=1e-8)
    
    # Normalize to FP4 range
    normalized = flat / scale
    normalized = normalized.clamp(-FP4_MAX, FP4_MAX)
    
    # Quantize to nearest FP4 value (sequential loop - slow)
    fp4_vals = _fp4_values_for(device)
    abs_normalized = normalized.abs()
    
    # Find nearest FP4 value
    distances = (abs_normalized.unsqueeze(-1) - fp4_vals).abs()
    indices = distances.argmin(dim=-1).byte()
    signs = (normalized < 0).byte()
    
    # Pack: 4-bit code = sign (1 bit) + magnitude index (3 bits)
    fp4_codes = (signs << 3) | indices
    
    # Pack pairs of 4-bit values into bytes
    flat_codes = fp4_codes
    if flat_codes.numel() % 2 != 0:
        flat_codes = F.pad(flat_codes, (0, 1))
    
    pairs = flat_codes.reshape(-1, 2)
    packed = (pairs[:, 0] << 4) | pairs[:, 1]
    
    # Return packed data and scalar scale
    scale_tensor = scale.reshape(1).to(dtype=dtype, device=device)
    return packed.to(torch.uint8), scale_tensor


def dequantize_fp4_baseline(
    packed_data: torch.Tensor,
    scale: torch.Tensor,
    original_shape: torch.Size,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Baseline FP4 dequantization - no caching."""
    device = packed_data.device
    signed_fp4_vals = _fp4_signed_values_for(device)
    
    # Unpack bytes
    unpacked = _unpack_fp4_codes(packed_data)
    
    # Decode FP4 directly from the packed sign+magnitude code.
    values = signed_fp4_vals[unpacked]
    
    # Apply scale
    dequantized = values * scale.to(values.dtype)
    
    # Reshape to original
    n_orig = math.prod(original_shape)
    return dequantized[:n_orig].reshape(original_shape).to(dtype)


class BaselineFP4Linear(nn.Module):
    """Baseline FP4 linear layer without optimizations.
    
    Issues:
    - Per-tensor scaling (loses precision)
    - Dequantizes on every forward pass
    - No FP8 tensor core acceleration
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Initialize FP16 weights
        weight = torch.empty(out_features, in_features, dtype=dtype)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        self.register_buffer('_weight_fp16', weight)
        self.register_buffer('weight_packed', None)
        self.register_buffer('weight_scale', None)
        self._quantized = False
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
    
    def quantize(self) -> None:
        """Quantize weights to FP4."""
        if self._weight_fp16 is not None:
            packed, scale = quantize_fp4_baseline(self._weight_fp16)
            self.weight_packed = packed
            self.weight_scale = scale
            self._quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with FP4 weights (dequantize every time - slow)."""
        if self._quantized:
            weight = dequantize_fp4_baseline(
                self.weight_packed,
                self.weight_scale,
                torch.Size([self.out_features, self.in_features]),
                self.dtype
            )
        else:
            weight = self._weight_fp16
        
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        return F.linear(x, weight, self.bias)


class NaiveFP16MLP(nn.Module):
    """Naive FP16 MLP with redundant operations (baseline anti-pattern).
    
    Simulates what happens without proper optimization:
    - Multiple redundant copies and casts
    - Explicit synchronization between ops (breaks pipelining)
    - FP32 intermediate computation
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        # Anti-pattern: Store weights in FP32, convert on every forward
        self.fc1 = nn.Linear(d_model, d_ff, dtype=torch.float32)
        self.fc2 = nn.Linear(d_ff, d_model, dtype=torch.float32)
        self.activation = nn.GELU()
        self._input_dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Anti-pattern: Convert to FP32 for computation
        x = x.float()  # FP16->FP32 conversion
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.to(self._input_dtype)  # FP32->FP16 conversion
        return x


class BaselineFP4MLP(nn.Module):
    """Baseline FP4 MLP that dequantizes on EVERY forward pass (slow).
    
    This demonstrates the anti-pattern of not caching dequantized weights.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.fc1 = BaselineFP4Linear(d_model, d_ff, dtype=dtype)
        self.fc2 = BaselineFP4Linear(d_ff, d_model, dtype=dtype)
        self.activation = nn.GELU()
    
    def quantize(self) -> None:
        """Quantize all FP4 layers."""
        self.fc1.quantize()
        self.fc2.quantize()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)  # Dequantizes weights every call
        x = self.activation(x)
        x = self.fc2(x)  # Dequantizes weights every call
        return x


class BaselineFP4WeightQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP4 with per-forward dequantization (slow).
    
    Anti-pattern: Dequantize weights on every forward pass.
    The optimized version caches the FP8 execution bridge on Blackwell.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        
        # Configuration for fair comparison
        self.batch_size = 16
        self.seq_len = 256
        self.d_model = 2048
        self.d_ff = 8192
        
        self.input: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup baseline FP4 model (dequant every forward)."""
        torch.manual_seed(42)
        
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        # Use FP4 MLP that dequantizes on every forward
        self.model = BaselineFP4MLP(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dtype=dtype,
        ).to(self.device)
        
        # Quantize weights to FP4
        self.model.quantize()
        self.model.eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.input = torch.randn(
            self.batch_size, self.seq_len, self.d_model,
            device=self.device, dtype=dtype
        )
        
        # Warmup
        with torch.inference_mode():
            for _ in range(5):
                _ = self.model(self.input)
        
    
    def benchmark_fn(self) -> None:
        """Benchmark naive MLP."""
        with self._nvtx_range("baseline_naive_mlp"):
            with torch.inference_mode():
                output = self.model(self.input)
                self.output = output
        if self.output is None or self.input is None or self.model is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.5, 5.0),
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": False,
            },
        )
    
    def teardown(self) -> None:
        """Clean up."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return baseline FP16 metrics."""
        # Weight memory calculations
        fp16_bytes = (self.d_model * self.d_ff + self.d_ff * self.d_model) * 2
        
        return {
            "precision.dtype": "fp16/bf16",
            "precision.weight_bytes": float(fp16_bytes),
            "precision.weight_mb": float(fp16_bytes / (1024 * 1024)),
            "precision.is_quantized": 0.0,
        }
    
    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        
        with torch.inference_mode():
            output = self.model(self.input[:1, :32])
            if torch.isnan(output).any():
                return "NaN in output"
        
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineFP4WeightQuantizationBenchmark()
