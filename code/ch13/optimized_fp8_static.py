"""
optimized_fp8_static.py - Static FP8 Quantization (Ch13)

WHAT: Static quantization calibrates scale factors ONCE during profiling,
then uses fixed scales during inference.

WHY: Dynamic quantization computes scales every forward pass:
  - Dynamic: Compute amax → derive scale → quantize → GEMM
  - Static: Use pre-computed scale → quantize → GEMM
  
Static saves compute and is deterministic for deployment.

WORKFLOW:
  1. Calibration pass: Run representative data, collect amax statistics
  2. Compute scales: scales = amax / fp8_max
  3. Freeze scales: Store as model attributes
  4. Inference: Use frozen scales, no amax computation

WHEN TO USE:
  - Production inference where latency matters
  - When input distribution is stable
  - Edge deployment where compute is limited

REQUIREMENTS:
  - PyTorch 2.1+ with FP8 support
  - Calibration dataset
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

_HAS_SCALED_MM = hasattr(torch, "_scaled_mm")
_HAS_FLOAT8_E4M3FN = hasattr(torch, "float8_e4m3fn")


@dataclass
class CalibrationStats:
    """Statistics collected during calibration."""
    amax_history: List[float] = field(default_factory=list)
    running_amax: float = 0.0
    num_samples: int = 0
    _amax_tensors: List[torch.Tensor] = field(default_factory=list, repr=False)
    _amax_materialize_buffer: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _amax_materialize_host_buffer: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    
    def update(self, tensor: torch.Tensor):
        self._amax_tensors.append(tensor.detach().abs().amax())
        self.num_samples += 1

    def _materialize_buffers(self, count: int, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._amax_materialize_buffer is None
            or self._amax_materialize_buffer.device != sample.device
            or self._amax_materialize_buffer.dtype != sample.dtype
            or self._amax_materialize_buffer.numel() < count
        ):
            self._amax_materialize_buffer = torch.empty(
                count,
                device=sample.device,
                dtype=sample.dtype,
            )
            self._amax_materialize_host_buffer = torch.empty(
                count,
                dtype=sample.dtype,
                device="cpu",
                pin_memory=sample.device.type == "cuda",
            )
        assert self._amax_materialize_host_buffer is not None
        return self._amax_materialize_buffer, self._amax_materialize_host_buffer

    def _materialize_amax_history(self) -> None:
        if not self._amax_tensors:
            return
        count = len(self._amax_tensors)
        value_buffer, host_buffer = self._materialize_buffers(count, self._amax_tensors[0])
        value_slice = value_buffer[:count]
        for idx, value in enumerate(self._amax_tensors):
            value_slice[idx].copy_(value)
        host_slice = host_buffer[:count]
        host_slice.copy_(value_slice)
        running_amax = self.running_amax
        for idx in range(count):
            value = float(host_slice[idx])
            self.amax_history.append(value)
            running_amax = max(running_amax, value)
        self.running_amax = running_amax
        self._amax_tensors.clear()
    
    def get_scale(self, fp8_max: float = 448.0, margin: float = 0.0) -> float:
        self._materialize_amax_history()
        amax = self.running_amax * (1.0 + margin)
        return max(amax / fp8_max, 1e-12)


class StaticFP8Linear(nn.Module):
    """Linear layer with static FP8 quantization via torch._scaled_mm."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        
        self.fp8_max = 448.0
        self.register_buffer('input_scale', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('weight_scale', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('is_calibrated', torch.tensor(False, device=device))
        self._is_calibrated = False
        self.register_buffer('weight_fp8', torch.empty(0, device=device, dtype=torch.float8_e4m3fn))
        self._weight_fp8_t: Optional[torch.Tensor] = None
        self.register_buffer(
            "_input_scaled_buffer",
            torch.empty(0, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_input_fp8_buffer",
            torch.empty(0, device=device, dtype=torch.float8_e4m3fn),
            persistent=False,
        )
        
        self._calibrating = False
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @contextmanager
    def calibration_mode(self):
        self._calibrating = True
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        try:
            yield
        finally:
            self._calibrating = False
    
    def freeze_scales(self, margin: float = 0.0):
        input_scale = self._input_stats.get_scale(self.fp8_max, margin)
        weight_scale = self._weight_stats.get_scale(self.fp8_max, margin)
        
        self.input_scale.fill_(input_scale)
        self.weight_scale.fill_(weight_scale)
        self.is_calibrated.fill_(True)
        self._is_calibrated = True
        weight_fp8 = (self.weight / self.weight_scale).to(torch.float8_e4m3fn)
        self.weight_fp8 = weight_fp8.contiguous()
        self._weight_fp8_t = self.weight_fp8.T
        
        return {"input_scale": input_scale, "weight_scale": weight_scale,
                "calibration_samples": self._input_stats.num_samples}

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        
        if self._calibrating:
            self._input_stats.update(x)
            self._weight_stats.update(self.weight)
            output = F.linear(x, self.weight, self.bias)
            
        elif self._is_calibrated:
            if not _HAS_SCALED_MM:
                raise RuntimeError("torch._scaled_mm is required for static FP8 benchmark")
            if not _HAS_FLOAT8_E4M3FN:
                raise RuntimeError("torch.float8_e4m3fn is required for static FP8 benchmark")
            if self.weight_fp8.numel() == 0 or self._weight_fp8_t is None:
                raise RuntimeError("freeze_scales() must be called before inference")

            batch_shape = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1])
            input_scaled, x_fp8 = self._activation_buffers(x_2d)
            torch.div(x_2d, self.input_scale, out=input_scaled)
            x_fp8.copy_(input_scaled)

            output_2d = torch._scaled_mm(
                x_fp8,
                self._weight_fp8_t,
                self.input_scale,
                self.weight_scale,
                out_dtype=torch.float32,
            )
            if self.bias is not None:
                output_2d.add_(self.bias)
            output = output_2d.reshape(*batch_shape, -1)
            output = output.to(original_dtype)
        else:
            output = F.linear(x, self.weight, self.bias)
        
        return output


#============================================================================
# Benchmark
#============================================================================

def benchmark():
    """Compare static vs dynamic FP8 quantization."""
    print("Static vs Dynamic FP8 Quantization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Create simple model
    dim = 4096
    batch_size, seq_len = 32, 512
    
    fp32_linear = nn.Linear(dim, dim).to(device)
    static_linear = StaticFP8Linear(dim, dim, device=device)
    
    # Copy weights
    with torch.inference_mode():
        static_linear.weight.copy_(fp32_linear.weight)
        static_linear.bias.copy_(fp32_linear.bias)
    
    # Calibrate
    print("\nCalibrating...")
    with static_linear.calibration_mode():
        for _ in range(50):
            x = torch.randn(batch_size, seq_len, dim, device=device)
            _ = static_linear(x)
    
    cal_info = static_linear.freeze_scales()
    print(f"  Calibration samples: {cal_info['calibration_samples']}")
    print(f"  Input scale: {cal_info['input_scale']:.6f}")
    print(f"  Weight scale: {cal_info['weight_scale']:.6f}")
    
    # Benchmark
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = fp32_linear(x)
        _ = static_linear(x)
    torch.cuda.synchronize()
    
    # FP32 baseline
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = fp32_linear(x)
    end.record()
    end.synchronize()
    fp32_ms = start.elapsed_time(end) / 100
    
    # Static FP8
    start.record()
    for _ in range(100):
        _ = static_linear(x)
    end.record()
    end.synchronize()
    static_ms = start.elapsed_time(end) / 100
    
    # Accuracy
    with torch.inference_mode():
        fp32_out = fp32_linear(x)
        static_out = static_linear(x)
    error = (static_out - fp32_out).abs().mean() / fp32_out.abs().mean() * 100
    
    print(f"\nResults:")
    print(f"  FP32: {fp32_ms:.3f} ms")
    print(f"  Static FP8: {static_ms:.3f} ms")
    print(f"  Speedup: {fp32_ms / static_ms:.2f}x")
    print(f"  Relative Error: {error:.4f}%")
    
    print("\nNote: Static FP8 avoids per-forward amax computation")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class StaticFP8Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for static FP8 quantization."""

    def __init__(self):
        super().__init__()
        self.static_linear = None
        self.x = None
        self.batch_size = 32
        self.seq_len = 512
        self.dim = 4096
        self._last = 0.0
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize and calibrate static FP8 linear."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.static_linear = StaticFP8Linear(self.dim, self.dim, device=self.device)
        self.parameter_count = sum(p.numel() for p in self.static_linear.parameters())
        
        # Calibrate
        with self.static_linear.calibration_mode():
            for _ in range(50):
                x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
                _ = self.static_linear(x)
        
        self.static_linear.freeze_scales()
        
        self.x = torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
        self._verify_input = self.x.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            self.dim,
            device=self.device,
            dtype=torch.float32,
        )
        
        # Warmup
        for _ in range(3):
            with torch.inference_mode():
                _ = self.static_linear(self.x)

    def benchmark_fn(self) -> None:
        """Benchmark: Static FP8 forward pass."""
        with torch.inference_mode():
            self.output = self.static_linear(self.x)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            :,
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.static_linear = None
        self.x = None
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
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
        if self.static_linear is None:
            return "Linear layer not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return StaticFP8Benchmark()
