"""
optimized_fp8_static_demo.py - Static FP8 Quantization (Ch13)

WHAT: Static quantization calibrates scale factors ONCE during profiling,
then uses fixed scales during inference.

WHY: Dynamic quantization computes scales every forward pass:
  - Dynamic: Compute amax → derive scale → quantize → GEMM
  - Static: Use pre-computed scale → quantize → GEMM
  
Static saves compute and is deterministic for deployment.

WORKFLOW:
  1. Calibration pass: Run representative data, collect amax statistics
  2. Compute scales: scales = amax / fp8_max (often with margin)
  3. Freeze scales: Store as model attributes
  4. Inference: Use frozen scales, no amax computation

WHEN TO USE:
  - Production inference where latency matters
  - When input distribution is stable (known batch sizes, similar inputs)
  - Edge deployment where compute is limited

TRADE-OFF:
  - Better latency than dynamic
  - Requires calibration dataset
  - Less accurate if inference data differs from calibration

REQUIREMENTS:
  - PyTorch 2.1+ with FP8 support
  - Calibration dataset
"""

from __future__ import annotations

import heapq
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


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
        """Update statistics with new tensor."""
        self._amax_tensors.append(tensor.detach().abs().amax())
        self.num_samples += 1

    def _materialize_buffers(self, count: int, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        """Move pending device-side amax samples to host history in one transfer."""
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
        """Compute final scale factor."""
        self._materialize_amax_history()
        amax = self.running_amax * (1.0 + margin)
        return max(amax / fp8_max, 1e-12)
    
    def get_percentile_scale(
        self,
        percentile: float = 99.9,
        fp8_max: float = 448.0,
    ) -> float:
        """Compute scale from percentile (handles outliers better)."""
        self._materialize_amax_history()
        if not self.amax_history:
            return 1e-12
        
        idx = int(len(self.amax_history) * percentile / 100)
        idx = max(0, min(idx, len(self.amax_history) - 1))
        if idx <= len(self.amax_history) // 2:
            amax = heapq.nsmallest(idx + 1, self.amax_history)[-1]
        else:
            amax = heapq.nlargest(len(self.amax_history) - idx, self.amax_history)[-1]
        return max(amax / fp8_max, 1e-12)


class StaticFP8Linear(nn.Module):
    """Linear layer with static FP8 quantization.
    
    Modes:
    - calibration: Collect amax statistics
    - inference: Use frozen scales
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
        
        # FP8 constants
        self.fp8_max = 448.0  # E4M3
        
        # Static scales (frozen after calibration)
        self.register_buffer('input_scale', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('weight_scale', torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer('is_calibrated', torch.tensor(False, device=device))
        self._is_calibrated = False
        self.register_buffer(
            '_calibration_info_values',
            torch.empty(3, dtype=torch.float32, device=device),
            persistent=False,
        )
        self._calibration_info_host: Optional[torch.Tensor] = None
        
        # Calibration state
        self._calibrating = False
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    @contextmanager
    def calibration_mode(self):
        """Context manager for calibration mode."""
        self._calibrating = True
        self._input_stats = CalibrationStats()
        self._weight_stats = CalibrationStats()
        try:
            yield
        finally:
            self._calibrating = False
    
    def freeze_scales(self, margin: float = 0.0, use_percentile: bool = True):
        """Freeze scales after calibration."""
        if use_percentile:
            input_scale = self._input_stats.get_percentile_scale(99.9, self.fp8_max)
            weight_scale = self._weight_stats.get_percentile_scale(99.9, self.fp8_max)
        else:
            input_scale = self._input_stats.get_scale(self.fp8_max, margin)
            weight_scale = self._weight_stats.get_scale(self.fp8_max, margin)
        
        self.input_scale.fill_(input_scale)
        self.weight_scale.fill_(weight_scale)
        self.is_calibrated.fill_(True)
        self._is_calibrated = True
        
        return {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "input_samples": self._input_stats.num_samples,
            "input_max_amax": self._input_stats.running_amax,
            "weight_max_amax": self._weight_stats.running_amax,
        }
    
    def _quantize_static(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize with pre-computed scale (no amax computation)."""
        scaled = tensor / scale
        clamped = torch.clamp(scaled, -self.fp8_max, self.fp8_max)
        return clamped.round()
    
    def _dequantize(
        self,
        tensor: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize output."""
        return tensor * input_scale * weight_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        In calibration mode: collect stats, run in full precision
        In inference mode: use static FP8 quantization
        """
        original_dtype = x.dtype
        
        if self._calibrating:
            # Calibration: collect statistics
            self._input_stats.update(x)
            self._weight_stats.update(self.weight)
            
            # Run in full precision during calibration
            output = torch.nn.functional.linear(x, self.weight, self.bias)
            
        elif self._is_calibrated:
            # Static FP8 inference
            x_q = self._quantize_static(x, self.input_scale)
            w_q = self._quantize_static(self.weight, self.weight_scale)
            
            output_q = torch.nn.functional.linear(x_q, w_q, bias=None)
            output = self._dequantize(output_q, self.input_scale, self.weight_scale)
            
            if self.bias is not None:
                output.add_(self.bias)
                
            output = output.to(original_dtype)
            
        else:
            # Not calibrated - run in full precision with warning
            output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        return output

    def _calibration_info_list(self) -> List[float]:
        values = self._calibration_info_values
        if values.device != self.input_scale.device:
            self._calibration_info_values = torch.empty(
                3,
                dtype=torch.float32,
                device=self.input_scale.device,
            )
            values = self._calibration_info_values
        values[0].copy_(self.is_calibrated)
        values[1].copy_(self.input_scale)
        values[2].copy_(self.weight_scale)
        needs_pinned = values.device.type == "cuda"
        if (
            self._calibration_info_host is None
            or (needs_pinned and not self._calibration_info_host.is_pinned())
        ):
            self._calibration_info_host = torch.empty(
                3,
                dtype=torch.float32,
                device="cpu",
                pin_memory=needs_pinned,
            )
        self._calibration_info_host.copy_(values)
        return [
            float(self._calibration_info_host[idx])
            for idx in range(self._calibration_info_host.numel())
        ]
    
    def get_calibration_info(self) -> dict:
        """Get calibration information."""
        is_calibrated, input_scale, weight_scale = self._calibration_info_list()
        return {
            "is_calibrated": bool(is_calibrated),
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "calibration_samples": self._input_stats.num_samples,
        }


class StaticFP8Model(nn.Module):
    """Wrapper to calibrate and convert a model to static FP8."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._fp8_layers: List[StaticFP8Linear] = []
        self._scale_values: Optional[torch.Tensor] = None
        self._scale_values_host: Optional[torch.Tensor] = None
        
        # Find all Linear layers and track for calibration
        self._find_fp8_layers(model)
    
    def _find_fp8_layers(self, module: nn.Module):
        """Find all StaticFP8Linear layers."""
        for child in module.children():
            if isinstance(child, StaticFP8Linear):
                self._fp8_layers.append(child)
            else:
                self._find_fp8_layers(child)
    
    @classmethod
    def convert_linear_layers(
        cls,
        model: nn.Module,
        dtype: torch.dtype = torch.float32,
    ) -> 'StaticFP8Model':
        """Convert all Linear layers to StaticFP8Linear."""
        
        def replace_linear(module: nn.Module) -> nn.Module:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Replace with StaticFP8Linear
                    fp8_linear = StaticFP8Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=dtype,
                    )
                    # Copy weights
                    with torch.inference_mode():
                        fp8_linear.weight.copy_(child.weight)
                        if child.bias is not None:
                            fp8_linear.bias.copy_(child.bias)
                    setattr(module, name, fp8_linear)
                else:
                    replace_linear(child)
            return module
        
        model = replace_linear(model)
        return cls(model)
    
    @contextmanager
    def calibration_mode(self):
        """Enable calibration mode for all FP8 layers."""
        contexts = [layer.calibration_mode() for layer in self._fp8_layers]
        for ctx in contexts:
            ctx.__enter__()
        try:
            yield
        finally:
            for ctx in contexts:
                ctx.__exit__(None, None, None)
    
    def calibrate(
        self,
        dataloader,
        num_batches: int = 100,
        device: torch.device = torch.device('cuda'),
    ) -> Dict[str, dict]:
        """Run calibration on a dataset.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to use for calibration
            device: Device to run calibration on
            
        Returns:
            Dictionary of layer calibration statistics
        """
        self.model.eval()
        
        with torch.inference_mode(), self.calibration_mode():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                elif isinstance(batch, dict):
                    x = batch['input_ids'].to(device) if 'input_ids' in batch else batch['x'].to(device)
                else:
                    x = batch.to(device)
                
                # Forward pass to collect statistics
                _ = self.model(x)
        
        # Freeze all scales
        results = {}
        for i, layer in enumerate(self._fp8_layers):
            results[f"layer_{i}"] = layer.freeze_scales()
        
        return results
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_all_scales(self) -> Dict[str, Tuple[float, float]]:
        """Get all frozen scales."""
        if not self._fp8_layers:
            return {}
        count = 2 * len(self._fp8_layers)
        first_scale = self._fp8_layers[0].input_scale
        if (
            self._scale_values is None
            or self._scale_values.device != first_scale.device
            or self._scale_values.numel() < count
        ):
            self._scale_values = torch.empty(
                count,
                dtype=torch.float32,
                device=first_scale.device,
            )
            self._scale_values_host = torch.empty(
                count,
                dtype=torch.float32,
                device="cpu",
                pin_memory=first_scale.device.type == "cuda",
            )
        assert self._scale_values_host is not None
        scale_slice = self._scale_values[:count]
        for idx, layer in enumerate(self._fp8_layers):
            scale_slice[2 * idx].copy_(layer.input_scale)
            scale_slice[2 * idx + 1].copy_(layer.weight_scale)
        scale_host = self._scale_values_host[:count]
        scale_host.copy_(scale_slice)
        scales = {}
        for i in range(len(self._fp8_layers)):
            scales[f"layer_{i}"] = (float(scale_host[2 * i]), float(scale_host[2 * i + 1]))
        return scales


#============================================================================
# Benchmark
#============================================================================

def benchmark_static_vs_dynamic():
    """Compare static vs dynamic FP8 quantization."""
    print("Static vs Dynamic FP8 Quantization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, dim: int, num_layers: int = 4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu_(layer(x))
            return x
    
    dim = 4096
    num_layers = 8
    batch_size = 32
    seq_len = 512
    
    # Create models
    baseline_model = SimpleModel(dim, num_layers).cuda()
    
    # Convert to static FP8
    static_model = StaticFP8Model.convert_linear_layers(
        SimpleModel(dim, num_layers).cuda()
    )
    
    # Copy weights
    for (name1, p1), (_name2, p2) in zip(
        baseline_model.named_parameters(),
        static_model.model.named_parameters()
    ):
        if 'weight' in name1 or 'bias' in name1:
            with torch.inference_mode():
                p2.copy_(p1)
    
    # Create calibration data
    print("Running calibration...")
    calibration_data = [
        torch.randn(batch_size, seq_len, dim, device=device)
        for _ in range(50)
    ]
    
    class SimpleDataLoader:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter(self.data)
    
    cal_results = static_model.calibrate(
        SimpleDataLoader(calibration_data),
        num_batches=50,
        device=device,
    )
    
    print(f"Calibrated {len(cal_results)} layers")
    print()
    
    # Benchmark
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = baseline_model(x)
        _ = static_model(x)
    torch.cuda.synchronize()
    
    # Benchmark baseline (FP32)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        _ = baseline_model(x)
    end.record()
    end.synchronize()
    
    baseline_ms = start.elapsed_time(end) / 100
    
    # Benchmark static FP8
    start.record()
    for _ in range(100):
        _ = static_model(x)
    end.record()
    end.synchronize()
    
    static_ms = start.elapsed_time(end) / 100
    
    # Check accuracy
    with torch.inference_mode():
        baseline_out = baseline_model(x)
        static_out = static_model(x)
    
    error = (static_out - baseline_out).abs().mean() / baseline_out.abs().mean()
    
    # Results
    print("Results:")
    print(f"  Baseline (FP32): {baseline_ms:.3f} ms")
    print(f"  Static FP8:      {static_ms:.3f} ms")
    print(f"  Speedup:         {baseline_ms / static_ms:.2f}x")
    print(f"  Relative Error:  {float((error * 100).detach().cpu()):.4f}%")
    print()
    
    # Print scale info
    print("Calibrated Scales (first 3 layers):")
    scales = static_model.get_all_scales()
    for name, (in_scale, w_scale) in list(scales.items())[:3]:
        print(f"  {name}: input={in_scale:.6f}, weight={w_scale:.6f}")
    
    print()
    print("Notes:")
    print("  - Static FP8 avoids amax computation every forward pass")
    print("  - Requires calibration dataset that represents inference distribution")
    print("  - Use percentile-based scaling to handle outliers")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class FP8StaticDemoBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for FP8 static quantization demo."""

    def __init__(self):
        super().__init__()
        self.model = None
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
        """Setup: Initialize and calibrate static FP8 model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self, dim: int, num_layers: int = 4):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(dim, dim) for _ in range(num_layers)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu_(layer(x))
                return x
        
        # Convert to static FP8
        self.model = StaticFP8Model.convert_linear_layers(
            SimpleModel(self.dim, 4).to(self.device)
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Calibrate
        calibration_data = [
            torch.randn(self.batch_size, self.seq_len, self.dim, device=self.device)
            for _ in range(10)
        ]
        
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
        
        self.model.calibrate(SimpleDataLoader(calibration_data), num_batches=10, device=self.device)
        
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
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Static FP8 inference."""
        with torch.inference_mode():
            self.output = self.model(self.x)
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
        if self.model is None:
            return "Model not initialized"
        return None

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.5, 5.0)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return FP8StaticDemoBenchmark()
