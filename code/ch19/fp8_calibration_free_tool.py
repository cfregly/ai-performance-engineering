#!/usr/bin/env python3
"""Optimized: Calibration-free FP8 serving for Blackwell.

Demonstrates FP8 inference without calibration phase using:
- Dynamic scaling based on tensor statistics
- Per-tensor quantization for activation and weights
- Automatic fallback to BF16 for problematic layers
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.utils import scalar_tensor_to_float
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)
_HAS_FLOAT8_E4M3FN = hasattr(torch, "float8_e4m3fn")

# Check for Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    TE_AVAILABLE = True
except Exception as exc:
    TE_AVAILABLE = False
    logger.warning(f"Transformer Engine not available, using fallback: {exc}")


class CalibrationFreeFP8Linear(nn.Module):
    """FP8 linear layer with dynamic scaling (no calibration)."""
    
    def __init__(self, in_features: int, out_features: int, use_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight in BF16 (master copy)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16)) if use_bias else None
        
        # Dynamic scaling factors (learned during forward passes)
        self.register_buffer('weight_scale', torch.ones(1, dtype=torch.float32))
        self.register_buffer('input_scale', torch.ones(1, dtype=torch.float32))
        
        # EMA smoothing for stability
        self.scale_ema = 0.9
        
        # Fallback flag for problematic layers
        self.use_fp8 = True
        self._weight_fp8_cache: Optional[torch.Tensor] = None
    
    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute FP8 scaling factor dynamically.
        
        FP8 E4M3 range: ~[-448, 448]
        Target: scale such that max(abs(x)) * scale ≈ 448
        """
        with torch.inference_mode():
            # Compute absmax
            absmax = x.abs().max()
            
            # FP8 E4M3 maximum representable value
            fp8_max = 448.0
            
            # Compute scale (with epsilon to avoid division by zero)
            scale = fp8_max / (absmax + 1e-12)
            
            # Clamp to reasonable range
            scale = torch.clamp(scale, min=1e-6, max=1e6)
        
        return scale
    
    def _quantize_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to FP8 E4M3.
        
        Args:
            x: Input tensor (BF16/FP32)
            scale: Scaling factor
        
        Returns:
            x_fp8: Quantized tensor (stored as FP8)
            scale: Updated scale factor
        """
        # Update scale with EMA
        new_scale = self._compute_scale(x)
        scale = self.scale_ema * scale + (1 - self.scale_ema) * new_scale
        
        # Scale and quantize
        x_scaled = x * scale
        
        # Simulate FP8 quantization (PyTorch native FP8 support)
        if hasattr(torch, 'float8_e4m3fn'):
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        else:
            # Fallback: clamp to FP8 range
            x_fp8 = torch.clamp(x_scaled, -448.0, 448.0)
        
        return x_fp8, scale

    def _weight_fp8(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._weight_fp8_cache is None
            or self._weight_fp8_cache.device != self.weight.device
            or tuple(self._weight_fp8_cache.shape) != tuple(self.weight.shape)
        ):
            weight_fp8, weight_scale = self._quantize_fp8(self.weight, self.weight_scale)
            self.weight_scale = weight_scale
            self._weight_fp8_cache = weight_fp8
        return self._weight_fp8_cache, self.weight_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic FP8 quantization.
        
        Args:
            x: [batch_size, seq_len, in_features]
        
        Returns:
            output: [batch_size, seq_len, out_features]
        """
        if not self.use_fp8 or not _HAS_FLOAT8_E4M3FN:
            # Fallback to BF16
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            return nn.functional.linear(x, self.weight, self.bias)
        
        # Quantize input
        x_fp8, self.input_scale = self._quantize_fp8(x, self.input_scale)
        
        # Static inference weights only need FP8 materialization once.
        weight_fp8, weight_scale = self._weight_fp8()
        
        try:
            # FP8 matrix multiplication
            # output = (x_fp8 / input_scale) @ (weight_fp8 / weight_scale).T
            output = torch.mm(
                x_fp8.view(-1, self.in_features).to(torch.float32) / self.input_scale,
                (weight_fp8.to(torch.float32) / weight_scale).T
            )
            
            output = output.view(*x.shape[:-1], self.out_features)
            
            if self.bias is not None:
                output.add_(self.bias)
            
            return output.to(x.dtype)
        
        except Exception as e:
            logger.warning(f"FP8 computation failed: {e}, falling back to BF16")
            self.use_fp8 = False
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
            return nn.functional.linear(x, self.weight, self.bias)


class OptimizedFP8CalibrationFree:
    """Calibration-free FP8 serving benchmark."""
    
    def __init__(
        self,
        batch_size: int = 8,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 4,
        use_te: bool = True,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_te = use_te and TE_AVAILABLE
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_slice: Optional[torch.Tensor] = None
        self._last_output: Optional[torch.Tensor] = None
        self._nan_output: Optional[torch.Tensor] = None
        self.output_mean: Optional[float] = None
        
        if self.use_te:
            logger.info("Using Transformer Engine FP8 with dynamic scaling")
        else:
            logger.info("Using custom FP8 implementation")
    
    def setup(self):
        """Initialize model with FP8 layers."""
        if self.use_te:
            # Transformer Engine with delayed scaling (no calibration)
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,  # E4M3 for forward, E5M2 for backward
                amax_history_len=16,  # Short history for dynamic scaling
                amax_compute_algo="max",  # Use max instead of moving average
            )
            
            self.layers = nn.ModuleList([
                te.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=False,
                    params_dtype=torch.bfloat16
                )
                for _ in range(self.num_layers)
            ]).to(self.device)
        else:
            # Custom FP8 implementation
            self.layers = nn.ModuleList([
                CalibrationFreeFP8Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]).to(self.device)
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        self._nan_output = torch.empty((), device=self.device, dtype=torch.float32)
        self._nan_output.fill_(float("inf"))
        
        logger.info(f"Setup complete: {self.num_layers} FP8 layers")
    
    def run(self) -> torch.Tensor:
        """Execute FP8 forward pass without calibration."""
        torch.cuda.synchronize()
        
        x = self.input
        
        if self.use_te:
            # Transformer Engine FP8 context
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                for layer in self.layers:
                    x = layer(x)
        else:
            # Custom FP8
            for layer in self.layers:
                x = layer(x)
        
        torch.cuda.synchronize()
        
        # Check output validity
        if torch.isnan(x).any():
            logger.error("NaN detected in output!")
            self._last_output = None
            if self._nan_output is None:
                raise RuntimeError("setup() must initialize NaN output sentinel")
            self.output_slice = self._nan_output
            return self.output_slice
        
        self._last_output = x
        self.output_slice = x[:1, :1, : min(16, x.shape[-1])]
        return self.output_slice
    
    def cleanup(self):
        """Clean up resources."""
        del self.layers
        del self.input
        self._nan_output = None
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 8,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 4,
    use_te: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run calibration-free FP8 benchmark."""
    
    benchmark = OptimizedFP8CalibrationFree(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        use_te=use_te,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="optimized_fp8_calibration_free"
    )
    
    benchmark.run()
    if benchmark._last_output is not None:
        benchmark.output_mean = scalar_tensor_to_float(benchmark._last_output.abs().mean())
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "output_mean": (
            benchmark.output_mean
            if benchmark.output_mean is not None
            else 0.0
        ),
        "use_te": benchmark.use_te,
        "num_layers": num_layers,
    }


class _FP8CalibrationFreeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Wrapper benchmark for calibration-free FP8."""

    allowed_benchmark_fn_antipatterns = ("host_transfer", "sync")

    def __init__(self) -> None:
        super().__init__()
        self._impl = OptimizedFP8CalibrationFree()
        self._output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_nan_output_buffer: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._payload_parameter_count = 0
        tokens = float(self._impl.batch_size * self._impl.seq_length)
        self.register_workload_metadata(requests_per_iteration=1.0, tokens_per_iteration=tokens)

    def setup(self) -> None:
        self._impl.setup()
        self._verify_output_buffer = torch.empty(
            1,
            1,
            min(16, self._impl.hidden_size),
            device=self._impl.device,
            dtype=torch.float32,
        )
        self._verify_nan_output_buffer = torch.empty((), device=self._impl.device, dtype=torch.float32)
        if hasattr(self._impl, "layers"):
            self._payload_parameter_count = sum(p.numel() for p in self._impl.layers.parameters())

    def benchmark_fn(self) -> None:
        output = self._impl.run()
        self._output = output
        if self._impl.input is None or self._output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self._impl.input is None
            or self._output is None
            or self._verify_output_buffer is None
            or self._verify_nan_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output.ndim == 0:
            verify_output = self._verify_nan_output_buffer
            verify_output.copy_(self._output, non_blocking=False)
        else:
            verify_output = self._verify_output_buffer
            output_slice = self._output[
                : verify_output.shape[0],
                : verify_output.shape[1],
                : verify_output.shape[2],
            ]
            verify_output.copy_(output_slice, non_blocking=False)
        self._set_verification_payload(
            inputs={"input": self._impl.input},
            output=verify_output,
            batch_size=self._impl.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
            precision_flags={"fp16": False, "bf16": True, "fp8": True, "tf32": False},
        )

    def teardown(self) -> None:
        self._impl.cleanup()
        self._output = None
        self._verify_output_buffer = None
        self._verify_nan_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

def get_benchmark() -> BaseBenchmark:
    return _FP8CalibrationFreeBenchmark()
