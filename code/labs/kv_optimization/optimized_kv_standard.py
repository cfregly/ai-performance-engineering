#!/usr/bin/env python3
"""Optimized: FP8 compressed KV cache for Blackwell.

Optimized KV cache with:
- FP8 E4M3 quantization (2× memory savings)
- Dynamic scaling per layer
- Optional NVFP4 for extreme compression (4× savings)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkMode,
)
from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedKVFP8Compressed(VerificationPayloadMixin, BaseBenchmark):
    """Optimized FP8 compressed KV cache."""

    signature_equivalence_group = "labs_kv_standard_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(
        self,
        batch_size: int = 8,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_seq_length: int = 8192,
        use_fp8: bool = True,
        use_fp4: bool = False,
        active_layers: int = 16,
        num_decode_steps: int = 256,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        if active_layers > num_layers:
            raise ValueError("active_layers must be <= num_layers")
        if num_decode_steps > max_seq_length:
            raise ValueError("num_decode_steps must be <= max_seq_length")
        self.use_fp8 = use_fp8
        self.use_fp4 = use_fp4
        self.active_layers = active_layers
        self.num_decode_steps = num_decode_steps
        self._last_metrics: Dict[str, Any] = {}
        self.output: Optional[torch.Tensor] = None
        self._pending_timing_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._generated_k_steps: Optional[torch.Tensor] = None
        self._generated_v_steps: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

        # Determine precision (fail fast if requested dtype is unavailable).
        if use_fp4:
            if not hasattr(torch, "float4_e2m1fn"):
                raise RuntimeError("FP4 requested but torch.float4_e2m1fn is unavailable")
            self.cache_dtype = torch.float4_e2m1fn
            self.bytes_per_element = 0.5
            compression_ratio = 4
        elif use_fp8:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("FP8 requested but torch.float8_e4m3fn is unavailable")
            self.cache_dtype = torch.float8_e4m3fn
            self.bytes_per_element = 1
            compression_ratio = 2
        else:
            raise RuntimeError("Optimized KV cache requires FP8 or FP4 compression")

        self.precision_label = str(self.cache_dtype).split(".")[-1]
        memory_per_token = num_layers * 2 * num_heads * head_dim * self.bytes_per_element
        total_memory_gb = (batch_size * max_seq_length * memory_per_token) / (1024**3)

        self._compression_ratio = compression_ratio
        self._estimated_memory_gb = total_memory_gb

    def setup(self):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        """Initialize compressed KV cache."""
        # Pre-allocate KV cache in compressed format
        self.kv_cache = torch.zeros(
            self.batch_size,
            self.num_layers,
            2,
            self.num_heads,
            self.max_seq_length,
            self.head_dim,
            device=self.device,
            dtype=self.cache_dtype
        )

        # Per-token scales needed to correctly dequantize older entries.
        self.k_scales = torch.ones(self.num_layers, self.max_seq_length, device=self.device, dtype=torch.float32)
        self.v_scales = torch.ones(self.num_layers, self.max_seq_length, device=self.device, dtype=torch.float32)

        # Sequence lengths
        self.seq_lengths = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self._generated_k_steps = torch.randn(
            self.num_decode_steps,
            self.batch_size,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._generated_v_steps = torch.randn_like(self._generated_k_steps)

        logger.debug(f"Optimized KV Cache ({self.cache_dtype})")
        logger.debug(f"  Compression: {self._compression_ratio}x")
        logger.debug(f"  Estimated memory: {self._estimated_memory_gb:.2f} GB")
        logger.debug("Compressed KV cache allocated")
    
    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamic scaling factor."""
        absmax = x.abs().amax().float()
        
        if self.use_fp4:
            max_val = 6.0  # FP4 range
        else:  # FP8
            max_val = 448.0  # FP8 E4M3 range
        
        return max_val / (absmax + 1e-12)
    
    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
        batch_indices: Optional[torch.Tensor] = None
    ):
        """Append K/V with compression."""
        if batch_indices is not None:
            raise RuntimeError("Optimized KV cache expects full-batch appends")
        if pos >= self.max_seq_length:
            raise RuntimeError("KV cache overflow in optimized append")
        
        # Quantize and store
        k_scale = self._compute_scale(k)
        v_scale = self._compute_scale(v)
        k_quantized = (k * k_scale).to(self.cache_dtype)
        v_quantized = (v * v_scale).to(self.cache_dtype)

        self.k_scales[layer_idx, pos] = k_scale
        self.v_scales[layer_idx, pos] = v_scale
        self.kv_cache[:, layer_idx, 0, :, pos].copy_(k_quantized)
        self.kv_cache[:, layer_idx, 1, :, pos].copy_(v_quantized)

    def append_active_layers(self, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Quantize once and append the same decode-step K/V across active layers."""
        if pos >= self.max_seq_length:
            raise RuntimeError("KV cache overflow in optimized append")

        k_scale = self._compute_scale(k)
        v_scale = self._compute_scale(v)
        k_quantized = (k * k_scale).to(self.cache_dtype)
        v_quantized = (v * v_scale).to(self.cache_dtype)

        active = slice(0, self.active_layers)
        self.k_scales[active, pos] = k_scale
        self.v_scales[active, pos] = v_scale
        self.kv_cache[:, active, 0, :, pos, :].copy_(k_quantized.unsqueeze(1))
        self.kv_cache[:, active, 1, :, pos, :].copy_(v_quantized.unsqueeze(1))
    
    def get_kv(
        self,
        layer_idx: int,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and dequantize K/V."""
        seq_len = self.seq_lengths[batch_idx].item()
        
        k_quantized = self.kv_cache[batch_idx, layer_idx, 0, :, :seq_len]
        v_quantized = self.kv_cache[batch_idx, layer_idx, 1, :, :seq_len]
        
        # Dequantize
        k_scale = self.k_scales[layer_idx, :seq_len].view(1, seq_len, 1)
        v_scale = self.v_scales[layer_idx, :seq_len].view(1, seq_len, 1)
        k = (k_quantized.float() / k_scale).to(torch.bfloat16)
        v = (v_quantized.float() / v_scale).to(torch.bfloat16)
        
        return k, v
    
    def benchmark_fn(self) -> None:
        """Benchmark compressed KV cache."""
        if self._generated_k_steps is None or self._generated_v_steps is None:
            raise RuntimeError("setup() must precompute decode-step inputs before benchmarking")
        num_decode_steps = self.num_decode_steps
        self.seq_lengths.zero_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for pos in range(num_decode_steps):
            new_k = self._generated_k_steps[pos]
            new_v = self._generated_v_steps[pos]
            self.append_active_layers(new_k, new_v, pos=pos)
            self.seq_lengths += 1

        end_event.record()
        self._pending_timing_pair = (start_event, end_event)

        # Verification output: dequantize the first token of layer 0 so we compare against the BF16 baseline.
        head_window = min(8, self.head_dim)
        kq0 = self.kv_cache[0, 0, 0, :, 0, :head_window]
        vq0 = self.kv_cache[0, 0, 1, :, 0, :head_window]
        k_scale0 = self.k_scales[0, 0]
        v_scale0 = self.v_scales[0, 0]
        k0 = (kq0.float() / k_scale0).unsqueeze(1)
        v0 = (vq0.float() / v_scale0).unsqueeze(1)
        view = torch.stack([k0, v0], dim=0).unsqueeze(0).unsqueeze(0)
        self.output = view.detach().clone()

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor([self.batch_size], dtype=torch.int64, device="cpu"),
                "seq_lengths": self.seq_lengths.detach().clone(),
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": self.cache_dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def finalize_iteration_metrics(self) -> Optional[Dict[str, Any]]:
        if self._pending_timing_pair is None:
            return None
        elapsed_ms_value = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
        elapsed_s = max(elapsed_ms_value, 1e-9) / 1000.0
        memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        tokens_per_sec = (self.batch_size * self.num_decode_steps) / elapsed_s

        logger.debug(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.debug(f"Memory: {memory_gb:.2f} GB (FP8 compressed)")

        self._last_metrics = {
            "latency_ms": elapsed_ms_value,
            "tokens_per_sec": tokens_per_sec,
            "memory_gb": memory_gb,
            "compression_ratio": 2.0 / self.bytes_per_element,
        }
        return None

    def get_custom_metrics(self) -> Dict[str, Any]:
        self.finalize_iteration_metrics()
        return self._last_metrics

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=True,
        )

    def teardown(self):
        """Clean up."""
        del self.kv_cache
        self._generated_k_steps = None
        self._generated_v_steps = None
        self.output = None
        super().teardown()


def run_benchmark(
    batch_size: int = 8,
    num_layers: int = 32,
    max_seq_length: int = 8192,
    use_fp8: bool = True,
    use_fp4: bool = False,
    active_layers: int = 16,
    num_decode_steps: int = 256,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized KV cache benchmark."""

    benchmark = OptimizedKVFP8Compressed(
        batch_size=batch_size,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        use_fp8=use_fp8,
        use_fp4=use_fp4,
        active_layers=active_layers,
        num_decode_steps=num_decode_steps,
    )

    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        profile_mode=profile,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    result = harness.benchmark(benchmark, name="optimized_kv_fp8_compressed")

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        "precision": benchmark.precision_label,
        **metrics,
    }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedKVFP8Compressed()
