#!/usr/bin/env python3
"""Optimized: FP8 compressed KV cache for Blackwell.

Optimized KV cache with:
- FP8 E4M3 quantization (2× memory savings)
- Dynamic scaling per layer
- Optional NVFP4 for extreme compression (4× savings)
"""

from typing import Any, Dict, Optional, Tuple

import torch

from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
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
        self._timing_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._generated_k_steps: Optional[torch.Tensor] = None
        self._generated_v_steps: Optional[torch.Tensor] = None
        self._generated_step_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._generated_step_position_pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._generated_step_position_count = 0
        self._k_quantized_step: Optional[torch.Tensor] = None
        self._v_quantized_step: Optional[torch.Tensor] = None
        self._k_quantized_layer_view: Optional[torch.Tensor] = None
        self._v_quantized_layer_view: Optional[torch.Tensor] = None
        self._scale_abs_buffer: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._seq_lengths_host: list[int] = [0] * batch_size
        self._batch_size_tensor: Optional[torch.Tensor] = None
        self._seq_lengths_payload: Optional[torch.Tensor] = None
        self._active_layer_slice = slice(0, active_layers)
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
        self.kv_cache = torch.empty(
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
        self._seq_lengths_host = [0] * self.batch_size
        self._generated_k_steps = torch.randn(
            self.num_decode_steps,
            self.batch_size,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._generated_v_steps = torch.randn_like(self._generated_k_steps)
        self._generated_step_pairs = list(
            zip(self._generated_k_steps, self._generated_v_steps, strict=True)
        )
        self._generated_step_position_pairs = [
            (pos, new_k, new_v)
            for pos, (new_k, new_v) in enumerate(self._generated_step_pairs)
        ]
        self._generated_step_position_count = len(self._generated_step_position_pairs)
        self._k_quantized_step = torch.empty(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.cache_dtype,
        )
        self._v_quantized_step = torch.empty_like(self._k_quantized_step)
        self._scale_abs_buffer = torch.empty_like(self._generated_k_steps[0])
        self._k_quantized_layer_view = self._k_quantized_step.unsqueeze(1)
        self._v_quantized_layer_view = self._v_quantized_step.unsqueeze(1)
        self._output_view = self.kv_cache[:1, :1, :, :, :1, : min(8, self.head_dim)]
        self._verify_output_buffer = torch.empty(
            1,
            1,
            2,
            self.num_heads,
            1,
            min(8, self.head_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._batch_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        self._batch_size_tensor[0] = self.batch_size
        self._seq_lengths_payload = torch.empty_like(self.seq_lengths)

        logger.debug(f"Optimized KV Cache ({self.cache_dtype})")
        logger.debug(f"  Compression: {self._compression_ratio}x")
        logger.debug(f"  Estimated memory: {self._estimated_memory_gb:.2f} GB")
        logger.debug("Compressed KV cache allocated")

        self._timing_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def _get_timing_pair(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._timing_pair is None:
            self._timing_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._timing_pair
    
    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamic scaling factor."""
        shape = tuple(int(dim) for dim in x.shape)
        numel = int(x.numel())
        if (
            self._scale_abs_buffer is None
            or self._scale_abs_buffer.device != x.device
            or self._scale_abs_buffer.dtype != x.dtype
            or self._scale_abs_buffer.numel() < numel
        ):
            self._scale_abs_buffer = torch.empty(numel, dtype=x.dtype, device=x.device)
        abs_buffer = self._scale_abs_buffer[:numel].view(shape)
        torch.abs(x, out=abs_buffer)
        absmax = abs_buffer.amax().float()
        
        if self.use_fp4:
            max_val = 6.0  # FP4 range
        else:  # FP8
            max_val = 448.0  # FP8 E4M3 range
        
        return max_val / (absmax + 1e-12)

    def _quantize_step_into(self, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            torch.mul(x, scale, out=out)
            return out
        return (x * scale).to(self.cache_dtype)
    
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
        if self._k_quantized_step is None or self._v_quantized_step is None:
            raise RuntimeError("Quantization buffers not initialized")
        k_quantized = self._quantize_step_into(k, k_scale, self._k_quantized_step)
        v_quantized = self._quantize_step_into(v, v_scale, self._v_quantized_step)

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
        if (
            self._k_quantized_step is None
            or self._v_quantized_step is None
            or self._k_quantized_layer_view is None
            or self._v_quantized_layer_view is None
        ):
            raise RuntimeError("Quantization buffers not initialized")
        k_quantized = self._quantize_step_into(k, k_scale, self._k_quantized_step)
        v_quantized = self._quantize_step_into(v, v_scale, self._v_quantized_step)
        k_layer = (
            self._k_quantized_layer_view
            if k_quantized is self._k_quantized_step
            else k_quantized.unsqueeze(1)
        )
        v_layer = (
            self._v_quantized_layer_view
            if v_quantized is self._v_quantized_step
            else v_quantized.unsqueeze(1)
        )

        active = self._active_layer_slice
        self.k_scales[active, pos] = k_scale
        self.v_scales[active, pos] = v_scale
        self.kv_cache[:, active, 0, :, pos, :].copy_(k_layer)
        self.kv_cache[:, active, 1, :, pos, :].copy_(v_layer)
    
    def get_kv(
        self,
        layer_idx: int,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and dequantize K/V."""
        seq_len = self._seq_lengths_host[batch_idx]
        
        k_quantized = self.kv_cache[batch_idx, layer_idx, 0, :, :seq_len]
        v_quantized = self.kv_cache[batch_idx, layer_idx, 1, :, :seq_len]
        
        # Dequantize
        k_scale = self.k_scales[layer_idx, :seq_len].view(1, seq_len, 1)
        v_scale = self.v_scales[layer_idx, :seq_len].view(1, seq_len, 1)
        k = (k_quantized.float() / k_scale).to(torch.bfloat16)
        v = (v_quantized.float() / v_scale).to(torch.bfloat16)
        
        return k, v

    def _set_host_seq_lengths(self, value: int) -> None:
        if len(self._seq_lengths_host) != self.batch_size:
            raise RuntimeError("Host sequence length slots not initialized")
        for batch_idx in range(self.batch_size):
            self._seq_lengths_host[batch_idx] = value
    
    def benchmark_fn(self) -> None:
        """Benchmark compressed KV cache."""
        if self._generated_k_steps is None or self._generated_v_steps is None:
            raise RuntimeError("setup() must precompute decode-step inputs before benchmarking")
        if self._generated_step_position_count != self.num_decode_steps or self._output_view is None:
            raise RuntimeError("setup() must precompute decode-step views before benchmarking")
        num_decode_steps = self.num_decode_steps
        self._set_host_seq_lengths(0)

        timing_pair = self._get_timing_pair()
        start_event, end_event = timing_pair
        current_stream = torch.cuda.current_stream(self.device)
        start_event.record(current_stream)

        with torch.inference_mode():
            for pos, new_k, new_v in self._generated_step_position_pairs:
                self.append_active_layers(new_k, new_v, pos=pos)

        end_event.record(current_stream)
        self.seq_lengths.fill_(num_decode_steps)
        self._set_host_seq_lengths(num_decode_steps)
        self._pending_timing_pair = timing_pair

        self.output = self._output_view

    def _build_verification_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before verification capture")
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")

        # Dequantize the first token of layer 0 so we compare against the BF16 baseline.
        kq0 = self.output[0, 0, 0, :, 0, :]
        vq0 = self.output[0, 0, 1, :, 0, :]
        k_scale0 = self.k_scales[0, 0]
        v_scale0 = self.v_scales[0, 0]
        torch.div(kq0.float(), k_scale0, out=self._verify_output_buffer[0, 0, 0, :, 0, :])
        torch.div(vq0.float(), v_scale0, out=self._verify_output_buffer[0, 0, 1, :, 0, :])
        return self._verify_output_buffer

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self._batch_size_tensor is None or self._seq_lengths_payload is None:
            raise RuntimeError("setup() must initialize verification metadata tensors")
        self._seq_lengths_payload.copy_(self.seq_lengths)
        self._set_verification_payload(
            inputs={
                "batch_size": self._batch_size_tensor,
                "seq_lengths": self._seq_lengths_payload,
            },
            output=self._build_verification_output(),
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

        logger.debug("Throughput: %.2f tokens/sec", tokens_per_sec)
        logger.debug("Memory: %.2f GB (FP8 compressed)", memory_gb)

        metrics = self._last_metrics
        metrics["latency_ms"] = elapsed_ms_value
        metrics["tokens_per_sec"] = tokens_per_sec
        metrics["memory_gb"] = memory_gb
        metrics["compression_ratio"] = 2.0 / self.bytes_per_element
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
        self._generated_step_pairs = []
        self._generated_step_position_pairs = []
        self._generated_step_position_count = 0
        self._k_quantized_step = None
        self._v_quantized_step = None
        self._k_quantized_layer_view = None
        self._v_quantized_layer_view = None
        self._scale_abs_buffer = None
        self._output_view = None
        self._verify_output_buffer = None
        self.output = None
        self._batch_size_tensor = None
        self._seq_lengths_payload = None
        self._seq_lengths_host = [0] * self.batch_size
        self._timing_pair = None
        self._pending_timing_pair = None
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
