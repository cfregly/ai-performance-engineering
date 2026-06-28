#!/usr/bin/env python3
"""Baseline: Standard KV cache without compression.

Standard KV cache using BF16 precision without optimization.
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


class BaselineKVStandard(VerificationPayloadMixin, BaseBenchmark):
    """Baseline KV cache (BF16, no compression).
    
    Goal: memory - This benchmark measures memory usage for KV cache.
    """

    signature_equivalence_group = "labs_kv_standard_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(
        self,
        batch_size: int = 8,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_seq_length: int = 8192,
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
        self.active_layers = active_layers
        self.num_decode_steps = num_decode_steps
        self._last_metrics: Dict[str, Any] = {}
        self.precision_label = "bf16"
        self.output: Optional[torch.Tensor] = None
        self._timing_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_pair: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._generated_k_steps: Optional[torch.Tensor] = None
        self._generated_v_steps: Optional[torch.Tensor] = None
        self._generated_step_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._output_view: Optional[torch.Tensor] = None
        self._seq_lengths_host: list[int] = [0] * batch_size
        self._active_layer_slice = slice(0, active_layers)
        self.register_workload_metadata(requests_per_iteration=1.0)

        memory_per_token = num_layers * 2 * num_heads * head_dim * 2  # 2 for K/V, 2 bytes for BF16
        total_memory_gb = (batch_size * max_seq_length * memory_per_token) / (1024**3)

        self._estimated_memory_gb = total_memory_gb

    def setup(self):
        """Initialize KV cache."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Pre-allocate KV cache
        # Shape: [batch, num_layers, 2, num_heads, max_seq, head_dim]
        self.kv_cache = torch.empty(
            self.batch_size,
            self.num_layers,
            2,  # K and V
            self.num_heads,
            self.max_seq_length,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16
        )

        # Current sequence lengths per batch
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
        self._output_view = self.kv_cache[:1, :1, :, :, :1, : min(8, self.head_dim)]

        logger.debug("Baseline KV Cache (BF16)")
        logger.debug(f"  Estimated memory: {self._estimated_memory_gb:.2f} GB")
        logger.debug("KV cache allocated")
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

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
        batch_indices: Optional[torch.Tensor] = None
    ):
        """Append K/V to cache."""
        if batch_indices is None:
            batch_indices = torch.arange(self.batch_size, device=self.device)
        if pos >= self.max_seq_length:
            raise RuntimeError("KV cache overflow in baseline append")

        for i, batch_idx in enumerate(batch_indices):
            self.kv_cache[batch_idx, layer_idx, 0, :, pos] = k[i]
            self.kv_cache[batch_idx, layer_idx, 1, :, pos] = v[i]

    def append_active_layers(self, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Append the same decode-step K/V tensors across all active layers."""
        if pos >= self.max_seq_length:
            raise RuntimeError("KV cache overflow in baseline append")

        active = self._active_layer_slice
        self.kv_cache[:, active, 0, :, pos, :].copy_(k.unsqueeze(1))
        self.kv_cache[:, active, 1, :, pos, :].copy_(v.unsqueeze(1))

    def get_kv(
        self,
        layer_idx: int,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V from cache."""
        seq_len = self._seq_lengths_host[batch_idx]
        k = self.kv_cache[batch_idx, layer_idx, 0, :, :seq_len]
        v = self.kv_cache[batch_idx, layer_idx, 1, :, :seq_len]
        return k, v

    def _set_host_seq_lengths(self, value: int) -> None:
        if len(self._seq_lengths_host) != self.batch_size:
            raise RuntimeError("Host sequence length slots not initialized")
        for batch_idx in range(self.batch_size):
            self._seq_lengths_host[batch_idx] = value

    def benchmark_fn(self) -> None:
        """Benchmark KV cache operations."""
        if self._generated_k_steps is None or self._generated_v_steps is None:
            raise RuntimeError("setup() must precompute decode-step inputs before benchmarking")
        if len(self._generated_step_pairs) != self.num_decode_steps or self._output_view is None:
            raise RuntimeError("setup() must precompute decode-step views before benchmarking")
        # Simulate decoding
        num_decode_steps = self.num_decode_steps
        self._set_host_seq_lengths(0)

        timing_pair = self._get_timing_pair()
        start_event, end_event = timing_pair
        start_event.record()

        for pos, (new_k, new_v) in enumerate(self._generated_step_pairs):
            self.append_active_layers(new_k, new_v, pos=pos)

        end_event.record()
        self.seq_lengths.fill_(num_decode_steps)
        self._set_host_seq_lengths(num_decode_steps)
        self._pending_timing_pair = timing_pair

        self.output = self._output_view.detach()

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor([self.batch_size], dtype=torch.int64, device="cpu"),
                "seq_lengths": self.seq_lengths.detach().clone(),
            },
            output=self.output.float().clone(),
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
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
        logger.debug(f"Memory: {memory_gb:.2f} GB")

        self._last_metrics = {
            "latency_ms": elapsed_ms_value,
            "tokens_per_sec": tokens_per_sec,
            "memory_gb": memory_gb,
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
        self._generated_step_pairs = []
        self._output_view = None
        self.output = None
        self._seq_lengths_host = [0] * self.batch_size
        self._timing_pair = None
        self._pending_timing_pair = None
        super().teardown()


def run_benchmark(
    batch_size: int = 8,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    max_seq_length: int = 8192,
    active_layers: int = 16,
    num_decode_steps: int = 256,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline KV cache benchmark."""

    benchmark = BaselineKVStandard(
        batch_size=batch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_length=max_seq_length,
        active_layers=active_layers,
        num_decode_steps=num_decode_steps,
    )

    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        profile_mode=profile,
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    result = harness.benchmark(benchmark, name="baseline_kv_standard")

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        "precision": benchmark.precision_label,
        **metrics,
    }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineKVStandard()
