"""optimized_attention.py - Flash Attention / SDPA optimized (optimized).

Chapter 10: Blackwell Software Optimizations

This optimized version uses PyTorch's scaled_dot_product_attention which:
- Automatically selects Flash Attention or Memory-Efficient Attention
- Uses FP16 for tensor core acceleration
- Fuses QK^T softmax and attention@V into a single kernel
- Has O(n) memory complexity (no materialized attention matrix)
- Optimized for GPU memory hierarchy

Key optimization: Memory-efficient attention algorithms avoid materializing
the full n×n attention matrix, dramatically reducing memory bandwidth.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized attention using Flash Attention/SDPA and FP16."""

    def __init__(self):
        super().__init__()
        self.query: Optional[torch.Tensor] = None
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        # Larger sizes to show tensor core optimization benefits
        self.batch_size = 16
        # Match the longer-context baseline so the Flash/SDPA benchmark shows
        # the intended memory-efficiency benefit on the same workload.
        self.seq_len = 1280
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Use FP16 for tensor core optimization
        # Create Q, K, V tensors in the format expected by SDPA
        self.query = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        self.key = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        self.value = torch.randn(
            self.batch_size, self.num_heads, self.seq_len, self.head_dim,
            device=self.device, dtype=torch.float16
        )
        self._verify_output_buffer = torch.empty_like(self.query, dtype=torch.float32)
        
        self._synchronize()
        tokens = float(self.batch_size * self.seq_len)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.batch_size),
        )

    def benchmark_fn(self) -> None:
        """Benchmark: Optimized attention using SDPA/Flash Attention.
        
        Optimizations:
        1. FP16 precision for tensor core acceleration
        2. Flash Attention fuses all operations into single kernel
        3. No materialized attention matrix (O(n) memory)
        4. Tiled computation for cache efficiency
        5. Optimized for modern GPU memory hierarchy
        """
        with self._nvtx_range("optimized_attention_flash"):
            with torch.inference_mode():
                # scaled_dot_product_attention automatically selects:
                # - Flash Attention V2 when available
                # - Memory-efficient attention as fallback
                self.output = torch.nn.functional.scaled_dot_product_attention(
                    self.query, self.key, self.value,
                    dropout_p=0.0,
                    is_causal=False,
                )
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "query": self.query,
                "key": self.key,
                "value": self.value,
            },
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.query = None
        self.key = None
        self.value = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual attention workload shape."""
        from ch10.flash_attention_common import compute_attention_workload_metrics

        metrics = compute_attention_workload_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            is_causal=False,
        )
        metrics["attention.uses_sdpa"] = 1.0
        metrics["attention.materializes_attn_matrix"] = 0.0
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.query is None or self.key is None or self.value is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> OptimizedAttentionBenchmark:
    return OptimizedAttentionBenchmark()
