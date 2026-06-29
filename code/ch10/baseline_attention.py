"""baseline_attention.py - Naive attention implementation (baseline).

Chapter 10: Blackwell Software Optimizations

This baseline demonstrates a naive attention implementation with:
- Explicit Q, K, V matrix multiplications
- Softmax with explicit exp/sum operations
- Multiple separate kernel launches
- No memory-efficient attention algorithms

The naive approach has O(n²) memory complexity and poor cache efficiency.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive attention with explicit matrix operations."""

    def __init__(self):
        super().__init__()
        self.query: Optional[torch.Tensor] = None
        self.key: Optional[torch.Tensor] = None
        self._key_t: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        # Larger sizes to show optimization benefits
        self.batch_size = 16
        # Retuned to a longer context so the eager-vs-SDPA gap reflects the
        # intended O(n^2) materialization penalty on modern decoder workloads.
        self.seq_len = 1280
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create Q, K, V tensors in FP16 so baseline/optimized compare the same precision.
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
        self._key_t = self.key.transpose(-2, -1)
        self._verify_output_buffer = torch.empty_like(self.query, dtype=torch.float32)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self._synchronize()
        tokens = float(self.batch_size * self.seq_len)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=float(self.batch_size),
        )

    def benchmark_fn(self) -> None:
        """Benchmark: Naive attention with explicit operations.
        
        Naive approach:
        1. Compute attention scores: QK^T (O(n²) memory)
        2. Scale scores
        3. Apply softmax (materializes full attention matrix)
        4. Compute output: Attention @ V
        
        This creates a full n×n attention matrix, causing:
        - O(n²) memory usage
        - Poor cache locality
        - Multiple kernel launches
        """
        with self._nvtx_range("baseline_attention_naive"):
            with torch.inference_mode():
                # Naive: Explicit matrix multiplications
                # Q @ K^T -> (batch, heads, seq, seq) attention matrix
                attn_scores = torch.matmul(self.query, self._key_t)
                attn_scores.mul_(self.scale)
                
                # Softmax over last dimension (materializes full attention matrix).
                # Keep the compute dtype aligned with the optimized SDPA path.
                attn_weights = torch.softmax(attn_scores, dim=-1)
                
                # Attention @ V -> output
                self.output = torch.matmul(attn_weights, self.value)
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
        self._key_t = None
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
        metrics["attention.uses_sdpa"] = 0.0
        metrics["attention.materializes_attn_matrix"] = 1.0
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.query is None or self.key is None or self.value is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaselineAttentionBenchmark:
    return BaselineAttentionBenchmark()
