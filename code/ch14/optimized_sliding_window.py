"""Optimized causal attention via Flash SDPA vs naive O(n²) attention.

This optimized variant uses PyTorch's scaled_dot_product_attention, which
leverages Flash Attention kernels for fused execution without explicit
materialization of the O(n²) attention score matrix. The historical
``sliding_window`` filename is kept for benchmark-pair continuity; the
optimized path is full causal SDPA rather than an explicit local-window mask.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedAttentionModule(nn.Module):
    """Optimized attention using full causal SDPA (Flash Attention).

    Uses PyTorch's scaled_dot_product_attention for fused execution with the
    same sequence shape as the baseline, but without explicit attention-score
    materialization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,  # Kept only for pair/file API compatibility
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized attention forward pass using SDPA/Flash Attention.
        
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use SDPA which leverages Flash Attention kernels
        # This is much faster than naive matmul-based attention
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class OptimizedSlidingWindowBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: SDPA/Flash Attention vs naive O(n²) matmul attention."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        # Match baseline dimensions for fair comparison
        self.batch_size = 4
        self.seq_len = 4096
        self.embed_dim = 1024
        self.num_heads = 16
        self.window_size = 512  # Kept for API compatibility
        # Match baseline dtype for strict signature/workload comparability.
        self.dtype = torch.float16
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize optimized attention model."""
        torch.manual_seed(42)

        self.model = OptimizedAttentionModule(
            self.embed_dim, self.num_heads, self.window_size
        ).to(self.device, self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        # Warmup
        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self.x)

    def benchmark_fn(self) -> None:
        """Benchmark fused full-sequence causal SDPA."""
        with torch.inference_mode():
            self.output = self.model(self.x)
        if self.output is None or self.x is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.x},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        total_flops = 2.0 * self.seq_len * self.seq_len * self.head_dim * self.num_heads
        total_bytes = float(self.seq_len * self.num_heads * self.head_dim * 3 * 2)
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="bf16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None or self.x is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedSlidingWindowBenchmark()
