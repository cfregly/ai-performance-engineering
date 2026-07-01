"""Baseline sliding window attention - Naive O(n²) explicit attention.

This baseline uses explicit matmul/softmax/matmul operations which have:
- O(n²) memory for attention scores matrix
- Multiple kernel launches (matmul, softmax, matmul)
- No memory optimization

Compare with optimized_sliding_window.py which uses Flash Attention
via scaled_dot_product_attention for O(n) memory and fused kernels.
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


class NaiveAttentionModule(nn.Module):
    """Naive O(n²) attention using explicit matmul operations.
    
    This is the slow baseline that Flash Attention optimizes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer(
            "_causal_mask",
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False,
        )

    def _causal_mask_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = self._causal_mask
        if mask.device != device or mask.size(0) < seq_len:
            pos = torch.arange(seq_len, device=device)
            self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)
            mask = self._causal_mask
        return mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Naive O(n²) attention with explicit matmul operations."""
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Explicit O(n²) attention - creates full S×S attention matrix
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = self._causal_mask_for(S, x.device)
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class BaselineSlidingWindowBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Naive O(n²) attention (slow)."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 4096  # Longer sequence to show difference
        self.embed_dim = 1024
        self.num_heads = 16
        self.dtype = torch.float16  # Use float16 for better Flash Attention support
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
        """Setup: Initialize naive attention model."""
        torch.manual_seed(42)
        
        self.model = NaiveAttentionModule(
            self.embed_dim, self.num_heads
        ).to(self.device, self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        # Proper warmup to avoid cold cache effects
        for _ in range(5):
            with torch.inference_mode():
                _ = self.model(self.x)

    def benchmark_fn(self) -> None:
        """Benchmark: Naive attention."""
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
        return None

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.x is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineSlidingWindowBenchmark()
