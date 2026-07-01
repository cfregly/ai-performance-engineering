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
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._attn_merge_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._qkv_weight_t = self.qkv_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()

    def _ensure_projection_buffers(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_shape = (batch_size, seq_len, 3 * self.embed_dim)
        merge_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        output_shape = (batch_size, seq_len, self.embed_dim)
        rows = int(batch_size * seq_len)
        if (
            self._qkv_buffer is None
            or self._qkv_buffer.size(0) < rows
            or self._qkv_buffer.device != x.device
            or self._qkv_buffer.dtype != x.dtype
        ):
            self._qkv_buffer = torch.empty(rows, qkv_shape[-1], device=x.device, dtype=x.dtype)
        if (
            self._attn_merge_buffer is None
            or self._attn_merge_buffer.size(0) < rows
            or self._attn_merge_buffer.device != x.device
            or self._attn_merge_buffer.dtype != x.dtype
        ):
            self._attn_merge_buffer = torch.empty(rows, output_shape[-1], device=x.device, dtype=x.dtype)
        if (
            self._output_buffer is None
            or self._output_buffer.size(0) < rows
            or self._output_buffer.device != x.device
            or self._output_buffer.dtype != x.dtype
        ):
            self._output_buffer = torch.empty(rows, output_shape[-1], device=x.device, dtype=x.dtype)
        return (
            self._qkv_buffer[:rows].view(qkv_shape),
            self._output_buffer[:rows].view(output_shape),
            self._attn_merge_buffer[:rows].view(merge_shape),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized attention forward pass using SDPA/Flash Attention.
        
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        B, S, _ = x.shape
        
        # QKV projection
        if torch.is_grad_enabled():
            qkv = self.qkv_proj(x)
            output_buffer = None
            attn_merge_buffer = None
        else:
            if self._qkv_weight_t is None or self._out_proj_weight_t is None:
                self.cache_weight_views()
            qkv_buffer, output_buffer, attn_merge_buffer = self._ensure_projection_buffers(x, B, S)
            qkv = torch.matmul(x, self._qkv_weight_t, out=qkv_buffer)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Use SDPA which leverages Flash Attention kernels
        # This is much faster than naive matmul-based attention
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and output projection
        if attn_merge_buffer is None:
            output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        else:
            attn_merge_buffer.copy_(output.transpose(1, 2))
            output = attn_merge_buffer.view(B, S, self.embed_dim)
        if output_buffer is not None:
            return torch.matmul(output, self._out_proj_weight_t, out=output_buffer)
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
        self.model.cache_weight_views()
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
