"""flash_attention_sdpa_bench.py - Auxiliary Flash Attention SDPA bench.

Auxiliary manual benchmark for the Flash Attention / SDPA implementation used
in Chapter 14.

This script intentionally avoids `baseline_` / `optimized_` naming so the
harness does not auto-discover it as a canonical pair. The paired harness
target lives in `baseline_sliding_window.py` / `optimized_sliding_window.py`.

It uses scaled_dot_product_attention which leverages Flash Attention for:
- O(n) memory instead of O(n²)
- Fused kernel (no intermediate materialization)
- Hardware-optimized attention computation
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


class FlashAttentionModule(nn.Module):
    """Optimized attention using Flash Attention via SDPA.
    
    Uses torch.nn.functional.scaled_dot_product_attention which
    automatically uses the Flash Attention backend when available.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.output = None
        self._verify_input = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self._attn_merge_buffer: Optional[torch.Tensor] = None

    def _ensure_attention_merge_buffer(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        rows = int(batch_size * seq_len)
        if (
            self._attn_merge_buffer is None
            or self._attn_merge_buffer.size(0) < rows
            or self._attn_merge_buffer.device != x.device
            or self._attn_merge_buffer.dtype != x.dtype
        ):
            self._attn_merge_buffer = torch.empty(rows, self.embed_dim, device=x.device, dtype=x.dtype)
        return self._attn_merge_buffer[:rows].view(batch_size, seq_len, self.num_heads, self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flash Attention forward pass - O(n) memory, fused kernel."""
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Flash Attention via SDPA - O(n) memory, no S×S matrix!
        output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0
        )
        
        # Output projection
        if torch.is_grad_enabled():
            output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        else:
            merge_buffer = self._ensure_attention_merge_buffer(x, B, S)
            merge_buffer.copy_(output.transpose(1, 2))
            output = merge_buffer.view(B, S, self.embed_dim)
        return self.out_proj(output)


class FlashAttentionSdpaBenchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Auxiliary Flash Attention via SDPA benchmark."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.batch_size = 4
        self.seq_len = 4096  # Same as baseline for fair comparison
        self.embed_dim = 1024
        self.num_heads = 16
        self.dtype = torch.float16  # Flash Attention works best with float16
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None

    def setup(self) -> None:
        """Setup: Initialize Flash Attention model."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = FlashAttentionModule(
            self.embed_dim, self.num_heads
        ).to(self.device, self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        self._verify_output_buffer = torch.empty_like(self.x)
        
        # Proper warmup to avoid cold cache effects
        for _ in range(5):
            with torch.inference_mode():
                _ = self.model(self.x)

    def benchmark_fn(self) -> None:
        """Benchmark: Flash Attention."""
        with torch.inference_mode():
            output = self.model(self.x)
            self.output = output
        if self.output is None or self.x is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.x},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.x = None
        self.output = None
        self._verify_output_buffer = None
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
    return FlashAttentionSdpaBenchBenchmark()
