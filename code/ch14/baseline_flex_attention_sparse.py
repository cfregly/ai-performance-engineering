"""baseline_flex_attention_sparse.py - Dense masked sliding-window attention baseline (Ch14).

Computes sliding-window causal attention with a dense score matrix + explicit masking.
This is intentionally inefficient (O(S^2)) and serves as the baseline for the
FlexAttention sparse implementation in optimized_flex_attention_sparse.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class DenseMaskedSlidingWindowAttention(nn.Module):
    """Sliding-window causal attention implemented with dense masking."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        scores.masked_fill_(~allowed_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class BaselineFlexAttentionSparseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: dense masked implementation of sliding-window causal attention."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[DenseMaskedSlidingWindowAttention] = None
        self.x: Optional[torch.Tensor] = None
        self.allowed_mask: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0

        self.batch_size = 1
        self.num_heads = 16
        self.head_dim = 64
        self.seq_len = 4096
        self.window_size = 128
        self.embed_dim = self.num_heads * self.head_dim
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for FlexAttention sparse baseline")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = DenseMaskedSlidingWindowAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
        ).to(self.device, dtype=self.dtype)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        idx = torch.arange(self.seq_len, device=self.device)
        q_idx = idx[:, None]
        kv_idx = idx[None, :]
        self.allowed_mask = (q_idx >= kv_idx) & ((q_idx - kv_idx) <= self.window_size)
        self.allowed_mask = self.allowed_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        self.x = torch.randn(self.batch_size, self.seq_len, self.embed_dim, device=self.device, dtype=self.dtype)
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            min(256, self.embed_dim),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self.x, self.allowed_mask)

    def benchmark_fn(self) -> None:
        if self.model is None or self.x is None or self.allowed_mask is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_flex_attention_sparse"):
            with torch.inference_mode():
                self.output = self.model(self.x, self.allowed_mask)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.x is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("capture_verification_payload() requires completed run")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self.x},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.x = None
        self.allowed_mask = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.output is None:
            return "Output not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlexAttentionSparseBenchmark()
