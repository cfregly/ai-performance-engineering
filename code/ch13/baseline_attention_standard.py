"""baseline_attention_standard.py - Standard attention baseline (baseline)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class StandardAttention(nn.Module):
    """Standard attention implementation."""
    
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=torch.float16)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)


class BaselineAttentionStandardBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Standard attention baseline - no optimizations."""
    
    def __init__(self):
        super().__init__()
        self.q = None
        self.k = None
        self._k_t = None
        self.v = None
        self.batch_size = 2
        self.seq_len = 8192
        self.hidden_dim = 1024
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._scale = 0.0
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        self.q = torch.randn(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        self._k_t = self.k.transpose(-2, -1)
        self._scale = 1.0 / (self.head_dim ** 0.5)
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            self.num_heads,
            min(128, self.seq_len),
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.q is None or self.k is None or self._k_t is None or self.v is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_attention_standard"):
            with torch.inference_mode():
                scores = torch.matmul(self.q, self._k_t)
                scores.mul_(self._scale)
                attn = torch.softmax(scores, dim=-1)
                self.output = torch.matmul(attn, self.v)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
            : self._verify_output_buffer.shape[3],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.2, 2.0),
        )

    def teardown(self) -> None:
        self.q = None
        self.k = None
        self._k_t = None
        self.v = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.q is None or self.k is None or self.v is None:
            return "Inputs not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output


def get_benchmark() -> BaselineAttentionStandardBenchmark:
    """Factory function for harness discovery."""
    return BaselineAttentionStandardBenchmark()
