"""Baseline dense attention - naive explicit attention.

This baseline uses explicit matmul/softmax/matmul operations which have:
- O(n²) memory for attention scores matrix
- Multiple kernel launches (matmul, softmax, matmul)
- No memory optimization

Compare with optimized_dense_attention_flash.py which uses Flash Attention
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
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineDenseAttentionFlashBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: naive dense attention with explicit score materialization."""

    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_seq_len = 4096  # Longer sequence to show Flash Attention benefit
        self.qkv_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self.dtype = torch.float16
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._causal_mask: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        
        tokens = self.batch_size * self.max_seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.qkv_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 3,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.out_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.inputs = torch.randn(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        pos = torch.arange(self.max_seq_len, device=self.device)
        self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._payload_parameter_count = sum(p.numel() for p in self.qkv_proj.parameters()) + sum(
            p.numel() for p in self.out_proj.parameters()
        )
        
        # Proper warmup
        for _ in range(5):
            with torch.inference_mode():
                self._forward_naive()
        torch.cuda.synchronize(self.device)

    def _forward_naive(self):
        """Naive O(n²) attention with explicit matmul."""
        qkv = self.qkv_proj(self.inputs)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for attention
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Explicit O(n²) attention - creates full S×S attention matrix
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        if self._causal_mask is None:
            raise RuntimeError("Causal mask not initialized")
        scores.masked_fill_(self._causal_mask[:S, :S], float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
        return self.out_proj(output)

    def benchmark_fn(self) -> None:
        """Benchmark: Naive attention."""
        with nvtx_range("baseline_dense_attention_flash", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self._forward_naive()
        if self._verify_input is None:
            raise RuntimeError("Verification input missing")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.qkv_proj = None
        self.out_proj = None
        self.inputs = None
        self._causal_mask = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        if self.qkv_proj is None or self.inputs is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDenseAttentionFlashBenchmark()
