"""baseline_sdpa_attention.py - Naive multi-kernel attention (low arithmetic intensity).

Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

This baseline demonstrates the naive approach to attention:
- Separate kernels for Q@K^T, softmax, and attn@V
- Multiple round-trips to global memory between operations
- Low arithmetic intensity due to unfused operations

The book mentions (line 164): "PyTorch's `scaled_dot_product_attention` (SDPA)
may dispatch to FlashAttention, memory-efficient, or cuDNN backends..."

This baseline intentionally uses separate operations to show why fusion matters.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def _attention_total_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> float:
    attention_elements = float(batch_size * num_heads * seq_len * seq_len)
    matmul_flops = 4.0 * attention_elements * float(head_dim)
    softmax_flops = 6.0 * attention_elements
    return matmul_flops + softmax_flops


def _naive_attention_total_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    element_size: int,
) -> float:
    qkv_output_elements = float(batch_size * num_heads * seq_len * head_dim)
    scores_elements = float(batch_size * num_heads * seq_len * seq_len)
    return float(element_size) * ((4.0 * qkv_output_elements) + (6.0 * scores_elements))


class BaselineSDPAAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Naive multi-kernel attention without fusion.
    
    Demonstrates low arithmetic intensity due to:
    1. Separate matmul for Q @ K^T (writes intermediate to HBM)
    2. Separate softmax kernel (reads/writes HBM)
    3. Separate matmul for attn_weights @ V (reads from HBM)
    
    Each operation has low FLOPS/byte due to intermediate memory traffic.
    """

    def __init__(self):
        super().__init__()
        # Typical LLM attention dimensions
        self.batch_size = 4
        self.num_heads = 32
        self.seq_len = 1024
        self.head_dim = 128
        
        self.query = None
        self.key = None
        self._key_t = None
        self.value = None
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._scale = 0.0
        
        # SDPA benchmark - fixed dimensions for attention comparison
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Create Q, K, V tensors in attention shape [B, H, S, D]
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.query = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.key = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.value = torch.randn(shape, device=self.device, dtype=torch.float16)
        self._key_t = self.key.transpose(-2, -1)
        self._verify_output_buffer = torch.empty_like(self.query)
        self._scale = 1.0 / (self.head_dim ** 0.5)
        
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Naive attention: 3 separate kernels, 2 HBM round-trips."""
        with self._nvtx_range("baseline_sdpa_attention"):
            with torch.inference_mode():
                # Kernel 1: Q @ K^T -> attn_scores (written to HBM)
                # Shape: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
                attn_scores = torch.matmul(
                    self.query, 
                    self._key_t
                )
                
                # Scale (fused with matmul in optimized version)
                attn_scores.mul_(self._scale)
                
                # Kernel 2: Softmax (reads attn_scores from HBM, writes back)
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Kernel 3: attn_weights @ V -> output
                # Shape: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
                self.output = torch.matmul(attn_weights, self.value)
                
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
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
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
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

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_roofline_metrics

        total_flops = _attention_total_flops(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
        )
        total_bytes = _naive_attention_total_bytes(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            element_size=2,
        )
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        if self.query is None:
            return "Query tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return BaselineSDPAAttentionBenchmark()
