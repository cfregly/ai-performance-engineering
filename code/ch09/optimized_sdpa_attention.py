"""optimized_sdpa_attention.py - Fused attention with high arithmetic intensity.

Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

This optimized version demonstrates the power of kernel fusion:
- Uses torch.nn.functional.scaled_dot_product_attention (SDPA)
- Can dispatch to FlashAttention, memory-efficient, or cuDNN backends
- Single fused kernel eliminates intermediate HBM traffic
- Much higher arithmetic intensity (FLOPS/byte)

The book mentions (line 164): "To control the backend selection, use
`torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`, for instance."

FlashAttention achieves high arithmetic intensity by:
1. Tiling in shared memory to avoid HBM writes of attention scores
2. Online softmax computation (never materializes full S×S matrix)
3. Single fused kernel for Q@K^T, softmax, attn@V
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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


def _fused_attention_total_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    element_size: int,
) -> float:
    qkv_output_elements = float(batch_size * num_heads * seq_len * head_dim)
    return float(element_size) * (4.0 * qkv_output_elements)


class OptimizedSDPAAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Fused SDPA attention with FlashAttention backend.
    
    Demonstrates high arithmetic intensity through:
    1. Single fused kernel (no intermediate HBM writes)
    2. Tiled computation in shared memory
    3. Online softmax (streaming, never materializes S×S matrix)
    
    This achieves 4-10x higher FLOPS/byte compared to naive attention.
    """

    def __init__(self):
        super().__init__()
        # Same dimensions as baseline for fair comparison
        self.batch_size = 4
        self.num_heads = 32
        self.seq_len = 1024
        self.head_dim = 128
        
        self.query = None
        self.key = None
        self.value = None
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        
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

        if torch.cuda.is_available() and not torch.backends.cuda.flash_sdp_enabled():
            raise RuntimeError("Flash SDP backend is disabled; enable flash attention for this benchmark.")
        
        # Create Q, K, V tensors in attention shape [B, H, S, D]
        shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.query = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.key = torch.randn(shape, device=self.device, dtype=torch.float16)
        self.value = torch.randn(shape, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty_like(self.query)
        
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Fused SDPA: Single kernel, minimal HBM traffic."""
        with self._nvtx_range("optimized_sdpa_attention"):
            with torch.inference_mode():
                # Force FlashAttention to avoid slow math fallback paths.
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    # This single call fuses Q@K^T, scale, softmax, attn@V
                    # and uses tiled shared memory to avoid HBM writes of S×S matrix.
                    self.output = F.scaled_dot_product_attention(
                        self.query,
                        self.key,
                        self.value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                
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
        total_bytes = _fused_attention_total_bytes(
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
    return OptimizedSDPAAttentionBenchmark()
