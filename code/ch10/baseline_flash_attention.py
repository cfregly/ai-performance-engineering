"""baseline_flash_attention.py - Baseline attention without FlashAttention in GEMM context."""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from ch10.flash_attention_common import (
    FLASH_ATTENTION_OUTPUT_TOLERANCE,
    compute_attention_workload_metrics,
)


class BaselineFlashAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Standard attention without FlashAttention optimizations.
    
    Uses manual attention computation (matmul + softmax + matmul) which
    materializes the full O(seq_len²) attention matrix.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self._causal_mask = None
        self.batch_size = 4
        self.seq_len = 1024  # Match optimized version
        self.hidden_dim = 512  # Match optimized version
        self.num_heads = 8  # Match optimized version
        self.use_causal = True  # Match optimized version
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize attention model without FlashAttention."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Projections
        # Use the same datatype as the optimized (FlashAttention/SDPA) path so the
        # comparison isolates the tiling/memory behavior rather than precision.
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Baseline still materializes the full attention matrix, but uses FP16
        # inputs like the optimized path.
        self.input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        if self.use_causal:
            pos = torch.arange(self.seq_len, device=self.device)
            self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        self._verify_output_buffer = torch.empty_like(self.input, dtype=torch.float32)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                self._manual_attention(self.input, is_causal=self.use_causal)
    
    def _manual_attention(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        """Manual attention that materializes full attention matrix."""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Manual attention: Q @ K^T -> O(seq_len²) memory
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if is_causal:
            # Apply a causal mask so the baseline matches the optimized SDPA path.
            if self._causal_mask is None:
                raise RuntimeError("Causal mask missing - setup() must initialize it")
            attn_weights.masked_fill_(self._causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # attn_weights @ V
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(output)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard attention without FlashAttention."""
        with self._nvtx_range("baseline_flash_attention"):
            with torch.inference_mode():
                self.output = self._manual_attention(self.input, is_causal=self.use_causal)
        if self.output is None or self.input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=FLASH_ATTENTION_OUTPUT_TOLERANCE,
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.input = None
        self._causal_mask = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return workload metrics derived from the real attention shape."""
        return compute_attention_workload_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            is_causal=self.use_causal,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.q_proj is None:
            return "Projections not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineFlashAttentionBenchmark()
