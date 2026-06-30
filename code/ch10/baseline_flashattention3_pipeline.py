"""Baseline FlashAttention-3 style attention without advanced pipelining.

This baseline demonstrates standard tiled attention computation without
the advanced pipelining techniques introduced in FlashAttention-3:
- No warp specialization (all warps do same work)
- Single-stage loading (no overlap of memory and compute)
- No persistent kernel patterns
- Standard online softmax without pipelining optimizations

FlashAttention-3 Key Innovations (not used in baseline):
1. Warp specialization: Producer warps for TMA, consumer warps for compute
2. 3-stage async pipeline: Memory → Softmax → Output stages overlap
3. Block scheduling: Persistent kernels with low-sync scheduling
4. Intra-warp pipelining: Overlapping GEMM with softmax within warps
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineFlashAttention3(nn.Module):
    """Baseline tiled attention without FA3 optimizations.
    
    This implementation:
    - Uses standard PyTorch operations
    - No explicit pipelining
    - Sequential loading and computation
    - Materializes intermediate attention scores
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        tile_size: int = 64,  # Not actually used for pipelining here
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads
        self.dropout = dropout
        self.tile_size = tile_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Standard QKV projections (not fused)
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
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
        
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass using naive attention (materializes full matrix).
        
        Baseline approach (O(n^2) memory):
        - Sequential Q, K, V projections
        - Computes full attention matrix
        - Memory: O(batch × heads × seq_len × seq_len)
        
        This is the pre-FlashAttention approach that FA3 improves upon.
        """
        batch_size, seq_len, _ = x.shape
        
        # Sequential projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # NAIVE ATTENTION: Materialize full attention matrix (O(n^2) memory)
        # This is what FlashAttention optimizes away
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            mask = self._causal_mask_for(seq_len, x.device)
            scores.masked_fill_(mask, float('-inf'))
        
        # Softmax (full matrix in memory)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class BaselineFlashAttention3Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline benchmark for FlashAttention-3 style attention.
    
    Measures performance of standard tiled attention without:
    - Warp specialization
    - 3-stage async pipelining
    - Persistent kernel scheduling
    - Intra-warp GEMM/softmax overlap
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        
        # FA3 benchmark config (larger sizes show pipeline benefits)
        self.batch_size = 4
        self.seq_len = 4096  # Long sequences benefit most from FA3
        self.hidden_dim = 2048
        self.num_heads = 32
        self.head_dim = 64
        self.use_causal = True
        
        self.input: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        
        tokens = self.batch_size * self.seq_len
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
        """Setup baseline attention model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.model = BaselineFlashAttention3(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=0.0,
        ).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Use BF16 for modern GPU workloads
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = self.model.to(dtype)

        # Reset RNG after model construction so baseline/optimized see identical weights and inputs.
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        weight_scale = 0.02
        q_weight = torch.randn(self.hidden_dim, self.hidden_dim, device=self.device, dtype=dtype) * weight_scale
        k_weight = torch.randn(self.hidden_dim, self.hidden_dim, device=self.device, dtype=dtype) * weight_scale
        v_weight = torch.randn(self.hidden_dim, self.hidden_dim, device=self.device, dtype=dtype) * weight_scale
        out_weight = torch.randn(self.hidden_dim, self.hidden_dim, device=self.device, dtype=dtype) * weight_scale

        with torch.inference_mode():
            self.model.q_proj.weight.copy_(q_weight)
            self.model.k_proj.weight.copy_(k_weight)
            self.model.v_proj.weight.copy_(v_weight)
            self.model.out_proj.weight.copy_(out_weight)

        self.input = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=dtype
        )

        self._verify_input = self.input.detach()
        self._verify_output_buffer = torch.empty_like(self.input)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = self.model(self.input, is_causal=self.use_causal)
        
    
    def benchmark_fn(self) -> None:
        """Benchmark baseline attention."""
        with self._nvtx_range("baseline_fa3_attention"):
            with torch.inference_mode():
                self.output = self.model(self.input, is_causal=self.use_causal)
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        dtype = self._verify_input.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-2),
        )
    
    def teardown(self) -> None:
        """Clean up."""
        self.model = None
        self.input = None
        self._verify_input = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual attention workload shape without approximate timing."""
        from ch10.flash_attention_common import compute_attention_workload_metrics

        metrics = compute_attention_workload_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            is_causal=False,
        )
        metrics["attention.uses_sdpa"] = 0.0
        metrics["attention.materializes_attn_matrix"] = 1.0
        return metrics
    
    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        
        with torch.inference_mode():
            output = self.model(self.input[:1, :128], is_causal=False)
            if torch.isnan(output).any():
                return "NaN in attention output"
        
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineFlashAttention3Benchmark()
