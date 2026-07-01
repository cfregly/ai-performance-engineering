"""
flex_attention_sparse_demo.py - FlexAttention with Block Sparsity (Ch14) (non-benchmark demo)

WHAT: FlexAttention is PyTorch's flexible attention API that allows custom
attention patterns via user-defined mask functions, compiled to efficient kernels.

WHY: Standard attention is O(n²) in memory and compute. Sparse patterns like:
  - Sliding window (local context only)
  - Block sparse (document boundaries)
  - Causal + local (LLM decoding)
  
reduce complexity to O(n) or O(n·w) where w is window size.

WHEN TO USE:
  - Long sequences where full attention OOMs
  - Document-aware attention (don't attend across docs)
  - Encoder-decoder with structured sparsity
  
REQUIREMENTS:
  - PyTorch 2.5+ (flex_attention API)
  - torch.compile for kernel generation
"""

from __future__ import annotations

from typing import Callable, Optional

from core.benchmark.utils import scalar_tensor_to_float

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    flex_attention,
)

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# Ensure we have the flex_attention API
try:
    from torch.nn.attention.flex_attention import flex_attention
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    print("Warning: flex_attention requires PyTorch 2.5+")


#============================================================================
# Mask Functions for FlexAttention
#============================================================================
# These mask functions use Torch operations to be compatible with tracing.
# They return boolean tensors/scalars that FlexAttention can compile.

def causal_mask(b, h, q_idx, kv_idx):
    """Standard causal mask: attend to current and previous positions only."""
    return q_idx >= kv_idx


def sliding_window_mask(window_size: int):
    """Create sliding window mask function.
    
    Attend only to positions within `window_size` of current position.
    """
    def mask_fn(b, h, q_idx, kv_idx):
        diff = q_idx - kv_idx
        return (diff >= -window_size) & (diff <= window_size)
    return mask_fn


def sliding_window_causal_mask(window_size: int):
    """Sliding window + causal: attend to last `window_size` tokens only."""
    def mask_fn(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) <= window_size
        return causal & in_window
    return mask_fn


def document_mask_fn(document_ids: torch.Tensor):
    """Document-aware mask: don't attend across document boundaries.
    
    Args:
        document_ids: [batch, seq_len] tensor where each position has a doc ID
        
    Note: This requires document_ids to be a compile-time constant or
    passed through FlexAttention's score_mod mechanism.
    """
    def mask_fn(b, h, q_idx, kv_idx):
        # For block mask creation, indices are integers
        return document_ids[b, q_idx] == document_ids[b, kv_idx]
    return mask_fn


def block_sparse_mask(block_size: int, sparse_ratio: float = 0.5):
    """Block sparse attention: attend to alternating blocks.
    
    Args:
        block_size: Size of attention blocks
        sparse_ratio: Fraction of blocks to keep (0.5 = every other block)
    """
    stride = max(1, int(1.0 / sparse_ratio))
    
    def mask_fn(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        # Attend within same block or to every Nth block
        same_block = (q_block == kv_block)
        global_block = ((kv_block % stride) == 0)
        return same_block | global_block
    return mask_fn


def prefix_lm_mask(prefix_length: int):
    """Prefix LM mask: bidirectional attention for prefix, causal after.
    
    Common for encoder-decoder style within decoder-only.
    """
    def mask_fn(b, h, q_idx, kv_idx):
        # In prefix region: bidirectional (always attend)
        in_prefix = (q_idx < prefix_length) & (kv_idx < prefix_length)
        # After prefix: causal
        causal = q_idx >= kv_idx
        return in_prefix | causal
    return mask_fn


#============================================================================
# FlexAttention Benchmark
#============================================================================

class FlexAttentionBenchmark:
    """Benchmark comparing FlexAttention patterns."""
    
    def __init__(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        head_dim: int = 128,
        seq_len: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.dtype = dtype
        self.device = device
        
        # Create test tensors
        self.q = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.k = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.v = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            dtype=dtype, device=device
        )
        
        # Use flex_attention directly (compilation happens internally)
        self._compiled_flex = flex_attention
        self._last_output: Optional[torch.Tensor] = None
        self._last_block_mask: Optional[torch.Tensor] = None
    
    def benchmark_pattern(
        self,
        name: str,
        mask_fn: Callable,
        num_warmup: int = 3,
        num_iterations: int = 10,
        store_output: bool = False,
    ) -> dict:
        """Benchmark a specific attention pattern."""
        
        if not HAS_FLEX_ATTENTION:
            return {"name": name, "error": "flex_attention not available"}
        
        # Create block mask from mask function
        block_mask = create_block_mask(
            mask_fn,
            B=self.batch_size,
            H=self.num_heads,
            Q_LEN=self.seq_len,
            KV_LEN=self.seq_len,
            device=self.device,
        )
        if store_output:
            self._last_block_mask = block_mask
        
        # Warmup
        for _ in range(num_warmup):
            _ = self._compiled_flex(self.q, self.k, self.v, block_mask=block_mask)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(self.device)
        
        start.record(current_stream)
        last_output = None
        for _ in range(num_iterations):
            last_output = self._compiled_flex(self.q, self.k, self.v, block_mask=block_mask)
        end.record(current_stream)
        end.synchronize()
        
        if store_output and last_output is not None:
            self._last_output = last_output
        elif store_output:
            self._last_output = None

        elapsed_ms = start.elapsed_time(end) / num_iterations
        
        # Calculate metrics
        total_flops = 4 * self.batch_size * self.num_heads * self.seq_len * self.seq_len * self.head_dim
        tflops = total_flops / (elapsed_ms / 1000) / 1e12
        
        # Estimate sparsity from block mask
        num_blocks = scalar_tensor_to_float(block_mask.kv_num_blocks.sum())
        max_blocks = (self.seq_len // _DEFAULT_SPARSE_BLOCK_SIZE) ** 2 * self.batch_size * self.num_heads
        sparsity = 1.0 - (num_blocks / max_blocks) if max_blocks > 0 else 0.0
        
        return {
            "name": name,
            "elapsed_ms": elapsed_ms,
            "tflops": tflops,
            "sparsity_pct": sparsity * 100,
            "block_size": _DEFAULT_SPARSE_BLOCK_SIZE,
        }
    
    def benchmark_full_attention(
        self,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> dict:
        """Benchmark full (dense) attention using SDPA for comparison."""
        
        # Use F.scaled_dot_product_attention for full attention baseline
        # This is more stable than torch.compile on custom attention
        
        # Warmup
        for _ in range(num_warmup):
            _ = torch.nn.functional.scaled_dot_product_attention(
                self.q, self.k, self.v, is_causal=False
            )
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(self.device)
        
        start.record(current_stream)
        for _ in range(num_iterations):
            _ = torch.nn.functional.scaled_dot_product_attention(
                self.q, self.k, self.v, is_causal=False
            )
        end.record(current_stream)
        end.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / num_iterations
        
        total_flops = 4 * self.batch_size * self.num_heads * self.seq_len * self.seq_len * self.head_dim
        tflops = total_flops / (elapsed_ms / 1000) / 1e12
        
        return {
            "name": "Full Attention (SDPA)",
            "elapsed_ms": elapsed_ms,
            "tflops": tflops,
            "sparsity_pct": 0.0,
        }
    
    def run_benchmarks(self) -> list:
        """Run benchmarks for all patterns."""
        results = []
        
        # Full attention baseline
        results.append(self.benchmark_full_attention())
        
        # Causal (50% sparsity)
        results.append(self.benchmark_pattern("Causal", causal_mask))
        
        # Sliding window variants
        for window in [128, 256, 512]:
            results.append(self.benchmark_pattern(
                f"Sliding Window (w={window})",
                sliding_window_mask(window)
            ))
        
        # Sliding window + causal
        results.append(self.benchmark_pattern(
            "Sliding Window Causal (w=256)",
            sliding_window_causal_mask(256)
        ))
        
        # Block sparse
        results.append(self.benchmark_pattern(
            "Block Sparse (50%)",
            block_sparse_mask(64, 0.5)
        ))
        
        # Prefix LM
        prefix_len = self.seq_len // 4
        results.append(self.benchmark_pattern(
            f"Prefix LM (prefix={prefix_len})",
            prefix_lm_mask(prefix_len)
        ))
        
        return results


#============================================================================
# SlidingWindowCausalAttention Module
#============================================================================

class SlidingWindowCausalAttention(nn.Module):
    """Production-ready sliding window causal attention.
    
    Combines:
    - Sliding window for memory efficiency
    - Causal masking for autoregressive generation
    - torch.compile for kernel fusion
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self._compiled_flex = torch.compile(flex_attention) if HAS_FLEX_ATTENTION else None
        self._block_mask_cache = {}
    
    def _get_block_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """Get or create cached block mask."""
        key = (batch_size, seq_len, str(device))
        
        if key not in self._block_mask_cache:
            mask_fn = sliding_window_causal_mask(self.window_size)
            self._block_mask_cache[key] = create_block_mask(
                mask_fn,
                B=batch_size,
                H=self.num_heads,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
        
        return self._block_mask_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sliding window causal attention.
        
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # FlexAttention with sliding window causal mask
        if self._compiled_flex is not None and HAS_FLEX_ATTENTION:
            block_mask = self._get_block_mask(batch_size, seq_len, x.device)
            output = self._compiled_flex(q, k, v, block_mask=block_mask)
        else:
            # Fallback to standard SDPA
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)


#============================================================================
# Main
#============================================================================

#============================================================================
# Benchmark Harness Integration
#============================================================================

class FlexAttentionSparseDemoBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for FlexAttention sparse demo."""

    def __init__(self):
        super().__init__()
        self.demo_benchmark = None
        self.batch_size = 2
        self.num_heads = 32
        self.head_dim = 128
        self.seq_len = 2048
        self._last = 0.0
        self.output = None
        self._block_mask = None
        self._payload_inputs = {}
        self._verification_payload = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize FlexAttention benchmark."""
        torch.manual_seed(42)
        self.demo_benchmark = FlexAttentionBenchmark(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            seq_len=self.seq_len,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device=str(self.device),
        )
        if HAS_FLEX_ATTENTION:
            self._block_mask = create_block_mask(
                causal_mask,
                B=self.batch_size,
                H=self.num_heads,
                Q_LEN=self.seq_len,
                KV_LEN=self.seq_len,
                device=self.device,
            )
        # Warmup with causal pattern
        if HAS_FLEX_ATTENTION:
            with torch.inference_mode():
                for _ in range(10):
                    _ = self.demo_benchmark._compiled_flex(
                        self.demo_benchmark.q,
                        self.demo_benchmark.k,
                        self.demo_benchmark.v,
                        block_mask=self._block_mask,
                    )
        payload_inputs = self._payload_inputs
        payload_inputs.clear()
        payload_inputs["q"] = self.demo_benchmark.q
        payload_inputs["k"] = self.demo_benchmark.k
        payload_inputs["v"] = self.demo_benchmark.v
        if self._block_mask is not None:
            payload_inputs["block_mask"] = self._block_mask
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: FlexAttention with causal mask."""
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError("FlexAttention not available (requires PyTorch 2.5+)")
        if self.demo_benchmark is None:
            raise RuntimeError("Benchmark not initialized")

        with torch.inference_mode():
            self.output = self.demo_benchmark._compiled_flex(
                self.demo_benchmark.q,
                self.demo_benchmark.k,
                self.demo_benchmark.v,
                block_mask=self._block_mask,
            )
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        inputs = self._payload_inputs
        self._set_verification_payload(
            inputs=inputs,
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.demo_benchmark.q.dtype == torch.float16,
                "bf16": self.demo_benchmark.q.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.demo_benchmark = None
        self._block_mask = None
        self._payload_inputs.clear()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        if not HAS_FLEX_ATTENTION:
            return "FlexAttention not available (requires PyTorch 2.5+)"
        if self.demo_benchmark is None:
            return "Benchmark not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return FlexAttentionSparseDemoBenchmark()
