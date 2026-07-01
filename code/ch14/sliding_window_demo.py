"""
sliding_window_demo.py - Sliding Window + Causal Attention (Ch14)

WHAT: Sliding window attention restricts attention to a local window around
each token, rather than attending to all previous tokens.

WHY: For long sequences, full attention is O(n²):
  - 8K tokens: 64M attention pairs
  - 32K tokens: 1B attention pairs
  
Sliding window with window_size=w is O(n·w):
  - 8K tokens, w=512: 4M pairs (16x reduction)
  - 32K tokens, w=512: 16M pairs (64x reduction)

WHEN TO USE:
  - Long-context inference (>4K tokens)
  - When most relevant context is local (e.g., code, prose)
  - Streaming/online inference with bounded memory
  
MODELS USING THIS:
  - Mistral 7B (w=4096)
  - LongT5
  - BigBird (combined with global tokens)
  - Longformer

REQUIREMENTS:
  - PyTorch 2.5+ for flex_attention
  - Or manual implementation with torch.compile
"""

from __future__ import annotations

import math
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


class SlidingWindowSelfAttention(nn.Module):
    """Efficient sliding window self-attention.
    
    Implements sliding window + causal masking with:
    - FlexAttention backend when available
    - Chunked computation fallback for memory efficiency
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 512,
        dropout: float = 0.0,
        use_flex_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.use_flex_attention = use_flex_attention
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Try to import flex_attention
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            self._has_flex = True
            self._flex_attention = torch.compile(flex_attention)
            self._create_block_mask = create_block_mask
        except ImportError:
            self._has_flex = False
            
        self._mask_cache = {}
    
    def _sliding_window_causal_mask(self, q_idx, kv_idx):
        """Mask function: causal + sliding window.
        
        Uses & instead of 'and' for torch tracing compatibility.
        """
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) <= self.window_size
        return causal & in_window
    
    def _get_block_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ):
        """Get or create cached block mask for FlexAttention."""
        key = (batch_size, seq_len, str(device))
        
        if key not in self._mask_cache:
            # Create mask function that captures window_size
            # Use & instead of 'and' for torch tracing compatibility
            window = self.window_size
            def mask_fn(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                in_window = (q_idx - kv_idx) <= window
                return causal & in_window
            
            self._mask_cache[key] = self._create_block_mask(
                mask_fn,
                B=batch_size,
                H=self.num_heads,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
        
        return self._mask_cache[key]
    
    def _chunked_sliding_attention(
        self,
        q: torch.Tensor,  # [B, H, S, D]
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked sliding window attention for memory efficiency.
        
        Processes attention in chunks to avoid materializing full O(n²) matrix.
        """
        B, H, S, D = q.shape
        window = self.window_size
        
        # Output accumulator
        output = torch.zeros_like(q)
        
        # Process each query position
        # In practice, chunk multiple positions together for efficiency
        chunk_size = min(64, S)
        
        for q_start in range(0, S, chunk_size):
            q_end = min(q_start + chunk_size, S)
            q_chunk = q[:, :, q_start:q_end]  # [B, H, chunk, D]
            
            # KV range: from (q_start - window) to q_end
            kv_start = max(0, q_start - window)
            kv_end = q_end
            
            k_chunk = k[:, :, kv_start:kv_end]
            v_chunk = v[:, :, kv_start:kv_end]
            
            # Compute attention scores
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
            
            # Create sliding window causal mask for this chunk
            q_positions = torch.arange(q_start, q_end, device=q.device)
            kv_positions = torch.arange(kv_start, kv_end, device=q.device)
            
            # [chunk_q, chunk_kv]
            pos_diff = q_positions[:, None] - kv_positions[None, :]
            
            # Causal: pos_diff >= 0, Window: pos_diff <= window
            mask = (pos_diff >= 0) & (pos_diff <= window)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, chunk_q, chunk_kv]
            
            # Apply mask
            scores.masked_fill_(~mask, float('-inf'))
            
            # Softmax and weighted sum
            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            
            output[:, :, q_start:q_end] = torch.matmul(attn, v_chunk)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, seq_len, embed_dim]
            kv: Optional separate KV tensor (for cross-attention)
            
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Choose attention implementation
        if self._has_flex and self.use_flex_attention:
            # Use FlexAttention with block sparse mask
            block_mask = self._get_block_mask(B, S, x.device)
            output = self._flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Fallback to chunked implementation
            output = self._chunked_sliding_attention(q, k, v)
        
        # Reshape and output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class SlidingWindowTransformerBlock(nn.Module):
    """Transformer block with sliding window attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        window_size: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = SlidingWindowSelfAttention(
            embed_dim, num_heads, window_size, dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


#============================================================================
# Benchmark
#============================================================================

def benchmark_sliding_window():
    """Compare sliding window vs full attention."""
    print("Sliding Window Attention Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()
    
    # Test config
    batch_size = 4
    embed_dim = 1024
    num_heads = 16
    window_size = 512
    dtype = torch.bfloat16
    
    # Create models
    sliding_attn = SlidingWindowSelfAttention(
        embed_dim, num_heads, window_size,
        use_flex_attention=True
    ).to(device, dtype)
    
    full_attn = nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True
    ).to(device, dtype)
    
    # Test various sequence lengths
    seq_lens = [1024, 2048, 4096, 8192]
    
    print(f"Config: B={batch_size}, D={embed_dim}, H={num_heads}, W={window_size}")
    print()
    print(f"{'Seq Len':<10} {'Full Attn':<15} {'Sliding':<15} {'Speedup':<10} {'Mem Saved':<12}")
    print("-" * 62)
    
    for seq_len in seq_lens:
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup sliding
        for _ in range(3):
            _ = sliding_attn(x)
        torch.cuda.synchronize()
        
        # Benchmark sliding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(device)
        
        start.record(current_stream)
        for _ in range(10):
            _ = sliding_attn(x)
        end.record(current_stream)
        end.synchronize()
        
        sliding_ms = start.elapsed_time(end) / 10
        sliding_mem = torch.cuda.max_memory_allocated() / 1e9
        
        # Clear cache for full attention
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark full attention (may OOM for long sequences)
        try:
            pos = torch.arange(seq_len, device=device)
            causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)
            
            # Warmup
            for _ in range(3):
                _ = full_attn(x, x, x, attn_mask=causal_mask)
            torch.cuda.synchronize()
            
            start.record(current_stream)
            for _ in range(10):
                _ = full_attn(x, x, x, attn_mask=causal_mask)
            end.record(current_stream)
            end.synchronize()
            
            full_ms = start.elapsed_time(end) / 10
            full_mem = torch.cuda.max_memory_allocated() / 1e9
            
            speedup = full_ms / sliding_ms
            mem_saved = (1 - sliding_mem / full_mem) * 100
            
            print(f"{seq_len:<10} {full_ms:<15.2f} {sliding_ms:<15.2f} {speedup:<10.2f}x {mem_saved:<12.1f}%")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{seq_len:<10} {'OOM':<15} {sliding_ms:<15.2f} {'∞':<10} {'100%':<12}")
                torch.cuda.empty_cache()
            else:
                raise
    
    print()
    print("Notes:")
    print(f"  - Window size: {window_size} (attend to last {window_size} tokens)")
    print("  - Sliding window is O(n·w) vs O(n²) for full attention")
    print("  - Memory savings increase with sequence length")
    print("  - Quality impact depends on task; local context often sufficient")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class OptimizedSDPAAttention(nn.Module):
    """Optimized attention using scaled_dot_product_attention (SDPA).
    
    SDPA automatically uses Flash Attention when available, providing:
    - O(n) memory instead of O(n²)
    - Fused kernel for matmul/softmax/matmul
    - 2-4x speedup over naive attention
    """
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using SDPA (Flash Attention backend)."""
        B, S, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # SDPA with Flash Attention - O(n) memory, fused kernels
        # is_causal=True enables causal masking efficiently
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Output projection
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(output)


class SlidingWindowDemoBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Flash Attention via SDPA for O(n) memory attention.
    
    Key optimization over baseline:
    - Uses scaled_dot_product_attention (SDPA)
    - Flash Attention backend: O(n) memory vs O(n²)
    - Fused kernel: single launch vs matmul/softmax/matmul
    - Native causal masking support
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.x = None
        self.output = None
        # Match baseline dimensions for fair comparison
        self.batch_size = 4
        self.num_heads = 16
        self.head_dim = 64  # embed_dim(1024) / num_heads(16)
        self.seq_len = 4096
        self.embed_dim = 1024
        self._last = 0.0
        self.parameter_count: int = 0
        self._verification_payload = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize SDPA-based attention module."""
        torch.manual_seed(42)
        
        # Match baseline dtype for strict signature/workload comparability.
        self.dtype = torch.float16
        
        self.model = OptimizedSDPAAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        ).to(self.device, self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        # Warmup - important for Flash Attention kernel selection
        for _ in range(10):
            with torch.inference_mode():
                _ = self.model(self.x)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: SDPA/Flash Attention forward pass."""
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
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.x = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)
    
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
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return SlidingWindowDemoBenchmark()
