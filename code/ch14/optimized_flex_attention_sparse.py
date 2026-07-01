"""
optimized_flex_attention_sparse.py - FlexAttention with block-sparse sliding-window masks (Ch14)

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

# Check for FlexAttention availability
HAS_FLEX_ATTENTION = False
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        _DEFAULT_SPARSE_BLOCK_SIZE,
    )
    HAS_FLEX_ATTENTION = True
except ImportError:
    _DEFAULT_SPARSE_BLOCK_SIZE = 128


#============================================================================
# Mask Functions for Different Sparsity Patterns
#============================================================================

def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Standard causal mask: attend to current and previous positions only."""
    return q_idx >= kv_idx


def sliding_window_mask(window_size: int):
    """Create sliding window mask function."""
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return abs(q_idx - kv_idx) <= window_size
    return mask_fn


def sliding_window_causal_mask(window_size: int):
    """Sliding window + causal: attend to last `window_size` tokens only."""
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        causal = q_idx >= kv_idx
        in_window = q_idx - kv_idx <= window_size
        return causal and in_window
    return mask_fn


def block_sparse_mask(block_size: int, sparse_ratio: float = 0.5):
    """Block sparse attention: attend to alternating blocks."""
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        stride = int(1.0 / sparse_ratio)
        return q_block == kv_block or kv_block % stride == 0
    return mask_fn


#============================================================================
# FlexAttention Module
#============================================================================

class SlidingWindowCausalAttention(nn.Module):
    """Production-ready sliding window causal attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._attn_merge_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None
        
        self._compiled_flex = torch.compile(flex_attention) if HAS_FLEX_ATTENTION else None
        self._block_mask_cache = {}

    def cache_weight_views(self) -> None:
        self._qkv_weight_t = self.qkv_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()

    def _ensure_projection_buffers(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_shape = (batch_size, seq_len, 3 * self.embed_dim)
        merge_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        output_shape = (batch_size, seq_len, self.embed_dim)
        rows = int(batch_size * seq_len)
        if (
            self._qkv_buffer is None
            or self._qkv_buffer.size(0) < rows
            or self._qkv_buffer.device != x.device
            or self._qkv_buffer.dtype != x.dtype
        ):
            self._qkv_buffer = torch.empty(rows, qkv_shape[-1], device=x.device, dtype=x.dtype)
        if (
            self._attn_merge_buffer is None
            or self._attn_merge_buffer.size(0) < rows
            or self._attn_merge_buffer.device != x.device
            or self._attn_merge_buffer.dtype != x.dtype
        ):
            self._attn_merge_buffer = torch.empty(rows, output_shape[-1], device=x.device, dtype=x.dtype)
        if (
            self._output_buffer is None
            or self._output_buffer.size(0) < rows
            or self._output_buffer.device != x.device
            or self._output_buffer.dtype != x.dtype
        ):
            self._output_buffer = torch.empty(rows, output_shape[-1], device=x.device, dtype=x.dtype)
        return (
            self._qkv_buffer[:rows].view(qkv_shape),
            self._output_buffer[:rows].view(output_shape),
            self._attn_merge_buffer[:rows].view(merge_shape),
        )
    
    def _get_block_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """Get or create cached block mask."""
        key = (batch_size, seq_len, str(device))
        
        if key not in self._block_mask_cache:
            window = self.window_size
            def mask_fn(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)
            
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
        batch_size, seq_len, _ = x.shape
        
        if torch.is_grad_enabled():
            qkv = self.qkv_proj(x)
            output_buffer = None
            attn_merge_buffer = None
        else:
            if self._qkv_weight_t is None or self._out_proj_weight_t is None:
                self.cache_weight_views()
            qkv_buffer, output_buffer, attn_merge_buffer = self._ensure_projection_buffers(
                x, batch_size, seq_len
            )
            qkv = torch.matmul(x, self._qkv_weight_t, out=qkv_buffer)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        if self._compiled_flex is None or not HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "FAIL FAST: optimized_flex_attention_sparse requires FlexAttention kernels; "
                "no SDPA fallback is allowed for this benchmark."
            )
        try:
            block_mask = self._get_block_mask(batch_size, seq_len, x.device)
            output = self._compiled_flex(q, k, v, block_mask=block_mask)
        except Exception as exc:
            raise RuntimeError(
                "FAIL FAST: optimized_flex_attention_sparse could not build or execute the "
                "FlexAttention block mask/kernels."
            ) from exc
        
        if attn_merge_buffer is None:
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        else:
            attn_merge_buffer.copy_(output.transpose(1, 2))
            output = attn_merge_buffer.view(batch_size, seq_len, self.embed_dim)
        if output_buffer is not None:
            return torch.matmul(output, self._out_proj_weight_t, out=output_buffer)
        return self.out_proj(output)


#============================================================================
# Benchmark
#============================================================================

def benchmark():
    """Benchmark FlexAttention sparse patterns."""
    print("FlexAttention Block Sparsity Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError(
            "FAIL FAST: FlexAttention requires PyTorch 2.5+; no SDPA fallback is allowed "
            "for optimized_flex_attention_sparse."
        )
    
    # Config
    batch_size, num_heads, head_dim, seq_len = 2, 32, 128, 4096
    dtype = torch.bfloat16
    
    print(f"\nConfig: B={batch_size}, H={num_heads}, D={head_dim}, S={seq_len}")
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Benchmark full attention via SDPA
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device)
    
    start.record(current_stream)
    for _ in range(10):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    end.record(current_stream)
    end.synchronize()
    
    sdpa_ms = start.elapsed_time(end) / 10
    
    print(f"\nResults:")
    print(f"  SDPA Causal: {sdpa_ms:.3f} ms")
    
    # FlexAttention if available
    if HAS_FLEX_ATTENTION:
        window_size = 512
        mask_fn = sliding_window_causal_mask(window_size)
        block_mask = create_block_mask(
            mask_fn, B=batch_size, H=num_heads,
            Q_LEN=seq_len, KV_LEN=seq_len, device=device
        )
        
        compiled_flex = torch.compile(flex_attention)
        
        for _ in range(3):
            _ = compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        
        start.record(current_stream)
        for _ in range(10):
            _ = compiled_flex(q, k, v, block_mask=block_mask)
        end.record(current_stream)
        end.synchronize()
        
        flex_ms = start.elapsed_time(end) / 10
        
        speedup = sdpa_ms / flex_ms
        sparsity = 1.0 - (window_size / seq_len)
        
        print(f"  FlexAttention (w={window_size}): {flex_ms:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Sparsity: {sparsity*100:.1f}%")
    
    print("\nNote: Sliding window reduces O(n²) to O(n·w)")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class FlexAttentionSparseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for FlexAttention sparse patterns."""

    def __init__(self):
        super().__init__()
        self.attn = None
        self.model = None
        self.x = None
        self.batch_size = 1
        self.num_heads = 16
        self.head_dim = 64
        self.hidden_dim = self.num_heads * self.head_dim
        self.seq_len = 4096
        self.window_size = 128
        self._last = 0.0
        self.parameter_count: int = 0
        self._verification_payload = None
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize sliding window causal attention."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for FlexAttention sparse benchmark")
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError("SKIPPED: FlexAttention requires PyTorch 2.5+ (torch.nn.attention.flex_attention)")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        embed_dim = self.num_heads * self.head_dim
        self.attn = SlidingWindowCausalAttention(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
        ).to(self.device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.attn.cache_weight_views()
        self.model = self.attn
        self.parameter_count = sum(p.numel() for p in self.attn.parameters())
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.x = torch.randn(
            self.batch_size, self.seq_len, embed_dim,
            device=self.device,
            dtype=dtype,
        )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, self.seq_len),
            min(256, self.hidden_dim),
            device=self.device,
            dtype=torch.float32,
        )

        self.attn._get_block_mask(self.batch_size, self.seq_len, self.device)
        
        # Warmup
        for _ in range(3):
            with torch.inference_mode():
                _ = self.attn(self.x)

    def benchmark_fn(self) -> None:
        """Benchmark: FlexAttention sliding window forward pass."""
        with nvtx_range("optimized_flex_attention_sparse", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.x)
        if self.output is None or self.x is None:
            raise RuntimeError("benchmark_fn() must produce output")
        self._payload_dtype = self.x.dtype

    def capture_verification_payload(self) -> None:
        if self.x is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("capture_verification_payload() requires completed run")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self.x},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.attn = None
        self.model = None
        self.x = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        total_flops = 2.0 * self.seq_len * self.seq_len * self.head_dim * self.num_heads
        total_bytes = float(self.seq_len * self.num_heads * self.head_dim * 3 * 2)
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="bf16",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Attention module not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return FlexAttentionSparseBenchmark()
