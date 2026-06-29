#!/usr/bin/env python3
"""FlexAttention with sliding window attention.

Demonstrates sliding window attention using FlexAttention for
memory-efficient long-context processing.
"""

import math
import time
from typing import Any, Callable, Dict

import torch

from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from core.utils.logger import get_logger

logger = get_logger(__name__)


def _time_region_ms(
    device: torch.device,
    fn: Callable[[], torch.Tensor],
) -> tuple[float, torch.Tensor]:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(device)
        start.record(current_stream)
        output = fn()
        end.record(current_stream)
        end.synchronize()
        return start.elapsed_time(end), output

    start = time.perf_counter()
    output = fn()
    return (time.perf_counter() - start) * 1000.0, output

# Check for FlexAttention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    logger.warning("FlexAttention not available, using fallback")


class SlidingWindowFlexAttention:
    """Sliding window attention with FlexAttention."""
    
    def __init__(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        seq_length: int = 16384,
        head_dim: int = 128,
        window_size: int = 1024,
        use_compile: bool = True,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.window_size = window_size
        self.use_compile = use_compile
        self._fallback_positions = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Sliding Window FlexAttention")
        logger.info(f"  Sequence: {seq_length}, Window: {window_size}")
        logger.info(f"  Memory savings: ~{self._calculate_memory_savings():.1f}×")
    
    def _calculate_memory_savings(self) -> float:
        """Calculate memory savings vs full attention."""
        # Full attention: seq_len × seq_len
        full_memory = self.seq_length * self.seq_length
        
        # Sliding window: seq_len × window_size
        window_memory = self.seq_length * self.window_size
        
        savings = full_memory / window_memory
        return savings
    
    def _create_sliding_window_mask(self) -> Callable:
        """Create sliding window mask function.
        
        Each position attends to:
        - window_size//2 tokens before
        - window_size//2 tokens after
        """
        half_window = self.window_size // 2
        
        def sliding_window_pattern(b, h, q_idx, kv_idx):
            # Distance between query and key positions
            distance = torch.abs(q_idx - kv_idx)
            
            # Within window
            return distance <= half_window
        
        return sliding_window_pattern
    
    def setup(self):
        """Initialize FlexAttention with sliding window."""
        # Create Q, K, V
        self.q = torch.randn(
            self.batch_size,
            self.num_heads,
            self.seq_length,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        
        if FLEX_ATTENTION_AVAILABLE:
            # Create sliding window mask
            mask_fn = self._create_sliding_window_mask()
            self.block_mask = create_block_mask(
                mask_fn,
                B=self.batch_size,
                H=self.num_heads,
                Q_LEN=self.seq_length,
                KV_LEN=self.seq_length,
                device=self.device
            )
            
            # Compile if requested
            if self.use_compile:
                logger.info("Compiling FlexAttention...")
                self.attention_fn = torch.compile(flex_attention)
            else:
                self.attention_fn = flex_attention
            
            logger.info("FlexAttention setup complete")
        else:
            self._fallback_positions = torch.arange(self.seq_length, device=self.device)
            logger.info("Using fallback (FlexAttention not available)")
    
    def run(self) -> float:
        """Execute sliding window attention."""
        if not FLEX_ATTENTION_AVAILABLE:
            # Fallback
            def _run_fallback() -> torch.Tensor:
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(self.q, self.k.transpose(-2, -1)) * scale

                # Apply window mask manually
                half_window = self.window_size // 2
                pos = self._fallback_positions
                if pos is None:
                    raise RuntimeError("setup() must run before fallback attention")
                q_pos = pos.unsqueeze(1)
                kv_pos = pos.unsqueeze(0)
                mask = (kv_pos < q_pos - half_window) | (kv_pos > q_pos + half_window)

                scores.masked_fill_(mask, float('-inf'))
                attn = torch.softmax(scores, dim=-1)
                return torch.matmul(attn, self.v)

            elapsed_ms, _ = _time_region_ms(self.device, _run_fallback)
            return elapsed_ms

        def _run_flex_attention() -> torch.Tensor:
            return self.attention_fn(
                self.q,
                self.k,
                self.v,
                block_mask=self.block_mask
            )

        elapsed_ms, _ = _time_region_ms(self.device, _run_flex_attention)
        
        logger.info(f"Sliding window attention: {elapsed_ms:.2f} ms")
        
        return elapsed_ms
    
    def cleanup(self):
        """Clean up resources."""
        del self.q, self.k, self.v
        if hasattr(self, 'block_mask'):
            del self.block_mask
        self._fallback_positions = None
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_length: int = 16384,
    head_dim: int = 128,
    window_size: int = 1024,
    use_compile: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run sliding window FlexAttention benchmark."""
    
    benchmark = SlidingWindowFlexAttention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_length=seq_length,
        head_dim=head_dim,
        window_size=window_size,
        use_compile=use_compile,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=5,
        warmup=10,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="flexattention_sliding_window"
    )
    
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "seq_length": seq_length,
        "window_size": window_size,
        "memory_savings": benchmark._calculate_memory_savings(),
    }


if __name__ == "__main__":
    print(run_benchmark())
