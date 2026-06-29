#!/usr/bin/env python3
"""FlexAttention with document-level attention masks.

Demonstrates document-aware attention where tokens attend only within
their document boundaries, useful for multi-document batching.
"""

import torch
import time
from typing import Dict, Any, Callable

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

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


class DocumentAttentionFlexAttention:
    """Document-level attention with FlexAttention."""
    
    def __init__(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        seq_length: int = 8192,
        head_dim: int = 128,
        num_documents: int = 8,
        use_compile: bool = True,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.num_documents = num_documents
        self.use_compile = use_compile
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.doc_length = seq_length // num_documents
        
        logger.info(f"Document Attention")
        logger.info(f"  Documents: {num_documents}, length: {self.doc_length}")
    
    def _create_document_mask(self) -> Callable:
        """Create document boundary mask."""
        doc_length = self.doc_length
        
        def document_pattern(b, h, q_idx, kv_idx):
            # Same document check
            q_doc = q_idx // doc_length
            kv_doc = kv_idx // doc_length
            return q_doc == kv_doc
        
        return document_pattern
    
    def setup(self):
        """Initialize FlexAttention."""
        self.q = torch.randn(
            self.batch_size, self.num_heads, self.seq_length, self.head_dim,
            device=self.device, dtype=torch.bfloat16
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        
        if FLEX_ATTENTION_AVAILABLE:
            mask_fn = self._create_document_mask()
            self.block_mask = create_block_mask(
                mask_fn,
                B=self.batch_size,
                H=self.num_heads,
                Q_LEN=self.seq_length,
                KV_LEN=self.seq_length,
                device=self.device
            )
            
            if self.use_compile:
                self.attention_fn = torch.compile(flex_attention)
            else:
                self.attention_fn = flex_attention
    
    def run(self) -> float:
        """Execute document attention."""
        if not FLEX_ATTENTION_AVAILABLE:
            return 0.0

        elapsed_ms, _ = _time_region_ms(
            self.device,
            lambda: self.attention_fn(self.q, self.k, self.v, block_mask=self.block_mask),
        )
        return elapsed_ms
    
    def cleanup(self):
        """Clean up."""
        del self.q, self.k, self.v
        if hasattr(self, 'block_mask'):
            del self.block_mask
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_length: int = 8192,
    num_documents: int = 8,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run document attention benchmark."""
    
    benchmark = DocumentAttentionFlexAttention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_length=seq_length,
        num_documents=num_documents,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(iterations=5, warmup=10, profile_mode=profile)
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    result = harness.benchmark(benchmark.run, name="flexattention_document")
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "num_documents": num_documents,
        "doc_length": seq_length // num_documents,
    }


if __name__ == "__main__":
    print(run_benchmark())
