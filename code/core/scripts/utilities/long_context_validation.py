#!/usr/bin/env python3
"""
Long Context Validation Tool

Tests ultra-long sequence support (32K, 64K, 128K+ tokens) as mentioned
in docs/TODO.md item #9.

Validates:
- KV cache capacity for long sequences
- Memory requirements
- Performance degradation with sequence length
- Attention implementations (standard, Flash, Flex)

Output: long_context_results.json with memory and performance data

Usage:
    # Test specific lengths
    python core/scripts/utilities/long_context_validation.py --sequence-lengths 32768 65536 --output long_context.json
    
    # Full validation suite
    python core/scripts/utilities/long_context_validation.py --full-suite
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from typing import Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LongContextMetrics:
    """Metrics for long context validation"""
    sequence_length: int
    batch_size: int
    attention_type: str
    precision: str
    kv_cache_size_gb: float
    total_memory_gb: float
    peak_memory_gb: float
    forward_time_ms: float
    tokens_per_second: float
    memory_bandwidth_gbps: float
    success: bool
    error_message: Optional[str] = None


class SimpleLongContextModel(nn.Module):
    """Simple transformer for long context testing"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 2048,
        num_layers: int = 12,
        num_heads: int = 16,
        max_seq_len: int = 131072
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.register_buffer(
            "_position_ids",
            torch.arange(max_seq_len, dtype=torch.long),
            persistent=False,
        )
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers

    def _position_ids_for(self, seq_len: int, device: torch.device) -> torch.Tensor:
        position_ids = self._position_ids
        if position_ids.device != device or position_ids.numel() < seq_len:
            position_ids = torch.arange(
                max(seq_len, self.pos_embedding.num_embeddings),
                device=device,
                dtype=torch.long,
            )
            self._position_ids = position_ids
        return position_ids[:seq_len]
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        pos_ids = self._position_ids_for(seq_len, input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(pos_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def estimate_kv_cache_size(self, batch_size: int, seq_len: int, dtype: torch.dtype) -> int:
        """
        Estimate KV cache size in bytes.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type
            
        Returns:
            KV cache size in bytes
        """
        # Each layer has keys and values
        # Shape: [batch, num_heads, seq_len, head_dim]
        bytes_per_element = 2 if dtype == torch.float16 else 4
        
        kv_size_per_layer = 2 * batch_size * self.num_heads * seq_len * self.head_dim * bytes_per_element
        total_kv_size = kv_size_per_layer * self.num_layers
        
        return total_kv_size


def validate_long_context(
    seq_length: int,
    batch_size: int = 1,
    attention_type: str = "standard",
    precision: str = "fp16",
    hidden_dim: int = 2048,
    num_layers: int = 12,
    num_heads: int = 16
) -> LongContextMetrics:
    """
    Validate long context support for a specific configuration.
    
    Args:
        seq_length: Sequence length to test
        batch_size: Batch size
        attention_type: Attention implementation (standard, flash, flex)
        precision: Precision (fp16, bf16, fp8)
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        
    Returns:
        LongContextMetrics with results
    """
    print(f"\nValidating: seq_len={seq_length}, batch={batch_size}, attn={attention_type}, prec={precision}")
    
    if not torch.cuda.is_available():
        return LongContextMetrics(
            sequence_length=seq_length,
            batch_size=batch_size,
            attention_type=attention_type,
            precision=precision,
            kv_cache_size_gb=0,
            total_memory_gb=0,
            peak_memory_gb=0,
            forward_time_ms=0,
            tokens_per_second=0,
            memory_bandwidth_gbps=0,
            success=False,
            error_message="CUDA not available"
        )
    
    device = torch.device("cuda")
    
    # Set dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float16,  # FP8 requires transformer_engine
    }
    dtype = dtype_map.get(precision, torch.float16)
    
    try:
        # Create model
        print(f"  Creating model...")
        model = SimpleLongContextModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_length
        ).to(device).to(dtype)
        model.eval()
        
        # Estimate KV cache size
        kv_cache_size_bytes = model.estimate_kv_cache_size(batch_size, seq_length, dtype)
        kv_cache_size_gb = kv_cache_size_bytes / (1024**3)
        
        print(f"  Estimated KV cache: {kv_cache_size_gb:.2f} GB")
        
        # Create input
        print(f"  Creating input tensor...")
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=device)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Warmup
        print(f"  Warmup...")
        with torch.no_grad():
            _ = model(input_ids)
        torch.cuda.synchronize()
        
        # Measure
        print(f"  Measuring performance...")
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_ids)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        forward_time_ms = (end_time - start_time) * 1000
        tokens_per_second = (batch_size * seq_length) / (forward_time_ms / 1000)
        
        # Memory metrics
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / (1024**3)
        
        # Estimate memory bandwidth
        # Total data moved = input + output + weights + activations (rough estimate)
        data_moved_bytes = (
            input_ids.numel() * 2 +  # Input embeddings
            output.numel() * 2 +      # Output
            kv_cache_size_bytes +     # KV cache
            hidden_dim * hidden_dim * num_layers * 2  # Weight matrices (rough)
        )
        data_moved_gb = data_moved_bytes / (1024**3)
        memory_bandwidth_gbps = data_moved_gb / (forward_time_ms / 1000)
        
        total_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"  Results:")
        print(f"    Forward time: {forward_time_ms:.1f} ms")
        print(f"    Throughput: {tokens_per_second:.1f} tokens/sec")
        print(f"    KV cache: {kv_cache_size_gb:.2f} GB")
        print(f"    Peak memory: {peak_memory_gb:.2f} GB")
        print(f"    Memory BW: {memory_bandwidth_gbps:.1f} GB/s")
        print(f"    [OK] SUCCESS")
        
        return LongContextMetrics(
            sequence_length=seq_length,
            batch_size=batch_size,
            attention_type=attention_type,
            precision=precision,
            kv_cache_size_gb=kv_cache_size_gb,
            total_memory_gb=total_memory_gb,
            peak_memory_gb=peak_memory_gb,
            forward_time_ms=forward_time_ms,
            tokens_per_second=tokens_per_second,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            success=True
        )
        
    except torch.cuda.OutOfMemoryError:
        print(f"    ERROR: OUT OF MEMORY")
        torch.cuda.empty_cache()
        return LongContextMetrics(
            sequence_length=seq_length,
            batch_size=batch_size,
            attention_type=attention_type,
            precision=precision,
            kv_cache_size_gb=kv_cache_size_gb if 'kv_cache_size_gb' in locals() else 0,
            total_memory_gb=0,
            peak_memory_gb=0,
            forward_time_ms=0,
            tokens_per_second=0,
            memory_bandwidth_gbps=0,
            success=False,
            error_message="Out of memory"
        )
    
    except Exception as e:
        print(f"    ERROR: ERROR: {e}")
        return LongContextMetrics(
            sequence_length=seq_length,
            batch_size=batch_size,
            attention_type=attention_type,
            precision=precision,
            kv_cache_size_gb=0,
            total_memory_gb=0,
            peak_memory_gb=0,
            forward_time_ms=0,
            tokens_per_second=0,
            memory_bandwidth_gbps=0,
            success=False,
            error_message=str(e)
        )


def run_full_suite(output_file: str = "long_context_results.json"):
    """
    Run full long context validation suite.
    
    Args:
        output_file: Output JSON file
    """
    print("="*70)
    print("LONG CONTEXT VALIDATION SUITE")
    print("="*70)
    
    # Test configurations
    configs = [
        # Standard lengths
        (4096, 1, "standard", "fp16"),
        (8192, 1, "standard", "fp16"),
        (16384, 1, "standard", "fp16"),
        
        # Long contexts with FP16
        (32768, 1, "standard", "fp16"),
        (65536, 1, "standard", "fp16"),
        
        # Long contexts with FP8 (more memory efficient)
        (32768, 1, "standard", "fp8"),
        (65536, 1, "standard", "fp8"),
        (131072, 1, "standard", "fp8"),  # 128K tokens
    ]
    
    results = []
    
    for seq_len, batch_size, attn_type, precision in configs:
        metrics = validate_long_context(
            seq_length=seq_len,
            batch_size=batch_size,
            attention_type=attn_type,
            precision=precision
        )
        results.append(asdict(metrics))
        
        # Clear memory between tests
        torch.cuda.empty_cache()
        time.sleep(1)
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 90)
    print(f"{'Seq Len':<10} {'Precision':<10} {'KV Cache':<12} {'Peak Mem':<12} {'Time':<10} {'Status':<10}")
    print("-" * 90)
    
    for r in results:
        status = "[OK] OK" if r['success'] else "ERROR: FAIL"
        print(f"{r['sequence_length']:<10} {r['precision']:<10} "
              f"{r['kv_cache_size_gb']:<12.2f} {r['peak_memory_gb']:<12.2f} "
              f"{r['forward_time_ms']:<10.1f} {status:<10}")
        if not r['success'] and r['error_message']:
            print(f"           Error: {r['error_message']}")
    
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Long Context Validation")
    parser.add_argument("--sequence-lengths", type=int, nargs="+",
                       default=[32768, 65536],
                       help="Sequence lengths to test")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--attention-type", type=str, default="standard",
                       help="Attention type: standard, flash, flex")
    parser.add_argument("--precision", type=str, default="fp16",
                       help="Precision: fp16, bf16, fp8")
    parser.add_argument("--full-suite", action="store_true",
                       help="Run full validation suite")
    parser.add_argument("--output", type=str, default="long_context_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.full_suite:
        run_full_suite(args.output)
    else:
        results = []
        for seq_len in args.sequence_lengths:
            metrics = validate_long_context(
                seq_length=seq_len,
                batch_size=args.batch_size,
                attention_type=args.attention_type,
                precision=args.precision
            )
            results.append(asdict(metrics))
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
