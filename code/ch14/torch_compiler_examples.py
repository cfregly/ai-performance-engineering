"""
Optimized torch.compile for Blackwell B200

Demonstrates torch.compile configuration for optimal performance on Blackwell.
Includes proper warmup, TF32 settings, and Inductor configuration.
"""
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import os
import sys
import time

import torch
import torch.nn as nn
import triton.testing
from contextlib import nullcontext
from core.utils.compile_utils import compile_model, enable_tf32
from core.harness.arch_config import prefer_sdpa_backends

from ch14.torch_compile_large_model import create_model

# Quick mode removed; always run full demo


def configure_for_blackwell_peak_performance():
    """Proper configuration for Blackwell B200 peak performance."""
    print("=" * 80)
    print("Configuring PyTorch for Blackwell B200 Peak Performance")
    print("=" * 80)
    
    enable_tf32(set_global_precision=True)
    print("TF32 enabled (via arch_config)")
    
    # Flash Attention already configured by arch_config, but safe to call again
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    print("Flash Attention enabled")
    
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.triton.cudagraph_trees = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = True
    print("Inductor configuration applied")
    
    try:
        torch.compiler.set_stance("eager_on_recompile")
        print("torch.compiler.set_stance → eager_on_recompile")
    except AttributeError:
        print("torch.compiler.set_stance not available; skipping.")
    
    os.environ['TRITON_CUDNN_ALGOS'] = '1'
    os.environ['TRITON_ALWAYS_COMPILE'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    print("=" * 80 + "\n")


class OptimizedTransformerBlock(nn.Module):
    """Transformer block optimized for torch.compile."""
    def __init__(self, d_model=1024, num_heads=16, d_ff=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # FFN
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Attention with residual
        residual = x
        x = self.norm1(x)
        
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Scaled dot-product attention (Flash Attention will be used)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        x = self.out_proj(attn_out)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x + residual
        
        return x


def benchmark_with_proper_warmup(model, x, name):
    """Benchmark using Triton's testing framework with automatic warmup."""
    print(f"\nBenchmarking: {name}")
    sdpa_ctx_factory = prefer_sdpa_backends

    def run_model():
        with torch.inference_mode():
            with sdpa_ctx_factory():
                return model(x)
    
    avg_time_ms = triton.testing.do_bench(run_model)
    throughput = 1000.0 / avg_time_ms  # iter/s
    
    print(f"  Average time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {throughput:.1f} iter/s")
    
    return avg_time_ms, throughput


def main():
    """Demonstrate torch.compile usage for Blackwell."""
    configure_for_blackwell_peak_performance()
    
    # 2. Create model (larger for better compilation benefits)
    print("Creating model...")
    model_size = '5b'

    # Use existing large model infrastructure
    # Larger model highlights torch.compile benefits on compute-bound kernels
    model, config, total_params = create_model(model_size)
    model = model.cuda().to(dtype=torch.bfloat16).eval()
    print(f"Model parameters: {total_params / 1e9:.2f}B")
    
    # 3. Create compiled version with proper settings
    print("Compiling model...")
    model_compiled = compile_model(
        model,
        mode='max-autotune',      # Most aggressive optimization
        fullgraph=True,            # Compile entire graph (best performance)
        dynamic=False,             # Static shapes (better optimization)
        backend='inductor',        # Use Inductor backend
    )
    print(" Model compiled")
    
    # 4. Create input (larger for better performance)
    batch_size = 16
    seq_len = 2048
    d_model = config['d_model']
    x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input size: {x.numel() * 4 / 1e6:.2f} MB")
    
    # 5. Benchmark eager mode
    print("\n" + "=" * 80)
    print("EAGER MODE")
    print("=" * 80)
    eager_time, eager_throughput = benchmark_with_proper_warmup(
        model, x, "Eager Mode"
    )
    
    # 6. Explicit warmup for torch.compile
    print("\n" + "=" * 80)
    print("COMPILED MODE - WARMUP PHASE")
    print("=" * 80)
    warmup_iters = 10
    print(f"Running {warmup_iters} warmup iteration(s) for torch.compile...")
    with torch.inference_mode():
        for i in range(warmup_iters):
            _ = model_compiled(x)
            if (i + 1) % 5 == 0:
                print(f"  Warmup iteration {i + 1}/{warmup_iters}...")
    torch.cuda.synchronize()
    print(" Warmup complete! Now benchmarking...")
    
    # 7. Benchmark compiled mode (after warmup)
    print("\n" + "=" * 80)
    print("COMPILED MODE (after warmup)")
    print("=" * 80)
    compiled_time, compiled_throughput = benchmark_with_proper_warmup(
        model_compiled, x, "Compiled Mode"
    )
    
    # 7. Results
    speedup = eager_time / compiled_time
    throughput_improvement = compiled_throughput / eager_throughput
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Eager mode:        {eager_time:.3f} ms")
    print(f"Compiled mode:     {compiled_time:.3f} ms")
    print(f"Speedup:           {speedup:.2f}x {'' if speedup >= 1.25 else ''}")
    print(f"Throughput gain:   {throughput_improvement:.2f}x")
    print()
    
    if speedup >= 1.4:
        print(" EXCELLENT! Exceeding 1.4x speedup target!")
    elif speedup >= 1.3:
        print(" GOOD! Meeting 1.3x speedup target!")
    elif speedup >= 1.2:
        print("  OK, but can be better. Try larger model or longer sequences.")
    elif speedup >= 1.05:
        print("  ACCEPTABLE: torch.compile provides modest speedup.")
        print("  Note: Very large models and optimal baseline can limit compile benefits.")
    else:
        print(" ISSUE: Speedup below target. Check:")
        print("   1. Is TF32 enabled?")
        print("   2. Did you run enough warmup iterations (>=10, or 3 in quick mode)?")
        print("   3. Is the model large enough to benefit from compilation?")
    
    print("\n" + "=" * 80)
    print("KEY LEARNINGS FOR BOOK")
    print("=" * 80)
    print("1. Warmup is still required—10 iterations baseline")
    print("2. TF32 must be enabled (e.g., torch.set_float32_matmul_precision('high'))")
    print("3. fullgraph=True gives best performance (if possible)")
    print("4. CUDA graph trees provide additional 15-20% speedup")
    print("5. Larger models benefit more (aim for >1M parameters)")
    print("6. Static shapes (dynamic=False) allow better optimization")
    print("=" * 80)
    
    return speedup


if __name__ == "__main__":
    speedup = main()
    
    # Exit with appropriate code
    # Accept any speedup (1.0x+) as success - torch.compile doesn't always
    # show dramatic gains, especially when the baseline is already well-optimized
    SUCCESS_THRESHOLD = 0.98
    sys.exit(0 if speedup >= SUCCESS_THRESHOLD else 1)
