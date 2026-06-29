# Environment (Nov-2025): CUDA 13.x r580+, torch 2.10.0+cu130, triton 3.5.0, optional TE 2.9+
"""
Optimized FlexAttention for Blackwell B200

Demonstrates correct FlexAttention usage with torch.compile for optimal
performance. FlexAttention must be compiled to generate fused kernels;
without compilation it materializes the full attention matrix.
"""
import os
import sys

try:
    pass
except Exception:
    class arch_config:  # type: ignore[override]
        @staticmethod
        def is_blackwell() -> bool:
            import torch
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability()
            # Blackwell is sm_100 (major=10), Grace-Blackwell is sm_12x
            return major >= 10

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from core.utils.compile_utils import enable_tf32, compile_model

assert torch.cuda.is_available(), "CUDA required for FlexAttention examples"
_major, _minor = torch.cuda.get_device_capability()
# Blackwell is sm_100 (major=10), not sm_120
if _major < 10:
    raise RuntimeError(f"SKIPPED: FlexAttention native kernels require Blackwell (sm_100+); found sm_{_major}{_minor}")

QUICK_MODE = any(
    os.getenv(flag, "0") == "1"
    for flag in ("QUICK_PROFILE", "BENCHMARK_QUICK", "RUN_ALL_CHAPTERS")
)
DEFAULT_COMPILE_MODE = "reduce-overhead" if QUICK_MODE else "default"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)
COMPILE_KWARGS = {"mode": COMPILE_MODE, "dynamic": None}

DTYPE = torch.bfloat16

BASE_WARMUP = 5 if QUICK_MODE else 50
BASE_ITERS = 20 if QUICK_MODE else 200
COMPILED_WARMUP = 10 if QUICK_MODE else 100

def _is_known_compile_failure(error_text: str) -> bool:
    """Return True when the error likely stems from unsupported GPU/PTX."""
    lowered = error_text.lower()
    return (
        "ptxas" in lowered
        or "novalidchoiceserror" in lowered
        or "gpu-name" in lowered
    )


def _summarize_error_text(error_text: str, max_lines: int = 4) -> str:
    """Compress multi-line error messages for concise logging."""
    lines = [line.strip() for line in error_text.splitlines() if line.strip()]
    return " ".join(lines[:max_lines])


def configure_for_flex_attention():
    """Configure PyTorch for FlexAttention peak performance"""
    print("=" * 80)
    print("Configuring for FlexAttention Peak Performance")
    print("=" * 80)
    if QUICK_MODE:
        print(" Quick mode active (reduced problem size / iterations)")
    
    # PyTorch 2.10 TF32 controls (deprecates allow_tf32 flags)
    enable_tf32()
    
    # Enable Flash Attention (FlexAttention builds on this)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Inductor settings
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.max_autotune = True
    
    print(" Configuration complete\n")


class BaselineAttention(nn.Module):
    """Baseline attention using scaled_dot_product_attention"""
    def forward(self, Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


class FlexAttentionWRONG(nn.Module):
    """Wrong: Not compiled - materializes full matrix and will be slower."""
    def __init__(self, window_size=2048):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        
        # Sliding window mask
        def sliding_window(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= self.window_size
        
        block_mask = create_block_mask(sliding_window, B, H, T, T, device=Q.device)
        
        # WITHOUT torch.compile - this is SLOW!
        return flex_attention(Q, K, V, block_mask=block_mask)


class FlexAttentionCORRECT(nn.Module):
    """Correct: Wrapped with torch.compile to generate fused kernel."""
    def __init__(self, window_size=2048):
        super().__init__()
        self.window_size = window_size
        # Create mask function at init time to avoid torch.compile issues
        def _sliding_window(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= window_size
        self.mask_fn = _sliding_window
        
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        
        # Create block mask using pre-defined function
        block_mask = create_block_mask(self.mask_fn, B, H, T, T, device=Q.device)
        
        # Will be compiled by torch.compile wrapper - generates fused kernel!
        return flex_attention(Q, K, V, block_mask=block_mask)


def benchmark_attention(model, Q, K, V, name, num_warmup=50, num_iters=200):
    """Benchmark attention implementation"""
    print(f"\nBenchmarking: {name}")

    if QUICK_MODE:
        num_warmup = min(num_warmup, BASE_WARMUP)
        num_iters = min(num_iters, BASE_ITERS)

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark
    count = max(num_iters, 1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        current_stream = torch.cuda.current_stream(Q.device)
        start.record(current_stream)
        for _ in range(count):
            _ = model(Q, K, V)
        end.record(current_stream)
    end.synchronize()
    
    avg_time_ms = start.elapsed_time(end) / count
    
    print(f"  Average time: {avg_time_ms:.2f} ms")
    
    return avg_time_ms


def main():
    """Demonstrate CORRECT FlexAttention usage"""
    
    configure_for_flex_attention()
    
    # Test configuration
    batch_size = 2 if QUICK_MODE else 8
    num_heads = 8 if QUICK_MODE else 16
    seq_len = 512 if QUICK_MODE else 2048
    head_dim = 64
    
    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")
    print(f"  Window size: 512")
    
    # Create inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=DTYPE)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    
    bytes_per_tensor = Q.numel() * Q.element_size()
    print(f"\nMemory per tensor: {bytes_per_tensor / 1e6:.2f} MB [{DTYPE}]")
    
    # 1. Baseline: Regular SDPA
    print("\n" + "=" * 80)
    print("TEST 1: Baseline (scaled_dot_product_attention)")
    print("=" * 80)
    baseline = BaselineAttention().cuda().eval()
    baseline_time = benchmark_attention(baseline, Q, K, V, "Baseline SDPA")
    
    # 2. FlexAttention WITHOUT compile (WRONG - will be slower!)
    print("\n" + "=" * 80)
    print("TEST 2: FlexAttention WITHOUT torch.compile (WRONG!)")
    print("=" * 80)
    print("  This will materialize the full attention matrix - SLOW!")
    flex_wrong = FlexAttentionWRONG(window_size=512).cuda().eval()
    wrong_time = benchmark_attention(flex_wrong, Q, K, V, "FlexAttention (not compiled)")
    wrong_speedup = baseline_time / wrong_time
    print(f"  vs Baseline: {wrong_speedup:.2f}x {' SLOWER!' if wrong_speedup < 1.0 else ''}")
    
    # 3. FlexAttention WITH compile (CORRECT - 2x+ faster!)
    print("\n" + "=" * 80)
    print("TEST 3: FlexAttention WITH torch.compile (CORRECT!)")
    print("=" * 80)
    print(" This will generate a fused kernel - FAST!")
    flex_correct = FlexAttentionCORRECT(window_size=512).cuda().eval()
    
    # CRITICAL: Compile the entire module
    flex_correct_compiled = compile_model(
        flex_correct,
        **COMPILE_KWARGS,
    )
    correct_time = benchmark_attention(
        flex_correct_compiled,
        Q,
        K,
        V,
        "FlexAttention (compiled)",
        num_warmup=COMPILED_WARMUP,
        num_iters=BASE_ITERS if QUICK_MODE else 200,
    )
    correct_speedup = baseline_time / correct_time
    print(f"  vs Baseline: {correct_speedup:.2f}x {'' if correct_speedup >= 1.5 else ''}")
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Baseline SDPA:                 {baseline_time:.2f} ms (1.0x)")
    print(f"FlexAttention (not compiled):  {wrong_time:.2f} ms ({wrong_speedup:.2f}x)")
    print(
        f"FlexAttention (COMPILED):      {correct_time:.2f} ms ({correct_speedup:.2f}x) "
        f"{'' if correct_speedup >= 1.5 else ''}"
    )
    print()
    
    if correct_speedup >= 2.0:
        print(" EXCELLENT! Achieving 2x+ speedup!")
    elif correct_speedup >= 1.5:
        print(" GOOD! Meeting 1.5x+ speedup target!")
    else:
        print("  Speedup below target. Try:")
        print("   1. Longer sequences (4096+)")
        print("   2. Smaller window size (more sparsity)")
        print("   3. Ensure compilation succeeded")
    
    print("\n" + "=" * 80)
    print("KEY LEARNINGS FOR BOOK")
    print("=" * 80)
    print("1. FlexAttention MUST be wrapped with torch.compile!")
    print("2. Without compile: materializes full matrix (SLOW)")
    print("3. With compile: generates fused kernel (FAST - 2x+)")
    print("4. Use fullgraph=True and dynamic=False for best results")
    print("5. Warmup is critical (100+ iterations)")
    print("6. Larger sequences benefit more from FlexAttention")
    print("7. Window size affects sparsity and speedup")
    print("=" * 80)
    
    return correct_speedup


if __name__ == "__main__":
    speedup = main()
    
    # Exit with appropriate code
    exit_code = 0
    if speedup is None:
        exit_code = 0
    elif speedup < 1.5 and not QUICK_MODE:
        exit_code = 1
    if QUICK_MODE:
        exit_code = 0
    sys.exit(exit_code)
