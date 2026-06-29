"""
Enhanced FlexAttention with Custom Block Masks and Dynamic Shapes

Demonstrates advanced FlexAttention patterns for Blackwell B200/GB10:
- Sliding window + causal masking
- Variable-length sequences with dynamic shapes
- Custom attention patterns (local + sparse global)
- Proper torch.compile integration for 2-3x speedup

Architecture-aware optimizations:
- B200: Optimized for HBM3e bandwidth patterns
- GB10: Leverages NVLink-C2C for large sequence handling
"""
import os

import copy

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Tuple

from core.utils.compile_utils import enable_tf32, compile_model

QUICK_MODE = any(
    os.getenv(flag, "0") == "1"
    for flag in ("QUICK_PROFILE", "BENCHMARK_QUICK", "RUN_ALL_CHAPTERS")
)
DEFAULT_COMPILE_MODE = "reduce-overhead" if QUICK_MODE else "default"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)
COMPILE_KWARGS = {"mode": COMPILE_MODE, "dynamic": None}
DTYPE = torch.bfloat16
BASE_WARMUP = 5 if QUICK_MODE else 50
BASE_ITERS = 10 if QUICK_MODE else 200
COMPILED_WARMUP = 10 if QUICK_MODE else 100


def _is_known_compile_failure(error_text: str) -> bool:
    """Return True when the error likely stems from unsupported GPU/PTX."""
    lowered = error_text.lower()
    return (
        "ptxas" in lowered
        or "novalidchoiceserror" in lowered
        or "gpu-name" in lowered
        or "internaltorchdynamoerror" in lowered
    )


def _summarize_error_text(error_text: str, max_lines: int = 4) -> str:
    """Compress multi-line error messages for concise logging."""
    lines = [line.strip() for line in error_text.splitlines() if line.strip()]
    return " ".join(lines[:max_lines])


def configure_for_enhanced_flex_attention():
    """Configure PyTorch for Enhanced FlexAttention"""
    print("=" * 80)
    print("Enhanced FlexAttention Configuration")
    print("=" * 80)
    if QUICK_MODE:
        print(" Quick mode active (reduced sizes / iterations)")
    
    enable_tf32()
    
    # Enable all attention backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Disable fallback
    
    # Inductor settings for best compilation
    if hasattr(torch, "_inductor"):
        cfg = torch._inductor.config
        cfg.triton.cudagraphs = True
        cfg.max_autotune = True
        cfg.max_autotune_gemm_backends = "CUTLASS,TRITON,ATEN"
        if hasattr(cfg, "aggressive_fusion"):
            cfg.aggressive_fusion = True
    
    print("Configuration complete\n")


class SlidingWindowCausalAttention(nn.Module):
    """
    Sliding window + causal masking for efficient long-context attention
    
    Benefits:
    - Reduces O(n²) to O(n*w) where w is window size
    - Causal constraint for autoregressive models
    - 2-3x faster than full attention for long sequences
    """
    def __init__(self, window_size=2048):
        super().__init__()
        self.window_size = window_size
        
        # Pre-define mask function to avoid torch.compile issues
        def _mask_fn(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            window = (q_idx - kv_idx).abs() <= window_size
            return causal & window
        
        self.mask_fn = _mask_fn
    
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        block_mask = create_block_mask(self.mask_fn, B, H, T, T, device=Q.device)
        return flex_attention(Q, K, V, block_mask=block_mask)


class LocalGlobalAttention(nn.Module):
    """
    Hybrid local + sparse global attention
    
    Pattern:
    - Attend to local window (e.g., 512 tokens)
    - Plus sparse global tokens (every 64th token)
    
    Use case: Long documents where context is mostly local
    """
    def __init__(self, local_window=512, global_stride=64):
        super().__init__()
        self.local_window = local_window
        self.global_stride = global_stride
        
        def _mask_fn(b, h, q_idx, kv_idx):
            # Local window
            local = (q_idx - kv_idx).abs() <= local_window
            # Sparse global (every Nth token)
            global_token = (kv_idx % global_stride) == 0
            # Causal
            causal = q_idx >= kv_idx
            return causal & (local | global_token)
        
        self.mask_fn = _mask_fn
    
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        block_mask = create_block_mask(self.mask_fn, B, H, T, T, device=Q.device)
        return flex_attention(Q, K, V, block_mask=block_mask)


class DynamicSlidingWindowAttention(nn.Module):
    """
    Variable window size per head (for multi-head attention diversity)
    
    Different heads attend to different context windows:
    - Head 0-3: Small window (256)
    - Head 4-7: Medium window (512)
    - Head 8-11: Large window (1024)
    - Head 12-15: Full context
    """
    def __init__(self, num_heads=16, base_window=256):
        super().__init__()
        self.num_heads = num_heads
        self.base_window = base_window
        
        # Compute window size per head
        self.window_sizes = []
        for h in range(num_heads):
            if h < num_heads // 4:
                w = base_window
            elif h < num_heads // 2:
                w = base_window * 2
            elif h < 3 * num_heads // 4:
                w = base_window * 4
            else:
                w = 999999  # Full attention
            self.window_sizes.append(w)

        self.register_buffer(
            "window_sizes_tensor",
            torch.tensor(self.window_sizes, dtype=torch.int32),
            persistent=False,
        )

        def _mask_fn(b, h, q_idx, kv_idx):
            # Different window per head
            window_sizes = self.window_sizes_tensor
            if window_sizes.device != q_idx.device:
                window_sizes = window_sizes.to(q_idx.device)
            if isinstance(h, torch.Tensor):
                idx = h.to(torch.long)
                window_size = torch.take(window_sizes, idx)
            else:
                window_size = window_sizes[int(h)]
            window = (q_idx - kv_idx).abs() <= window_size
            causal = q_idx >= kv_idx
            return causal & window

        self.mask_fn = _mask_fn
    
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        block_mask = create_block_mask(self.mask_fn, B, H, T, T, device=Q.device)
        return flex_attention(Q, K, V, block_mask=block_mask)


def compile_module(module: nn.Module, label: str) -> Tuple[nn.Module, bool]:
    """Compile module with torch.compile and propagate failures."""
    try:
        module_to_compile = copy.deepcopy(module)
    except Exception:
        module_to_compile = module
    compiled = compile_model(
        module_to_compile,
        **COMPILE_KWARGS,
    )
    return compiled, True


def benchmark_attention(
    model,
    Q,
    K,
    V,
    name,
    *,
    num_warmup: int = 50,
    num_iters: int = 200,
):
    """Benchmark attention implementation. Returns (time_ms, used_compiled)."""
    print(f"\nBenchmarking: {name}")
    if QUICK_MODE:
        num_warmup = min(num_warmup, BASE_WARMUP)
        num_iters = min(num_iters, BASE_ITERS)

    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(Q, K, V)
    torch.cuda.synchronize()

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

    used_compiled = True
    if isinstance(model, torch.nn.Module) and getattr(model, "_orig_mod", None) is None:
        used_compiled = False

    return avg_time_ms, used_compiled


def detect_architecture():
    """Detect GPU architecture"""
    if not torch.cuda.is_available():
        return "cpu", 0, 0
    
    props = torch.cuda.get_device_properties(0)
    
    if props.major == 12:
        return "gb10", props.major, props.minor
    elif props.major == 10 and props.minor == 0:
        return "b200", props.major, props.minor
    else:
        return "other", props.major, props.minor


def main():
    """Demonstrate Enhanced FlexAttention patterns"""
    
    configure_for_enhanced_flex_attention()
    
    arch_type, major, minor = detect_architecture()
    print(f"Architecture: ", end="")
    if arch_type == "gb10":
        print(f"Grace-Blackwell GB10 (SM {major}.{minor})")
        print("Optimizations: NVLink-C2C coherent memory for large sequences")
    elif arch_type == "b200":
        print(f"Blackwell B200 (SM {major}.{minor})")
        print("Optimizations: HBM3e bandwidth patterns")
    else:
        print(f"Generic GPU (SM {major}.{minor})")
    
    # Test configuration
    batch_size = 2 if QUICK_MODE else 4
    num_heads = 8 if QUICK_MODE else 16
    seq_len = 1024 if QUICK_MODE else 4096  # Longer sequence to show benefits
    head_dim = 64
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")
    
    # Create inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=DTYPE)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    bytes_per_tensor = Q.numel() * Q.element_size()
    
    print(f"  Memory per tensor: {bytes_per_tensor / 1e6:.2f} MB [{DTYPE}]")
    print(f"  Total memory: {3 * bytes_per_tensor / 1e6:.2f} MB")
    
    # Baseline: Regular SDPA (full attention)
    print("\n" + "=" * 80)
    print("Baseline: Full Attention (scaled_dot_product_attention)")
    print("=" * 80)
    baseline_time = None
    try:
        baseline_fn = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        baseline_time, _ = benchmark_attention(
            lambda Q, K, V: baseline_fn(Q, K, V),
            Q,
            K,
            V,
            "Full Attention",
        )
    except Exception as e:
        print(f"Baseline failed: {e}")
        baseline_time = None
    
    results: dict[str, dict[str, object]] = {}
    
    # Test 1: Sliding Window + Causal
    print("\n" + "=" * 80)
    print("Test 1: Sliding Window + Causal (window=1024)")
    print("=" * 80)
    
    model1 = SlidingWindowCausalAttention(window_size=1024).cuda().eval()
    model1_compiled, model1_compiled_ok = compile_module(model1, "Sliding Window + Causal")
    time1, used_compiled1 = benchmark_attention(
        model1_compiled,
        Q,
        K,
        V,
        "Sliding Window + Causal",
        num_warmup=COMPILED_WARMUP,
    )
    results["sliding_window"] = {"time": time1, "compiled": model1_compiled_ok and used_compiled1}
    if baseline_time:
        print(f"  Speedup vs baseline: {baseline_time / time1:.2f}x")
    
    # Test 2: Local + Global Sparse
    print("\n" + "=" * 80)
    print("Test 2: Local + Global Sparse (local=512, global_stride=64)")
    print("=" * 80)
    
    model2 = LocalGlobalAttention(local_window=512, global_stride=64).cuda().eval()
    model2_compiled, model2_compiled_ok = compile_module(model2, "Local + Global Sparse")
    time2, used_compiled2 = benchmark_attention(
        model2_compiled,
        Q,
        K,
        V,
        "Local + Global Sparse",
        num_warmup=COMPILED_WARMUP,
    )
    results["local_global"] = {"time": time2, "compiled": model2_compiled_ok and used_compiled2}
    if baseline_time:
        print(f"  Speedup vs baseline: {baseline_time / time2:.2f}x")
    
    # Test 3: Dynamic per-head windows
    print("\n" + "=" * 80)
    print("Test 3: Dynamic Per-Head Windows (256/512/1024/full)")
    print("=" * 80)
    
    if QUICK_MODE:
        print("  Skipping dynamic windows benchmark in quick mode.")
        results["dynamic_windows"] = {"time": None, "compiled": False}
    else:
        model3 = DynamicSlidingWindowAttention(num_heads=num_heads, base_window=256).cuda().eval()
        candidate_module, candidate_ok = compile_module(copy.deepcopy(model3).cuda().eval(), "Dynamic Windows")
        model3_compiled = candidate_module
        model3_compiled_ok = candidate_ok
        if candidate_ok:
            with torch.inference_mode():
                candidate_module(Q[:, :, :1], K[:, :, :1], V[:, :, :1])
        else:
            raise RuntimeError("SKIPPED: Dynamic windows compilation unavailable.")

        time3, used_compiled3 = benchmark_attention(
            model3_compiled,
            Q,
            K,
            V,
            "Dynamic Windows",
            num_warmup=COMPILED_WARMUP,
        )
        results["dynamic_windows"] = {"time": time3, "compiled": model3_compiled_ok and used_compiled3}
        if baseline_time:
            print(f"  Speedup vs baseline: {baseline_time / time3:.2f}x")
    
    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if baseline_time:
        print(f"Baseline (Full Attention):     {baseline_time:.2f} ms (1.00x)")
    else:
        print(f"Baseline (Full Attention):     N/A")
    
    for name, entry in results.items():
        time_ms = entry.get("time")
        compiled_ok = entry.get("compiled", False)
        note = "" if compiled_ok else " (eager)"
        if time_ms is None:
            print(f"{name:30s}: unavailable{note}")
        else:
            speedup = baseline_time / time_ms if baseline_time else 1.0
            print(f"{name:30s}: {time_ms:6.2f} ms ({speedup:.2f}x){note}")
    
    # Architecture-specific insights
    print("\n" + "=" * 80)
    print("Architecture-Specific Insights")
    print("=" * 80)
    
    if arch_type == "gb10":
        print("[OK] GB10 (Grace-Blackwell) Benefits:")
        print("  • NVLink-C2C enables efficient handling of very long sequences")
        print("  • Can keep embeddings/KV cache in CPU memory (900 GB/s access)")
        print("  • Ideal for: 16K-128K token context windows")
    elif arch_type == "b200":
        print("[OK] B200 (Blackwell) Benefits:")
        print("  • HBM3e (7.8 TB/s) bandwidth for attention patterns")
        print("  • Optimal for: 4K-16K token sequences with sparse patterns")
        print("  • FlexAttention reduces memory from O(n²) to O(n*k)")
    
    print("\n" + "=" * 80)
    print("KEY PATTERNS FOR PRODUCTION")
    print("=" * 80)
    print("1. Sliding Window + Causal:  Best for autoregressive generation")
    print("2. Local + Global Sparse:    Best for long documents (summarization)")
    print("3. Dynamic Windows:          Best for diverse attention needs per head")
    print("4. Always use torch.compile: 2-3x speedup vs non-compiled")
    print("5. Longer sequences = more benefit from sparse patterns")
    print("=" * 80)


if __name__ == "__main__":
    main()
