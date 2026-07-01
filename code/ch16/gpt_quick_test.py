#!/usr/bin/env python3

"""
Quick GPT-style model test - NO HEAVY COMPILATION
Shows realistic torch.compile speedup on B200
"""

from core.utils.compile_utils import enable_tf32
from core.utils.warning_filters import suppress_benchmark_import_warnings

with suppress_benchmark_import_warnings(context="ch16.gpt_quick_test torch import"):
    import torch
    import torch.nn as nn

class SimpleGPTBlock(nn.Module):
    def __init__(self, d_model=4096, n_heads=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_input = self.ln1(x)
        x = x + self.attn(attn_input, attn_input, attn_input)[0]
        x = x + self.mlp(self.ln2(x))
        return x

def benchmark_quick(model, x, name, num_iters=20):
    """Quick benchmark"""
    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    count = max(num_iters, 1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(x.device)
    with torch.inference_mode():
        start.record(current_stream)
        for _ in range(count):
            _ = model(x)
        end.record(current_stream)
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    elapsed = elapsed_ms / 1000.0
    
    avg_ms = elapsed_ms / count
    tokens_per_sec = (x.shape[0] * x.shape[1] * count) / elapsed
    
    print(f"\n{name}:")
    print(f"  Time: {avg_ms:.2f} ms")
    print(f"  Throughput: {tokens_per_sec/1000:.1f}K tokens/sec")
    
    return avg_ms

def main():
    # NEW PyTorch 2.10 API (fixes warnings!)
    with suppress_benchmark_import_warnings(context="ch16.gpt_quick_test enable_tf32"):
        enable_tf32()
    
    print("=" * 80)
    print("QUICK GPT TEST ON B200")
    print("=" * 80)
    
    # Test different sizes to find sweet spot
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    prefers_bfloat16 = capability[0] >= 9
    run_dtype = torch.bfloat16 if prefers_bfloat16 else torch.float16

    configs = [
        # Tuned for quick turnaround but still large enough to show compile gains
        (4, 1024, 4, 512),
        (6, 1536, 4, 512),
        (8, 2048, 4, 512),
    ]
    
    for n_layers, d_model, batch, seq_len in configs:
        print(f"\n{'=' * 80}")
        print(f"Config: {n_layers} layers, d_model={d_model}, batch={batch}, seq={seq_len}")
        print(f"{'=' * 80}")
        
        # Create model
        blocks = [SimpleGPTBlock(d_model=d_model, n_heads=max(4, d_model // 256)) for _ in range(n_layers)]
        model = nn.Sequential(*blocks).cuda().to(run_dtype).eval()
        
        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Parameters: {params:.2f}B")
        
        # Input
        x = torch.randn(batch, seq_len, d_model, device='cuda', dtype=run_dtype)
        bytes_per_elem = torch.finfo(run_dtype).bits // 8
        mem = x.numel() * bytes_per_elem / 1e9
        print(f"Input size: {mem:.2f} GB")
        
        # Eager
        eager_time = benchmark_quick(model, x, "Eager Mode")
        
        # Compiled (reduce-overhead mode - faster compilation)
        print("\n[Compiling... this may take 30-60 seconds]")
        with suppress_benchmark_import_warnings(context="ch16.gpt_quick_test torch.compile"):
            model_compiled = torch.compile(model, mode='reduce-overhead')
        print("[Compilation done, now benchmarking...]")
        compiled_time = benchmark_quick(model_compiled, x, "Compiled Mode")
        
        speedup = eager_time / compiled_time
        print(f"\n>>> Speedup: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("[OK] GOOD speedup!")
            break
        else:
            print("WARNING: Too small, trying larger...")
        del model_compiled
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
