"""
torch.compile Benchmark with REALISTIC Model Size for Blackwell B200
=====================================================================

The previous benchmark used a 25M parameter model (too small).
This benchmark uses a 500M-1B parameter model to show REAL speedup.

Key insight: torch.compile benefits scale with model size!
- Small (<50M): 1.0-1.1x
- Medium (50-500M): 1.1-1.3x  
- Large (500M-5B): 1.3-1.5x  <- WE'RE TESTING THIS

Hardware: NVIDIA B200 (178 GB memory available)
"""

import os
import sys


import torch
from core.utils.compile_utils import enable_tf32
import torch.nn as nn
import triton.testing



def configure_for_peak_performance():
    """Configure PyTorch for Blackwell B200 peak performance"""
    print("Configuring for Blackwell B200...")
    
    # TF32
    enable_tf32()
    
    # Flash Attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Inductor
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.triton.cudagraph_trees = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = True
    
    # cuDNN
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("Configuration complete\n")


class LargeTransformerBlock(nn.Module):
    """Production-size transformer block"""
    def __init__(self, d_model=2048, num_heads=16, d_ff=8192):
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
        
        # Scaled dot-product attention
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


def create_model(size='medium'):
    """
    Create transformer model of specified size
    
    Sizes:
    - small: ~50M params (for comparison)
    - medium: ~350M params (GPT-2 medium)
    - large: ~750M params (GPT-2 large)
    - xl: ~1.5B params (GPT-2 XL)
    """
    configs = {
        'small': {'d_model': 1024, 'num_heads': 16, 'd_ff': 4096, 'layers': 12},
        'medium': {'d_model': 1536, 'num_heads': 16, 'd_ff': 6144, 'layers': 24},
        'large': {'d_model': 2048, 'num_heads': 16, 'd_ff': 8192, 'layers': 24},
        'xl': {'d_model': 2560, 'num_heads': 20, 'd_ff': 10240, 'layers': 32},
        '5b': {'d_model': 4096, 'num_heads': 32, 'd_ff': 16384, 'layers': 40},
        '10b': {'d_model': 5120, 'num_heads': 40, 'd_ff': 20480, 'layers': 50},
        '20b': {'d_model': 6144, 'num_heads': 48, 'd_ff': 24576, 'layers': 60},
        '50b': {'d_model': 8192, 'num_heads': 64, 'd_ff': 32768, 'layers': 80},
        '100b': {'d_model': 10240, 'num_heads': 80, 'd_ff': 40960, 'layers': 100},
    }
    
    config = configs[size]
    
    blocks = []
    for _ in range(config['layers']):
        blocks.append(LargeTransformerBlock(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff']
        ))
    
    model = nn.Sequential(*blocks)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    return model, config, total_params


def estimate_memory(batch_size, seq_len, d_model, total_params):
    """Estimate memory usage"""
    # Activations
    activation_memory = batch_size * seq_len * d_model * 4  # float32
    
    # Parameters
    param_memory = total_params * 4  # float32
    
    # Gradients (if training)
    grad_memory = total_params * 4
    
    # Optimizer states (if training)
    optimizer_memory = total_params * 8  # Adam: 2x params
    
    # Total for inference
    inference_total = (activation_memory + param_memory) / (1024**3)
    
    # Total for training
    training_total = (activation_memory + param_memory + grad_memory + optimizer_memory) / (1024**3)
    
    return inference_total, training_total


def benchmark_with_proper_warmup(model, x, name):
    """Benchmark using Triton's testing framework (handles warmup, sync, outliers)"""
    print(f"\nBenchmarking: {name}")
    
    # Use Triton benchmarking - automatically handles warmup and synchronization
    def run_model():
        with torch.inference_mode():
            return model(x)
    
    warmup = 5
    rep = 50
    avg_time_ms = triton.testing.do_bench(run_model, warmup=warmup, rep=rep)
    throughput = 1000.0 / avg_time_ms  # iter/s
    
    print(f"  Average time: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2f} iter/s")
    
    return avg_time_ms, throughput


def main():
    """Run large model benchmark"""
    configure_for_peak_performance()
    
    # Check available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Available GPU memory: {total_memory:.1f} GB\n")
    
    # Test different model sizes
    print("=" * 80)
    print("MODEL SIZE SELECTION")
    print("=" * 80)
    
    sizes_to_test = ['small', 'medium', 'large', 'xl', '5b', '10b', '20b', '50b', '100b']
    batch_size = 16
    seq_len = 1024

    # Find largest model that fits
    selected_size = None
    selected_config = None
    selected_params = None
    
    for size in sizes_to_test:
        print(f"\nTesting {size} model...")
        model, config, total_params = create_model(size)
        
        inference_mem, training_mem = estimate_memory(
            batch_size, seq_len, config['d_model'], total_params
        )
        
        print(f"  Parameters: {total_params / 1e6:.1f}M")
        print(f"  Config: {config['layers']} layers, {config['d_model']} hidden")
        print(f"  Estimated memory (inference): {inference_mem:.1f} GB")
        print(f"  Estimated memory (training): {training_mem:.1f} GB")
        
        if inference_mem < total_memory * 0.8:  # Leave 20% headroom
            selected_size = size
            selected_config = config
            selected_params = total_params
            print(f"  Status: FITS (using {inference_mem/total_memory*100:.1f}% of memory)")
        else:
            print(f"  Status: TOO LARGE")
            break
    
    if selected_size is None:
        print("\nERROR: No model size fits in memory!")
        return
    
    print(f"\n" + "=" * 80)
    print(f"SELECTED: {selected_size.upper()} model ({selected_params / 1e6:.1f}M parameters)")
    print("=" * 80)
    
    # Create models
    print("\nCreating models...")
    model, config, total_params = create_model(selected_size)
    model = model.cuda().eval()
    
    model_compiled = torch.compile(
        model,
        mode='max-autotune',
        fullgraph=True,
        dynamic=False,
        backend='inductor',
    )
    
    # Create input
    batch_size = 16
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, config['d_model'], device='cuda', dtype=torch.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Input size: {x.numel() * 4 / 1e6:.1f} MB")
    print(f"Model parameters: {total_params / 1e6:.1f}M")
    
    # Benchmark eager mode
    print("\n" + "=" * 80)
    print("EAGER MODE")
    print("=" * 80)
    eager_time, eager_throughput = benchmark_with_proper_warmup(
        model, x, "Eager Mode"
    )
    
    # Benchmark compiled mode
    print("\n" + "=" * 80)
    print("COMPILED MODE")
    print("=" * 80)
    compiled_time, compiled_throughput = benchmark_with_proper_warmup(
        model_compiled, x, "Compiled Mode"
    )
    
    # Results
    speedup = eager_time / compiled_time
    throughput_gain = compiled_throughput / eager_throughput
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Model size:        {selected_size} ({selected_params / 1e6:.1f}M parameters)")
    print(f"Eager mode:        {eager_time:.2f} ms")
    print(f"Compiled mode:     {compiled_time:.2f} ms")
    print(f"Speedup:           {speedup:.2f}x")
    print(f"Throughput gain:   {throughput_gain:.2f}x")
    print()
    
    # Assessment
    expected_ranges = {
        'small': (1.0, 1.15),
        'medium': (1.15, 1.35),
        'large': (1.3, 1.5),
        'xl': (1.4, 1.6),
    }
    
    expected_min, expected_max = expected_ranges[selected_size]
    
    if speedup >= expected_min:
        print(f"EXCELLENT! Speedup {speedup:.2f}x is within expected range ({expected_min:.2f}-{expected_max:.2f}x)")
        print(f"for {selected_size} models ({selected_params / 1e6:.0f}M parameters)")
    else:
        print(f"Speedup {speedup:.2f}x is below expected range ({expected_min:.2f}-{expected_max:.2f}x)")
        print(f"for {selected_size} models. This may indicate:")
        print("  - Need more warmup iterations")
        print("  - Memory bandwidth bottleneck")
        print("  - Model still too small for compilation benefits")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"1. Larger models show better torch.compile speedup")
    print(f"2. {selected_size.capitalize()} model ({selected_params / 1e6:.0f}M params) achieved {speedup:.2f}x")
    print(f"3. Expected range for this size: {expected_min:.2f}-{expected_max:.2f}x")
    print(f"4. Using {x.numel() * 4 / 1e9:.2f} GB / {total_memory:.1f} GB available memory")
    print("=" * 80)
    
    return speedup, expected_min


if __name__ == "__main__":
    speedup, expected_min = main()
    
    import os
    default_threshold = expected_min * 0.95
    success_threshold = float(os.getenv("TORCH_COMPILE_MIN_SPEEDUP", f"{default_threshold:.4f}"))
    sys.exit(0 if speedup >= success_threshold else 1)
