import math

import torch

from core.utils.architecture_runtime import (
    get_arch_config,
)

_ARCH_CFG = get_arch_config()

"""fusion_pytorch.py
Chapter 9: Kernel Efficiency & Arithmetic Intensity Examples

Kernel fusion demonstrations targeting Blackwell GPUs."""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

GELU_TANH_SCALE = math.sqrt(2.0 / math.pi)

# Custom fused activation functions
class FusedGELU(nn.Module):
    """Fused GELU activation using torch.compile for kernel fusion"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Standard GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + torch.tanh(
            GELU_TANH_SCALE * (x + 0.044715 * torch.pow(x, 3))
        ))

class FusedLayerNormGELU(nn.Module):
    """Fused LayerNorm + GELU for better performance"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        # Fused LayerNorm + GELU
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        normalized.mul_(self.weight).add_(self.bias)
        scaled = normalized
        
        # Apply GELU activation
        return 0.5 * scaled * (1.0 + torch.tanh(
            GELU_TANH_SCALE * (scaled + 0.044715 * torch.pow(scaled, 3))
        ))

class FusedLinearBiasGELU(nn.Module):
    """Fused Linear + Bias + GELU"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # Fuse linear transformation with GELU activation
        x = self.linear(x)
        return F.gelu(x)  # PyTorch's optimized GELU

def unfused_operations(x, weight, bias):
    """Unfused version - multiple separate operations"""
    # Each operation launches a separate kernel
    x = torch.matmul(x, weight.t())  # Linear transformation
    x = x + bias                     # Bias addition
    x = F.gelu(x)                   # GELU activation
    x = F.layer_norm(x, [x.size(-1)]) # Layer normalization
    return x

@torch.compile
def fused_operations(x, weight, bias, ln_weight, ln_bias):
    """Fused version using torch.compile"""
    # torch.compile will fuse these operations into fewer kernels
    x = torch.matmul(x, weight.t())
    x = x + bias
    x = F.gelu(x)
    x = F.layer_norm(x, [x.size(-1)], weight=ln_weight, bias=ln_bias)
    return x

def manual_fused_linear_gelu(x, weight, bias):
    """Manually fused linear + bias + GELU in a single operation"""
    # This encourages the compiler to fuse the operations
    return F.gelu(F.linear(x, weight, bias))

class AttentionFusion(nn.Module):
    """Example of attention computation with fusion opportunities"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward_unfused(self, x):
        """Unfused attention - many separate operations"""
        batch_size, seq_len = x.shape[:2]
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    @torch.compile
    def forward_fused(self, x):
        """Fused attention using torch.compile"""
        batch_size, seq_len = x.shape[:2]
        
        # Same operations but compiled for fusion
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output

def benchmark_fusion():
    """Benchmark different fusion approaches"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Problem dimensions
    batch_size = 32
    seq_len = 512
    hidden_size = 768
    num_heads = 12
    
    print(f"=== Kernel Fusion Benchmark ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Hidden size: {hidden_size}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight = torch.randn(hidden_size, hidden_size, device=device)
    bias = torch.randn(hidden_size, device=device)
    ln_weight = torch.randn(hidden_size, device=device)
    ln_bias = torch.randn(hidden_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = unfused_operations(x, weight, bias)
        _ = fused_operations(x, weight, bias, ln_weight, ln_bias)
    
    torch.cuda.synchronize()
    
    # Benchmark unfused operations using CUDA Events (GPU timing)
    iterations = 100
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device)
    
    start_event.record(current_stream)
    for _ in range(iterations):
        result_unfused = unfused_operations(x, weight, bias)
    end_event.record(current_stream)
    end_event.synchronize()
    unfused_time = start_event.elapsed_time(end_event) / iterations
    
    # Benchmark fused operations using CUDA Events
    start_event.record(current_stream)
    for _ in range(iterations):
        result_fused = fused_operations(x, weight, bias, ln_weight, ln_bias)
    end_event.record(current_stream)
    end_event.synchronize()
    fused_time = start_event.elapsed_time(end_event) / iterations
    
    print(f"Linear + Bias + GELU + LayerNorm:")
    print(f"  Unfused time: {unfused_time:.2f} ms")
    print(f"  Fused time:   {fused_time:.2f} ms")
    print(f"  Speedup:      {unfused_time / fused_time:.2f}x")
    print()
    
    # Benchmark attention
    attention = AttentionFusion(hidden_size, num_heads).to(device)
    
    # Warmup
    for _ in range(5):
        _ = attention.forward_unfused(x)
        _ = attention.forward_fused(x)
    
    torch.cuda.synchronize()
    
    # Unfused attention using CUDA Events
    attn_iterations = iterations // 10  # Fewer iterations for complex operation
    start_event.record(current_stream)
    for _ in range(attn_iterations):
        result_attn_unfused = attention.forward_unfused(x)
    end_event.record(current_stream)
    end_event.synchronize()
    attn_unfused_time = start_event.elapsed_time(end_event) / attn_iterations
    
    # Fused attention using CUDA Events
    start_event.record(current_stream)
    for _ in range(attn_iterations):
        result_attn_fused = attention.forward_fused(x)
    end_event.record(current_stream)
    end_event.synchronize()
    attn_fused_time = start_event.elapsed_time(end_event) / attn_iterations
    
    print(f"Multi-Head Attention:")
    print(f"  Unfused time: {attn_unfused_time:.2f} ms")
    print(f"  Fused time:   {attn_fused_time:.2f} ms")
    print(f"  Speedup:      {attn_unfused_time / attn_fused_time:.2f}x")
    print()
    
    # Test custom fused modules
    fused_ln_gelu = FusedLayerNormGELU(hidden_size).to(device)
    fused_linear_gelu = FusedLinearBiasGELU(hidden_size, hidden_size).to(device)
    
    # Compile custom modules
    fused_ln_gelu_compiled = torch.compile(fused_ln_gelu)
    fused_linear_gelu_compiled = torch.compile(fused_linear_gelu)
    
    # Warmup custom modules
    for _ in range(10):
        _ = fused_ln_gelu_compiled(x)
        _ = fused_linear_gelu_compiled(x)
    
    torch.cuda.synchronize()
    
    # Benchmark custom modules using CUDA Events
    start_event.record(current_stream)
    for _ in range(iterations):
        result_custom = fused_ln_gelu_compiled(x)
    end_event.record(current_stream)
    end_event.synchronize()
    custom_time = start_event.elapsed_time(end_event) / iterations
    
    print(f"Custom Fused LayerNorm + GELU: {custom_time:.2f} ms")
    
    # Memory usage analysis
    print(f"\n=== Memory Usage ===")
    torch.cuda.reset_peak_memory_stats()
    _ = fused_operations(x, weight, bias, ln_weight, ln_bias)
    fused_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    torch.cuda.reset_peak_memory_stats()
    _ = unfused_operations(x, weight, bias)
    unfused_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Unfused peak memory: {unfused_memory:.1f} MB")
    print(f"Fused peak memory:   {fused_memory:.1f} MB")
    print(f"Memory reduction:    {(unfused_memory - fused_memory) / unfused_memory * 100:.1f}%")
    
    print(f"\n=== Profiling Commands ===")
    print("To profile kernel fusion with PyTorch Profiler:")
    print("python -c \"import torch.profiler; prof = torch.profiler.profile(...); prof.start(); your_code(); prof.stop(); prof.export_chrome_trace('trace.json')\"")
    print("\nTo analyze with Nsight Systems:")
    print("nsys profile --force-overwrite=true -o fusion_pytorch python fusion_pytorch.py")

def demonstrate_torch_compile_fusion():
    """Demonstrate torch.compile fusion capabilities"""
    print("=== torch.compile Fusion Examples ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1024, 512, device=device)
    
    # Example 1: Simple element-wise operations
    def unfused_elementwise(x):
        y = torch.sin(x)
        z = torch.cos(y)
        return torch.exp(z)
    
    @torch.compile
    def fused_elementwise(x):
        y = torch.sin(x)
        z = torch.cos(y)
        return torch.exp(z)
    
    # Example 2: Reduction followed by broadcast
    def unfused_reduction_broadcast(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-5)
    
    @torch.compile
    def fused_reduction_broadcast(x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-5)
    
    # Benchmark examples
    iterations = 1000
    
    # Warmup
    for _ in range(10):
        unfused_elementwise(x)
        fused_elementwise(x)
        unfused_reduction_broadcast(x)
        fused_reduction_broadcast(x)
    
    torch.cuda.synchronize()
    
    # Create CUDA Events for benchmarking
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device)
    
    # Elementwise benchmark using CUDA Events
    start_event.record(current_stream)
    for _ in range(iterations):
        unfused_elementwise(x)
    end_event.record(current_stream)
    end_event.synchronize()
    unfused_elem_time = start_event.elapsed_time(end_event) / iterations
    
    start_event.record(current_stream)
    for _ in range(iterations):
        fused_elementwise(x)
    end_event.record(current_stream)
    end_event.synchronize()
    fused_elem_time = start_event.elapsed_time(end_event) / iterations
    
    # Reduction/broadcast benchmark using CUDA Events
    start_event.record(current_stream)
    for _ in range(iterations):
        unfused_reduction_broadcast(x)
    end_event.record(current_stream)
    end_event.synchronize()
    unfused_rb_time = start_event.elapsed_time(end_event) / iterations
    
    start_event.record(current_stream)
    for _ in range(iterations):
        fused_reduction_broadcast(x)
    end_event.record(current_stream)
    end_event.synchronize()
    fused_rb_time = start_event.elapsed_time(end_event) / iterations
    
    print(f"Element-wise operations (sin->cos->exp):")
    print(f"  Unfused: {unfused_elem_time:.3f} ms")
    print(f"  Fused:   {fused_elem_time:.3f} ms")
    print(f"  Speedup: {unfused_elem_time / fused_elem_time:.2f}x")
    print()
    
    print(f"Reduction + Broadcast (normalization):")
    print(f"  Unfused: {unfused_rb_time:.3f} ms")
    print(f"  Fused:   {fused_rb_time:.3f} ms")
    print(f"  Speedup: {unfused_rb_time / fused_rb_time:.2f}x")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print()
        
        benchmark_fusion()
        print()
        demonstrate_torch_compile_fusion()
    else:
        print("CUDA not available. Running on CPU for demonstration.")
        benchmark_fusion()
