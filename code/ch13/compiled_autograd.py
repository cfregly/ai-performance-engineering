#!/usr/bin/env python3
from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
"""
Compiled Autograd with PyTorch 2.10

PyTorch 2.10 can compile the backward pass for improved training efficiency.
Requires PyTorch 2.10+ and CUDA 13.0+.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import compiled_autograd
import time
from core.common.device_utils import get_preferred_device

# Check if compiled autograd is available
try:
    from torch._dynamo import compiled_autograd
    COMPILED_AUTOGRAD_AVAILABLE = True
except ImportError:
    COMPILED_AUTOGRAD_AVAILABLE = False
    print("WARNING: Compiled autograd not available. Requires PyTorch 2.10+")


class BenchmarkModel(nn.Module):
    """Model for benchmarking compiled autograd."""
    
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.GELU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def benchmark_standard_autograd() -> float:
    """Benchmark standard autograd (baseline)."""
    device_obj, cuda_err = get_preferred_device()
    device = device_obj.type
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU for baseline autograd benchmark.")

    model = BenchmarkModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device == "cuda"))
    
    batch_size = 32
    seq_len = 256
    hidden_dim = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
    else:
        start_time = time.perf_counter()
    
    num_iters = 20
    for _ in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        end.record(current_stream)
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / num_iters
    else:
        elapsed_ms = (time.perf_counter() - start_time) / num_iters * 1000
    
    return elapsed_ms


def benchmark_compiled_autograd() -> float:
    """Benchmark with compiled autograd."""
    if not COMPILED_AUTOGRAD_AVAILABLE:
        return 0.0

    device_obj, cuda_err = get_preferred_device()
    device = device_obj.type
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU for compiled autograd benchmark.")

    model = BenchmarkModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(device == "cuda"))
    
    batch_size = 32
    seq_len = 256
    hidden_dim = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Enable compiled autograd
    compiled_autograd.enable(compiler="inductor")
    
    try:
        # Warmup
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            current_stream = torch.cuda.current_stream()
            start.record(current_stream)
        else:
            start_time = time.perf_counter()
        
        num_iters = 20
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if device == "cuda":
            end.record(current_stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / num_iters
        else:
            elapsed_ms = (time.perf_counter() - start_time) / num_iters * 1000
        
        return elapsed_ms
    
    finally:
        # Disable compiled autograd
        compiled_autograd.disable()


def benchmark_forward_and_backward_compiled() -> tuple[float, float]:
    """Benchmark with both forward and backward compiled."""
    if not COMPILED_AUTOGRAD_AVAILABLE:
        return 0.0, 0.0

    device_obj, cuda_err = get_preferred_device()
    device = device_obj.type
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU for forward+backward compiled benchmark.")

    model = BenchmarkModel().to(device)

    # Compile forward pass when CUDA is available; CPU uses eager mode.
    compiled_model = torch.compile(model, mode="max-autotune") if device == "cuda" else model

    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4, fused=(device == "cuda"))
    
    batch_size = 32
    seq_len = 256
    hidden_dim = 4096
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    target = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Enable compiled autograd
    compiled_autograd.enable(compiler="inductor")
    
    try:
        # Warmup
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            output = compiled_model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            current_stream = torch.cuda.current_stream()
            start.record(current_stream)
        else:
            start_time = time.perf_counter()
        
        num_iters = 20
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            output = compiled_model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if device == "cuda":
            end.record(current_stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / num_iters
        else:
            elapsed_ms = (time.perf_counter() - start_time) / num_iters * 1000
        
        # Memory usage
        if device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0.0
        
        return elapsed_ms, memory_mb
    
    finally:
        compiled_autograd.disable()


def demonstrate_compiled_autograd_context() -> None:
    """Demonstrate using compiled autograd as a context manager."""
    if not COMPILED_AUTOGRAD_AVAILABLE:
        print("Compiled autograd not available")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BenchmarkModel(hidden_dim=2048, num_layers=4).to(device)
    
    x = torch.randn(16, 128, 2048, device=device)
    target = torch.randn(16, 128, 2048, device=device)
    
    print("\n" + "=" * 80)
    print("Compiled Autograd Context Manager Example")
    print("=" * 80)
    
    # Method 1: Global enable/disable
    print("\nMethod 1: Global enable/disable")
    compiled_autograd.enable(compiler="inductor")
    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    compiled_autograd.disable()
    print(" Used global enable/disable")
    
    # Method 2: Context manager (cleaner)
    print("\nMethod 2: Context manager (recommended)")
    torch.cuda.empty_cache() if device == "cuda" else None
    
    with compiled_autograd.enable(compiler="inductor"):
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
    print(" Used context manager")
    
    # Method 3: Selective compilation
    print("\nMethod 3: Selective compilation")
    print("  - Forward pass: compiled")
    print("  - Backward pass: compiled autograd")
    
    compiled_model = torch.compile(model, mode="reduce-overhead")
    
    with compiled_autograd.enable(compiler="inductor"):
        output = compiled_model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
    print(" Both forward and backward compiled")
    
    print("=" * 80)


def main() -> None:
    """Run compiled autograd benchmarks."""
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU (slower).")
    
    if not COMPILED_AUTOGRAD_AVAILABLE:
        print("\n" + "!" * 80)
        print("Compiled autograd not available in your PyTorch installation.")
        print("This feature requires PyTorch 2.10-dev or later.")
        print("\nTo install PyTorch 2.10 nightly:")
        print("  pip install torch==2.10.0.dev*+cu130 --index-url https://download.pytorch.org/whl/nightly/cu130")
        print("!" * 80)
        return
    
    print("=" * 80)
    print("Compiled Autograd Benchmark (PyTorch 2.10-dev)")
    print("=" * 80)
    print()
    
    # Benchmark 1: Standard autograd
    print("1. Benchmarking standard autograd (baseline)...")
    standard_time = benchmark_standard_autograd()
    print(f"   Time: {standard_time:.2f} ms/iter")
    
    # Benchmark 2: Compiled autograd
    print("\n2. Benchmarking compiled autograd...")
    compiled_time = benchmark_compiled_autograd()
    if compiled_time > 0:
        print(f"   Time: {compiled_time:.2f} ms/iter")
        speedup = standard_time / compiled_time
        print(f"   Speedup: {speedup:.2f}x")
    
    # Benchmark 3: Both forward and backward compiled
    print("\n3. Benchmarking forward+backward compiled...")
    both_time, memory_mb = benchmark_forward_and_backward_compiled()
    if both_time > 0:
        print(f"   Time: {both_time:.2f} ms/iter")
        speedup = standard_time / both_time
        print(f"   Speedup: {speedup:.2f}x")
        if memory_mb > 0:
            print(f"   Peak memory: {memory_mb:.1f} MB")
    
    # Context manager demo
    demonstrate_compiled_autograd_context()
    
    print("\n" + "=" * 80)
    print("Summary: Compiled Autograd")
    print("=" * 80)
    print("PyTorch 2.10 compiled autograd provides:")
    print(" 20-30% faster backward pass")
    print(" Better kernel fusion")
    print(" Reduced Python overhead")
    print(" Compatible with torch.compile for forward pass")
    print("\nBest for:")
    print("- Large models with complex backward graphs")
    print("- Training on Blackwell GPUs")
    print("- Production training workloads")
    print("\nUsage:")
    print("  with compiled_autograd.enable(compiler='inductor'):")
    print("      loss.backward()")
    print("=" * 80)


if __name__ == "__main__":
    main()
