"""
Stage 7: Auto-Select Best Configuration
=======================================

This module provides automatic selection of the best kernel configuration.
It benchmarks all available kernels and caches the optimal choice per problem size.

How it works:
1. Tries all available optimizations (stages 2-6)
2. Measures actual performance on your hardware
3. Caches the winner for instant selection next time
4. Adapts to different matrix sizes (small vs large may prefer different kernels)
"""

import heapq
import json
import os
from typing import Callable, Dict, Optional, Union

import torch

# Import all kernel variants
from tcgen05_loader import (
    matmul_tcgen05,
    matmul_tcgen05_pipelined,
    matmul_tcgen05_3stage,
    matmul_tcgen05_swizzled,
    matmul_tcgen05_cluster,
    matmul_tcgen05_warp_spec,
)

# Cache file location
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_CACHE_DIR, ".autotune_cache.json")


DeviceLike = Union[torch.device, str, int]


def _resolve_cuda_device(device: Optional[DeviceLike] = None) -> torch.device:
    """Resolve an explicit CUDA device, defaulting to the current device."""
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Autotuning requires a CUDA device, got {resolved}")
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _get_device_key(device: Optional[DeviceLike] = None) -> str:
    """Generate a cache key for the GPU that will execute the benchmark."""
    resolved = _resolve_cuda_device(device)
    props = torch.cuda.get_device_properties(resolved.index)
    return f"{props.name}_{props.major}.{props.minor}"


def _load_cache() -> Dict:
    """Load the autotune cache from disk."""
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_cache(cache: Dict):
    """Save the autotune cache to disk."""
    try:
        with open(_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass


def _benchmark_kernel(fn: Callable, A: torch.Tensor, B: torch.Tensor, 
                      warmup: int = 3, iters: int = 10) -> float:
    """Benchmark a kernel and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        _ = fn(A, B)
    torch.cuda.synchronize(A.device)

    # Timed runs: keep the k+1 smallest samples, whose max is the upper median.
    target_heap_size = iters // 2 + 1
    upper_median_heap = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(A.device)
    for _ in range(iters):
        start_event.record(current_stream)
        _ = fn(A, B)
        end_event.record(current_stream)
        end_event.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        heapq.heappush(upper_median_heap, -elapsed_ms)
        if len(upper_median_heap) > target_heap_size:
            heapq.heappop(upper_median_heap)

    return -upper_median_heap[0]


# Available kernels with names (in order of progressive optimization)
KERNELS = {
    "basic": matmul_tcgen05,
    "2stage": matmul_tcgen05_pipelined,
    "3stage": matmul_tcgen05_3stage,
    "swizzled": matmul_tcgen05_swizzled,
    "cluster": matmul_tcgen05_cluster,
    "4stage": matmul_tcgen05_warp_spec,  # Deep 4-stage pipeline
}


def autotune(
    M: int,
    N: int,
    K: int,
    verbose: bool = True,
    device: Optional[DeviceLike] = None,
) -> str:
    """
    Autotune to find the best kernel for the given problem size.
    
    Args:
        M, N, K: Matrix dimensions (A is MxK, B is NxK)
        verbose: Print tuning progress
    
    Returns:
        Name of the best kernel
    """
    resolved_device = _resolve_cuda_device(device)
    device_key = _get_device_key(resolved_device)
    size_key = f"{M}x{N}x{K}"
    cache_key = f"{device_key}_{size_key}"
    
    # Check cache
    cache = _load_cache()
    if cache_key in cache:
        winner = cache[cache_key]
        if verbose:
            print(f"  [Autotune cache hit: {winner}]")
        return winner
    
    if verbose:
        print(f"\n  ★ AUTOTUNING for {M}x{N}x{K} on {device_key} ★")
        print(f"  Testing {len(KERNELS)} configurations...")
    
    results = {}
    failures: Dict[str, str] = {}
    # Several extension variants launch on at::cuda::getCurrentCUDAStream()
    # and compile for the current device. Keep the current CUDA context aligned
    # with the tensors being tuned, including cleanup of that device's cache.
    with torch.cuda.device(resolved_device):
        A = torch.randn(M, K, device=resolved_device, dtype=torch.float16)
        B = torch.randn(N, K, device=resolved_device, dtype=torch.float16)

        try:
            for name, fn in KERNELS.items():
                try:
                    t = _benchmark_kernel(fn, A, B)
                    results[name] = t
                    if verbose:
                        tflops = 2 * M * N * K / t / 1e9
                        print(f"    {name:12s}: {t:>7.3f} ms ({tflops:>6.1f} TFLOPS)")
                except Exception as exc:
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    if verbose:
                        print(f"    {name:12s}: FAILED ({exc})")
        finally:
            del A, B
            torch.cuda.empty_cache()

    if not results:
        failure_summary = "; ".join(
            f"{name}={message}" for name, message in sorted(failures.items())
        )
        raise RuntimeError(
            f"Autotuning {M}x{N}x{K} on {device_key} failed: "
            f"all {len(KERNELS)} kernels failed ({failure_summary})"
        )
    
    # Find winner
    winner = min(results, key=results.get)
    
    if verbose:
        print(f"  ★ Winner: {winner} ({results[winner]:.3f} ms) ★\n")
    
    # Update cache
    cache[cache_key] = winner
    _save_cache(cache)
    
    return winner


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Execute GEMM using the autotuned best kernel.
    
    The first call for a given size will run autotuning (takes a few seconds).
    Subsequent calls use the cached optimal kernel.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor (result of A @ B^T)
    """
    M, K = a.shape
    N = b.shape[0]
    if a.device != b.device:
        raise ValueError(f"Input tensors must be on the same device, got {a.device} and {b.device}")
    if a.device.type != "cuda":
        raise ValueError(f"Autotuned tcgen05 kernels require CUDA tensors, got {a.device}")
    
    # Get best kernel (from cache or autotune)
    best = autotune(M, N, K, verbose=False, device=a.device)
    
    # Execute under the input device's context because the extension variants
    # use the current CUDA stream for their launches.
    with torch.cuda.device(a.device):
        return KERNELS[best](a, b)


def clear_cache():
    """Clear the autotune cache to force re-tuning."""
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
        print("Autotune cache cleared.")


def show_cache():
    """Display the current autotune cache."""
    cache = _load_cache()
    if not cache:
        print("Autotune cache is empty.")
        return
    
    print("\nAutotune Cache:")
    print("-" * 50)
    for key, value in sorted(cache.items()):
        print(f"  {key}: {value}")
    print("-" * 50)


if __name__ == "__main__":
    # Demo: autotune for common sizes
    print("=" * 60)
    print("  AUTOTUNE DEMO")
    print("=" * 60)
    
    sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    for M, N, K in sizes:
        autotune(M, N, K, verbose=True)
    
    print("\nFinal cache:")
    show_cache()
