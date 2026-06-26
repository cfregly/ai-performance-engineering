"""Benchmark Conservative vs Aggressive TMA Configurations

Demonstrates the performance impact of using optimal Blackwell configs instead of conservative workaround configurations.
"""
import torch
import triton
import triton.language as tl
import triton.testing
from triton.runtime import _allocation as triton_allocation
import os

# Setup allocator for TMA
class _TorchCudaBuffer:
    __slots__ = ("_ptr",)
    
    def __init__(self, ptr: int):
        self._ptr = ptr
    
    def data_ptr(self) -> int:
        return self._ptr
    
    def __del__(self):
        if self._ptr:
            torch.cuda.caching_allocator_delete(self._ptr)
            self._ptr = 0


class _TorchCudaAllocator:
    def __call__(self, size: int, alignment: int, stream: int | None):
        if size == 0:
            return _TorchCudaBuffer(0)
        
        if stream is None:
            current_stream = torch.cuda.current_stream()
            stream = current_stream.cuda_stream
            device_idx = current_stream.device.index
        else:
            device_idx = torch.cuda.current_device()
        
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        
        ptr = torch.cuda.caching_allocator_alloc(size, device_idx, stream=stream)
        return _TorchCudaBuffer(ptr)


if torch.cuda.is_available():
    current = triton_allocation._allocator.get()
    if isinstance(current, triton_allocation.NullAllocator):
        triton.set_allocator(_TorchCudaAllocator())


@triton.jit
def tma_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Flexible TMA GEMM for testing different configs"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, K, BLOCK_K):
        a = A_desc.load([m0, k0])
        b = B_desc.load([k0, n0])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def benchmark_config(M, N, K, config_name, block_m, block_n, block_k, num_warps, num_stages):
    """Benchmark a specific configuration"""
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    
    # Warmup
    for _ in range(10):
        tma_gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    time_ms = triton.testing.do_bench(lambda: tma_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    ))
    
    # Calculate TFLOPS
    flops = 2 * M * N * K  # Multiply-add counts as 2 ops
    tflops = flops / (time_ms * 1e-3) / 1e12
    
    # Calculate bandwidth utilization (rough estimate)
    bytes_read = (M * K + K * N) * 2  # FP16 input
    bytes_write = M * N * 4  # FP32 output
    total_bytes = bytes_read + bytes_write
    bandwidth_gbs = total_bytes / (time_ms * 1e-3) / 1e9
    hbm3e_peak = 7800  # GB/s for B200
    bandwidth_pct = (bandwidth_gbs / hbm3e_peak) * 100
    
    return {
        'config': config_name,
        'time_ms': time_ms,
        'tflops': tflops,
        'bandwidth_gbs': bandwidth_gbs,
        'bandwidth_pct': bandwidth_pct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Triton Version: {triton.__version__}\n")
    
    if device_props.major >= 12 and device_props.minor >= 1:
        print("Grace-Blackwell SM 12.x does not expose TMA instructions yet (CUDA 13.0). Skipping benchmark.")
        return
    
    print("=" * 90)
    print("TMA Configuration Benchmark: Conservative vs Aggressive")
    print("=" * 90)
    
    quick_mode = os.getenv("BENCHMARK_QUICK", "0") not in ("0", "false", "False", "")
    test_sizes = [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    if quick_mode:
        test_sizes = [(1024, 1024, 1024)]
    
    for M, N, K in test_sizes:
        print(f"\nMatrix Size: {M}x{K} × {K}x{N} = {M}x{N}")
        print("-" * 90)
        
        # Conservative config (current workaround)
        conservative = benchmark_config(
            M, N, K,
            config_name="Conservative",
            block_m=64,
            block_n=64,
            block_k=32,
            num_warps=4,
            num_stages=1
        )
        
        # Aggressive config (optimal for Blackwell)
        aggressive = benchmark_config(
            M, N, K,
            config_name="Aggressive",
            block_m=128,
            block_n=128,
            block_k=128,
            num_warps=16,
            num_stages=4
        )
        
        # Display results
        print(f"{'Configuration':<15} {'Time (ms)':<12} {'TFLOPS':<12} {'BW (GB/s)':<12} {'BW %':<8} {'Speedup':<8}")
        print("-" * 90)
        for result in [conservative, aggressive]:
            speedup = conservative['time_ms'] / result['time_ms'] if result != conservative else 1.0
            print(f"{result['config']:<15} "
                  f"{result['time_ms']:>10.2f} "
                  f"{result['tflops']:>10.1f} "
                  f"{result['bandwidth_gbs']:>10.1f} "
                  f"{result['bandwidth_pct']:>6.1f} "
                  f"{speedup:>6.2f}x")
        
        # Calculate improvement
        speedup = conservative['time_ms'] / aggressive['time_ms']
        tflops_improvement = (aggressive['tflops'] / conservative['tflops'] - 1) * 100
        bw_improvement = (aggressive['bandwidth_pct'] / conservative['bandwidth_pct'] - 1) * 100
        
        print("\n" + " IMPROVEMENT WITH AGGRESSIVE CONFIG ".ljust(90, "="))
        print(f" Speedup: {speedup:.2f}x faster ({(speedup-1)*100:.0f}% improvement)")
        print(f" TFLOPS: +{tflops_improvement:.0f}% higher throughput")
        print(f" Bandwidth: +{bw_improvement:.0f}% better utilization")
    
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    print("Update triton_tma_blackwell.py GEMM configs to use aggressive settings:")
    print(" - BLOCK_K: 32 → 128")
    print(" - num_stages: 1 → 4")
    print(" - num_warps: 4 → 16")
    print("\nExpected benefit: 2-3x performance improvement on large matrices")
    print("=" * 90)


if __name__ == "__main__":
    main()
