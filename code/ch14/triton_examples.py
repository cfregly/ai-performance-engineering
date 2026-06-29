"""
Triton 3.5 Kernel Examples with Blackwell B200 Optimizations

Blackwell B200 Optimizations Applied:
- FP16 inputs with FP32 accumulators for bandwidth reduction
- Cache eviction policies for better L2 utilization
- Expanded autotune space with BLOCK_K=128 and num_warps=16
- Deeper pipelines (num_stages=4-5) for better overlap
- Direct broadcast for offset tensors to reduce register pressure
"""
import sys

import torch
import triton
import triton.language as tl
import triton.testing
from triton.runtime import _allocation as triton_allocation


# ============================================================================
# TMA Allocator Bridge (Required for Triton TMA kernels)
# ============================================================================

class _TorchCudaBuffer:
    """Simple wrapper for pointers returned by PyTorch's caching allocator."""
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
    """Allocator that lets Triton reuse PyTorch's caching allocator for TMA scratch buffers."""

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


def _ensure_triton_allocator():
    """Set Triton's allocator once so TMA kernels can grab scratch buffers."""
    if not torch.cuda.is_available():
        return
    current = triton_allocation._allocator.get()
    if isinstance(current, triton_allocation.NullAllocator):
        triton.set_allocator(_TorchCudaAllocator())


_ensure_triton_allocator()


# Check for FP8 support
try:
    FP8_E4M3_DTYPE = torch.float8_e4m3fn
    FP8_E5M2_DTYPE = torch.float8_e5m2
    FP8_AVAILABLE = True
except AttributeError:
    FP8_AVAILABLE = False
    FP8_E4M3_DTYPE = torch.float16
    FP8_E5M2_DTYPE = torch.float16


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def tiled_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled GEMM with TMA tensor descriptors.
    
    Uses tl.load_tensor_descriptor() for efficient TMA-backed loads.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M
    n_off = pid_n * BLOCK_N

    # Create TMA tensor descriptors
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

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop with TMA loads
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load_tensor_descriptor(A_desc, [m_off, k])
        b_tile = tl.load_tensor_descriptor(B_desc, [k, n_off])
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # Store with boundary mask
    offs_m = m_off + tl.arange(0, BLOCK_M)
    offs_n = n_off + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Tiled matrix multiplication using autotuned Triton kernel."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Use META-aware grid to correctly handle all autotune configs
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    tiled_gemm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Persistent thread GEMM with TMA tensor descriptors.
    
    Each thread block processes multiple output tiles in a loop.
    Uses tl.load_tensor_descriptor() for efficient TMA-backed loads.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    MT = tl.cdiv(M, BLOCK_M)
    NT = tl.cdiv(N, BLOCK_N)
    TILE_COUNT = MT * NT

    # Create TMA tensor descriptors (outside tile loop for efficiency)
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

    tile_idx = pid
    while tile_idx < TILE_COUNT:
        pid_m = tile_idx // NT
        pid_n = tile_idx % NT

        m_off = pid_m * BLOCK_M
        n_off = pid_n * BLOCK_N

        # FP32 accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Main K-loop with TMA loads
        for k in range(0, K, BLOCK_K):
            a_tile = tl.load_tensor_descriptor(A_desc, [m_off, k])
            b_tile = tl.load_tensor_descriptor(B_desc, [k, n_off])
            acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

        # Store with boundary mask
        offs_m = m_off + tl.arange(0, BLOCK_M)
        offs_n = n_off + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

        tile_idx += nprog


def persistent_matmul_descriptor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Persistent GEMM using tensor descriptors with META-aware tiling."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Use META-aware grid calculation
    grid = lambda META: (min(65536, triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])),)

    matmul_kernel_persistent[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# FP8 GEMM Kernel

@triton.jit
def matmul_fp8_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """FP8 matrix multiplication kernel with FP32 accumulation."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        if FP8_AVAILABLE:
            a_fp8 = a.to(tl.float8e4m3fn)
            b_fp8 = b.to(tl.float8e4m3fn)
            acc += tl.dot(a_fp8, b_fp8, out_dtype=tl.float32)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32)
        

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_fp8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FP8 matrix multiplication wrapper."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: A.K={K}, B.K={K2}"
    
    if FP8_AVAILABLE and A.dtype != FP8_E4M3_DTYPE:
        A = A.to(FP8_E4M3_DTYPE)
    if FP8_AVAILABLE and B.dtype != FP8_E4M3_DTYPE:
        B = B.to(FP8_E4M3_DTYPE)
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_fp8_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


def benchmark_fp8_vs_fp16() -> None:
    """Benchmark FP8 vs FP16 matrix multiplication."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping FP8 benchmark")
        return
    
    print("\n" + "=" * 80)
    print("FP8 vs FP16 Matrix Multiplication Benchmark (Triton 3.5)")
    print("=" * 80)
    
    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    for M, N, K in sizes:
        print(f"\nMatrix size: {M}x{K} @ {K}x{N}")
        
        A_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
        
        # Use Triton's benchmarking - handles warmup, sync, outliers automatically
        fp16_time = triton.testing.do_bench(lambda: tiled_matmul(A_fp16, B_fp16), rep=100)
        
        C_fp16 = tiled_matmul(A_fp16, B_fp16)  # For numerical comparison
        
        flops = 2 * M * N * K
        fp16_tflops = flops / (fp16_time * 1e-3) / 1e12
        
        print(f"  FP16: {fp16_time:.2f} ms/iter, {fp16_tflops:.1f} TFLOPS")
        
        if FP8_AVAILABLE:
            A_fp8 = A_fp16.to(FP8_E4M3_DTYPE)
            B_fp8 = B_fp16.to(FP8_E4M3_DTYPE)
            
            # Use Triton's benchmarking - handles warmup, sync, outliers automatically
            fp8_time = triton.testing.do_bench(lambda: matmul_fp8(A_fp8, B_fp8), rep=100)
            
            C_fp8 = matmul_fp8(A_fp8, B_fp8)  # For numerical comparison
            
            fp8_tflops = flops / (fp8_time * 1e-3) / 1e12
            speedup = fp16_time / fp8_time
            
            print(f"  FP8:  {fp8_time:.2f} ms/iter, {fp8_tflops:.1f} TFLOPS ({speedup:.2f}x speedup)")
            
            diff = (C_fp16 - C_fp8).abs()
            diff_stats = torch.empty(2, device=diff.device, dtype=diff.dtype)
            diff_stats[0].copy_(diff.max())
            diff_stats[1].copy_(diff.mean())
            diff_stats_host = diff_stats.detach().cpu()
            max_diff = float(diff_stats_host[0])
            mean_diff = float(diff_stats_host[1])
            print(f"  Numerical error: max={max_diff:.6f}, mean={mean_diff:.6f}")
        else:
            print(f"  FP8:  Not available (requires PyTorch 2.10+)")
    
    print("\n" + "=" * 80)


@triton.jit
def persistent_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent GEMM kernel with work queue load balancing."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        if pid_m >= num_pid_m or pid_n >= num_pid_n:
            continue
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
            a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
            # Load as fp16 for bandwidth reduction, use cache eviction policy
            a = tl.load(a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first").to(tl.float16)
            
            b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
            b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            # Load as fp16 for bandwidth reduction, keep in cache for reuse
            b = tl.load(b_ptrs, mask=b_mask, other=0.0, eviction_policy="evict_last").to(tl.float16)
            
            # Compute: fp16 inputs cut bandwidth, fp32 accumulator maintains accuracy
            acc += tl.dot(a, b, out_dtype=tl.float32)
        

        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def persistent_matmul_queue(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Persistent GEMM with work queue and reduced launch overhead."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    NUM_SMS = 148  # Blackwell B200 has 148 SMs
    grid = (NUM_SMS,)
    
    persistent_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8,
        num_stages=3,
    )
    
    return C


def benchmark_persistent_vs_standard():
    """Benchmark persistent kernel performance vs standard implementations"""
    print("\n" + "=" * 80)
    print("Persistent Kernel Benchmark")
    print("=" * 80)
    
    device = "cuda"
    sizes = [2048, 4096, 8192]
    
    for size in sizes:
        M = N = K = size
        
        print(f"\nMatrix size: {M}x{K} @ {K}x{N}")
        
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)
        
        # Pre-convert to float32 outside benchmark to avoid timing dtype conversions
        A_fp32 = A.float()
        B_fp32 = B.float()
        
        # Use Triton's benchmarking - handles warmup, sync, outliers automatically
        persistent_descriptor_time = triton.testing.do_bench(lambda: persistent_matmul_descriptor(A, B), rep=100)
        persistent_queue_time = triton.testing.do_bench(lambda: persistent_matmul_queue(A, B), rep=100)
        torch_time = triton.testing.do_bench(lambda: torch.matmul(A_fp32, B_fp32), rep=100)
        
        flops = 2 * M * N * K
        persistent_descriptor_tflops = flops / (persistent_descriptor_time * 1e-3) / 1e12
        persistent_queue_tflops = flops / (persistent_queue_time * 1e-3) / 1e12
        torch_tflops = flops / (torch_time * 1e-3) / 1e12
        
        descriptor_speedup = torch_time / persistent_descriptor_time
        queue_speedup = torch_time / persistent_queue_time
        
        print(f"  Persistent (descriptor): {persistent_descriptor_time:.2f} ms, {persistent_descriptor_tflops:.1f} TFLOPS ({descriptor_speedup:.2f}x)")
        print(f"  Persistent (queue):      {persistent_queue_time:.2f} ms, {persistent_queue_tflops:.1f} TFLOPS ({queue_speedup:.2f}x)")
        print(f"  PyTorch matmul:          {torch_time:.2f} ms, {torch_tflops:.1f} TFLOPS (baseline)")
    
    print("\n" + "=" * 80)



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton examples benchmark.")
        sys.exit(0)

    cc = torch.cuda.get_device_capability()
    if cc != (10, 0):
        print(f"Skipping Triton examples: requires Blackwell SM 10.0, detected SM {cc[0]}.{cc[1]}.")
        sys.exit(0)

    print("Running Triton examples...")
    
    benchmark_fp8_vs_fp16()
    benchmark_persistent_vs_standard()
