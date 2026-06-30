"""
triton_persistent_demo.py - Triton Persistent Kernels (Ch14)

WHAT: Persistent kernels keep thread blocks alive for multiple "tiles" of work,
rather than launching one block per tile.

WHY: Benefits include:
  - Reduced kernel launch overhead (1 launch vs thousands)
  - Better L2 cache reuse across tiles
  - Enables work-stealing for load balancing
  - Critical for small tiles where launch overhead dominates

PATTERN:
  Standard:  grid=(M/TILE_M, N/TILE_N) → one block per tile
  Persistent: grid=(NUM_SMS, 1) → each block processes many tiles via loop

WHEN TO USE:
  - GEMM with small tiles or small matrices
  - Operations where launch overhead is significant
  - When you need dynamic load balancing
  - StreamK-style algorithms

REQUIREMENTS:
  - Triton 2.0+
  - Understanding of tile iteration patterns
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from ch14.triton_persistent_batched import matmul_persistent_batched
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

#============================================================================
# Standard (Non-Persistent) GEMM Kernel
#============================================================================

@triton.jit
def gemm_standard_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard GEMM: one block per output tile."""
    
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output tile bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to first tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K-loop
    for k in range(0, K, BLOCK_K):
        # Load tiles
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


#============================================================================
# Persistent GEMM Kernel
#============================================================================

@triton.jit
def gemm_persistent_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile counts
    num_tiles_m, num_tiles_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent GEMM: each SM processes multiple tiles via work-stealing loop."""
    
    # Each program processes multiple tiles
    pid = tl.program_id(0)
    
    # Total number of tiles
    num_tiles = num_tiles_m * num_tiles_n
    
    # Work-stealing loop: process tiles in strided pattern
    # Each SM gets tiles: pid, pid + NUM_SMS, pid + 2*NUM_SMS, ...
    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Convert linear tile_id to 2D tile indices
        # Use swizzle pattern for better L2 cache reuse
        # Group tiles in clusters to maximize spatial locality
        GROUP_M: tl.constexpr = 8
        
        # Swizzled tile indexing
        group_id = tile_id // (GROUP_M * num_tiles_n)
        first_tile_m = group_id * GROUP_M
        group_size_m = min(num_tiles_m - first_tile_m, GROUP_M)
        
        tile_in_group = tile_id % (GROUP_M * num_tiles_n)
        pid_m = first_tile_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m
        
        # Compute output tile bounds
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Pointers to first K-tile
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K-loop
        for k in range(0, K, BLOCK_K):
            mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
            mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
            
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            
            acc += tl.dot(a, b)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        # Store output
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask_c)


#============================================================================
# Persistent GEMM with Atomic Work Queue
#============================================================================

@triton.jit
def gemm_persistent_atomic_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    work_counter_ptr,  # Atomic counter for work distribution
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile counts
    num_tiles_m, num_tiles_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_TILES_PER_SM: tl.constexpr,
):
    """Persistent GEMM with atomic work queue for perfect load balancing.
    
    Instead of static strided assignment, each SM atomically claims the next
    available tile. This provides optimal load balancing when tile costs vary.
    
    Note: Triton doesn't support 'break' in while loops, so we use a for loop
    with a maximum iteration count and early-exit via conditional computation.
    """
    
    num_tiles = num_tiles_m * num_tiles_n
    
    # Persistent loop - use for loop since Triton doesn't support break in while
    for _ in range(MAX_TILES_PER_SM):
        # Atomically claim next tile
        tile_id = tl.atomic_add(work_counter_ptr, 1)
        
        # Check if all work is done - use tl.where for conditional execution
        valid_tile = tile_id < num_tiles
        
        # Only process if we have a valid tile
        # Convert to 2D indices (simple row-major for atomic version)
        pid_m = tl.where(valid_tile, tile_id // num_tiles_n, 0)
        pid_n = tl.where(valid_tile, tile_id % num_tiles_n, 0)
        
        # Compute output tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            mask_a = valid_tile & (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
            mask_b = valid_tile & (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
            
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            
            acc += tl.dot(a, b)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = valid_tile & (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask_c)


#============================================================================
# Wrapper Functions
#============================================================================

def matmul_standard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard GEMM launch."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    gemm_standard_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, num_sms: int = 108) -> torch.Tensor:
    """Persistent GEMM launch."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    
    # Launch only NUM_SMS blocks - they will process all tiles
    grid = (num_sms,)
    
    gemm_persistent_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_tiles_m, num_tiles_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=num_sms,
    )
    
    return c


def matmul_persistent_atomic(a: torch.Tensor, b: torch.Tensor, num_sms: int = 108) -> torch.Tensor:
    """Persistent GEMM with atomic work queue."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n
    
    # Max tiles any single SM might process (generous upper bound)
    max_tiles_per_sm = (num_tiles + num_sms - 1) // num_sms + 16
    
    # Atomic counter for work distribution
    work_counter = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    grid = (num_sms,)
    
    gemm_persistent_atomic_kernel[grid](
        a, b, c,
        work_counter,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_tiles_m, num_tiles_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MAX_TILES_PER_SM=max_tiles_per_sm,
    )
    
    return c


#============================================================================
# Benchmark
#============================================================================

def benchmark_kernels():
    """Compare standard vs persistent GEMM."""
    print("Triton Persistent Kernel Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    
    print(f"Device: {props.name}")
    print(f"SMs: {num_sms}")
    print()
    
    # Test various sizes
    sizes = [
        (512, 512, 512),    # Small - launch overhead dominates
        (1024, 1024, 1024), # Medium
        (2048, 2048, 2048), # Large - compute dominates
        (4096, 4096, 4096), # Very large
    ]
    
    print(f"{'Size':<20} {'Standard':<12} {'Persistent':<12} {'Atomic':<12} {'Speedup':<10}")
    print("-" * 66)
    
    for M, N, K in sizes:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = matmul_standard(a, b)
            _ = matmul_persistent(a, b, num_sms)
        torch.cuda.synchronize()
        
        # Benchmark standard
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(device)
        
        start.record(current_stream)
        for _ in range(20):
            _ = matmul_standard(a, b)
        end.record(current_stream)
        end.synchronize()
        
        ms_standard = start.elapsed_time(end) / 20
        
        # Benchmark persistent
        start.record(current_stream)
        for _ in range(20):
            _ = matmul_persistent(a, b, num_sms)
        end.record(current_stream)
        end.synchronize()
        
        ms_persistent = start.elapsed_time(end) / 20
        
        # Benchmark atomic
        start.record(current_stream)
        for _ in range(20):
            _ = matmul_persistent_atomic(a, b, num_sms)
        end.record(current_stream)
        end.synchronize()
        
        ms_atomic = start.elapsed_time(end) / 20
        
        # Speedup
        speedup = ms_standard / ms_persistent
        
        size_str = f"{M}x{N}x{K}"
        print(f"{size_str:<20} {ms_standard:<12.3f} {ms_persistent:<12.3f} {ms_atomic:<12.3f} {speedup:<10.2f}x")
    
    print()
    print("Note: Persistent kernels shine for small/medium sizes where")
    print("launch overhead is significant. For large sizes, the benefit diminishes.")
    print("Atomic version provides better load balancing but has atomic overhead.")
    
    # Verify correctness
    print("\nVerifying correctness...")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    
    ref = torch.mm(a, b)
    c_std = matmul_standard(a, b)
    c_pers = matmul_persistent(a, b, num_sms)
    c_atom = matmul_persistent_atomic(a, b, num_sms)
    
    error_stats = torch.empty(3, device=ref.device, dtype=torch.float32)
    error_stats[0].copy_((c_std - ref).abs().max())
    error_stats[1].copy_((c_pers - ref).abs().max())
    error_stats[2].copy_((c_atom - ref).abs().max())
    error_stats_host = error_stats.detach().cpu()
    std_err = float(error_stats_host[0])
    pers_err = float(error_stats_host[1])
    atom_err = float(error_stats_host[2])
    
    print(f"Standard max error: {std_err:.6f}")
    print(f"Persistent max error: {pers_err:.6f}")
    print(f"Atomic max error: {atom_err:.6f}")
    

#============================================================================
# Benchmark Harness Integration
#============================================================================

class TritonPersistentDemoBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for Triton persistent kernels demo."""

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.output = None
        self._output_buffer = None
        self.num_sms = 0
        # Match baseline dimensions for fair comparison (baseline uses batch_size=32, M=N=K=256)
        self.batch_size = 32
        self.M = 256
        self.N = 256
        self.K = 256
        self._last = 0.0
        self._verification_payload = None
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize matrices and detect SM count."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        props = torch.cuda.get_device_properties(self.device)
        self.num_sms = props.multi_processor_count
        
        self.a = torch.randn(self.batch_size, self.M, self.K, device=self.device, dtype=torch.float16)
        self.b = torch.randn(self.batch_size, self.K, self.N, device=self.device, dtype=torch.float16)
        self._output_buffer = torch.empty(
            (self.batch_size, self.M, self.N), device=self.device, dtype=torch.float16
        )
        
        # Warmup
        for _ in range(3):
            _ = matmul_persistent_batched(self.a, self.b, self.num_sms, out=self._output_buffer)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Persistent GEMM kernel."""
        self.output = matmul_persistent_batched(self.a, self.b, self.num_sms, out=self._output_buffer)
        if self.output is None or self.a is None or self.b is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        if self.a is None or self.b is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return TritonPersistentDemoBenchmark()
