"""optimized_persistent_matmul_tma.py

Persistent matmul using Triton TMA (Tensor Memory Accelerator) on Blackwell GPUs.
Uses tensor descriptors for efficient asynchronous memory transfers.

This implementation uses the stable Triton 3.5+ TMA API:
- tl.make_tensor_descriptor() - Create TMA descriptors
- tl.load_tensor_descriptor() - Load tiles via TMA hardware
- tl.store_tensor_descriptor() - Store tiles via TMA hardware

Requires SM100 (Blackwell) or newer for TMA hardware acceleration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ch10.persistent_matmul_tma_common import persistent_program_count, persistent_tile_count
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin

try:
    import triton
    import triton.language as tl
    from triton.runtime import _allocation as triton_allocation
    TRITON_AVAILABLE = True
except ImportError as exc:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is required for this example") from exc

# Verify TMA tensor descriptor API is available
REQUIRED_TMA_FEATURES = ['make_tensor_descriptor', 'load_tensor_descriptor', 'store_tensor_descriptor']
_missing_features = [f for f in REQUIRED_TMA_FEATURES if not hasattr(tl, f)]
if _missing_features:
    raise RuntimeError(
        f"FAIL FAST: Triton {triton.__version__} is missing required TMA features: {_missing_features}. "
        f"This benchmark requires Triton 3.5+ with TMA tensor descriptor support."
    )


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
    """Set Triton's allocator once so TMA kernels can grab scratch buffers.
    
    Must be called AFTER CUDA is initialized (after first tensor creation).
    """
    if not torch.cuda.is_available():
        return
    current = triton_allocation._allocator.get()
    if isinstance(current, triton_allocation.NullAllocator):
        # Ensure CUDA context is initialized only when we actually need to install
        # the allocator in this thread context.
        torch.cuda.init()
        torch.cuda.current_stream()  # Force stream creation
        triton.set_allocator(_TorchCudaAllocator())


# Don't call at import time - let setup() handle it


# ============================================================================
# TMA Persistent Matmul Kernel
# ============================================================================

BLOCK_M = 128
BLOCK_N = 256
BLOCK_K = 64
# GB300 (sm_103) re-tune. The earlier B200-screened narrow config (BLOCK_N=128,
# GROUP_M=2, 4 warps) left the Blackwell tensor cores under-fed at 4096^3. A config
# sweep on GB300 favored a wider N-tile + more warps: a 5-trial A/B put this config
# at 1390 +/- 4 TFLOPS vs the old 1232 +/- 2 (1.128x), exact match vs torch.mm.
GROUP_M = 4
NUM_WARPS = 8
NUM_STAGES = 4

@triton.jit
def persistent_matmul_tma(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """TMA-backed persistent matmul kernel using tensor descriptors.

    Persistent scheduling: launch one program per SM and have each program
    iterate over multiple output tiles. This amortizes scheduling overhead and
    improves L2 locality while keeping the math identical to the baseline.
    """
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = grid_m * grid_n
    max_tiles_per_sm = (num_tiles + NUM_SMS - 1) // NUM_SMS
    width = GROUP_M * grid_n

    # Create TMA tensor descriptors for A and B
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

    # Iterate over all tiles assigned to this SM.
    for tile_iter in range(0, max_tiles_per_sm):
        tile_id = start_pid + tile_iter * NUM_SMS
        valid = tile_id < num_tiles
        safe_tile_id = tl.where(valid, tile_id, 0)

        group_id = safe_tile_id // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (safe_tile_id % group_size)
        pid_n = (safe_tile_id % width) // group_size
        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_tile = tl.load_tensor_descriptor(A_desc, [rm, k])
            b_tile = tl.load_tensor_descriptor(B_desc, [k, rn])
            acc += tl.dot(a_tile, b_tile)

        offs_m = rm + tl.arange(0, BLOCK_M)
        offs_n = rn + tl.arange(0, BLOCK_N)
        c_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = valid & (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr, acc.to(tl.float16), mask=mask)


def run_optimized(M=1024, N=1024, K=1024):
    """Run TMA-accelerated persistent matmul.
    
    Uses Triton tensor descriptors for efficient memory access on Blackwell.
    """
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    grid_sms = persistent_program_count(M, N, BLOCK_M, BLOCK_N, num_sms)
    grid = (grid_sms,)

    persistent_matmul_tma[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        GROUP_M=GROUP_M,
        NUM_SMS=grid_sms,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c


# ============================================================================
# Benchmark Harness Integration
# ============================================================================

class PersistentMatmulTMABenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark harness wrapper for TMA persistent matmul."""

    def __init__(self, M: int = 4096, N: int = 4096, K: int = 4096):
        super().__init__()
        self.a = None
        self.b = None
        self.c = None
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        # Match baseline dimensions for fair verification
        self.M = M
        self.N = N
        self.K = K
        self._last = 0.0
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.M * self.N),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize matrices and warmup TMA kernel."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Initialize CUDA context first - CRITICAL: must happen before Triton allocator
        torch.cuda.init()
        _ = torch.empty(1, device='cuda')  # Force CUDA context creation
        torch.cuda.current_stream()  # Ensure stream is ready
        
        # Set allocator BEFORE creating tensors (required for Triton TMA autotuning)
        _ensure_triton_allocator()
        
        if self.M % BLOCK_M != 0 or self.N % BLOCK_N != 0 or self.K % BLOCK_K != 0:
            raise RuntimeError(
                f"FAIL FAST: persistent_matmul_tma requires M/N/K divisible by {BLOCK_M}."
            )

        self.a = torch.randn(self.M, self.K, device=self.device, dtype=torch.float16)
        self.b = torch.randn(self.K, self.N, device=self.device, dtype=torch.float16)
        self.c = torch.empty(self.M, self.N, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty(
            min(128, self.M),
            min(256, self.N),
            device=self.device,
            dtype=torch.float32,
        )
        
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: TMA persistent matmul."""
        # Triton's allocator lives in a ContextVar (thread-local); the harness may
        # execute benchmark_fn inside watchdog threads, so ensure the allocator is
        # installed in the active thread context every call.
        _ensure_triton_allocator()
        num_sms = torch.cuda.get_device_properties(self.device.index or 0).multi_processor_count
        grid_sms = persistent_program_count(self.M, self.N, BLOCK_M, BLOCK_N, num_sms)
        grid = (grid_sms,)
        persistent_matmul_tma[grid](
            self.a, self.b, self.c,
            self.M, self.N, self.K,
            self.a.stride(0), self.a.stride(1),
            self.b.stride(0), self.b.stride(1),
            self.c.stride(0), self.c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            GROUP_M=GROUP_M,
            NUM_SMS=grid_sms,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        self.output = self.c
        if self.c is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.a is None or self.b is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"A": self.a, "B": self.b},
            output=self._verify_output_buffer,
            batch_size=self.M,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.05, 0.05),
        )

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        self.c = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return numeric workload metrics for the persistent TMA kernel."""
        from core.benchmark.metrics import compute_gemm_metrics

        num_sms = torch.cuda.get_device_properties(self.device.index or 0).multi_processor_count
        scheduled_programs = persistent_program_count(self.M, self.N, BLOCK_M, BLOCK_N, num_sms)
        tile_count = persistent_tile_count(self.M, self.N, BLOCK_M, BLOCK_N)
        metrics = compute_gemm_metrics(
            m=self.M,
            n=self.N,
            k=self.K,
            precision="fp16",
            bytes_per_element=2,
        )
        metrics["gemm.uses_tma"] = 1.0
        metrics["gemm.persistent_kernel"] = 1.0
        metrics["gemm.scheduled_programs"] = float(scheduled_programs)
        metrics["gemm.tile_count"] = float(tile_count)
        metrics["gemm.programs_per_sm"] = float(scheduled_programs / num_sms)
        metrics["gemm.num_stages"] = float(NUM_STAGES)
        metrics["gemm.num_warps"] = float(NUM_WARPS)
        return metrics

    def validate_result(self) -> Optional[str]:
        if not TRITON_AVAILABLE:
            return "Triton not available"
        if self.a is None or self.b is None or self.c is None:
            return "Input/output buffers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return PersistentMatmulTMABenchmark()
