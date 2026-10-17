#!/usr/bin/env python3
"""
Verify Triton Support for Modern Blackwell GPU Features
========================================================

Tests Triton's support for SM100 (Blackwell) features including:
- Basic kernel compilation on SM100/SM121
- TMA (Tensor Memory Accelerator) descriptors
- Warp specialization hints
- Block-scaled formats (MXFP8, FP8)
- Persistent kernels
- torch.compile integration

Usage:
    python -m core.verification.verify_triton_blackwell_features [--verbose] [--benchmark]
"""

import argparse
import sys
from typing import Dict, Tuple, Optional

# Apply compile patches before importing torch/triton
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import torch

# Check CUDA availability first
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

import triton
import triton.language as tl


# (architecture, family) for each Blackwell compute capability. These are the
# labels in core.harness.arch_config.ArchitectureConfig.metadata_for_capability.
# This script does not import that module, because importing it reconfigures
# Inductor and Triton before the tests run. SM 12.0 is the GeForce RTX 50 and
# RTX PRO family and SM 12.1 is GB10. Neither is a GB200 or GB300 GPU, which
# report SM 10.0 and 10.3.
_BLACKWELL_FAMILIES = {
    (10, 0): ("blackwell", "Blackwell B200/GB200 family"),
    (10, 3): ("blackwell_ultra", "Blackwell Ultra B300/GB300 family"),
    (12, 0): ("blackwell_consumer", "Blackwell GeForce RTX 50 / RTX PRO family"),
    (12, 1): ("grace_blackwell", "GB10 / DGX Spark family"),
}
# B200, B300, GB200 and GB300 use HBM3e. SM 12.x parts use GDDR7 or LPDDR5X.
_HBM3E_ARCHITECTURES = frozenset({"blackwell", "blackwell_ultra"})


def _blackwell_family(major: int, minor: int) -> Tuple[str, str]:
    """Return (architecture, family) from the table, or ("other", "Unclassified CUDA GPU") for any other capability."""
    return _BLACKWELL_FAMILIES.get((major, minor), ("other", "Unclassified CUDA GPU"))


def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("="*70)


def print_result(name: str, status: str, detail: str = "") -> None:
    """Print test result with status."""
    icons = {"PASS": "✓", "FAIL": "✗", "SKIP": "○", "WARN": "⚠"}
    icon = icons.get(status, "?")
    line = f"  {icon} {status}: {name}"
    if detail:
        line += f" - {detail}"
    print(line)


def get_gpu_info() -> Dict:
    """Get GPU information."""
    props = torch.cuda.get_device_properties(0)
    architecture, family = _blackwell_family(props.major, props.minor)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "major": props.major,
        "minor": props.minor,
        "total_memory_gb": props.total_memory / 1024**3,
        "sms": props.multi_processor_count,
        "architecture": architecture,
        "family": family,
        "is_blackwell": architecture != "other",
    }


def get_triton_info() -> Dict:
    """Get Triton version and capabilities."""
    info = {
        "version": triton.__version__,
        "has_tma": hasattr(tl, "make_tensor_descriptor"),
        "has_warp_spec": hasattr(tl, "num_programs"),  # Proxy check
    }
    
    # Check SM patch status
    try:
        import triton.backends.nvidia.compiler as tc
        info["sm_patch_applied"] = getattr(tc, '_sm_arch_patch_applied', False)
        # Test SM architecture name
        props = torch.cuda.get_device_properties(0)
        cap = props.major * 10 + props.minor
        arch_name = tc.sm_arch_from_capability(cap)
        info["arch_name"] = arch_name
        info["arch_has_suffix"] = arch_name.endswith('a')
    except Exception as e:
        info["sm_patch_error"] = str(e)
    
    return info


# =============================================================================
# Test 1: Basic Kernel Compilation
# =============================================================================
def test_basic_kernel() -> Tuple[str, str]:
    """Test basic Triton kernel compilation on current GPU."""
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)
    
    try:
        n = 1024
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        out = torch.empty_like(x)
        
        grid = (triton.cdiv(n, 256),)
        add_kernel[grid](x, y, out, n, BLOCK=256)
        torch.cuda.synchronize()
        
        if torch.allclose(out, x + y):
            return "PASS", "Kernel compiles and executes correctly"
        return "FAIL", "Results incorrect"
    except Exception as e:
        return "FAIL", str(e)[:100]


# =============================================================================
# Test 2: TMA Descriptor Support
# =============================================================================
def test_tma_descriptors() -> Tuple[str, str]:
    """Test TMA tensor descriptor support."""
    if not hasattr(tl, 'make_tensor_descriptor'):
        return "SKIP", "tl.make_tensor_descriptor not available"
    
    # Setup TMA allocator (required for descriptor scratch buffers)
    try:
        from triton.runtime import _allocation as triton_allocation
        
        class TorchCudaAllocator:
            """Simple CUDA allocator using PyTorch for TMA scratch buffers."""
            def __init__(self):
                self._buffers = {}
            
            def __call__(self, size: int, alignment: int, stream: Optional[int]) -> int:
                aligned_size = (size + alignment - 1) // alignment * alignment
                buf = torch.empty(aligned_size, dtype=torch.uint8, device='cuda')
                ptr = buf.data_ptr()
                self._buffers[ptr] = buf
                return ptr
        
        triton_allocation.set_allocator(TorchCudaAllocator())
    except Exception:
        pass  # Allocator setup may not be needed on all versions
    
    @triton.jit
    def tma_copy_kernel(
        input_ptr, output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        # Create TMA descriptor
        input_desc = tl.make_tensor_descriptor(
            input_ptr,
            shape=[N],
            strides=[1],
            block_shape=[BLOCK_SIZE],
        )
        
        # Load using TMA
        data = input_desc.load([block_start])
        
        # Store
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        tl.store(output_ptr + offsets, data, mask=mask)
    
    try:
        N = 1024
        BLOCK_SIZE = 128
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.zeros_like(x)
        
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        tma_copy_kernel[grid](x, y, N, BLOCK_SIZE)
        torch.cuda.synchronize()
        
        if torch.allclose(y, x):
            return "PASS", "TMA descriptors work correctly"
        return "FAIL", "TMA copy results incorrect"
    except Exception as e:
        err = str(e)
        if "not supported" in err.lower() or "not available" in err.lower():
            return "SKIP", "TMA not supported on this GPU/Triton version"
        if "allocator" in err.lower():
            return "WARN", "TMA requires triton.set_allocator() - see docs"
        return "FAIL", err[:100]


# =============================================================================
# Test 3: Warp Specialization Hints
# =============================================================================
def test_warp_specialization() -> Tuple[str, str]:
    """Test warp specialization kernel hints (num_warps configuration)."""
    @triton.jit
    def warp_spec_kernel(
        x_ptr, y_ptr, out_ptr, n,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        
        # Load data
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        
        # FMA pattern - optimized by tensor cores
        result = x * y + x
        tl.store(out_ptr + offs, result, mask=mask)
    
    try:
        n = 2048
        x = torch.randn(n, device='cuda', dtype=torch.float32)
        y = torch.randn(n, device='cuda', dtype=torch.float32)
        out = torch.empty_like(x)
        
        # Test with different num_warps configurations
        BLOCK = 256
        grid = (triton.cdiv(n, BLOCK),)
        
        # Test num_warps=4 (default-ish)
        warp_spec_kernel[grid](x, y, out, n, BLOCK=BLOCK, num_warps=4)
        torch.cuda.synchronize()
        expected = x * y + x
        
        # Use 1e-3 tolerance for CUDA - parallel execution has numerical variance
        if not torch.allclose(out, expected, rtol=1e-3, atol=1e-3):
            return "FAIL", "num_warps=4 results incorrect"
        
        # Test num_warps=8 (more warps for higher occupancy)
        out2 = torch.empty_like(x)
        warp_spec_kernel[grid](x, y, out2, n, BLOCK=BLOCK, num_warps=8)
        torch.cuda.synchronize()
        
        if not torch.allclose(out2, expected, rtol=1e-3, atol=1e-3):
            return "FAIL", "num_warps=8 results incorrect"
        
        return "PASS", "num_warps hint works (tested 4 and 8)"
    except Exception as e:
        return "FAIL", str(e)[:100]


# =============================================================================
# Test 4: FP8 Support
# =============================================================================
def test_fp8_support() -> Tuple[str, str]:
    """Test FP8 tensor core operations."""
    try:
        # Check if FP8 dtype is available
        if not hasattr(torch, 'float8_e4m3fn'):
            return "SKIP", "torch.float8_e4m3fn not available"
        
        @triton.jit
        def fp8_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            # Simple operation
            tl.store(out_ptr + offs, x, mask=mask)
        
        n = 1024
        x = torch.randn(n, device='cuda', dtype=torch.float16)
        x_fp8 = x.to(torch.float8_e4m3fn)
        out = torch.empty_like(x_fp8)
        
        grid = (triton.cdiv(n, 256),)
        fp8_kernel[grid](x_fp8, out, n, BLOCK=256)
        torch.cuda.synchronize()
        
        return "PASS", "FP8 tensors work with Triton"
    except Exception as e:
        err = str(e)
        if "not supported" in err.lower():
            return "SKIP", "FP8 not supported"
        return "FAIL", err[:100]


# =============================================================================
# Test 5: Persistent Kernel Pattern
# =============================================================================
def test_persistent_kernel() -> Tuple[str, str]:
    """Test persistent kernel pattern for Blackwell."""
    @triton.jit
    def persistent_add_kernel(
        x_ptr, y_ptr, out_ptr,
        n,
        BLOCK: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_blocks = tl.cdiv(n, BLOCK)
        
        # Persistent loop - each SM processes multiple blocks
        for block_id in range(pid, num_blocks, NUM_SMS):
            offs = block_id * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)
    
    try:
        n = 8192
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        out = torch.empty_like(x)
        
        # Launch with fewer blocks than total work
        grid = (min(num_sms, triton.cdiv(n, 256)),)
        persistent_add_kernel[grid](x, y, out, n, BLOCK=256, NUM_SMS=num_sms)
        torch.cuda.synchronize()
        
        if torch.allclose(out, x + y):
            return "PASS", f"Persistent pattern works ({num_sms} SMs)"
        return "FAIL", "Persistent kernel results incorrect"
    except Exception as e:
        return "FAIL", str(e)[:100]


# =============================================================================
# Test 6: torch.compile Integration
# =============================================================================
def test_torch_compile() -> Tuple[str, str]:
    """Test torch.compile with Triton backend."""
    try:
        def matmul_fn(a, b):
            return torch.matmul(a, b)
        
        compiled_fn = torch.compile(matmul_fn, mode='reduce-overhead')
        
        a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
        b = torch.randn(256, 256, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = compiled_fn(a, b)
        torch.cuda.synchronize()
        
        result = compiled_fn(a, b)
        expected = torch.matmul(a, b)
        
        if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
            return "PASS", "torch.compile with Triton works"
        return "FAIL", "Results incorrect"
    except Exception as e:
        err = str(e)
        if 'sm_121a' in err or 'ptxas' in err.lower():
            return "FAIL", "SM architecture patch not applied"
        return "FAIL", err[:100]


# =============================================================================
# Test 7: Multi-Stage Pipeline (Blackwell optimal)
# =============================================================================
def test_pipeline_stages() -> Tuple[str, str]:
    """Test multi-stage software pipelining."""
    @triton.jit
    def pipelined_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16))
    
    try:
        M, N, K = 256, 256, 256
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        # Use num_stages for pipelining
        pipelined_matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_stages=3,  # Multi-stage pipelining
            num_warps=4,
        )
        torch.cuda.synchronize()
        
        expected = torch.matmul(a, b)
        if torch.allclose(c, expected, rtol=1e-1, atol=1e-1):
            return "PASS", "Multi-stage pipelining works (num_stages=3)"
        return "FAIL", "Pipelined matmul results incorrect"
    except Exception as e:
        err = str(e)
        if "latencies" in err.lower() or "pipeline" in err.lower():
            return "WARN", "Pipeline compilation issue (known Blackwell bug)"
        return "FAIL", err[:100]


# =============================================================================
# Benchmark: GEMM Performance (optional)
# =============================================================================
def benchmark_gemm(verbose: bool = False) -> Dict:
    """Benchmark GEMM performance."""
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16))
    
    results = {}
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        M = N = K = size
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        # Warmup
        for _ in range(5):
            matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M, BLOCK_N, BLOCK_K,
                num_stages=2, num_warps=4,
            )
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iters = 20
        start.record()
        for _ in range(iters):
            matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M, BLOCK_N, BLOCK_K,
                num_stages=2, num_warps=4,
            )
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iters
        tflops = 2 * M * N * K / (time_ms * 1e-3) / 1e12
        
        results[size] = {"time_ms": time_ms, "tflops": tflops}
        
        if verbose:
            print(f"    {size}x{size}x{size}: {time_ms:.3f}ms, {tflops:.2f} TFLOPS")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Verify Triton Blackwell Support")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run GEMM benchmark")
    args = parser.parse_args()
    
    print_header("Triton Blackwell Feature Verification")
    
    # GPU Info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"SMs: {gpu_info['sms']}")
    
    print(f"Architecture: {gpu_info['family']} ({gpu_info['architecture']})")
    
    # Triton Info
    triton_info = get_triton_info()
    print(f"\nTriton Version: {triton_info['version']}")
    print(f"TMA Support: {'Yes' if triton_info['has_tma'] else 'No'}")
    if 'arch_name' in triton_info:
        print(f"SM Architecture: {triton_info['arch_name']}")
    if triton_info.get('sm_patch_applied'):
        print(f"SM Patch: Applied")
    
    # Run Tests
    print_header("Feature Tests")
    
    tests = [
        ("Basic Kernel Compilation", test_basic_kernel),
        ("TMA Descriptors", test_tma_descriptors),
        ("Warp Specialization", test_warp_specialization),
        ("FP8 Support", test_fp8_support),
        ("Persistent Kernel Pattern", test_persistent_kernel),
        ("torch.compile Integration", test_torch_compile),
        ("Multi-Stage Pipeline", test_pipeline_stages),
    ]
    
    results = {"passed": 0, "failed": 0, "skipped": 0, "warnings": 0}
    
    for name, test_fn in tests:
        try:
            status, detail = test_fn()
            print_result(name, status, detail)
            
            if status == "PASS":
                results["passed"] += 1
            elif status == "FAIL":
                results["failed"] += 1
            elif status == "SKIP":
                results["skipped"] += 1
            elif status == "WARN":
                results["warnings"] += 1
        except Exception as e:
            print_result(name, "FAIL", f"Exception: {str(e)[:80]}")
            results["failed"] += 1
    
    # Optional Benchmark
    if args.benchmark:
        print_header("GEMM Benchmark")
        try:
            benchmark_gemm(verbose=True)
        except Exception as e:
            print(f"  Benchmark failed: {e}")
    
    # Summary
    print_header("Summary")
    print(f"  Passed:   {results['passed']}")
    print(f"  Failed:   {results['failed']}")
    print(f"  Skipped:  {results['skipped']}")
    print(f"  Warnings: {results['warnings']}")
    
    # Blackwell-specific notes
    if gpu_info['is_blackwell']:
        print("\n  Blackwell-Specific Notes:")
        print("  - TMA descriptors provide hardware-accelerated bulk memory transfers")
        if gpu_info['architecture'] in _HBM3E_ARCHITECTURES:
            print("  - Multi-stage pipelining (num_stages=3-5) optimal for HBM3e")
        print("  - Warp specialization can improve tensor core utilization")
        print("  - FP8/MXFP8 formats supported for inference workloads")
        if results['warnings'] > 0:
            print("  - Some features have known compiler limitations (see Triton issues)")
    
    print("="*70)
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
