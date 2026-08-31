#!/usr/bin/env python3
"""
TMA Verification Script for Grace-Blackwell GB10
=================================================

This script runs bounded correctness and sample-build checks on GB10 (SM 12.1).
Compilation and torch.compile success do not establish TMA instruction use.
The checks cover:
1. CUDA C++ kernels
2. Triton kernels
3. PyTorch torch.compile

Usage:
    python -m core.verification.verify_tma_sm121

Requirements:
    - Grace-Blackwell GB10 GPU (SM 12.1)
    - CUDA 13.0+
    - Repository-pinned PyTorch and Triton stack
"""

import os
import sys

import subprocess
import torch
import triton
import triton.language as tl
from triton.runtime.errors import PTXASError
from typing import Any, Dict, List, Tuple

from core.harness.hardware_capabilities import (
    detect_capabilities,
    ensure_tma_box_supported,
    format_capability_report,
)

# Import architecture configuration
try:
    from core.harness.arch_config import ArchitectureConfig, configure_optimizations
    configure_optimizations()
except ImportError:
    print("WARNING: Warning: Could not import arch_config, continuing without optimizations")
    ArchitectureConfig = None


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def check_gpu_info() -> Dict[str, Any]:
    """Check GPU information and TMA support."""
    print_section("GPU Information")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return {"available": False}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    info = {
        "available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "major": props.major,
        "minor": props.minor,
        "total_memory": props.total_memory / (1024**3),  # GB
        "multi_processor_count": props.multi_processor_count,
    }
    
    cap = detect_capabilities()
    if cap:
        print(format_capability_report(cap))
        info["tma_supported"] = cap.tma_supported
        info["tma_ready"] = cap.tma_ready
        info["capabilities"] = cap
    else:
        print(f"GPU: {info['name']}")
        print(f"Compute Capability: {info['compute_capability']}")
        print(f"Total Memory: {info['total_memory']:.2f} GB")
        print(f"SM Count: {info['multi_processor_count']}")
        info["tma_supported"] = False
        info["tma_ready"] = False
        info["capabilities"] = None
    
    return info


def check_software_versions():
    """Check software versions."""
    print_section("Software Versions")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA Runtime: {torch.version.cuda}")
    
    # Check for CUDA compiler
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"NVCC: {line.strip()}")
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("WARNING: NVCC not found or not accessible")
    
    # Note: TMA is enabled automatically via compute capability and API usage.
    # No environment variables are required or checked.


def test_triton_tma_basic() -> str:
    """Test basic Triton TMA functionality."""
    print_section("Triton TMA Test")
    
    try:
        # Simple TMA copy kernel
        @triton.jit
        def tma_copy_kernel(
            input_ptr, output_ptr,
            N: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            """Simple copy kernel using TMA descriptors."""
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            
            # Create TMA descriptor for input
            input_desc = tl.make_tensor_descriptor(
                input_ptr,
                shape=[N],
                strides=[1],
                block_shape=[BLOCK_SIZE],
            )
            
            # Load using TMA
            data = input_desc.load([block_start])
            
            # Store directly (no TMA descriptor for output to simplify)
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N
            tl.store(output_ptr + offsets, data, mask=mask)
        
        # Test with small data
        N = 1024
        cap = detect_capabilities()
        BLOCK_SIZE = min(256, getattr(getattr(cap, "tma_limits", None), "max_1d_elements", 256) or 256)
        try:
            ensure_tma_box_supported(
                (BLOCK_SIZE,),
                capability=cap,
                description="verify_tma_sm121 basic copy",
            )
        except RuntimeError as exc:
            print("WARNING: Triton TMA basic copy: SKIPPED")
            print(f"   Reason: {exc}")
            return "skip"
        
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.zeros_like(x)
        
        grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
        tma_copy_kernel[grid](x, y, N, BLOCK_SIZE)
        
        # Verify correctness
        if torch.allclose(x, y, rtol=1e-3):
            print("Triton TMA basic copy: PASSED")
            print(f"  - Copied {N} elements using TMA descriptors")
            print(f"  - Block size: {BLOCK_SIZE}")
            return "pass"
        else:
            print("ERROR: Triton TMA basic copy: FAILED (incorrect results)")
            return "fail"
            
    except PTXASError as e:
        msg = str(e)
        if "tensormap.replace" in msg and "not supported on .target 'sm_121'" in msg:
            print("WARNING: Triton TMA basic copy: SKIPPED")
            print("   Reason: CUDA 13.0 ptxas does not enable tensormap instructions for sm_121.")
            print("   Action: upgrade to a CUDA toolkit that advertises TMA for SM 12.1.")
            return "skip"
        print(f"ERROR: Triton TMA basic copy: FAILED")
        print(f"   PTXAS error: {e}")
        return "fail"
    except Exception as e:
        print(f"ERROR: Triton TMA basic copy: FAILED")
        print(f"   Error: {e}")
        return "fail"


def test_triton_tma_gemm() -> str:
    """Test Triton TMA GEMM (conservative config due to Triton 3.5 bug)."""
    print_section("Triton TMA GEMM Test")
    
    try:
        # Conservative configuration to avoid Triton 3.5 compiler bug
        cap = detect_capabilities()
        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_K = 32
        limits = getattr(cap, "tma_limits", None)
        if limits:
            if limits.max_2d_width:
                BLOCK_M = min(BLOCK_M, limits.max_2d_width)
            if limits.max_2d_height:
                BLOCK_N = min(BLOCK_N, limits.max_2d_height)
            if limits.max_1d_elements:
                BLOCK_K = min(BLOCK_K, limits.max_1d_elements)
        try:
            ensure_tma_box_supported((BLOCK_M, BLOCK_K), capability=cap, description="verify_tma_sm121 GEMM A tile")
            ensure_tma_box_supported((BLOCK_K, BLOCK_N), capability=cap, description="verify_tma_sm121 GEMM B tile")
        except RuntimeError as exc:
            print("WARNING: Triton TMA GEMM: SKIPPED")
            print(f"   Reason: {exc}")
            return "skip"
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
            """Conservative TMA GEMM kernel."""
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            
            m0 = pid_m * BLOCK_M
            n0 = pid_n * BLOCK_N
            
            # Create TMA descriptors
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
            
            # Main loop
            for k0 in range(0, K, BLOCK_K):
                a = A_desc.load([m0, k0])
                b = B_desc.load([k0, n0])
                acc += tl.dot(a, b, out_dtype=tl.float32)
            
            # Store result (no TMA descriptor to avoid additional issues)
            offs_m = m0 + tl.arange(0, BLOCK_M)
            offs_n = n0 + tl.arange(0, BLOCK_N)
            c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)
        
        # Test with small matrices
        M, N, K = 512, 512, 512
        # Block sizes were clamped earlier based on hardware limits
        
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        tma_gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
        
        # Verify against PyTorch
        C_ref = torch.matmul(A, B)
        if torch.allclose(C, C_ref, rtol=1e-3, atol=1e-3):
            print("Triton TMA GEMM: PASSED")
            print(f"  - Matrix size: {M}x{K} @ {K}x{N}")
            print(f"  - Block config: {BLOCK_M}x{BLOCK_N}x{BLOCK_K} (conservative)")
            print(f"  - Note: Using conservative config due to Triton 3.5 compiler bug")
            return "pass"
        else:
            max_diff = torch.max(torch.abs(C - C_ref)).item()
            print(f"ERROR: Triton TMA GEMM: FAILED (max diff: {max_diff})")
            return "fail"
            
    except PTXASError as e:
        msg = str(e)
        if "tensormap.replace" in msg and "not supported on .target 'sm_121'" in msg:
            print("WARNING: Triton TMA GEMM: SKIPPED")
            print("   Reason: CUDA 13.0 ptxas does not enable tensormap instructions for sm_121.")
            print("   Action: upgrade to a CUDA toolkit that advertises TMA for SM 12.1.")
            return "skip"
        print(f"ERROR: Triton TMA GEMM: FAILED")
        print(f"   PTXAS error: {e}")
        return "fail"
    except Exception as e:
        print(f"ERROR: Triton TMA GEMM: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return "fail"


def test_pytorch_compile_tma() -> str:
    """Test PyTorch torch.compile with TMA engagement."""
    print_section("PyTorch torch.compile GEMM correctness (TMA instructions not inspected)")
    
    try:
        # Simple GEMM function
        def matmul_fn(a, b):
            return torch.matmul(a, b)
        
        # Autotuning may choose a library or generated kernel; inspect code separately for TMA.
        compiled_fn = torch.compile(matmul_fn, mode='max-autotune')
        
        # Test with moderate-sized matrices
        M, N, K = 1024, 1024, 1024
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(3):
            _ = compiled_fn(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(10):
            C = compiled_fn(A, B)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Verify correctness
        C_ref = torch.matmul(A, B)
        if torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2):
            tflops = (2 * M * N * K * 10) / (elapsed * 1e12)
            print("PyTorch torch.compile: PASSED")
            print(f"  - Matrix size: {M}x{K} @ {K}x{N}")
            print(f"  - Performance: {tflops:.2f} TFLOPS")
            print(f"  - Average time: {elapsed/10*1000:.2f} ms")
            print("  - Mode: max-autotune; TMA instruction use was not inspected")
            return "pass"
        else:
            print("ERROR: PyTorch torch.compile: FAILED (incorrect results)")
            return "fail"
            
    except Exception as e:
        print(f"ERROR: PyTorch torch.compile: FAILED")
        print(f"   Error: {e}")
        return "fail"


def test_cuda_tma_compilation() -> str:
    """Check real sample compilation/execution on the explicitly supported target."""
    print_section("CUDA TMA Sample Compilation and Execution")
    if not torch.cuda.is_available():
        print("SKIPPED: no CUDA device; SM121 sample cannot be validated")
        return "skip"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) != (12, 1):
        print(f"SKIPPED: requires SM121, observed SM{props.major}{props.minor}")
        return "skip"
    from core.verification.tma_cuda_probe import run_cuda_tma_sample
    return run_cuda_tma_sample()


def print_summary(results: Dict[str, str]):
    """Print test summary."""
    print_section("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v == "pass")
    failed = sum(1 for v in results.values() if v == "fail")
    skipped = sum(1 for v in results.values() if v == "skip")
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if skipped:
        print(f"Skipped: {skipped}")
    print()
    
    for test_name, result in results.items():
        if result == "pass":
            status = "PASS"
        elif result == "skip":
            status = "WARNING: SKIP"
        else:
            status = "ERROR: FAIL"
        print(f"  {status}: {test_name}")
    
    print()
    if failed == 0:
        if skipped:
            print("WARNING: Validation is incomplete because some checks were skipped.")
        else:
            print("All checks passed within their stated scope. Confirm TMA instruction use separately.")
    else:
        print("WARNING: Some tests failed. Review the output above for details.")
    
    return failed == 0 and skipped == 0


def main():
    """Main verification routine."""
    print("="*80)
    print("  TMA Verification for Grace-Blackwell GB10")
    print("="*80)
    
    # Check GPU info
    gpu_info = check_gpu_info()
    if not gpu_info.get('available'):
        print("\nERROR: Cannot proceed without CUDA GPU")
        return 1
    
    if (gpu_info.get("major"), gpu_info.get("minor")) != (12, 1):
        print("SKIPPED: this verifier requires an SM121 / GB10 target")
        return 3

    if not gpu_info.get('tma_supported'):
        print("\nERROR: GPU does not support TMA")
        return 1
    
    if not gpu_info.get('tma_ready'):
        print("\nERROR: TMA hardware present but compiler/toolchain support is disabled. "
              "Upgrade to a CUDA release that enables tensormap instructions for this architecture.")
        return 1
    
    # Check software versions
    check_software_versions()
    
    # Run tests
    results = {}
    
    # Test 1: Triton TMA basic
    results["Triton TMA Basic"] = test_triton_tma_basic()
    
    # Test 2: Triton TMA GEMM
    results["Triton TMA GEMM"] = test_triton_tma_gemm()
    
    # Test 3: PyTorch compile
    results["PyTorch torch.compile correctness (TMA uninspected)"] = test_pytorch_compile_tma()
    
    # Test 4: CUDA compilation
    results["CUDA sample compile/run (not full-output validation)"] = test_cuda_tma_compilation()
    
    # Print summary
    all_passed = print_summary(results)
    
    if all_passed:
        return 0
    return 1 if "fail" in results.values() else 3


if __name__ == "__main__":
    sys.exit(main())
