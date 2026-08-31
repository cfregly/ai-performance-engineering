#!/usr/bin/env python3
"""Dump comprehensive hardware capabilities for the current system.

This script outputs all hardware capabilities including:
- GPU architecture and compute capability
- CUDA version and features
- CUTLASS availability
- Triton support
- CUDA extension compilation support
- Memory and bandwidth specifications
"""

from __future__ import annotations

import argparse

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.harness.hardware_capabilities import detect_capabilities, format_capability_report
from core.harness import arch_config  # noqa: E402
try:
    import importlib_metadata  # type: ignore[import-not-found]  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - stdlib fallback
    from importlib import metadata as importlib_metadata  # noqa: E402
import torch  # noqa: E402


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump GPU hardware capabilities.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow validation steps (CUDA extensions + torch.compile).",
    )
    return parser.parse_args()


def dump_gpu_info():
    """Dump GPU hardware information."""
    print_section("GPU Hardware")
    
    cap = detect_capabilities()
    if cap is None:
        print("ERROR: CUDA not available")
        return
    
    print(format_capability_report(cap))
    print("")
    print(f"Max Threads per Block: {cap.max_threads_per_block}")
    print(f"Max Threads per SM: {cap.max_threads_per_sm}")
    print(f"Warp Size: {cap.warp_size}")
    print(f"Shared Memory per Block: {cap.max_shared_mem_per_block / 1024:.1f} KB")
    print(f"Shared Memory per SM: {cap.max_shared_mem_per_sm / 1024:.1f} KB")
    l2_cache = f"{cap.l2_cache_kb:.1f} KB" if cap.l2_cache_kb else "Unknown"
    print(f"L2 Cache Size: {l2_cache}")


def dump_cuda_info():
    """Dump CUDA version and toolkit information."""
    print_section("CUDA Toolkit")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"CUDA Available: [OK]")
    
    # Check for CUDA toolkit components
    import subprocess
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nNVCC Compiler:")
            for line in result.stdout.split('\n')[:3]:
                if line.strip():
                    print(f"  {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("WARNING: NVCC not found (CUDA extensions may not compile)")


def dump_cutlass_info():
    """Dump CUTLASS backend information."""
    print_section("CUTLASS Backend")
    
    try:
        import cutlass
        print(f"[OK] CUTLASS module found: {cutlass.__file__}")
    except ImportError:
        print("ERROR: CUTLASS module not found")
        return
    
    try:
        version = importlib_metadata.version("nvidia-cutlass-dsl")
        print(f"[OK] nvidia-cutlass-dsl version: {version}")
        
        # Check version
        parts = [int(x) for x in version.split('.')]
        if len(parts) >= 2 and (parts[0] > 4 or (parts[0] == 4 and parts[1] >= 2)):
            print("[OK] CUTLASS 4.2+ detected (Blackwell support)")
        else:
            print("WARNING: CUTLASS < 4.2 (upgrade recommended for Blackwell)")
    except importlib_metadata.PackageNotFoundError:
        print("WARNING: nvidia-cutlass-dsl package not found")
    
    # Check torch.compile CUTLASS backend configuration
    try:
        import torch._inductor.config as config
        print(f"\ntorch.compile CUTLASS Configuration:")
        print(f"  max_autotune_gemm_backends: {getattr(config, 'max_autotune_gemm_backends', 'N/A')}")
        print(f"  cutlass_enabled_ops: {getattr(config.cuda, 'cutlass_enabled_ops', 'N/A')}")
        
        if hasattr(config.cuda, 'cutlass_dir'):
            cutlass_dir = getattr(config.cuda, 'cutlass_dir', None)
            if cutlass_dir:
                print(f"  cutlass_dir: {cutlass_dir}")
            else:
                print(f"  cutlass_dir: Not configured")
    except Exception as e:
        print(f"WARNING: Could not check torch.compile CUTLASS config: {e}")


def dump_triton_info():
    """Dump Triton compiler information."""
    print_section("Triton Compiler")
    
    try:
        import triton
        print(f"[OK] Triton version: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not available")
        return
    
    # Check for SM architecture patching
    try:
        arch_cfg = arch_config.ArchitectureConfig()
        print(f"[OK] Architecture detected: {arch_cfg.arch}")
        
        # Check if SM 12.1 patching is enabled
        patch_enabled = arch_config.ENABLE_TRITON_PATCH
        if patch_enabled:
            print("[OK] Triton SM architecture patching: ENABLED")
            print("   (Fixes sm_121a → sm_121 for GB10)")
        else:
            print("WARNING: Triton SM architecture patching: DISABLED")
    except Exception as e:
        print(f"WARNING: Could not check arch_config: {e}")


def dump_cuda_extensions_info(run_check: bool):
    """Dump CUDA extension compilation support."""
    print_section("CUDA Extensions")
    
    if not run_check:
        print("SKIPPED: CUDA extension build check disabled (--fast).")
        return
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - extensions cannot be compiled")
        return
    
    print("Testing CUDA extension compilation support...")
    
    try:
        from ch12.cuda_extensions import load_graph_bandwidth_extension
        try:
            ext = load_graph_bandwidth_extension()
            print("[OK] graph_bandwidth extension: Can compile and load")
            print(f"   Module: {ext}")
        except Exception as e:
            print(f"ERROR: graph_bandwidth extension: Compilation failed")
            print(f"   Error: {e}")
    except ImportError as e:
        print(f"WARNING: Could not import extension loader: {e}")


def dump_torch_compile_info(run_check: bool):
    """Dump torch.compile capabilities."""
    print_section("torch.compile Support")
    
    if not run_check:
        print("SKIPPED: torch.compile quick check disabled (--fast).")
        return
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - torch.compile requires GPU")
        return
    
    print(f"PyTorch Version: {torch.__version__}")
    
    print("\nTesting basic torch.compile...")
    try:
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        
        def test_fn(x):
            return torch.mm(x, x.t())
        
        compiled_fn = torch.compile(test_fn, mode="reduce-overhead")
        result = compiled_fn(x)
        
        print("[OK] torch.compile: Working")
        print(f"   Test result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: torch.compile: Failed")
        print(f"   Error: {e}")


def dump_summary():
    """Print summary of capabilities."""
    print_section("Capability Summary")
    
    cap = detect_capabilities()
    capabilities = []
    
    if cap is None:
        capabilities.append("ERROR: CUDA not available")
    else:
        capabilities.append(f"[OK] GPU: {cap.device_name} ({cap.compute_capability})")
        if cap.tma_ready:
            capabilities.append("[OK] TMA hardware + compiler support")
        elif cap.tma_supported:
            capabilities.append("WARNING: TMA hardware present but compiler support disabled")
        else:
            capabilities.append("WARNING: TMA unsupported on this GPU")
        if cap.cluster.has_dsmem:
            capabilities.append("[OK] Thread block clusters with DSMEM")
        elif cap.cluster.supports_clusters:
            capabilities.append("WARNING: Clusters available but DSMEM disabled")
        else:
            capabilities.append("WARNING: Thread block clusters unavailable")
    
    try:
        import cutlass
        capabilities.append("[OK] CUTLASS Backend Available")
    except ImportError:
        capabilities.append("ERROR: CUTLASS Backend Not Available")
    
    try:
        import triton
        capabilities.append(f"[OK] Triton Compiler (v{triton.__version__})")
    except ImportError:
        capabilities.append("ERROR: Triton Compiler Not Available")
    
    for cap in capabilities:
        print(cap)


def main() -> None:
    """Main function to dump all hardware capabilities."""
    args = parse_args()
    
    print("=" * 80)
    print("  Hardware Capabilities Report")
    print("=" * 80)
    
    dump_gpu_info()
    dump_cuda_info()
    dump_cutlass_info()
    dump_triton_info()
    dump_cuda_extensions_info(run_check=not args.fast)
    dump_torch_compile_info(run_check=not args.fast)
    dump_summary()
    
    print("\n" + "=" * 80)
    print("  End of Hardware Capabilities Report")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
