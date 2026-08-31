#!/usr/bin/env python3
"""Pre-compile CUDA extensions to avoid runtime segfaults.

This script compiles all CUDA extensions before running benchmarks,
ensuring hardware compatibility is checked upfront rather than during
benchmark execution.
"""

import sys

from core.harness.arch_config import configure_optimizations as _configure_arch_optimizations

_configure_arch_optimizations()

import torch

def precompile_extensions():
    """Pre-compile all CUDA extensions."""
    print("=" * 80)
    print("Pre-compiling CUDA Extensions")
    print("=" * 80)
    print()
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - skipping extension compilation")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    success = True
    compiled_count = 0
    failed_count = 0
    
    # Pre-compile ch06 extensions
    print("Compiling ch06 CUDA extensions...")
    try:
        from ch06.cuda_extensions import (
            load_bank_conflicts_extension,
            load_coalescing_extension,
            load_ilp_extension,
            load_launch_bounds_extension,
        )
        
        extensions = [
            ("bank_conflicts", load_bank_conflicts_extension),
            ("coalescing", load_coalescing_extension),
            ("ilp", load_ilp_extension),
            ("launch_bounds", load_launch_bounds_extension),
        ]
        
        for name, load_func in extensions:
            try:
                print(f"  Compiling {name} extension...")
                ext = load_func()
                print(f"    [OK] {name} extension compiled successfully")
                compiled_count += 1
            except Exception as e:
                print(f"    ERROR: Failed to compile {name} extension: {e}")
                failed_count += 1
                success = False
    except ImportError as e:
        print(f"  WARNING: Could not import ch06 extensions: {e}")
        print("    ch06 benchmarks may compile extensions at runtime")
    
    # Pre-compile ch12 extensions
    print("\nCompiling ch12 CUDA extensions...")
    try:
        from ch12.cuda_extensions import load_graph_bandwidth_extension
        
        try:
            print("  Compiling graph_bandwidth extension...")
            ext = load_graph_bandwidth_extension()
            print("    [OK] graph_bandwidth extension compiled successfully")
            compiled_count += 1
        except Exception as e:
            print(f"    ERROR: Failed to compile graph_bandwidth extension: {e}")
            failed_count += 1
            success = False
    except ImportError as e:
        print(f"  WARNING: Could not import ch12 extensions: {e}")
        print("    ch12 benchmarks may compile extensions at runtime")
    
    print()
    print(f"Summary: {compiled_count} compiled, {failed_count} failed")
    
    if success and compiled_count > 0:
        print("[OK] All CUDA extensions pre-compiled successfully")
        print("   Extensions are now cached and ready for use")
    elif compiled_count > 0:
        print("WARNING: Some extensions failed to compile")
        print("   Benchmarks using these extensions may fail or compile at runtime")
    else:
        print("INFO: No extensions were compiled (may not be needed)")
    
    return success


if __name__ == "__main__":
    success = precompile_extensions()
    sys.exit(0 if success else 1)
