#!/usr/bin/env python3
"""
Verify CUTLASS setup for Blackwell (SM100a) compatibility.

This script checks:
1. CUTLASS version in third_party/cutlass is >= 4.5.2
2. TransformerEngine's bundled CUTLASS is symlinked to the top-level CUTLASS
3. SM100a-specific headers exist
4. No duplicate/conflicting CUTLASS installations

Run this after setup.sh or whenever encountering CUTLASS-related build failures.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


class CutlassInfo(NamedTuple):
    path: Path
    version: tuple[int, int, int]
    is_symlink: bool
    symlink_target: Path | None
    has_sm100_headers: bool


def parse_cutlass_version(version_h: Path) -> tuple[int, int, int] | None:
    """Extract CUTLASS version from version.h."""
    if not version_h.exists():
        return None
    
    content = version_h.read_text()
    major = minor = patch = 0
    
    for line in content.splitlines():
        if match := re.match(r'#define\s+CUTLASS_MAJOR\s+(\d+)', line):
            major = int(match.group(1))
        elif match := re.match(r'#define\s+CUTLASS_MINOR\s+(\d+)', line):
            minor = int(match.group(1))
        elif match := re.match(r'#define\s+CUTLASS_PATCH\s+(\d+)', line):
            patch = int(match.group(1))
    
    return (major, minor, patch)


def check_sm100_headers(cutlass_include: Path) -> bool:
    """Check if critical SM100a headers exist."""
    required_headers = [
        "cute/arch/tmem_allocator_sm100.hpp",
        "cute/arch/mma_sm100_umma.hpp",
        "cute/atom/copy_traits_sm100.hpp",
        "cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp",
    ]
    
    for header in required_headers:
        if not (cutlass_include / header).exists():
            return False
    return True


def analyze_cutlass_path(path: Path) -> CutlassInfo | None:
    """Analyze a CUTLASS installation directory."""
    if not path.exists():
        return None
    
    is_symlink = path.is_symlink()
    symlink_target = path.resolve() if is_symlink else None
    
    include_dir = path / "include"
    version_h = include_dir / "cutlass" / "version.h"
    
    version = parse_cutlass_version(version_h)
    if version is None:
        return None
    
    has_sm100 = check_sm100_headers(include_dir)
    
    return CutlassInfo(
        path=path,
        version=version,
        is_symlink=is_symlink,
        symlink_target=symlink_target,
        has_sm100_headers=has_sm100,
    )


def main() -> int:
    # Find project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    third_party = project_root / "third_party"
    
    print("=" * 70)
    print("CUTLASS Setup Verification for Blackwell (SM100a)")
    print("=" * 70)
    print()
    
    issues: list[str] = []
    warnings: list[str] = []
    
    # Check main CUTLASS installation
    main_cutlass = third_party / "cutlass"
    main_info = analyze_cutlass_path(main_cutlass)
    
    print("1. Main CUTLASS (third_party/cutlass)")
    print("-" * 40)
    if main_info:
        ver_str = f"{main_info.version[0]}.{main_info.version[1]}.{main_info.version[2]}"
        print(f"   Version: {ver_str}")
        print(f"   SM100 Headers: {'✓ Present' if main_info.has_sm100_headers else '✗ MISSING'}")
        
        if main_info.version < (4, 5, 2):
            issues.append(f"Main CUTLASS is {ver_str}, need >= 4.5.2 for the validated Blackwell setup")
        
        if not main_info.has_sm100_headers:
            issues.append("Main CUTLASS is missing SM100a headers")
    else:
        print("   ✗ NOT FOUND")
        issues.append("Main CUTLASS not found at third_party/cutlass")
    
    print()
    
    # Check TransformerEngine's CUTLASS
    te_cutlass = third_party / "TransformerEngine" / "3rdparty" / "cutlass"
    te_info = analyze_cutlass_path(te_cutlass)
    
    print("2. TransformerEngine CUTLASS (TransformerEngine/3rdparty/cutlass)")
    print("-" * 40)
    if te_info:
        ver_str = f"{te_info.version[0]}.{te_info.version[1]}.{te_info.version[2]}"
        print(f"   Version: {ver_str}")
        print(f"   Symlink: {'Yes -> ' + str(te_info.symlink_target) if te_info.is_symlink else 'No (standalone copy)'}")
        print(f"   SM100 Headers: {'✓ Present' if te_info.has_sm100_headers else '✗ MISSING'}")
        
        if not te_info.is_symlink:
            warnings.append("TE CUTLASS is not a symlink - may diverge from main CUTLASS")
        
        if te_info.is_symlink and te_info.symlink_target != main_cutlass.resolve():
            issues.append(f"TE CUTLASS symlink points to wrong target: {te_info.symlink_target}")
        
        if not te_info.has_sm100_headers:
            issues.append("TE CUTLASS is missing SM100a headers - TransformerEngine builds will fail on Blackwell")
        
        if main_info and te_info.version != main_info.version:
            if te_info.is_symlink:
                warnings.append(f"Version mismatch but symlink exists - may be stale analysis")
            else:
                issues.append(f"TE CUTLASS ({ver_str}) differs from main CUTLASS - potential conflicts")
    else:
        print("   ✗ NOT FOUND or BROKEN")
        if (third_party / "TransformerEngine").exists():
            issues.append("TransformerEngine exists but its CUTLASS is missing/broken")
        else:
            warnings.append("TransformerEngine not installed")
    
    print()
    
    # Check for nvidia-cutlass-dsl (pip package)
    print("3. Python Package (nvidia-cutlass-dsl)")
    print("-" * 40)
    try:
        import cutlass
        pip_version = getattr(cutlass, '__version__', 'unknown')
        pip_path = Path(cutlass.__file__).parent
        print(f"   Version: {pip_version}")
        print(f"   Path: {pip_path}")
        
        # Check if pip package version matches source
        if main_info and pip_version != 'unknown':
            main_ver = f"{main_info.version[0]}.{main_info.version[1]}.{main_info.version[2]}"
            if pip_version != main_ver and not pip_version.startswith(main_ver):
                warnings.append(f"Pip CUTLASS ({pip_version}) differs from source ({main_ver})")
    except ImportError:
        print("   Not installed (nvidia-cutlass-dsl)")
        warnings.append("nvidia-cutlass-dsl pip package not installed - torch.compile CUTLASS backend unavailable")
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if issues:
        print("\n❌ ISSUES (must fix):")
        for issue in issues:
            print(f"   • {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed! CUTLASS setup is correct for Blackwell.")
    elif not issues:
        print("\n✓ No critical issues. Warnings above are informational.")
    
    print()
    
    # Recommendations
    if issues:
        print("RECOMMENDED FIXES:")
        print("-" * 40)
        
        if any("TE CUTLASS" in i and "symlink" in i.lower() for i in issues) or \
           any("SM100" in i and "TE" in i for i in issues):
            print("""
1. Re-create the TE CUTLASS symlink:
   
   cd /mnt/dev-fin-03/ai-performance-engineering/code/third_party
   rm -rf TransformerEngine/3rdparty/cutlass
   ln -s ../../../cutlass TransformerEngine/3rdparty/cutlass
""")
        
        if any("4.5.2" in i for i in issues):
            print("""
2. Update main CUTLASS to 4.5.2:
   
   ./core/scripts/install_cutlass.sh
""")
        
        print()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
