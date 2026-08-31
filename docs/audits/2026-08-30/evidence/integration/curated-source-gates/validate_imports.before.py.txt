#!/usr/bin/env python3
"""Validate that all benchmark files can be imported without errors.

This script attempts to import each benchmark module and reports any failures.
Useful for catching missing dependencies, syntax errors, or broken imports.

Usage:
    python -m core.scripts.validate_imports              # Validate all
    python -m core.scripts.validate_imports --chapter 7  # Validate ch07 only
    python -m core.scripts.validate_imports --verbose    # Show successful imports too
"""
from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Optional


def get_module_name(filepath: Path, root: Path) -> str:
    """Convert file path to module name."""
    relative = filepath.relative_to(root)
    parts = list(relative.parts)
    # Remove .py extension
    parts[-1] = parts[-1].replace('.py', '')
    return '.'.join(parts)


def try_import(module_name: str) -> Tuple[bool, Optional[str]]:
    """Try to import a module.
    
    Returns (success, error_message).
    """
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as e:
        # Get just the last line of the traceback
        tb = traceback.format_exc()
        error_line = tb.strip().split('\n')[-1]
        return False, error_line


def validate_benchmarks(
    root: Path,
    chapter: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Validate all benchmark imports.
    
    Returns (total, passed, failures) where failures is [(module, error), ...].
    """
    # Find benchmark files
    if chapter:
        patterns = [f'ch{chapter}/baseline_*.py', f'ch{chapter}/optimized_*.py']
    else:
        patterns = ['ch*/baseline_*.py', 'ch*/optimized_*.py']
    
    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    
    # Sort for consistent output
    files = sorted(set(files))
    
    total = 0
    passed = 0
    failures = []
    
    for filepath in files:
        if not filepath.is_file():
            continue
        if '__pycache__' in str(filepath):
            continue
        
        module_name = get_module_name(filepath, root)
        total += 1
        
        success, error = try_import(module_name)
        
        if success:
            passed += 1
            if verbose:
                print(f"  ✅ {module_name}")
        else:
            failures.append((module_name, error))
            print(f"  ❌ {module_name}")
            print(f"     {error}")
    
    return total, passed, failures


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark imports")
    parser.add_argument("--chapter", type=int, help="Validate specific chapter only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results")
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("Benchmark Import Validation")
    print("=" * 70)
    
    if args.chapter:
        print(f"Validating chapter {args.chapter}...")
    else:
        print("Validating all chapters...")
    print()
    
    total, passed, failures = validate_benchmarks(
        root,
        chapter=args.chapter,
        verbose=args.verbose,
    )
    
    print()
    print("=" * 70)
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
    
    if failures:
        print(f"\n❌ {len(failures)} failures:")
        
        # Group by error type
        error_types = {}
        for module, error in failures:
            error_type = error.split(':')[0] if ':' in error else error
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(module)
        
        for error_type, modules in sorted(error_types.items()):
            print(f"\n  {error_type} ({len(modules)} files):")
            for module in modules[:5]:  # Show first 5
                print(f"    - {module}")
            if len(modules) > 5:
                print(f"    ... and {len(modules) - 5} more")
        
        sys.exit(1)
    else:
        print("\n✅ All imports successful!")
        sys.exit(0)


if __name__ == "__main__":
    main()


