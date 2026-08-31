#!/usr/bin/env python3
"""Automatic pre-warming of CUDA extensions at import time.

This module can be imported early to trigger background compilation of
all CUDA extensions, so they're ready when benchmarks run.

Usage:
    # Option 1: Import directly (background mode)
    import core.utils.extension_prewarm
    
    # Option 2: Via environment variable (set before any imports)
    export PREWARM_CUDA_EXTENSIONS=1
    
    # Option 3: Explicit control
    from core.utils.extension_prewarm import prewarm_extensions
    prewarm_extensions(background=True)  # Non-blocking
    prewarm_extensions(background=False) # Blocking

Environment Variables:
    PREWARM_CUDA_EXTENSIONS: Set to "1" to enable auto-prewarm on import
    PREWARM_VERBOSE: Set to "1" to see build progress
    PREWARM_BACKGROUND: Set to "0" to build synchronously (default: "1")
"""

import os
import sys
import threading
import logging
import time as _time
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils.python_entrypoints import find_repo_root, temporary_sys_path

logger = logging.getLogger(__name__)

# State tracking
_PREWARM_STARTED = False
_PREWARM_COMPLETE = False
_PREWARM_THREAD: Optional[threading.Thread] = None
_PREWARM_RESULTS: Dict[str, Tuple[bool, str]] = {}
_PREWARM_TIMES: Dict[str, float] = {}  # Build times in seconds
_PREWARM_LOCK = threading.Lock()


def _get_repo_root() -> Path:
    """Find repository root."""
    return find_repo_root()


def _repo_import_context():
    """Temporarily expose the repo root for extension module imports."""
    return temporary_sys_path(_get_repo_root())


def _build_extension(name: str, import_path: str, verbose: bool = False) -> Tuple[bool, str, float]:
    """Build a single extension module.
    
    Returns:
        Tuple of (success, message, build_time_seconds)
    """
    start = _time.time()
    try:
        import importlib
        if verbose:
            print(f"  Building {name}...", flush=True)
        
        # Import triggers build
        importlib.import_module(import_path)
        
        elapsed = _time.time() - start
        if verbose:
            print(f"  ✓ {name} ({elapsed:.1f}s)")
        return (True, "OK", elapsed)
    except Exception as e:
        elapsed = _time.time() - start
        error_msg = str(e)[:200]
        if verbose:
            print(f"  ✗ {name}: {error_msg}")
        return (False, error_msg, elapsed)


def _do_prewarm(verbose: bool = False, parallel: bool = True) -> Dict[str, Tuple[bool, str]]:
    """Execute pre-warming of all extensions.
    
    Args:
        verbose: Print progress information
        parallel: Build independent extensions in parallel (faster)
    
    Returns:
        Dictionary mapping extension names to (success, message) tuples
    """
    global _PREWARM_RESULTS, _PREWARM_COMPLETE, _PREWARM_TIMES
    
    total_start = _time.time()
    
    # Extensions to prebuild
    # Format: (display_name, import_path, requires_sm100)
    extensions = [
        ("ch06.cuda_extensions", "ch06.cuda_extensions", False),
        ("ch12.cuda_extensions", "ch12.cuda_extensions", False),
        ("core.common.tcgen05", "core.common.tcgen05", True),  # SM100+ only
    ]
    
    if verbose:
        mode = "parallel" if parallel else "sequential"
        print(f"Pre-warming CUDA extensions ({mode})...", flush=True)
    
    results = {}
    times = {}
    
    # Check GPU capability for SM100+ extensions
    sm100_available = False
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            sm100_available = major >= 10
    except Exception:
        pass
    
    # Filter extensions based on hardware
    to_build = []
    for name, import_path, requires_sm100 in extensions:
        if requires_sm100 and not sm100_available:
            if verbose:
                print(f"  ⊘ {name} (skipped, requires SM100+)")
            results[name] = (True, "Skipped (requires SM100+)")
            times[name] = 0.0
        else:
            to_build.append((name, import_path))
    
    with _repo_import_context():
        if parallel and len(to_build) > 1:
            # Build extensions in parallel using thread pool
            # Note: GIL is released during compilation, so this helps
            max_workers = min(len(to_build), int(os.environ.get("MAX_JOBS", "4")))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_build_extension, name, import_path, verbose): name
                    for name, import_path in to_build
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        success, msg, elapsed = future.result()
                        results[name] = (success, msg)
                        times[name] = elapsed
                    except Exception as e:
                        results[name] = (False, str(e)[:200])
                        times[name] = 0.0
                        if verbose:
                            print(f"  ✗ {name}: {e}")
        else:
            # Sequential build
            for name, import_path in to_build:
                success, msg, elapsed = _build_extension(name, import_path, verbose)
                results[name] = (success, msg)
                times[name] = elapsed
    
    _PREWARM_RESULTS = results
    _PREWARM_TIMES = times
    _PREWARM_COMPLETE = True
    
    total_time = _time.time() - total_start
    
    if verbose:
        successes = sum(1 for ok, _ in results.values() if ok)
        failures = sum(1 for ok, _ in results.values() if not ok)
        print(f"Pre-warm complete: {successes} succeeded, {failures} failed (total: {total_time:.1f}s)", flush=True)
    
    return results


def prewarm_extensions(
    background: bool = True,
    verbose: Optional[bool] = None,
    wait: bool = False,
    parallel: bool = True,
) -> Optional[Dict[str, Tuple[bool, str]]]:
    """Pre-warm all CUDA extensions.
    
    Args:
        background: If True, build in background thread (non-blocking)
        verbose: Print progress (default: from PREWARM_VERBOSE env var)
        wait: If background=True, wait for completion before returning
        parallel: Build independent extensions in parallel (faster)
        
    Returns:
        Results dict if background=False or wait=True, else None
    """
    global _PREWARM_STARTED, _PREWARM_THREAD
    
    if verbose is None:
        verbose = os.environ.get("PREWARM_VERBOSE", "0") == "1"
    
    with _PREWARM_LOCK:
        if _PREWARM_STARTED:
            if wait and _PREWARM_THREAD is not None:
                _PREWARM_THREAD.join()
                return _PREWARM_RESULTS
            return None if background else _PREWARM_RESULTS
        _PREWARM_STARTED = True
    
    if background:
        def _background_prewarm():
            try:
                _do_prewarm(verbose=verbose, parallel=parallel)
            except Exception as e:
                logger.warning(f"Background prewarm failed: {e}")
        
        _PREWARM_THREAD = threading.Thread(
            target=_background_prewarm,
            name="cuda-extension-prewarm",
            daemon=True,
        )
        _PREWARM_THREAD.start()
        
        if wait:
            _PREWARM_THREAD.join()
            return _PREWARM_RESULTS
        return None
    else:
        return _do_prewarm(verbose=verbose, parallel=parallel)


def wait_for_prewarm(timeout: Optional[float] = None) -> Dict[str, Tuple[bool, str]]:
    """Wait for background pre-warming to complete.
    
    Args:
        timeout: Maximum time to wait in seconds (None = wait forever)
        
    Returns:
        Results dictionary
    """
    global _PREWARM_THREAD
    
    if _PREWARM_THREAD is not None:
        _PREWARM_THREAD.join(timeout=timeout)
    
    return _PREWARM_RESULTS


def is_prewarm_complete() -> bool:
    """Check if pre-warming has completed."""
    return _PREWARM_COMPLETE


def get_prewarm_results() -> Dict[str, Tuple[bool, str]]:
    """Get results of pre-warming (empty if not started/complete)."""
    return _PREWARM_RESULTS.copy()


def get_build_times() -> Dict[str, float]:
    """Get build times in seconds for each extension."""
    return _PREWARM_TIMES.copy()


def health_check(verbose: bool = True) -> bool:
    """Run a health check on all CUDA extensions.
    
    This verifies that:
    1. Extension modules can be imported
    2. Extension loader functions exist
    3. Extensions can be loaded without error
    
    Args:
        verbose: Print progress information
        
    Returns:
        True if all checks pass, False otherwise
    """
    with _repo_import_context():
        return _health_check_with_repo_imports(verbose=verbose)


def _health_check_with_repo_imports(verbose: bool = True) -> bool:
    """Run extension imports while the repository root is temporarily visible."""
    all_ok = True
    
    if verbose:
        print("Running CUDA extension health check...", flush=True)
    
    # Check ch06 extensions
    try:
        from ch06 import cuda_extensions as ch06_ext
        # Verify loader functions exist
        loaders = ['load_coalescing_extension', 'load_bank_conflicts_extension', 
                   'load_ilp_extension', 'load_launch_bounds_extension']
        for fn in loaders:
            if hasattr(ch06_ext, fn):
                if verbose:
                    print(f"  ✓ ch06.{fn}")
            else:
                if verbose:
                    print(f"  ✗ ch06.{fn}: Missing")
                all_ok = False
    except Exception as e:
        if verbose:
            print(f"  ✗ ch06.cuda_extensions: {e}")
        all_ok = False
    
    # Check ch12 extensions
    try:
        from ch12 import cuda_extensions as ch12_ext
        loaders = ['load_kernel_fusion_extension', 'load_graph_bandwidth_extension', 
                   'load_cuda_graphs_extension', 'load_work_queue_extension']
        for fn in loaders:
            if hasattr(ch12_ext, fn):
                if verbose:
                    print(f"  ✓ ch12.{fn}")
            else:
                if verbose:
                    print(f"  ✗ ch12.{fn}: Missing")
                all_ok = False
    except Exception as e:
        if verbose:
            print(f"  ✗ ch12.cuda_extensions: {e}")
        all_ok = False
    
    # Check tcgen05 (SM100+ only)
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 10:
                try:
                    from core.common import tcgen05
                    if hasattr(tcgen05, 'load_tiling_extension') or hasattr(tcgen05, '_load_extension'):
                        if verbose:
                            print(f"  ✓ core.common.tcgen05")
                    else:
                        if verbose:
                            print(f"  ⚠ core.common.tcgen05: Module loaded")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ core.common.tcgen05: {e}")
                    all_ok = False
            else:
                if verbose:
                    print(f"  ⊘ core.common.tcgen05 (skipped, requires SM100+)")
    except Exception as e:
        if verbose:
            print(f"  ⚠ tcgen05 check skipped: {e}")
    
    if verbose:
        status = "PASSED" if all_ok else "FAILED"
        print(f"Health check: {status}", flush=True)
    
    return all_ok


# Auto-prewarm on import if enabled
if os.environ.get("PREWARM_CUDA_EXTENSIONS", "0") == "1":
    _verbose = os.environ.get("PREWARM_VERBOSE", "0") == "1"
    _background = os.environ.get("PREWARM_BACKGROUND", "1") == "1"
    
    prewarm_extensions(background=_background, verbose=_verbose)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA Extension Pre-warming")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # prewarm command
    prewarm_parser = subparsers.add_parser("prewarm", help="Pre-warm all CUDA extensions")
    prewarm_parser.add_argument("--sequential", action="store_true", help="Build sequentially (default: parallel)")
    
    # health command
    health_parser = subparsers.add_parser("health", help="Run health check on extensions")
    
    # times command
    times_parser = subparsers.add_parser("times", help="Show build times after prewarm")
    
    args = parser.parse_args()
    
    if args.command == "prewarm":
        results = prewarm_extensions(
            background=False, 
            verbose=True, 
            parallel=not args.sequential
        )
        
        print()
        print("Build times:")
        times = get_build_times()
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name}: {t:.2f}s")
        
        failures = sum(1 for ok, _ in results.values() if not ok)
        sys.exit(1 if failures > 0 else 0)
    
    elif args.command == "health":
        ok = health_check(verbose=True)
        sys.exit(0 if ok else 1)
    
    elif args.command == "times":
        # Run prewarm first to get times
        prewarm_extensions(background=False, verbose=True)
        print()
        print("Build times:")
        times = get_build_times()
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name}: {t:.2f}s")
    
    else:
        parser.print_help()
