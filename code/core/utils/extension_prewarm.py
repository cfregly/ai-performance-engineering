#!/usr/bin/env python3
"""Automatic pre-warming of CUDA extensions at import time.

This module can be imported early to trigger background compilation of
the registered CUDA extension inventory, so it is ready when benchmarks run.

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

import logging
import os
import sys
import threading
import time as _time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from core.utils.python_entrypoints import find_repo_root, temporary_sys_path

logger = logging.getLogger(__name__)

# State tracking
PrewarmState = Literal["idle", "running", "complete", "failed"]
_PREWARM_STATE: PrewarmState = "idle"
_PREWARM_THREAD: threading.Thread | None = None
_PREWARM_RESULTS: dict[str, "ExtensionResult"] = {}
_PREWARM_TIMES: dict[str, float] = {}  # Build times in seconds
_PREWARM_LOCK = threading.Lock()
_PREWARM_CONDITION = threading.Condition(_PREWARM_LOCK)
_PREWARM_SETUP_RESULT_NAME = "__prewarm_setup__"
_PREWARM_WAIT_RESULT_NAME = "__prewarm_wait__"


@dataclass(frozen=True)
class ExtensionRegistration:
    """A core-owned or explicitly registered extension build target."""

    name: str
    loader: Callable[[], object]
    probe: Callable[[], tuple[bool, str]]
    supported_capabilities: frozenset[tuple[int, int]] | None = None
    owner: str = "core"


@dataclass(frozen=True)
class ExtensionResult:
    """Outcome of one build target without conflating skips and success."""

    status: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in {"built", "failed", "skipped"}:
            raise ValueError(f"invalid extension result status: {self.status}")


_REGISTRY_LOCK = threading.Lock()
_REGISTERED_EXTENSIONS: dict[str, ExtensionRegistration] = {}


def _validate_registration(registration: ExtensionRegistration) -> None:
    if (
        not registration.name
        or not callable(registration.loader)
        or not callable(registration.probe)
    ):
        raise ValueError("extension registrations require a name, loader, and probe")
    if not registration.owner or not (
        registration.name == registration.owner
        or registration.name.startswith(f"{registration.owner}.")
    ):
        raise ValueError("extension registration names must be scoped to their owner")


def _validate_application_registration(registration: ExtensionRegistration) -> None:
    """Reject application registrations that impersonate the core namespace."""
    _validate_registration(registration)
    reserved_owner = registration.owner == "core" or registration.owner.startswith("core.")
    reserved_name = registration.name == "core" or registration.name.startswith("core.")
    core_names = {candidate.name for candidate in _core_extensions()}
    if reserved_owner or reserved_name or registration.name in core_names:
        raise ValueError("application registrations cannot claim the reserved core namespace")


def _invalidate_default_state_for_registry_change() -> None:
    """Invalidate cached default results while the lifecycle lock is held."""
    global _PREWARM_RESULTS, _PREWARM_TIMES, _PREWARM_STATE, _PREWARM_THREAD
    _PREWARM_STATE = "idle"
    _PREWARM_THREAD = None
    _PREWARM_RESULTS = {}
    _PREWARM_TIMES = {}
    _PREWARM_CONDITION.notify_all()


def register_extension(registration: ExtensionRegistration) -> None:
    """Register an application-owned extension without importing it from core."""
    _validate_application_registration(registration)
    with _PREWARM_CONDITION:
        if _PREWARM_STATE == "running":
            raise RuntimeError("cannot register extensions while default prewarm is running")
        with _REGISTRY_LOCK:
            existing = _REGISTERED_EXTENSIONS.get(registration.name)
            if existing is not None:
                raise ValueError(f"extension {registration.name!r} is already registered")
            _REGISTERED_EXTENSIONS[registration.name] = registration
        _invalidate_default_state_for_registry_change()


def unregister_extension(name: str) -> None:
    """Remove an application registration; core defaults cannot be removed."""
    with _PREWARM_CONDITION:
        with _REGISTRY_LOCK:
            if name not in _REGISTERED_EXTENSIONS:
                return
            if _PREWARM_STATE == "running":
                raise RuntimeError("cannot unregister extensions while default prewarm is running")
            _REGISTERED_EXTENSIONS.pop(name)
        _invalidate_default_state_for_registry_change()


def _load_core_tcgen05(loader_name: str) -> object:
    from core.common import tcgen05

    return getattr(tcgen05, loader_name)()


def _probe_core_tcgen05(loader_name: str) -> tuple[bool, str]:
    try:
        from core.common import tcgen05

        loader = getattr(tcgen05, loader_name, None)
        return (callable(loader), "OK" if callable(loader) else f"missing callable {loader_name}")
    except Exception as exc:
        return False, str(exc)[:200]


def _core_extensions() -> tuple[ExtensionRegistration, ...]:
    capabilities = frozenset({(10, 0), (10, 3)})
    loaders = (
        "load_matmul_tcgen05_module",
        "load_tiling_tcgen05_module",
        "load_tcgen05_basic_module",
        "load_tcgen05_pipelined_module",
        "load_tcgen05_cluster_module",
        "load_tcgen05_warp_specialized_module",
        "load_tcgen05_warp_specialized_cutlass_module",
        "load_tcgen05_warpgroup_specialized_module",
    )
    return tuple(
        ExtensionRegistration(
            name=f"core.common.tcgen05.{loader_name}",
            loader=partial(_load_core_tcgen05, loader_name),
            probe=partial(_probe_core_tcgen05, loader_name),
            supported_capabilities=capabilities,
            owner="core",
        )
        for loader_name in loaders
    )


def _extension_inventory() -> tuple[ExtensionRegistration, ...]:
    core_extensions = _core_extensions()
    with _REGISTRY_LOCK:
        registered = tuple(_REGISTERED_EXTENSIONS[name] for name in sorted(_REGISTERED_EXTENSIONS))
    inventory = (*core_extensions, *registered)
    names: set[str] = set()
    for registration in inventory:
        if registration.name in names:
            raise ValueError(f"duplicate extension inventory name: {registration.name}")
        names.add(registration.name)
    return inventory


def _get_repo_root() -> Path:
    """Find repository root."""
    return find_repo_root()


def _repo_import_context():
    """Temporarily expose the repo root for extension module imports."""
    return temporary_sys_path(_get_repo_root())


def _build_extension(
    registration: ExtensionRegistration,
    verbose: bool = False,
) -> tuple[ExtensionResult, float]:
    """Build a single extension module.

    Returns:
        Tuple of (success, message, build_time_seconds)
    """
    start = _time.time()
    try:
        if verbose:
            print(f"  Building {registration.name}...", flush=True)

        artifact = registration.loader()
        if artifact is None:
            raise RuntimeError("extension loader returned no build artifact")

        elapsed = _time.time() - start
        if verbose:
            print(f"  ✓ {registration.name} ({elapsed:.1f}s)")
        return ExtensionResult("built", "OK"), elapsed
    except Exception as e:
        elapsed = _time.time() - start
        error_msg = str(e)[:200]
        if verbose:
            print(f"  ✗ {registration.name}: {error_msg}")
        return ExtensionResult("failed", error_msg), elapsed


def _do_prewarm(
    verbose: bool = False,
    parallel: bool = True,
    registrations: tuple[ExtensionRegistration, ...] | None = None,
    *,
    record_default_state: bool = True,
) -> dict[str, ExtensionResult]:
    """Execute pre-warming of the default or explicitly supplied inventory.

    Args:
        verbose: Print progress information
        parallel: Build independent extensions in parallel (faster)

    Returns:
        Dictionary mapping extension names to (success, message) tuples
    """
    total_start = _time.time()

    extensions = registrations if registrations is not None else _extension_inventory()

    if verbose:
        mode = "parallel" if parallel else "sequential"
        print(f"Pre-warming CUDA extensions ({mode})...", flush=True)

    results = {}
    times = {}

    capability: tuple[int, int] | None = None
    try:
        import torch

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
    except Exception:
        pass

    # Filter extensions based on hardware
    to_build = []
    for registration in extensions:
        supported = registration.supported_capabilities
        if supported is not None and capability not in supported:
            supported_text = ", ".join(f"SM{major}{minor}" for major, minor in sorted(supported))
            if verbose:
                print(f"  ⊘ {registration.name} (skipped, requires {supported_text})")
            results[registration.name] = ExtensionResult("skipped", f"requires {supported_text}")
            times[registration.name] = 0.0
        else:
            to_build.append(registration)

    with _repo_import_context():
        if parallel and len(to_build) > 1:
            # Build extensions in parallel using thread pool
            # Note: GIL is released during compilation, so this helps
            max_workers = min(len(to_build), int(os.environ.get("MAX_JOBS", "4")))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_build_extension, registration, verbose): registration.name
                    for registration in to_build
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result, elapsed = future.result()
                        results[name] = result
                        times[name] = elapsed
                    except Exception as e:
                        results[name] = ExtensionResult("failed", str(e)[:200])
                        times[name] = 0.0
                        if verbose:
                            print(f"  ✗ {name}: {e}")
        else:
            # Sequential build
            for registration in to_build:
                result, elapsed = _build_extension(registration, verbose)
                results[registration.name] = result
                times[registration.name] = elapsed

    total_time = _time.time() - total_start

    if verbose:
        built = sum(1 for result in results.values() if result.status == "built")
        skipped = sum(1 for result in results.values() if result.status == "skipped")
        failures = sum(1 for result in results.values() if result.status == "failed")
        print(
            f"Pre-warm complete: {built} built, {skipped} skipped, {failures} failed "
            f"(total: {total_time:.1f}s)",
            flush=True,
        )

    if record_default_state:
        _record_default_success(results, times)

    return results


def prewarm_extension_registrations(
    registrations: tuple[ExtensionRegistration, ...],
    *,
    verbose: bool = False,
    parallel: bool = True,
) -> dict[str, ExtensionResult]:
    """Build an explicit application-provided inventory without core back-edges."""
    names: set[str] = set()
    for registration in registrations:
        _validate_application_registration(registration)
        if registration.name in names:
            raise ValueError(f"duplicate extension registration: {registration.name}")
        names.add(registration.name)
    return _do_prewarm(
        verbose=verbose,
        parallel=parallel,
        registrations=registrations,
        record_default_state=False,
    )


def all_extensions_built(
    results: dict[str, ExtensionResult],
    expected_names: tuple[str, ...] | None = None,
) -> bool:
    """Return true only when every requested target has a distinct built result."""
    expected = set(expected_names) if expected_names is not None else set(results)
    if not expected or set(results) != expected:
        return False
    return all(results[name].status == "built" for name in expected)


def _record_default_success(
    results: dict[str, ExtensionResult],
    times: dict[str, float],
) -> None:
    """Publish a successful lifecycle completion while holding the state lock."""
    global _PREWARM_RESULTS, _PREWARM_TIMES, _PREWARM_STATE, _PREWARM_THREAD
    with _PREWARM_CONDITION:
        _PREWARM_RESULTS = results.copy()
        _PREWARM_TIMES = times.copy()
        _PREWARM_STATE = "complete"
        _PREWARM_THREAD = None
        _PREWARM_CONDITION.notify_all()


def _record_default_failure(exc: Exception) -> dict[str, ExtensionResult]:
    """Publish an unexpected setup failure as a terminal, retryable result."""
    global _PREWARM_RESULTS, _PREWARM_TIMES, _PREWARM_STATE, _PREWARM_THREAD
    message = str(exc)[:200] or type(exc).__name__
    results = {
        _PREWARM_SETUP_RESULT_NAME: ExtensionResult("failed", message),
    }
    with _PREWARM_CONDITION:
        _PREWARM_RESULTS = results.copy()
        _PREWARM_TIMES = {_PREWARM_SETUP_RESULT_NAME: 0.0}
        _PREWARM_STATE = "failed"
        _PREWARM_THREAD = None
        _PREWARM_CONDITION.notify_all()
    logger.warning("Extension prewarm setup failed: %s", message)
    return results


def _wait_for_terminal_state(timeout: float | None = None) -> bool:
    """Wait for the active lifecycle to reach complete or failed."""
    with _PREWARM_CONDITION:
        return _PREWARM_CONDITION.wait_for(
            lambda: _PREWARM_STATE != "running",
            timeout=timeout,
        )


def prewarm_extensions(
    background: bool = True,
    verbose: bool | None = None,
    wait: bool = False,
    parallel: bool = True,
) -> dict[str, ExtensionResult] | None:
    """Pre-warm the registered default CUDA extension inventory.

    Args:
        background: If True, build in background thread (non-blocking)
        verbose: Print progress (default: from PREWARM_VERBOSE env var)
        wait: If background=True, wait for completion before returning
        parallel: Build independent extensions in parallel (faster)

    Returns:
        Results dict if background=False or wait=True, else None
    """
    global _PREWARM_RESULTS, _PREWARM_TIMES, _PREWARM_STATE, _PREWARM_THREAD

    if verbose is None:
        verbose = os.environ.get("PREWARM_VERBOSE", "0") == "1"

    thread_to_start: threading.Thread | None = None
    with _PREWARM_CONDITION:
        if _PREWARM_STATE == "running":
            should_wait = wait or not background
            if not should_wait:
                return None
            _PREWARM_CONDITION.wait_for(lambda: _PREWARM_STATE != "running")
            return _PREWARM_RESULTS.copy()

        if _PREWARM_STATE == "complete":
            return _PREWARM_RESULTS.copy() if wait or not background else None

        # The first run starts from idle. A later call after a terminal setup
        # failure is the explicit safe retry policy; per-target build failures
        # still produce a normal "complete" lifecycle and are not auto-retried.
        _PREWARM_STATE = "running"
        _PREWARM_RESULTS = {}
        _PREWARM_TIMES = {}

        if background:

            def _background_prewarm() -> None:
                try:
                    _do_prewarm(verbose=verbose, parallel=parallel)
                except Exception as exc:
                    _record_default_failure(exc)

            try:
                thread_to_start = threading.Thread(
                    target=_background_prewarm,
                    name="cuda-extension-prewarm",
                    daemon=True,
                )
                # Publish the handle before releasing the lifecycle lock so a
                # concurrent waiter can never observe running-without-a-thread.
                _PREWARM_THREAD = thread_to_start
            except Exception as exc:
                message = str(exc)[:200] or type(exc).__name__
                _PREWARM_RESULTS = {
                    _PREWARM_SETUP_RESULT_NAME: ExtensionResult("failed", message),
                }
                _PREWARM_TIMES = {_PREWARM_SETUP_RESULT_NAME: 0.0}
                _PREWARM_STATE = "failed"
                _PREWARM_THREAD = None
                _PREWARM_CONDITION.notify_all()
                logger.warning("Extension prewarm thread setup failed: %s", message)
                return _PREWARM_RESULTS.copy() if wait else None

    if background:
        assert thread_to_start is not None
        try:
            thread_to_start.start()
        except Exception as exc:
            results = _record_default_failure(exc)
            return results if wait else None

        if wait:
            _wait_for_terminal_state()
            with _PREWARM_CONDITION:
                return _PREWARM_RESULTS.copy()
        return None

    try:
        return _do_prewarm(verbose=verbose, parallel=parallel)
    except Exception as exc:
        return _record_default_failure(exc)


def wait_for_prewarm(timeout: float | None = None) -> dict[str, ExtensionResult]:
    """Wait for background pre-warming to complete.

    Args:
        timeout: Maximum time to wait in seconds (None = wait forever)

    Returns:
        Results dictionary
    """
    reached_terminal = _wait_for_terminal_state(timeout)
    with _PREWARM_CONDITION:
        results = _PREWARM_RESULTS.copy()
    if not reached_terminal:
        timeout_text = "without a deadline" if timeout is None else f"after {timeout:.3f}s"
        results[_PREWARM_WAIT_RESULT_NAME] = ExtensionResult(
            "failed",
            f"timed out waiting for extension prewarm {timeout_text}",
        )
    return results


def is_prewarm_complete() -> bool:
    """Return whether pre-warming reached a complete or failed terminal state."""
    with _PREWARM_CONDITION:
        return _PREWARM_STATE in {"complete", "failed"}


def get_prewarm_results() -> dict[str, ExtensionResult]:
    """Get results of pre-warming (empty if not started/complete)."""
    with _PREWARM_CONDITION:
        return _PREWARM_RESULTS.copy()


def get_build_times() -> dict[str, float]:
    """Get build times in seconds for each extension."""
    with _PREWARM_CONDITION:
        return _PREWARM_TIMES.copy()


def health_check(verbose: bool = True) -> bool:
    """Validate the loader registrations without compiling extensions.

    Actual compilation is performed by :func:`prewarm_extensions`.

    Args:
        verbose: Print progress information

    Returns:
        True if all checks pass, False otherwise
    """
    with _repo_import_context():
        return _health_check_with_repo_imports(verbose=verbose)


def _health_check_with_repo_imports(verbose: bool = True) -> bool:
    """Probe extension loaders while the repository root is temporarily visible."""
    all_ok = True

    if verbose:
        print("Running CUDA extension health check...", flush=True)

    for registration in _extension_inventory():
        try:
            ok, message = registration.probe()
        except Exception as exc:
            ok, message = False, str(exc)[:200]
        all_ok = all_ok and ok
        if verbose:
            marker = "✓" if ok else "✗"
            print(f"  {marker} {registration.name}: {message}")

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
    prewarm_parser = subparsers.add_parser(
        "prewarm", help="Pre-warm the registered CUDA extension inventory"
    )
    prewarm_parser.add_argument(
        "--sequential", action="store_true", help="Build sequentially (default: parallel)"
    )

    # health command
    health_parser = subparsers.add_parser("health", help="Run health check on extensions")

    # times command
    times_parser = subparsers.add_parser("times", help="Show build times after prewarm")

    args = parser.parse_args()

    if args.command == "prewarm":
        results = prewarm_extensions(background=False, verbose=True, parallel=not args.sequential)

        print()
        print("Build times:")
        times = get_build_times()
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name}: {t:.2f}s")

        sys.exit(0 if all_extensions_built(results) else 1)

    elif args.command == "health":
        ok = health_check(verbose=True)
        sys.exit(0 if ok else 1)

    elif args.command == "times":
        # Run prewarm first to get times
        results = prewarm_extensions(background=False, verbose=True)
        print()
        print("Build times:")
        times = get_build_times()
        for name, t in sorted(times.items(), key=lambda x: -x[1]):
            print(f"  {name}: {t:.2f}s")
        sys.exit(0 if all_extensions_built(results) else 1)

    else:
        parser.print_help()
