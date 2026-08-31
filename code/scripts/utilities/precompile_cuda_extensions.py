#!/usr/bin/env python3
"""Precompile the chapter CUDA extensions used by the benchmark smoke path."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial

import torch

from core.harness.arch_config import configure_optimizations as _configure_arch_optimizations
from core.utils.extension_prewarm import (
    ExtensionRegistration,
    all_extensions_built,
    prewarm_extension_registrations,
)

_configure_arch_optimizations()


def _probe_loader(loader: Callable[[], object]) -> tuple[bool, str]:
    return callable(loader), "ready" if callable(loader) else "loader is not callable"


def chapter_extension_registrations() -> tuple[ExtensionRegistration, ...]:
    """Return the explicit chapter-owned build inventory for this command."""
    from ch06.cuda_extensions import (
        load_bank_conflicts_extension,
        load_coalescing_extension,
        load_ilp_extension,
        load_launch_bounds_extension,
    )
    from ch12.cuda_extensions import (
        load_bias_relu_residual_extension,
        load_cuda_graphs_extension,
        load_graph_bandwidth_extension,
        load_kernel_fusion_extension,
        load_work_queue_extension,
    )

    entries = (
        ("ch06.cuda_extensions.bank_conflicts", "ch06", load_bank_conflicts_extension),
        ("ch06.cuda_extensions.coalescing", "ch06", load_coalescing_extension),
        ("ch06.cuda_extensions.ilp", "ch06", load_ilp_extension),
        ("ch06.cuda_extensions.launch_bounds", "ch06", load_launch_bounds_extension),
        ("ch12.cuda_extensions.bias_relu_residual", "ch12", load_bias_relu_residual_extension),
        ("ch12.cuda_extensions.cuda_graphs", "ch12", load_cuda_graphs_extension),
        ("ch12.cuda_extensions.graph_bandwidth", "ch12", load_graph_bandwidth_extension),
        ("ch12.cuda_extensions.kernel_fusion", "ch12", load_kernel_fusion_extension),
        ("ch12.cuda_extensions.work_queue", "ch12", load_work_queue_extension),
    )
    return tuple(
        ExtensionRegistration(
            name=name,
            owner=owner,
            loader=loader,
            probe=partial(_probe_loader, loader),
        )
        for name, owner, loader in entries
    )


def precompile_extensions(*, parallel: bool = True) -> bool:
    """Build the command's declared extension inventory and report each status."""
    print("=" * 80)
    print("Pre-compiling CUDA Extensions")
    print("=" * 80)
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; no extensions were compiled")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    registrations = chapter_extension_registrations()
    results = prewarm_extension_registrations(
        registrations,
        verbose=True,
        parallel=parallel,
    )

    print()
    for registration in registrations:
        result = results[registration.name]
        print(f"{registration.name}: {result.status} ({result.message})")

    built = sum(result.status == "built" for result in results.values())
    skipped = sum(result.status == "skipped" for result in results.values())
    failed = sum(result.status == "failed" for result in results.values())
    print(f"Summary: {built} built, {skipped} skipped, {failed} failed")
    expected_names = tuple(registration.name for registration in registrations)
    return all_extensions_built(results, expected_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Build one extension at a time instead of using the parallel registry runner.",
    )
    args = parser.parse_args()
    return 0 if precompile_extensions(parallel=not args.sequential) else 1


if __name__ == "__main__":
    raise SystemExit(main())
