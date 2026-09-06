"""Explicit torchrun entrypoint for the cache-aware multi-GPU benchmark pair."""

from __future__ import annotations

from labs.cache_aware_disagg_inference.cache_aware_disagg_multigpu_common import run_cli


def main() -> None:
    run_cli()
