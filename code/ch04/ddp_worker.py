#!/usr/bin/env python3
"""Torchrun worker for the chapter 4 DDP overlap benchmark pair."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ch04.ddp_overlap_result import (
    DdpOverlapWorkloadResult,
    write_ddp_overlap_child_result,
)
from ch04.distributed_helper import run_main_with_skip_status


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DDP overlap benchmark variant.")
    parser.add_argument(
        "--variant",
        choices=("no-overlap", "overlap"),
        required=True,
        help="Communication schedule to measure.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def _resolve_worker(variant: str) -> Callable[[int, int], DdpOverlapWorkloadResult]:
    if variant == "no-overlap":
        from ch04.ddp_no_overlap import _run_worker

        return _run_worker

    from ch04.ddp_overlap import _run_worker

    return _run_worker


def main() -> None:
    args = _parse_args()
    result = _resolve_worker(args.variant)(args.iterations, args.warmup)
    write_ddp_overlap_child_result(result)
    if result.rank == 0:
        tokens_per_iteration = float(
            result.contract.batch_size * result.contract.hidden_size
        )
        tokens_per_second = tokens_per_iteration * 1000.0 / result.time_per_iter_ms
        print(f"rank0 tokens/s: {tokens_per_second:.2f} tokens/s", flush=True)
        print(
            f"rank0 time_per_iter_ms: {result.time_per_iter_ms:.9f}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
