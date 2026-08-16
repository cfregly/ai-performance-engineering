#!/usr/bin/env python3
"""Torchrun worker for the chapter 4 DDP overlap benchmark pair."""

from __future__ import annotations

import argparse
from collections.abc import Callable

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


def _resolve_worker(variant: str) -> Callable[[int, int], None]:
    if variant == "no-overlap":
        from ch04.ddp_no_overlap import _run_worker

        return _run_worker

    from ch04.ddp_overlap import _run_worker

    return _run_worker


def main() -> None:
    args = _parse_args()
    _resolve_worker(args.variant)(args.iterations, args.warmup)


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
