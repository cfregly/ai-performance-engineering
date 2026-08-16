#!/usr/bin/env python3
"""Create an evidence-first optimization campaign workspace."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from core.optimization.campaign import main as campaign_main


def build_parser() -> argparse.ArgumentParser:
    """Build the campaign example parser."""
    parser = argparse.ArgumentParser(
        description="Initialize a measured optimization campaign from frozen specs."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--initial-control-commit", required=True)
    parser.add_argument("--direction", choices=("higher", "lower"), default="lower")
    parser.add_argument("--primary-case", action="append", required=True)
    parser.add_argument("--frozen-case", action="append", required=True)
    parser.add_argument("--workload-spec", type=Path, required=True)
    parser.add_argument("--environment-spec", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--min-trials", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Initialize a campaign through the supported controller."""
    args = build_parser().parse_args(argv)
    command = [
        "init",
        str(args.workspace),
        "--objective",
        args.objective,
        "--metric",
        args.metric,
        "--initial-control-commit",
        args.initial_control_commit,
        "--direction",
        args.direction,
        "--beam-width",
        str(args.beam_width),
        "--min-trials",
        str(args.min_trials),
        "--workload-spec",
        str(args.workload_spec),
        "--environment-spec",
        str(args.environment_spec),
    ]
    for case_name in args.primary_case:
        command.extend(["--primary-case", case_name])
    for case_name in args.frozen_case:
        command.extend(["--frozen-case", case_name])
    return campaign_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
