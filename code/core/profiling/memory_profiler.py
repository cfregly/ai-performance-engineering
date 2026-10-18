"""Torch profiler helper focused on CUDA memory diagnostics."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from core.profiling.torch_profiler_compat import prepare_torch_cuda_profiler
from core.utils.python_entrypoints import find_repo_root, temporary_sys_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Memory profiler wrapper")
    parser.add_argument("script", help="Python script to execute under the profiler")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the script")
    parser.add_argument("--trace", type=Path, help="Optional path to write Chrome trace JSON")
    parser.add_argument("--sort", default="self_cuda_memory_usage", help="Metric to sort by in the summary table")
    parser.add_argument("--row-limit", type=int, default=25, help="Rows to display in summary table")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable assignments applied before running the script",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    script_dir = script_path.parent
    repo_root = find_repo_root(script_dir)

    # Always enable Python fault handler for stack traces on crashes
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    
    # Apply environment overrides
    for assignment in args.env:
        if "=" not in assignment:
            raise SystemExit(f"Invalid --env assignment '{assignment}' (expected KEY=VALUE)")
        key, value = assignment.split("=", 1)
        os.environ[key] = value

    activities = [ProfilerActivity.CPU]
    cuda_profile_requested = torch.cuda.is_available()
    if cuda_profile_requested:
        activities.append(ProfilerActivity.CUDA)
    prepare_torch_cuda_profiler(
        cuda_profile_requested=cuda_profile_requested
    )

    print(f"Profiling memory usage for {script_path} ...")
    sys.argv = [str(script_path)] + args.script_args
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with temporary_sys_path(script_dir, repo_root):
            runpy.run_path(str(script_path), run_name="__main__")

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(f"Chrome trace written to {args.trace}")

    table = prof.key_averages().table(sort_by=args.sort, row_limit=args.row_limit)
    print("\n=== Memory Profiling Summary ===")
    print(table)


if __name__ == "__main__":
    main()
