#!/usr/bin/env python3
"""Torchrun worker for the chapter 4 NVSHMEM benchmark wrappers."""

from __future__ import annotations

import argparse
import importlib
import sys

import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status

_WORKLOAD_MODULES = {
    "pipeline": "ch04.nvshmem_pipeline_parallel_multigpu",
    "training-example": "ch04.nvshmem_training_example",
    "training-patterns": "ch04.nvshmem_training_patterns",
    "collective": "ch04.nvshmem_vs_nccl_benchmark",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run one NVSHMEM benchmark variant.")
    parser.add_argument("--workload", choices=tuple(_WORKLOAD_MODULES), required=True)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    return parser.parse_known_args()


def main() -> None:
    args, workload_args = _parse_args()
    if args.variant == "optimized":
        from ch04.optimized_nvshmem_pipeline_parallel_multigpu import (
            _configure_blackwell_nccl,
        )

        _configure_blackwell_nccl()

    module_name = _WORKLOAD_MODULES[args.workload]
    module = importlib.import_module(module_name)
    workload_main = getattr(module, "main")

    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *workload_args]
        workload_main()
    finally:
        sys.argv = previous_argv
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
