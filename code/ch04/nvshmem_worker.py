#!/usr/bin/env python3
"""Torchrun worker for the chapter 4 NVSHMEM benchmark wrappers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status
from ch04.nvshmem_child_result import (
    NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV,
    NVSHMEMWorkloadResult,
    write_nvshmem_child_result,
)
from ch04.nvshmem_pipeline_result import (
    NVSHMEMPipelineWorkloadResult,
    write_nvshmem_pipeline_child_result,
)

_WORKLOAD_MODULES = {
    "pipeline": "ch04.nvshmem_pipeline_parallel_multigpu",
    "training-example": "ch04.nvshmem_training_example",
    "training-patterns": "ch04.nvshmem_training_patterns",
    "collective": "ch04.nvshmem_vs_nccl_benchmark",
    "symmetric-ring": "ch04.symmetric_memory_example",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run one NVSHMEM benchmark variant.")
    parser.add_argument("--workload", choices=tuple(_WORKLOAD_MODULES), required=True)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    return parser.parse_known_args()


def _run_workload() -> None:
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
        result = workload_main()
        if result is None:
            if os.environ.get(NVSHMEM_CHILD_RESULT_LAUNCH_WALL_NS_ENV):
                raise RuntimeError(
                    f"NVSHMEM workload {args.workload!r} did not return its measured output"
                )
            return
        if isinstance(result, NVSHMEMPipelineWorkloadResult):
            if args.workload != "pipeline":
                raise RuntimeError(
                    "NVSHMEM pipeline result was returned for non-pipeline workload "
                    f"{args.workload!r}"
                )
            write_nvshmem_pipeline_child_result(result, variant=args.variant)
        elif isinstance(result, NVSHMEMWorkloadResult):
            if result.workload != args.workload:
                raise RuntimeError(
                    "NVSHMEM worker workload mismatch: "
                    f"{result.workload!r} != {args.workload!r}"
                )
            write_nvshmem_child_result(result, variant=args.variant)
        else:
            raise TypeError(
                f"NVSHMEM workload {args.workload!r} returned unsupported result "
                f"{type(result).__name__}"
            )
        if result.rank == 0:
            print(
                f"rank0 time_per_iter_ms: {result.time_per_iter_ms:.9f}",
                flush=True,
            )
    finally:
        sys.argv = previous_argv
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    """Run one worker and preserve the chapter's explicit skip exit status."""
    return run_main_with_skip_status(_run_workload)


if __name__ == "__main__":
    raise SystemExit(main())
