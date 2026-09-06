"""Torchrun wrapper for benchmark launches.

This wrapper exists to enforce harness-level invariants inside torchrun-launched
multi-process benchmarks.

Currently enforced:
- RNG seed immutability: benchmarks must not reseed away from the harness-
  configured seeds (default seed=42).

The harness launches torchrun with this wrapper as the entrypoint and passes the
original benchmark script path + args through unchanged.
"""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.harness.backend_policy import BackendPolicyName, apply_backend_policy
from core.utils.python_entrypoints import temporary_sys_path


def _apply_backend_policy(deterministic: bool) -> None:
    apply_backend_policy(BackendPolicyName.PERFORMANCE, deterministic)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {value!r}") from exc


def _resolve_local_rank() -> int:
    value = os.environ.get("LOCAL_RANK")
    if value is not None and value != "":
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(f"Invalid LOCAL_RANK value: {value!r}") from exc
    world_size = os.environ.get("WORLD_SIZE")
    if world_size not in (None, "", "1"):
        raise RuntimeError("LOCAL_RANK must be set when WORLD_SIZE > 1")
    return 0


def _run_target_script(script_path: Path, argv: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = [str(script_path), *argv]
        with temporary_sys_path(script_path.parent):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = previous_argv


def _run_target_module(module_name: str, argv: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv


def _run_profiled_target(
    script_path: Optional[Path], module_name: Optional[str], argv: list[str]
) -> None:
    def run_target() -> None:
        try:
            if script_path is not None:
                _run_target_script(script_path, argv)
            else:
                _run_target_module(module_name or "", argv)
        except SystemExit as exc:
            if exc.code is not None and exc.code != 0:
                raise
            # A normal CLI exit must still close its profile and check seeds.

    output = os.environ.get("AISP_TORCH_PROFILE_OUTPUT")
    if not output:
        run_target()
        return
    from torch.profiler import ProfilerActivity, profile

    rank = _parse_int_env("RANK") or 0
    trace_path = Path(output)
    if rank:
        trace_path = trace_path.with_name(f"{trace_path.stem}.rank{rank}{trace_path.suffix}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiler:
        profiler.add_metadata("aisp_profile_scope", "torchrun_worker_lifetime")
        profiler.add_metadata("aisp_rank", str(rank))
        run_target()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--aisp-target-script",
        required=False,
        help="Path to the benchmark script to execute under torchrun.",
    )
    parser.add_argument(
        "--aisp-target-module",
        required=False,
        help="Module name of the benchmark entrypoint to execute under torchrun.",
    )
    parser.add_argument(
        "--aisp-expected-torch-seed",
        required=True,
        type=int,
        help="Expected torch.initial_seed() after benchmark completes.",
    )
    parser.add_argument(
        "--aisp-expected-cuda-seed",
        required=False,
        type=int,
        help="Expected torch.cuda.initial_seed() after benchmark completes (if CUDA is available).",
    )
    parser.add_argument(
        "--aisp-deterministic",
        action="store_true",
        help="Enable deterministic algorithms (mirrors harness deterministic mode).",
    )
    args, remainder = parser.parse_known_args(argv)

    if bool(args.aisp_target_script) == bool(args.aisp_target_module):
        raise RuntimeError(
            "Specify exactly one of --aisp-target-script or --aisp-target-module."
        )

    script_path: Optional[Path] = None
    module_name: Optional[str] = None
    if args.aisp_target_script:
        script_path = Path(args.aisp_target_script).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Target script not found: {script_path}")
    else:
        module_name = str(args.aisp_target_module)

    _apply_backend_policy(bool(args.aisp_deterministic))
    _set_seeds(int(args.aisp_expected_torch_seed))

    expected_torch_seed = int(args.aisp_expected_torch_seed)
    expected_cuda_seed: Optional[int] = args.aisp_expected_cuda_seed

    lock_requested = os.environ.get("AISP_LOCK_GPU_CLOCKS") == "1"
    ramp_requested = os.environ.get("AISP_RAMP_GPU_CLOCKS", "1") == "1"
    local_rank = _resolve_local_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if lock_requested and torch.cuda.is_available():
        from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks

        lock_ctx = lock_gpu_clocks(
            device=local_rank,
            sm_clock_mhz=_parse_int_env("AISP_GPU_SM_CLOCK_MHZ"),
            mem_clock_mhz=_parse_int_env("AISP_GPU_MEM_CLOCK_MHZ"),
        )
        with lock_ctx:
            if ramp_requested:
                ramp_gpu_clocks(device=local_rank)
            _run_profiled_target(script_path, module_name, remainder)
    else:
        _run_profiled_target(script_path, module_name, remainder)

    current_torch_seed = int(torch.initial_seed())
    if current_torch_seed != expected_torch_seed:
        raise RuntimeError(
            "Seed mutation detected during torchrun benchmark execution. "
            f"Expected torch.initial_seed()={expected_torch_seed}, got {current_torch_seed}. "
            "Benchmarks MUST NOT reseed; rely on harness-configured seeds."
        )

    if torch.cuda.is_available():
        if expected_cuda_seed is None:
            raise RuntimeError(
                "torch.cuda.is_available() is true but --aisp-expected-cuda-seed was not provided."
            )
        current_cuda_seed = int(torch.cuda.initial_seed())
        if current_cuda_seed != int(expected_cuda_seed):
            raise RuntimeError(
                "CUDA seed mutation detected during torchrun benchmark execution. "
                f"Expected torch.cuda.initial_seed()={int(expected_cuda_seed)}, got {current_cuda_seed}. "
                "Benchmarks MUST NOT reseed; rely on harness-configured seeds."
            )


if __name__ == "__main__":
    main()
