#!/usr/bin/env python3
"""Explicit two-rank worker for Chapter 4 gradient-compression pairs."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from ch04.distributed_helper import run_main_with_skip_status
from ch04.gradient_compression_multigpu_result import (
    GradientCompressionResultContract,
    assert_close_full,
    child_result_requested,
    distributed_topology,
    make_result_contract,
    write_gradient_compression_child_result,
)
from core.benchmark.distributed_work_contract import (
    DistributedRankWorkReceipt,
    DistributedWorkRecorder,
)
from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from core.profiling.nvtx_helper import nvtx_range

WORLD_SIZE = 2
TENSOR_SIZE_MB = 1024
SEED = 42


@dataclass
class _Buffers:
    compressed: torch.Tensor | None
    float_scratch: torch.Tensor | None
    fp32_output: torch.Tensor
    prequantized_scale: torch.Tensor | None = None


class _ObservedCollective:
    """Adapt a functional collective tensor to the core wait-recorder protocol."""

    def __init__(self, value: torch.Tensor) -> None:
        self._value: torch.Tensor | None = value
        self._result: torch.Tensor | None = None

    def wait(self) -> torch.Tensor:
        if self._result is None:
            if self._value is None:
                raise RuntimeError("Functional collective result was already released")
            wait = getattr(self._value, "wait", None)
            if not callable(wait):
                raise TypeError("Functional collective result does not expose wait()")
            self._result = wait()
            self._value = None
        return self._result

    def take_result(self) -> torch.Tensor:
        result = self.wait()
        self._result = None
        return result


def _profile_range(contract: GradientCompressionResultContract) -> str:
    suffix = "_comm_only" if contract.comm_only else ""
    return (
        f"{contract.variant}_gradient_compression_"
        f"{contract.pair_compression}{suffix}_multigpu"
    )


def _make_input(
    contract: GradientCompressionResultContract,
    *,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    # Each original CUDA device was seeded with the same seed. A local generator
    # preserves those exact per-rank inputs without mutating the harness seed.
    generator = torch.Generator(device=device)
    generator.manual_seed(contract.seed)
    return torch.randn(
        contract.numel,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _make_buffers(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
) -> _Buffers:
    compression = contract.effective_compression
    compressed: torch.Tensor | None = None
    float_scratch: torch.Tensor | None = None
    if compression == "fp16":
        compressed = torch.empty_like(initial_input, dtype=torch.float16)
    elif compression == "int8":
        compressed = torch.empty_like(initial_input, dtype=torch.int8)
        float_scratch = torch.empty_like(initial_input)
    return _Buffers(
        compressed=compressed,
        float_scratch=float_scratch,
        fp32_output=torch.empty_like(initial_input),
    )


def _collective(
    value: torch.Tensor,
    *,
    reduce_op: str,
    group: Any,
    recorder: DistributedWorkRecorder | None,
) -> _ObservedCollective:
    collective_group = dist.group.WORLD if group is None else group
    work = _ObservedCollective(
        funcol.all_reduce(value, reduce_op, collective_group)
    )
    if recorder is not None:
        recorder.record_async_collective(work)
    return work


def _complete(
    works: list[_ObservedCollective],
    recorder: DistributedWorkRecorder | None,
) -> list[torch.Tensor]:
    if recorder is not None:
        recorder.wait_for_async_collectives()
    return [work.take_result() for work in works]


def _inplace_sum(
    value: torch.Tensor,
    *,
    group: Any,
    recorder: DistributedWorkRecorder | None,
) -> Any:
    """Launch the public in-place process-group SUM and retain its real Work."""

    work = dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
        group=group,
        async_op=True,
    )
    if not callable(getattr(work, "wait", None)):
        raise RuntimeError("Asynchronous gradient all-reduce returned no Work handle")
    if recorder is not None:
        recorder.record_async_collective(work)
    return work


def _bucket_slices(contract: GradientCompressionResultContract):
    bucket_numel = max(1, contract.gradient_bucket_bytes // torch.float32.itemsize)
    for start in range(0, contract.numel, bucket_numel):
        yield slice(start, min(start + bucket_numel, contract.numel))


def _global_int8_scale(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
    *,
    group: Any,
    recorder: DistributedWorkRecorder | None,
) -> torch.Tensor:
    local_max = initial_input.abs().max()
    work = _collective(
        local_max,
        reduce_op="max",
        group=group,
        recorder=recorder,
    )
    global_max = _complete([work], recorder)[0]
    limit = max(1, 127 // contract.world_size)
    scale = global_max / float(limit)
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def _compress_int8(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
    buffers: _Buffers,
    scale: torch.Tensor,
) -> torch.Tensor:
    if buffers.compressed is None or buffers.float_scratch is None:
        raise RuntimeError("INT8 compression buffers are not initialized")
    limit = max(1, 127 // contract.world_size)
    sections = (
        _bucket_slices(contract)
        if contract.variant == "baseline" and not contract.comm_only
        else (slice(None),)
    )
    for section in sections:
        buffers.float_scratch[section].copy_(initial_input[section])
        buffers.float_scratch[section].div_(scale)
        buffers.float_scratch[section].round_()
        buffers.float_scratch[section].clamp_(-limit, limit)
        buffers.compressed[section].copy_(buffers.float_scratch[section])
    return buffers.compressed


def _prepare_comm_only(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
    buffers: _Buffers,
    *,
    group: Any,
) -> None:
    if not contract.comm_only or contract.effective_compression == "none":
        return
    if buffers.compressed is None:
        raise RuntimeError("Communication-only compression buffer is not initialized")
    if contract.effective_compression == "fp16":
        buffers.compressed.copy_(initial_input)
        return
    scale = _global_int8_scale(
        contract,
        initial_input,
        group=group,
        recorder=None,
    )
    _compress_int8(contract, initial_input, buffers, scale)
    buffers.prequantized_scale = scale


def _reduce_payload(
    contract: GradientCompressionResultContract,
    payload: torch.Tensor,
    buffers: _Buffers,
    *,
    group: Any,
    recorder: DistributedWorkRecorder | None,
) -> torch.Tensor:
    sections = list(_bucket_slices(contract))
    if not contract.comm_only:
        # Compression rewrites every element before every measured reduction.
        # Reducing the preallocated buffer slices in place therefore needs no
        # reset and leaves one contiguous full output, matching the original
        # torch.cuda.nccl out-buffer path without a baseline-only assembly copy.
        works = [
            _inplace_sum(
                payload[section],
                group=group,
                recorder=recorder,
            )
            for section in sections
        ]
        if recorder is not None:
            recorder.wait_for_async_collectives()
        else:
            for work in works:
                work.wait()
        return payload

    # Communication-only inputs are prepared once outside timing, so use the
    # functional out-of-place form to keep them unchanged across iterations.
    # Both control and candidate have one full payload and require no assembly.
    works = [
        _collective(
            payload[section],
            reduce_op="sum",
            group=group,
            recorder=recorder,
        )
        for section in sections
    ]
    results = _complete(works, recorder)
    if len(results) == 1:
        return results[0]
    raise RuntimeError("Communication-only gradient compression requires one full payload")


def _run_iteration(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
    buffers: _Buffers,
    *,
    group: Any,
    recorder: DistributedWorkRecorder | None,
) -> torch.Tensor:
    compression = contract.effective_compression
    scale: torch.Tensor | None = buffers.prequantized_scale
    if compression == "none":
        payload = initial_input
    elif contract.comm_only:
        if buffers.compressed is None:
            raise RuntimeError("Precompressed communication payload is missing")
        payload = buffers.compressed
    elif compression == "fp16":
        if buffers.compressed is None:
            raise RuntimeError("FP16 compression buffer is missing")
        buffers.compressed.copy_(initial_input)
        payload = buffers.compressed
    else:
        scale = _global_int8_scale(
            contract,
            initial_input,
            group=group,
            recorder=recorder,
        )
        payload = _compress_int8(contract, initial_input, buffers, scale)

    reduced = _reduce_payload(
        contract,
        payload,
        buffers,
        group=group,
        recorder=recorder,
    )
    if contract.comm_only and compression != "none":
        return reduced
    if compression == "none":
        return reduced
    buffers.fp32_output.copy_(reduced)
    if compression == "int8":
        if scale is None:
            raise RuntimeError("INT8 reduction scale is missing")
        buffers.fp32_output.mul_(scale)
    return buffers.fp32_output


def _dequantize_comm_only(
    contract: GradientCompressionResultContract,
    output: torch.Tensor,
    buffers: _Buffers,
) -> torch.Tensor:
    if not contract.comm_only:
        return output
    compression = contract.effective_compression
    if compression == "none":
        return output
    buffers.fp32_output.copy_(output)
    if compression == "int8":
        if buffers.prequantized_scale is None:
            raise RuntimeError("INT8 communication-only scale is missing")
        buffers.fp32_output.mul_(buffers.prequantized_scale)
    return buffers.fp32_output


def _build_independent_reference(
    contract: GradientCompressionResultContract,
    initial_input: torch.Tensor,
    *,
    group: Any,
) -> torch.Tensor:
    """Gather rank inputs and apply compression math outside the timed range."""

    gathered = torch.empty(
        contract.world_size * contract.numel,
        dtype=torch.float32,
        device=initial_input.device,
    )
    dist.all_gather_into_tensor(gathered, initial_input, group=group)
    rank_inputs = [
        gathered[rank * contract.numel : (rank + 1) * contract.numel]
        for rank in range(contract.world_size)
    ]
    compression = contract.effective_compression
    if compression == "none":
        reference = torch.zeros_like(initial_input)
        for rank_input in rank_inputs:
            reference.add_(rank_input)
        return reference
    if compression == "fp16":
        reference_half = torch.zeros_like(initial_input, dtype=torch.float16)
        for rank_input in rank_inputs:
            reference_half.add_(rank_input.to(torch.float16))
        return reference_half.float()

    maximum = torch.zeros((), dtype=torch.float32, device=initial_input.device)
    for rank_input in rank_inputs:
        maximum = torch.maximum(maximum, rank_input.abs().max())
    limit = max(1, 127 // contract.world_size)
    scale = maximum / float(limit)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    reference_int = torch.zeros_like(initial_input, dtype=torch.int32)
    for rank_input in rank_inputs:
        reference_int.add_(
            rank_input.div(scale).round().clamp(-limit, limit).to(torch.int32)
        )
    return reference_int.float().mul_(scale)


def execute_workload(
    contract: GradientCompressionResultContract,
    *,
    rank: int,
    device: torch.device,
    group: Any = None,
    profile_rank: int | None = None,
    publish_result: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor, DistributedRankWorkReceipt]:
    """Execute one complete worker workload on an initialized process group."""

    contract.validate()
    if not dist.is_initialized():
        raise RuntimeError("Gradient-compression process group is not initialized")
    if dist.get_world_size(group) != contract.world_size:
        raise RuntimeError("Gradient-compression process-group world size mismatch")
    if dist.get_rank(group) != rank:
        raise RuntimeError("Gradient-compression process-group rank mismatch")
    if profile_rank is not None and not 0 <= profile_rank < contract.world_size:
        raise ValueError("profile_rank must identify a participating rank")

    initial_input = _make_input(contract, rank=rank, device=device)
    buffers = _make_buffers(contract, initial_input)
    _prepare_comm_only(contract, initial_input, buffers, group=group)
    for _ in range(contract.warmup):
        _run_iteration(
            contract,
            initial_input,
            buffers,
            group=group,
            recorder=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier(group=group)

    topology = distributed_topology(contract)
    backend = str(dist.get_backend(group))
    recorder = DistributedWorkRecorder(topology, rank=rank, backend=backend)
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        started_ns = time.perf_counter_ns()
    recorder.begin_timed_region()
    final_output: torch.Tensor | None = None
    with nvtx_range(
        _profile_range(contract),
        enable=device.type == "cuda" and (profile_rank is None or rank == profile_rank),
    ):
        for _ in range(contract.iterations):
            final_output = _run_iteration(
                contract,
                initial_input,
                buffers,
                group=group,
                recorder=recorder,
            )
        recorder.run_final_barrier(lambda: dist.barrier(group=group))
    receipt = recorder.close_timed_region()
    if device.type == "cuda":
        assert start_event is not None and end_event is not None
        end_event.record()
        end_event.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event))
    else:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    elapsed_max = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX, group=group)
    time_per_iter_ms = float(elapsed_max.item()) / contract.iterations
    if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
        raise RuntimeError(
            "Gradient-compression worker timing must be finite and positive"
        )
    if final_output is None:
        raise RuntimeError("Gradient-compression worker produced no timed output")
    timed_output = _dequantize_comm_only(contract, final_output, buffers)

    # Numerical verification and payload serialization stay outside the measured
    # region. Profiler-only launches do not create multi-gigabyte success-shaped
    # artifacts because no parent freshness callback can consume them.
    reference = _build_independent_reference(
        contract,
        initial_input,
        group=group,
    )
    assert_close_full(
        timed_output,
        reference,
        rtol=contract.output_rtol,
        atol=contract.output_atol,
        label=f"rank {rank} worker output/reference",
    )
    if publish_result:
        write_gradient_compression_child_result(
            contract=contract,
            rank=rank,
            initial_input=initial_input,
            reference_output=reference,
            timed_output=timed_output,
            work_receipt=receipt,
        )
    return time_per_iter_ms, timed_output, reference, receipt


def _init_distributed() -> tuple[int, torch.device]:
    require_min_gpus(WORLD_SIZE)
    if not dist.is_available():
        raise RuntimeError("SKIPPED: torch.distributed is unavailable")
    missing = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ]
    if missing:
        raise RuntimeError(
            "SKIPPED: gradient-compression worker requires torchrun rank context"
        )
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=300),
            device_id=local_rank,
        )
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(
            f"SKIPPED: gradient-compression worker requires exactly {WORLD_SIZE} ranks"
        )
    return rank, torch.device("cuda", local_rank)


def run_worker(contract: GradientCompressionResultContract, *, profile_rank: int | None) -> None:
    rank, device = _init_distributed()
    try:
        time_per_iter_ms, _, _, _ = execute_workload(
            contract,
            rank=rank,
            device=device,
            profile_rank=profile_rank,
            publish_result=child_result_requested(),
        )
        if rank == 0:
            print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}", flush=True)
            print(
                "[gradient_compression] "
                f"variant={contract.variant} pair={contract.pair_compression} "
                f"comm_only={contract.comm_only} tensor_mb={contract.tensor_size_mb} "
                f"bucket_mb={contract.bucket_mb} ranks={contract.world_size} "
                f"process_collective_mode={contract.process_collective_mode} "
                f"algorithm_evidence=declared_only raw_result_tensor_bytes="
                f"{contract.raw_result_tensor_bytes if child_result_requested() else 0}",
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--pair-compression", choices=("fp16", "int8"), required=True)
    parser.add_argument("--comm-only", action="store_true")
    parser.add_argument("--tensor-size-mb", type=int, default=TENSOR_SIZE_MB)
    parser.add_argument("--bucket-mb", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-rtol", type=float, required=True)
    parser.add_argument("--output-atol", type=float, required=True)
    parser.add_argument("--profile-rank", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = make_result_contract(
        variant=args.variant,
        pair_compression=args.pair_compression,
        comm_only=args.comm_only,
        world_size=WORLD_SIZE,
        tensor_size_mb=args.tensor_size_mb,
        bucket_mb=args.bucket_mb,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        output_tolerance=(args.output_rtol, args.output_atol),
    )
    run_worker(contract, profile_rank=args.profile_rank)


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))


__all__ = [
    "SEED",
    "TENSOR_SIZE_MB",
    "WORLD_SIZE",
    "execute_workload",
    "main",
]
