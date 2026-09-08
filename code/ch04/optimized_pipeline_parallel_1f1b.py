#!/usr/bin/env python3
"""Optimized: Pipeline Parallelism (1F1B schedule, single GPU).

Interleaves forward and backward micro-batches to reduce pipeline bubbles.
Launched via torchrun.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import resolve_local_rank
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.profiling.nvtx_helper import nvtx_range
from core.utils.logger import get_logger
from core.utils.worker_seed import apply_worker_seed

from ch04.pipeline_parallel_common import (
    PIPELINE_RESULT_CALLBACK,
    PipelineIterationCapture,
    PipelineParallelChildResultMixin,
    pipeline_child_result_requested,
    run_1f1b_iteration,
    verify_and_concatenate_pipeline_capture,
    write_pipeline_child_result,
)

logger = get_logger(__name__)

PROFILE_NVTX_RANGE = "compute_kernel:pipeline_parallel_1f1b"
PIPELINE_SOURCE = "ch04.optimized_pipeline_parallel_1f1b"
PIPELINE_VARIANT = "optimized"
PIPELINE_SCHEDULE = "1f1b-paired-p2p"

_DEFAULT_BATCH = 32
_DEFAULT_SEQ = 2048
_DEFAULT_HIDDEN = 4096
_DEFAULT_LAYERS = 8
_DEFAULT_MICRO_BATCHES = 8


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for pipeline-parallel benchmark")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires >=2 GPUs.")
    return world_size


def _resolve_num_layers(num_layers: Optional[int], stage_count: int) -> int:
    base = _DEFAULT_LAYERS if num_layers is None else int(num_layers)
    if base % stage_count == 0:
        return base
    if num_layers is not None:
        raise ValueError("num_layers must be divisible by stage_count")
    return stage_count * ((base + stage_count - 1) // stage_count)


def _resolve_batch_config(
    batch_size: Optional[int],
    num_micro_batches: Optional[int],
    stage_count: int,
) -> tuple[int, int]:
    if num_micro_batches is None:
        micro_batches = max(_DEFAULT_MICRO_BATCHES, stage_count)
    else:
        micro_batches = int(num_micro_batches)
    batch = _DEFAULT_BATCH if batch_size is None else int(batch_size)
    if batch % micro_batches == 0:
        return batch, micro_batches
    if batch_size is not None:
        raise ValueError("batch_size must be divisible by num_micro_batches")
    adjusted_batch = micro_batches * ((batch + micro_batches - 1) // micro_batches)
    return adjusted_batch, micro_batches


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def _build_stage_layers(hidden: int, layers_per_stage: int, stage_count: int, device: torch.device):
    fwd_stages = nn.ModuleList()
    bwd_stages = nn.ModuleList()
    for _ in range(stage_count):
        fwd = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False)
            for _ in range(layers_per_stage)
        ]).to(device).to(torch.bfloat16)
        bwd = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False)
            for _ in range(layers_per_stage)
        ]).to(device).to(torch.bfloat16)
        fwd_stages.append(fwd)
        bwd_stages.append(bwd)
    return fwd_stages, bwd_stages


def _run_stage_inplace(stage_layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in stage_layers:
        x = layer(x)
        x.relu_()
    return x


def _run_rank_stage_inplace(
    stages: nn.ModuleList,
    rank: int,
    x: torch.Tensor,
) -> torch.Tensor:
    if rank < 0 or rank >= len(stages):
        raise IndexError(f"pipeline rank {rank} is outside {len(stages)} stages")
    return _run_stage_inplace(stages[rank], x)


def _run_contiguous_1f1b_iterations(
    *,
    rank: int,
    world_size: int,
    micro_batches_per_iteration: int,
    iteration_count: int,
    get_rank0_microbatch: Callable[[int], torch.Tensor],
    recv_forward_buffers: Sequence[torch.Tensor],
    recv_backward_buffers: Sequence[torch.Tensor],
    forward_step: Callable[[torch.Tensor], torch.Tensor],
    backward_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    activation_slots: list[Optional[tuple[int, torch.Tensor]]],
    capture: Optional[PipelineIterationCapture] = None,
) -> None:
    """Amortize boundaries across fixed-state iterations with no intervening update.

    This is valid for this benchmark's fixed weights and repeated input batch. A
    training loop that updates parameters between iterations must drain before
    each update instead of using this helper across that boundary.
    """

    if iteration_count <= 0:
        raise ValueError("iteration_count must be positive")
    if micro_batches_per_iteration <= 0:
        raise ValueError("micro_batches_per_iteration must be positive")
    if micro_batches_per_iteration < world_size:
        raise ValueError("Each logical iteration needs at least one microbatch per stage")
    scheduled_micro_batches = micro_batches_per_iteration * iteration_count

    def get_repeated_rank0_microbatch(index: int) -> torch.Tensor:
        return get_rank0_microbatch(index % micro_batches_per_iteration)

    run_1f1b_iteration(
        rank=rank,
        world_size=world_size,
        num_micro_batches=scheduled_micro_batches,
        get_rank0_microbatch=get_repeated_rank0_microbatch,
        # Only references are repeated. The same fixed tensor slots are safe to
        # receive into after their prior microbatch has completed its stage.
        recv_forward_buffers=list(recv_forward_buffers) * iteration_count,
        recv_backward_buffers=list(recv_backward_buffers) * iteration_count,
        forward_step=forward_step,
        backward_step=backward_step,
        activation_slots=activation_slots,
        capture=capture,
    )


def _run_worker(
    iters: int,
    warmup: int,
    batch_size: Optional[int],
    seq_length: int,
    hidden: int,
    num_layers: Optional[int],
    num_micro_batches: Optional[int],
    seed: int,
) -> None:
    require_min_gpus(2, "optimized_pipeline_parallel_1f1b.py")
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires >=2 GPUs.")
    num_layers = _resolve_num_layers(num_layers, world_size)
    batch_size, num_micro_batches = _resolve_batch_config(batch_size, num_micro_batches, world_size)

    apply_worker_seed(seed)

    device = torch.device(f"cuda:{local_rank}")
    layers_per_stage = num_layers // world_size
    micro_batch_size = batch_size // num_micro_batches

    fwd_layers, bwd_layers = _build_stage_layers(hidden, layers_per_stage, world_size, device)

    if rank == 0:
        inputs = torch.randn(batch_size, seq_length, hidden, device=device, dtype=torch.bfloat16)
    else:
        inputs = None
    recv_micro_batches: list[torch.Tensor] = []
    recv_micro_batch: Optional[torch.Tensor] = None
    if rank > 0:
        for _ in range(num_micro_batches):
            recv_micro_batch = torch.empty(
                micro_batch_size,
                seq_length,
                hidden,
                device=device,
                dtype=torch.bfloat16,
            )
            recv_micro_batches.append(recv_micro_batch)
    recv_grads: list[torch.Tensor] = []
    recv_grad: Optional[torch.Tensor] = None
    if rank < world_size - 1:
        for _ in range(num_micro_batches):
            recv_grad = torch.empty(
                micro_batch_size,
                seq_length,
                hidden,
                device=device,
                dtype=torch.bfloat16,
            )
            recv_grads.append(recv_grad)

    def _forward(micro_batch: torch.Tensor) -> torch.Tensor:
        return _run_rank_stage_inplace(fwd_layers, rank, micro_batch)

    def _backward(_activation: torch.Tensor, grad_in: torch.Tensor) -> torch.Tensor:
        return _run_rank_stage_inplace(bwd_layers, rank, grad_in)

    if rank == 0 and inputs is None:
        raise RuntimeError("rank zero pipeline input is missing")

    def _get_rank0_microbatch(micro_idx: int) -> torch.Tensor:
        if inputs is None:
            raise RuntimeError("Only rank zero owns pipeline input microbatches")
        start_idx = micro_idx * micro_batch_size
        return inputs.narrow(0, start_idx, micro_batch_size)

    warmup_steps = min(world_size - rank - 1, num_micro_batches)
    activation_slots: list[Optional[tuple[int, torch.Tensor]]] = [None] * max(
        warmup_steps, 1
    )

    def _run_contiguous_iterations(
        iteration_count: int,
        capture: Optional[PipelineIterationCapture] = None,
    ) -> None:
        _run_contiguous_1f1b_iterations(
            rank=rank,
            world_size=world_size,
            micro_batches_per_iteration=num_micro_batches,
            iteration_count=iteration_count,
            get_rank0_microbatch=_get_rank0_microbatch,
            recv_forward_buffers=recv_micro_batches,
            recv_backward_buffers=recv_grads,
            forward_step=_forward,
            backward_step=_backward,
            activation_slots=activation_slots,
            capture=capture,
        )

    result_requested = pipeline_child_result_requested()
    captured_iteration: Optional[PipelineIterationCapture] = None
    with torch.inference_mode():
        warmup_iterations = max(warmup, 0)
        if warmup_iterations:
            _run_contiguous_iterations(warmup_iterations)
        torch.cuda.synchronize(device)

        with nvtx_range(PROFILE_NVTX_RANGE, enable=True):
            start = time.perf_counter()
            measured_iterations = max(iters, 1)
            if result_requested:
                captured_iteration = PipelineIterationCapture.create(
                    num_micro_batches,
                    first_microbatch_index=(measured_iterations - 1)
                    * num_micro_batches,
                )
            _run_contiguous_iterations(measured_iterations, captured_iteration)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

        time_per_iter_ms = (elapsed / measured_iterations) * 1000.0
        if result_requested:
            if captured_iteration is None:
                raise RuntimeError("Pipeline timed iteration did not retain its actual outputs")
            forward_stages = [
                (lambda value, stage=stage: _run_stage_inplace(stage, value))
                for stage in fwd_layers
            ]
            backward_stages = [
                (lambda value, stage=stage: _run_stage_inplace(stage, value))
                for stage in bwd_layers
            ]
            verify_input, verify_output = verify_and_concatenate_pipeline_capture(
                rank=rank,
                capture=captured_iteration,
                forward_stages=forward_stages,
                backward_stages=backward_stages,
            )
            if not write_pipeline_child_result(
                verify_input=verify_input,
                verify_output=verify_output,
                completed_iterations=measured_iterations,
                time_per_iter_ms=time_per_iter_ms,
                reference_verified=True,
            ):
                raise RuntimeError("Pipeline child-result request disappeared before publication")

    if rank == 0:
        print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized 1F1B pipeline parallel benchmark")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global batch size (defaults to a world_size-aligned value).",
    )
    parser.add_argument("--seq-length", type=int, default=_DEFAULT_SEQ)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULT_HIDDEN)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Layer count (defaults to a world_size-aligned value).",
    )
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=None,
        help="Micro-batch count (defaults to a world_size-aligned value).",
    )
    args = parser.parse_args()
    _run_worker(
        args.iters,
        args.warmup,
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        args.num_layers,
        args.micro_batches,
        seed=args.seed,
    )


class OptimizedPipelineParallelBenchmark(
    PipelineParallelChildResultMixin,
    VerificationPayloadMixin,
    BaseBenchmark,
):
    preferred_ncu_replay_mode = "app-range"

    """Harness entry that launches this module via torchrun."""
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_SEQ)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)
        self._fwd_layers: Optional[nn.ModuleList] = None
        self._bwd_layers: Optional[nn.ModuleList] = None
        self._input: Optional[torch.Tensor] = None
        self._micro_batch: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._world_size = 1
        self._world_size_range = range(self._world_size)
        self._num_layers = _DEFAULT_LAYERS
        self._batch_size = _DEFAULT_BATCH
        self._micro_batches = _DEFAULT_MICRO_BATCHES
        self._layers_per_stage = _DEFAULT_LAYERS

    def setup(self) -> None:
        require_min_gpus(2, "optimized_pipeline_parallel_1f1b.py")
        self._world_size = torch.cuda.device_count()
        self._world_size_range = range(self._world_size)
        self._num_layers = _resolve_num_layers(None, self._world_size)
        self._batch_size, self._micro_batches = _resolve_batch_config(None, None, self._world_size)
        self._layers_per_stage = self._num_layers // self._world_size
        self._fwd_layers, self._bwd_layers = _build_stage_layers(
            _DEFAULT_HIDDEN,
            self._layers_per_stage,
            self._world_size,
            self.device,
        )
        self._input = torch.randn(
            self._batch_size,
            _DEFAULT_SEQ,
            _DEFAULT_HIDDEN,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._micro_batch = self._input.narrow(0, 0, self._batch_size // self._micro_batches)

    def benchmark_fn(self) -> None:
        if (
            self._input is None
            or self._micro_batch is None
            or self._fwd_layers is None
            or self._bwd_layers is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        if len(self._fwd_layers) != self._world_size or len(self._bwd_layers) != self._world_size:
            raise RuntimeError("pipeline stage count does not match world size")
        x = self._micro_batch
        stage_layers = iter(self._fwd_layers)
        for _ in self._world_size_range:
            x = _run_stage_inplace(next(stage_layers), x)
        stage_layers = reversed(self._bwd_layers)
        for _ in self._world_size_range:
            x = _run_stage_inplace(next(stage_layers), x)
        self._output = x

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        param_count = 2 * self._num_layers * (_DEFAULT_HIDDEN * _DEFAULT_HIDDEN)
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=self._batch_size,
            parameter_count=int(param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
            output_tolerance=(0.1, 1.0),
            signature_overrides={
                "world_size": self._world_size,
                "pipeline_stages": self._world_size,
                "pipeline_stage_boundaries": [
                    (stage_idx * self._layers_per_stage, (stage_idx + 1) * self._layers_per_stage - 1)
                    for stage_idx in range(self._world_size)
                ],
                "collective_type": "send_recv",
            },
        )

    def teardown(self) -> None:
        self.retain_failed_pipeline_child_result()
        self._fwd_layers = None
        self._bwd_layers = None
        self._input = None
        self._micro_batch = None
        self._output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=max(torch.cuda.device_count(), 1),
            iterations=3,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
            nsys_nvtx_include=[PROFILE_NVTX_RANGE],
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        effective_config = config or self.get_config()
        world_size = int(effective_config.nproc_per_node or max(torch.cuda.device_count(), 1))
        num_layers = _resolve_num_layers(None, world_size)
        result_env = self.prepare_pipeline_child_result(
            source=PIPELINE_SOURCE,
            variant=PIPELINE_VARIANT,
            schedule=PIPELINE_SCHEDULE,
            world_size=world_size,
            iterations=int(effective_config.iterations),
            batch_size=_DEFAULT_BATCH,
            seq_length=_DEFAULT_SEQ,
            hidden=_DEFAULT_HIDDEN,
            num_layers=num_layers,
        )
        return TorchrunLaunchSpec(
            module_name="core.harness.benchmark_worker",
            script_args=["--module", "ch04.optimized_pipeline_parallel_1f1b", "--callable", "main", "--"],
            env=result_env,
            multi_gpu_required=True,
            name="optimized_pipeline_parallel_1f1b",
            result_callback=PIPELINE_RESULT_CALLBACK,
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
                "seed": "--seed",
            },
            timing_source="rank0_time_per_iter_ms",
            timing_iterations_per_sample=int(effective_config.iterations),
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineParallelBenchmark()
