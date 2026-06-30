#!/usr/bin/env python3
"""Optimized: Pipeline Parallelism (1F1B schedule, single GPU).

Interleaves forward and backward micro-batches to reduce pipeline bubbles.
Launched via torchrun.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import os

from core.common.device_utils import resolve_local_rank
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from core.benchmark.gpu_requirements import require_min_gpus
from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    LaunchVia,
    TorchrunLaunchSpec,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

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


def _run_worker(
    iters: int,
    warmup: int,
    batch_size: Optional[int],
    seq_length: int,
    hidden: int,
    num_layers: Optional[int],
    num_micro_batches: Optional[int],
) -> None:
    require_min_gpus(2, "optimized_pipeline_parallel_1f1b.py")
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("optimized_pipeline_parallel_1f1b requires >=2 GPUs.")
    num_layers = _resolve_num_layers(num_layers, world_size)
    batch_size, num_micro_batches = _resolve_batch_config(batch_size, num_micro_batches, world_size)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    layers_per_stage = num_layers // world_size
    micro_batch_size = batch_size // num_micro_batches

    fwd_layers, bwd_layers = _build_stage_layers(hidden, layers_per_stage, world_size, device)

    if rank == 0:
        inputs = torch.randn(batch_size, seq_length, hidden, device=device, dtype=torch.bfloat16)
    else:
        inputs = None
    recv_micro_batch: Optional[torch.Tensor] = None
    if rank > 0:
        recv_micro_batch = torch.empty(
            micro_batch_size,
            seq_length,
            hidden,
            device=device,
            dtype=torch.bfloat16,
        )
    recv_grad: Optional[torch.Tensor] = None
    if rank < world_size - 1:
        recv_grad = torch.empty(
            micro_batch_size,
            seq_length,
            hidden,
            device=device,
            dtype=torch.bfloat16,
        )

    def _forward(micro_batch: torch.Tensor) -> torch.Tensor:
        x = micro_batch
        return _run_stage_inplace(fwd_layers[0], x)

    def _backward(grad_in: torch.Tensor) -> torch.Tensor:
        x = grad_in
        return _run_stage_inplace(bwd_layers[0], x)

    warmup_steps = min(world_size, num_micro_batches)
    activation_slots: list[Optional[torch.Tensor]] = [None] * warmup_steps

    def _run_iteration() -> None:
        activation_head = 0
        activation_count = 0

        for micro_idx in range(warmup_steps):
            if rank == 0:
                start_idx = micro_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                micro_batch = inputs[start_idx:end_idx]
            else:
                if recv_micro_batch is None:
                    raise RuntimeError("recv microbatch buffer missing")
                micro_batch = recv_micro_batch
                dist.recv(micro_batch, src=rank - 1)

            out = _forward(micro_batch)
            activation_slots[(activation_head + activation_count) % warmup_steps] = out
            activation_count += 1
            if rank < world_size - 1:
                dist.send(out, dst=rank + 1)

        for micro_idx in range(warmup_steps, num_micro_batches):
            activation = activation_slots[activation_head]
            if activation is None:
                raise RuntimeError("activation slot missing")
            activation_slots[activation_head] = None
            activation_head = (activation_head + 1) % warmup_steps
            activation_count -= 1
            if rank < world_size - 1:
                if recv_grad is None:
                    raise RuntimeError("recv grad buffer missing")
                grad_in = recv_grad
                dist.recv(grad_in, src=rank + 1)
            else:
                grad_in = activation

            grad = _backward(grad_in)
            if rank > 0:
                dist.send(grad, dst=rank - 1)

            if rank == 0:
                start_idx = micro_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                micro_batch = inputs[start_idx:end_idx]
            else:
                if recv_micro_batch is None:
                    raise RuntimeError("recv microbatch buffer missing")
                micro_batch = recv_micro_batch
                dist.recv(micro_batch, src=rank - 1)

            out = _forward(micro_batch)
            activation_slots[(activation_head + activation_count) % warmup_steps] = out
            activation_count += 1
            if rank < world_size - 1:
                dist.send(out, dst=rank + 1)

        while activation_count:
            activation = activation_slots[activation_head]
            if activation is None:
                raise RuntimeError("activation slot missing")
            activation_slots[activation_head] = None
            activation_head = (activation_head + 1) % warmup_steps
            activation_count -= 1
            if rank < world_size - 1:
                if recv_grad is None:
                    raise RuntimeError("recv grad buffer missing")
                grad_in = recv_grad
                dist.recv(grad_in, src=rank + 1)
            else:
                grad_in = activation

            grad = _backward(grad_in)
            if rank > 0:
                dist.send(grad, dst=rank - 1)

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _run_iteration()
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            _run_iteration()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if rank == 0:
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized 1F1B pipeline parallel benchmark")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
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
    )


class OptimizedPipelineParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
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
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
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
        x = self._micro_batch
        for _ in self._world_size_range:
            for layer in self._fwd_layers:
                x = layer(x)
                x.relu_()
        for _ in self._world_size_range:
            for layer in self._bwd_layers:
                x = layer(x)
                x.relu_()
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

    def _prepare_verification_payload(self) -> None:
        if hasattr(self, "_subprocess_verify_output"):
            return
        self.setup()
        try:
            self.benchmark_fn()
            self.capture_verification_payload()
            self._subprocess_verify_output = self.get_verify_output()
            self._subprocess_output_tolerance = self.get_output_tolerance()
            self._subprocess_input_signature = self.get_input_signature()
        finally:
            self.teardown()

    def teardown(self) -> None:
        self._fwd_layers = None
        self._bwd_layers = None
        self._input = None
        self._micro_batch = None
        self._output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=max(torch.cuda.device_count(), 1),
            iterations=3,
            warmup=5,
            multi_gpu_required=True,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=True,
            name="optimized_pipeline_parallel_1f1b",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedPipelineParallelBenchmark()
