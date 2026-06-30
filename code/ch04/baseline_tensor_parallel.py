#!/usr/bin/env python3
"""Baseline: Tensor Parallelism without communication overlap (single GPU).

Synchronous all-gather after each shard computation; launched via torchrun.
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

_DEFAULT_BATCH = 8
_DEFAULT_SEQ = 4096
_DEFAULT_HIDDEN = 4096
_DEFAULT_LAYERS = 4


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for tensor-parallel benchmark")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("baseline_tensor_parallel requires >=2 GPUs.")
    return world_size


def _resolve_hidden(hidden: Optional[int], world_size: int) -> int:
    base = _DEFAULT_HIDDEN if hidden is None else int(hidden)
    if base % world_size == 0:
        return base
    if hidden is not None:
        raise ValueError("hidden_size must be divisible by world_size")
    return world_size * ((base + world_size - 1) // world_size)


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("baseline_tensor_parallel requires torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def _build_layers(hidden: int, hidden_per_rank: int, num_layers: int, device: torch.device):
    shard = nn.ModuleList([
        nn.Linear(hidden, hidden_per_rank, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    proj = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    aux = nn.ModuleList([
        nn.Linear(hidden, hidden, bias=False)
        for _ in range(num_layers)
    ]).to(device).to(torch.bfloat16)
    return shard, proj, aux


def _replicate_tensor_parallel_shard(
    local_out: torch.Tensor,
    world_size: int,
    full_out: torch.Tensor,
) -> torch.Tensor:
    hidden_per_rank = local_out.shape[-1]
    full_out.view(*local_out.shape[:-1], world_size, hidden_per_rank).copy_(
        local_out.unsqueeze(-2)
    )
    return full_out


def _run_worker(
    iters: int,
    warmup: int,
    batch: int,
    seq_length: int,
    hidden: Optional[int],
    num_layers: int,
) -> None:
    require_min_gpus(2, "baseline_tensor_parallel.py")
    rank, world_size, local_rank = _init_distributed()
    if world_size < 2:
        raise RuntimeError("baseline_tensor_parallel requires >=2 GPUs.")
    hidden = _resolve_hidden(hidden, world_size)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    hidden_per_rank = hidden // world_size

    shard_layers, proj_layers, aux_layers = _build_layers(hidden, hidden_per_rank, num_layers, device)
    inputs = torch.randn(batch, seq_length, hidden, device=device, dtype=torch.bfloat16)
    gather_list = [torch.empty(batch, seq_length, hidden_per_rank, device=device, dtype=torch.bfloat16)
                   for _ in range(world_size)]
    full_out = torch.empty(batch, seq_length, hidden, device=device, dtype=torch.bfloat16)
    layer_range = range(num_layers)

    def _step() -> None:
        x = inputs
        for layer_idx in layer_range:
            local_out = shard_layers[layer_idx](x)
            dist.all_gather(gather_list, local_out)
            torch.cat(gather_list, dim=-1, out=full_out)
            aux_out = aux_layers[layer_idx](x)
            proj_out = proj_layers[layer_idx](full_out)
            x = proj_out + aux_out

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _step()
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            _step()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch * seq_length
    tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
    if rank == 0:
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline tensor parallel benchmark")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--seq-length", type=int, default=_DEFAULT_SEQ)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size (defaults to a world_size-aligned value).",
    )
    parser.add_argument("--num-layers", type=int, default=_DEFAULT_LAYERS)
    args = parser.parse_args()
    _run_worker(
        args.iters,
        args.warmup,
        args.batch_size,
        args.seq_length,
        args.hidden_size,
        args.num_layers,
    )


class BaselineTensorParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_SEQ)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)
        self._shard_layers: Optional[nn.ModuleList] = None
        self._proj_layers: Optional[nn.ModuleList] = None
        self._aux_layers: Optional[nn.ModuleList] = None
        self._input: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._full_out: Optional[torch.Tensor] = None
        self._world_size = 1
        self._hidden = _DEFAULT_HIDDEN
        self._hidden_per_rank = _DEFAULT_HIDDEN
        self._layer_range = range(0)
        self._payload_parameter_count = 0

    def setup(self) -> None:
        require_min_gpus(2, "baseline_tensor_parallel.py")
        self._world_size = torch.cuda.device_count()
        self._hidden = _resolve_hidden(None, self._world_size)
        self._hidden_per_rank = self._hidden // self._world_size
        self._layer_range = range(_DEFAULT_LAYERS)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._shard_layers, self._proj_layers, self._aux_layers = _build_layers(
            self._hidden,
            self._hidden_per_rank,
            _DEFAULT_LAYERS,
            self.device,
        )
        self._payload_parameter_count = 3 * _DEFAULT_LAYERS * (self._hidden * self._hidden)
        self._input = torch.randn(
            _DEFAULT_BATCH,
            _DEFAULT_SEQ,
            self._hidden,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._full_out = torch.empty(
            _DEFAULT_BATCH,
            _DEFAULT_SEQ,
            self._hidden,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def benchmark_fn(self) -> None:
        if (
            self._input is None
            or self._shard_layers is None
            or self._proj_layers is None
            or self._aux_layers is None
            or self._full_out is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        x = self._input
        for layer_idx in self._layer_range:
            local_out = self._shard_layers[layer_idx](x)
            _replicate_tensor_parallel_shard(local_out, self._world_size, self._full_out)
            aux_out = self._aux_layers[layer_idx](x)
            proj_out = self._proj_layers[layer_idx](self._full_out)
            x = proj_out + aux_out
        self._output = x

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=_DEFAULT_BATCH,
            parameter_count=self._payload_parameter_count,
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
            output_tolerance=(0.1, 1.0),
            signature_overrides={
                "world_size": self._world_size,
                "collective_type": "all_gather",
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
        self._shard_layers = None
        self._proj_layers = None
        self._aux_layers = None
        self._input = None
        self._output = None
        self._full_out = None
        self._layer_range = range(0)
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
            name="baseline_tensor_parallel",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineTensorParallelBenchmark()
