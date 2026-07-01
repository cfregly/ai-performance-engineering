"""optimized_torchcomms.py - Modern torchcomms API patterns (single GPU)."""

from __future__ import annotations

from pathlib import Path

import argparse
import os

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from ch04.torchcomms_common import BufferedTorchcommsBlock
from core.common.device_utils import resolve_local_rank
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

_DEFAULT_BATCH = 512
_DEFAULT_HIDDEN = 4096

try:
    from torch.distributed._functional_collectives import all_reduce as functional_all_reduce
except ImportError as exc:  # pragma: no cover - torchcomms required for this benchmark
    functional_all_reduce = None
    _TORCHCOMMS_IMPORT_ERROR = exc
else:
    _TORCHCOMMS_IMPORT_ERROR = None


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("optimized_torchcomms requires torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def _require_torchcomms() -> None:
    if functional_all_reduce is None:
        raise RuntimeError("torchcomms functional collectives are required for optimized_torchcomms")


def _resolve_world_size() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for torchcomms benchmark")
    return 1


def _build_block(hidden: int, device: torch.device) -> nn.Module:
    return BufferedTorchcommsBlock(hidden).to(device).eval()


def _run_worker(iters: int, warmup: int, batch: int, hidden: int) -> None:
    _require_torchcomms()
    rank, world_size, local_rank = _init_distributed()
    if world_size < 1:
        raise RuntimeError("optimized_torchcomms requires >=1 GPU.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    comm_block = _build_block(hidden, device)
    aux_block = _build_block(hidden, device)
    inputs = torch.randn(batch, hidden, device=device)
    comm_block.prepare_forward_buffers(inputs)
    aux_block.prepare_forward_buffers(inputs)
    comm_stream = torch.cuda.Stream()

    def _step() -> None:
        with torch.inference_mode():
            comm_out = comm_block.forward_prepared(inputs)
            if world_size > 1:
                current_stream = torch.cuda.current_stream()
                comm_stream.wait_stream(current_stream)
                with torch.cuda.stream(comm_stream):
                    reduced = functional_all_reduce(
                        comm_out,
                        reduceOp="avg",
                        group=dist.group.WORLD,
                    )
            else:
                reduced = comm_out
            aux_out = aux_block.forward_prepared(inputs)
            if world_size > 1:
                current_stream.wait_stream(comm_stream)
            reduced.add_(aux_out)

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch * hidden
    tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))

    if rank == 0:
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters,1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized torchcomms benchmark")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--hidden", type=int, default=_DEFAULT_HIDDEN)
    args = parser.parse_args()
    _run_worker(args.iters, args.warmup, args.batch, args.hidden)


class OptimizedTorchcommsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness entry that launches this module via torchrun."""

    def __init__(self) -> None:
        super().__init__()
        tokens = float(_DEFAULT_BATCH * _DEFAULT_HIDDEN)
        self.register_workload_metadata(requests_per_iteration=float(_DEFAULT_BATCH), tokens_per_iteration=tokens)
        self._comm_block: Optional[nn.Module] = None
        self._aux_block: Optional[nn.Module] = None
        self._input: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._world_size = _resolve_world_size()
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._comm_block = _build_block(_DEFAULT_HIDDEN, self.device)
        self._aux_block = _build_block(_DEFAULT_HIDDEN, self.device)
        self._payload_parameter_count = sum(p.numel() for p in self._comm_block.parameters())
        self._payload_parameter_count += sum(p.numel() for p in self._aux_block.parameters())
        self._input = torch.randn(
            _DEFAULT_BATCH,
            _DEFAULT_HIDDEN,
            device=self.device,
            dtype=torch.float32,
        )
        self._comm_block.prepare_forward_buffers(self._input)
        self._aux_block.prepare_forward_buffers(self._input)

    def benchmark_fn(self) -> None:
        if self._comm_block is None or self._aux_block is None or self._input is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode():
            comm_out = self._comm_block.forward_prepared(self._input)
            aux_out = self._aux_block.forward_prepared(self._input)
            comm_out.add_(aux_out)
            self._output = comm_out

    def capture_verification_payload(self) -> None:
        if self._output is None or self._input is None or self._comm_block is None or self._aux_block is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._input},
            output=self._output,
            batch_size=_DEFAULT_BATCH,
            parameter_count=self._payload_parameter_count,
            precision_flags=PrecisionFlags(tf32=False),
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={
                "world_size": self._world_size,
                "collective_type": "all_reduce",
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
        self._comm_block = None
        self._aux_block = None
        self._input = None
        self._output = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self._output is None:
            return "No output captured"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            launch_via=LaunchVia.TORCHRUN,
            nproc_per_node=1,
            iterations=50,
            warmup=5,
            multi_gpu_required=False,
            measurement_timeout_seconds=900,
        )

    def get_torchrun_spec(self, config: Optional[BenchmarkConfig] = None) -> TorchrunLaunchSpec:
        self._prepare_verification_payload()
        return TorchrunLaunchSpec(
            script_path=Path(__file__).resolve(),
            script_args=[],
            multi_gpu_required=False,
            name="optimized_torchcomms",
            config_arg_map={
                "iterations": "--iters",
                "warmup": "--warmup",
            },
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedTorchcommsBenchmark()
