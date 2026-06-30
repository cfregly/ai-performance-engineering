"""Shared helpers for the NCCL/NIXL/NVSHMEM lab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.util
import math
from pathlib import Path
import shutil
from typing import Any, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.symmetric_memory_patch import symmetric_memory_available


_DEFAULT_TOTAL_BLOCKS = 512
_DEFAULT_SELECTED_BLOCKS = 96
_DEFAULT_BLOCK_KIB = 64
_DEFAULT_INNER_ITERATIONS = 8
_DEFAULT_SEED = 7
_FLOAT_SIZE_BYTES = 4


@dataclass(frozen=True)
class TierHandoffWorkload:
    """Workload for the single-GPU memory-tier handoff analogue."""

    total_blocks: int = _DEFAULT_TOTAL_BLOCKS
    selected_blocks: int = _DEFAULT_SELECTED_BLOCKS
    block_kib: int = _DEFAULT_BLOCK_KIB
    inner_iterations: int = _DEFAULT_INNER_ITERATIONS
    seed: int = _DEFAULT_SEED

    @property
    def block_elements(self) -> int:
        return (self.block_kib * 1024) // _FLOAT_SIZE_BYTES

    @property
    def selected_bytes(self) -> int:
        return self.selected_blocks * self.block_kib * 1024

    @property
    def bytes_per_iteration(self) -> int:
        # D2H and H2D roundtrip for the selected blocks.
        return self.selected_bytes * 2 * self.inner_iterations


def default_workload() -> TierHandoffWorkload:
    return TierHandoffWorkload()


def _override_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--total-blocks", type=int, default=None)
    parser.add_argument("--selected-blocks", type=int, default=None)
    parser.add_argument("--block-kib", type=int, default=None)
    parser.add_argument("--inner-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def apply_cli_overrides(workload: TierHandoffWorkload, argv: list[str]) -> TierHandoffWorkload:
    parsed, _ = _override_parser().parse_known_args(argv)
    values: dict[str, int] = {}
    for field_name in ("total_blocks", "selected_blocks", "block_kib", "inner_iterations", "seed"):
        value = getattr(parsed, field_name)
        if value is not None:
            values[field_name] = int(value)
    updated = replace(workload, **values)
    if updated.total_blocks <= 0:
        raise ValueError("--total-blocks must be positive")
    if updated.selected_blocks <= 0:
        raise ValueError("--selected-blocks must be positive")
    if updated.selected_blocks > updated.total_blocks:
        raise ValueError("--selected-blocks must be <= --total-blocks")
    if updated.block_kib <= 0:
        raise ValueError("--block-kib must be positive")
    if updated.inner_iterations <= 0:
        raise ValueError("--inner-iterations must be positive")
    return updated


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_nccl_tests_binary() -> Optional[Path]:
    candidate = _repo_root() / "tools" / "nccl-tests" / "build" / "all_reduce_perf"
    return candidate if candidate.exists() else None


def _find_nvshmem_launcher() -> Optional[str]:
    for candidate in ("nvshmemrun", "nvshmrun", "nvshmrun.hydra"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    for root in sorted(Path("/usr/bin").glob("nvshmem_*")):
        if not root.is_dir():
            continue
        for candidate in ("nvshmrun", "nvshmrun.hydra"):
            resolved = root / candidate
            if resolved.exists():
                return str(resolved)
    return None


def probe_communication_stack() -> dict[str, Any]:
    """Return an honest runtime matrix for this lab."""
    nccl_binary = shutil.which("all_reduce_perf")
    repo_nccl = _repo_nccl_tests_binary()
    effective_nccl = nccl_binary or (str(repo_nccl) if repo_nccl is not None else None)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count else None
    dist_available = bool(torch.distributed.is_available())
    nccl_available = bool(dist_available and torch.distributed.is_nccl_available())
    nixl_import = importlib.util.find_spec("nixl") is not None
    nvshmem_import = importlib.util.find_spec("nvshmem") is not None
    nvshmem_launcher = _find_nvshmem_launcher()
    symm_mem = bool(symmetric_memory_available())

    blockers: list[str] = []
    if gpu_count < 2:
        blockers.append("NCCL and NVSHMEM data-path validation need >=2 GPUs; this host has 1 GPU.")
    if not nixl_import:
        blockers.append("NIXL Python bindings are not importable on this host.")
    if nvshmem_launcher is None:
        blockers.append("NVSHMEM launcher is unavailable; expected nvshmemrun, nvshmrun, or a versioned /usr/bin/nvshmem_*/nvshmrun path.")

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "torch_distributed_available": dist_available,
        "torch_nccl_available": nccl_available,
        "nccl_tests_binary": effective_nccl,
        "nixl_import_available": nixl_import,
        "nvshmem_import_available": nvshmem_import,
        "nvshmem_launcher": nvshmem_launcher,
        "symmetric_memory_available": symm_mem,
        "can_run_multi_gpu_nccl": bool(nccl_available and gpu_count >= 2 and effective_nccl),
        "can_run_nvshmem_one_sided": bool(symm_mem and gpu_count >= 2 and nvshmem_launcher),
        "recommended_local_path": "Use the tier_handoff benchmark pair and standalone compare/sweep runner on this 1-GPU host.",
        "blockers": blockers,
    }


def require_stack(probe: dict[str, Any], stack: str) -> None:
    """Fail clearly when a requested stack is unavailable."""
    if stack == "nccl":
        if not probe["can_run_multi_gpu_nccl"]:
            raise RuntimeError(
                "Requested NCCL validation is unavailable: need torch.distributed NCCL, "
                "repo/path nccl-tests binary, and >=2 visible GPUs."
            )
        return
    if stack == "nixl":
        if not probe["nixl_import_available"]:
            raise RuntimeError("Requested NIXL validation is unavailable: `nixl` is not importable.")
        return
    if stack == "nvshmem":
        if not probe["can_run_nvshmem_one_sided"]:
            raise RuntimeError(
                "Requested NVSHMEM validation is unavailable: need symmetric memory, "
                "an NVSHMEM launcher (`nvshmemrun` or `nvshmrun`), and >=2 visible GPUs."
            )
        return
    if stack == "symmetric-memory":
        if not probe["symmetric_memory_available"]:
            raise RuntimeError("Requested symmetric-memory validation is unavailable on this host.")
        return
    raise ValueError(f"Unknown stack requirement: {stack}")


def _selected_indices(workload: TierHandoffWorkload, device: torch.device) -> torch.Tensor:
    return _selected_indices_cpu(workload).to(device=device)


def _selected_indices_cpu(workload: TierHandoffWorkload) -> torch.Tensor:
    stride = 17
    while math.gcd(stride, workload.total_blocks) != 1:
        stride += 2
    indices = (torch.arange(workload.selected_blocks, dtype=torch.long) * stride + 7) % workload.total_blocks
    return indices


class TierHandoffBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU analogue for staged vs transport-aware memory-tier movement."""

    allowed_benchmark_fn_antipatterns = ("host_transfer", "sync")
    preferred_ncu_replay_mode = "application"

    def __init__(self, *, optimized: bool, workload: Optional[TierHandoffWorkload] = None) -> None:
        super().__init__()
        self.optimized = optimized
        self.workload = workload or default_workload()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.host_stage: Optional[torch.Tensor] = None
        self.gpu_stage: Optional[torch.Tensor] = None
        self.packed_stage: Optional[torch.Tensor] = None
        self.selected_idx: Optional[torch.Tensor] = None
        self.selected_cpu: Optional[list[int]] = None
        self.selected_copy_pairs: Optional[list[tuple[int, int]]] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._expected_buffer: Optional[torch.Tensor] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.copy_ready: Optional[torch.cuda.Event] = None
        self.baseline_copy_ready: Optional[torch.cuda.Event] = None
        self.output: Optional[torch.Tensor] = None
        self._selected_blocks_metric = 0.0
        self._block_kib_metric = 0.0
        self._inner_iterations_metric = 0.0
        self._bytes_per_iteration_mb = 0.0
        self._inner_iteration_range = range(0)
        self._baseline_copy_calls_metric = 0.0
        self._optimized_copy_calls_metric = 0.0
        self._metrics: dict[str, float] = {
            "tier_handoff.selected_blocks": 0.0,
            "tier_handoff.block_kib": 0.0,
            "tier_handoff.inner_iterations": 0.0,
            "tier_handoff.copy_calls": 0.0,
            "tier_handoff.uses_copy_stream": 0.0,
            "tier_handoff.bytes_per_iteration_mb": 0.0,
        }
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        bytes_per_iteration = float(self.workload.bytes_per_iteration)
        inner_iterations = int(self.workload.inner_iterations)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=bytes_per_iteration,
        )
        self._selected_blocks_metric = float(self.workload.selected_blocks)
        self._block_kib_metric = float(self.workload.block_kib)
        self._inner_iterations_metric = float(inner_iterations)
        self._bytes_per_iteration_mb = bytes_per_iteration / (1024.0 * 1024.0)
        self._inner_iteration_range = range(inner_iterations)
        self._baseline_copy_calls_metric = float(self.workload.selected_blocks * 2 * inner_iterations)
        self._optimized_copy_calls_metric = float(2 * inner_iterations)

    def _reset_metrics(self) -> None:
        for key in self._metrics:
            self._metrics[key] = 0.0

    def set_workload(self, workload: TierHandoffWorkload) -> None:
        self.workload = workload
        self._refresh_workload_metadata()

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA is required for labs/nccl_nixl_nvshmem")

        block_elems = self.workload.block_elements
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.workload.seed)
        self.src = torch.randn(
            self.workload.total_blocks,
            block_elems,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        self.dst = torch.empty_like(self.src)
        self.host_stage = torch.empty(
            self.workload.selected_blocks,
            block_elems,
            dtype=torch.float32,
            pin_memory=True,
        )
        self.gpu_stage = torch.empty(
            self.workload.selected_blocks,
            block_elems,
            device=self.device,
            dtype=torch.float32,
        )
        self.packed_stage = torch.empty_like(self.gpu_stage) if self.optimized else None
        self._output_buffer = torch.empty_like(self.gpu_stage)
        self._expected_buffer = torch.empty_like(self.gpu_stage)
        selected_cpu = _selected_indices_cpu(self.workload)
        self.selected_idx = selected_cpu.to(device=self.device)
        self.selected_cpu = [int(idx) for idx in selected_cpu.tolist()] if not self.optimized else None
        self.selected_copy_pairs = list(enumerate(self.selected_cpu)) if not self.optimized else None
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.optimized else None
        self.copy_ready = torch.cuda.Event() if self.optimized else None
        self.baseline_copy_ready = torch.cuda.Event() if not self.optimized else None
        self.output = None
        self._reset_metrics()
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.src is None
            or self.dst is None
            or self.host_stage is None
            or self.gpu_stage is None
            or self.selected_idx is None
            or self._output_buffer is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        if not self.optimized:
            if self.selected_cpu is None or self.selected_copy_pairs is None or self.baseline_copy_ready is None:
                raise RuntimeError("Baseline path requires setup() to cache selected CPU indices")
            selected_copy_pairs = self.selected_copy_pairs
            copy_ready = self.baseline_copy_ready
            current_stream = torch.cuda.current_stream(self.device)
            for _ in self._inner_iteration_range:
                for slot, block_idx in selected_copy_pairs:
                    self.host_stage[slot].copy_(self.src[block_idx], non_blocking=True)
                    copy_ready.record(current_stream)
                    copy_ready.synchronize()
                    self.dst[block_idx].copy_(self.host_stage[slot], non_blocking=True)
                    copy_ready.record(current_stream)
                    copy_ready.synchronize()
            copy_calls = self._baseline_copy_calls_metric
            uses_copy_stream = 0.0
        else:
            if self.copy_stream is None or self.copy_ready is None or self.packed_stage is None:
                raise RuntimeError("Optimized path requires a copy stream and event")
            src = self.src
            dst = self.dst
            selected_idx = self.selected_idx
            packed_stage = self.packed_stage
            host_stage = self.host_stage
            gpu_stage = self.gpu_stage
            copy_stream = self.copy_stream
            copy_ready = self.copy_ready
            current_stream = torch.cuda.current_stream(self.device)
            for _ in self._inner_iteration_range:
                torch.index_select(src, 0, selected_idx, out=packed_stage)
                copy_stream.wait_stream(current_stream)
                with torch.cuda.stream(copy_stream):
                    host_stage.copy_(packed_stage, non_blocking=True)
                    gpu_stage.copy_(host_stage, non_blocking=True)
                    copy_ready.record(copy_stream)
                current_stream.wait_event(copy_ready)
                dst.index_copy_(0, selected_idx, gpu_stage)
            copy_calls = self._optimized_copy_calls_metric
            uses_copy_stream = 1.0

        torch.index_select(self.dst, 0, self.selected_idx, out=self._output_buffer)
        self.output = self._output_buffer
        metrics = self._metrics
        metrics["tier_handoff.selected_blocks"] = self._selected_blocks_metric
        metrics["tier_handoff.block_kib"] = self._block_kib_metric
        metrics["tier_handoff.inner_iterations"] = self._inner_iterations_metric
        metrics["tier_handoff.copy_calls"] = copy_calls
        metrics["tier_handoff.uses_copy_stream"] = uses_copy_stream
        metrics["tier_handoff.bytes_per_iteration_mb"] = self._bytes_per_iteration_mb

    def capture_verification_payload(self) -> None:
        if self.src is None or self.output is None or self.selected_idx is None or self._expected_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        torch.index_select(self.src, 0, self.selected_idx, out=self._expected_buffer)
        self._set_verification_payload(
            inputs={
                "selected_source": self._expected_buffer,
                "selected_idx": self.selected_idx,
            },
            output=self.output,
            batch_size=self.workload.selected_blocks,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": 1,
                "collective_type": "tier_handoff",
            },
        )

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        self.host_stage = None
        self.gpu_stage = None
        self.packed_stage = None
        self.selected_idx = None
        self.selected_cpu = None
        self.selected_copy_pairs = None
        self._output_buffer = None
        self._expected_buffer = None
        self.copy_stream = None
        self.copy_ready = None
        self.baseline_copy_ready = None
        self.output = None
        self._reset_metrics()
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            measurement_timeout_seconds=60,
            setup_timeout_seconds=30,
            profiling_warmup=0,
            profiling_iterations=1,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._metrics)

    def validate_result(self) -> Optional[str]:
        if self.src is None or self.output is None or self.selected_idx is None or self._expected_buffer is None:
            return "Output not produced"
        torch.index_select(self.src, 0, self.selected_idx, out=self._expected_buffer)
        if self.output.shape != self._expected_buffer.shape:
            return "Unexpected output shape"
        if not torch.equal(self.output, self._expected_buffer):
            return "Selected blocks changed during handoff"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.set_workload(apply_cli_overrides(self.workload, argv))
