"""Shared helpers and benchmarks for the memory-bandwidth-patterns lab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from typing import Callable, Optional

import torch

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.memory_bandwidth_patterns.bandwidth_patterns_extension import (
    load_memory_bandwidth_patterns_extension,
)

DECK_TITLE = (
    "NVIDIA GTC 2026 - Maximize Memory Bandwidth on Modern GPUs: "
    "Practical Techniques, Patterns, and Working Examples"
)

DEFAULT_ROWS = 4096
DEFAULT_COLS = 8192


@dataclass(frozen=True)
class BandwidthLabConfig:
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    @property
    def bytes_per_tensor(self) -> int:
        return self.numel * 4

    @property
    def bytes_per_iteration(self) -> int:
        return self.bytes_per_tensor * 2


def _workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rows", type=int, default=None, help="Input matrix rows")
    parser.add_argument("--cols", type=int, default=None, help="Input matrix cols")
    return parser


def is_async_copy_supported(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


def require_async_copy_supported(device: Optional[torch.device] = None) -> None:
    if is_async_copy_supported(device):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The async copy milestone requires CUDA.")
    major, minor = torch.cuda.get_device_capability(device)
    raise RuntimeError(
        "The async copy milestone requires cp.async-capable hardware "
        f"(sm80+). Got sm_{major}{minor}."
    )


def load_lab_config_from_env() -> BandwidthLabConfig:
    return BandwidthLabConfig(
        rows=int(os.getenv("AISP_MEMORY_BANDWIDTH_ROWS", str(DEFAULT_ROWS))),
        cols=int(os.getenv("AISP_MEMORY_BANDWIDTH_COLS", str(DEFAULT_COLS))),
    )


def apply_workload_overrides(config: BandwidthLabConfig, argv: list[str]) -> BandwidthLabConfig:
    args, _ = _workload_parser().parse_known_args(argv)
    return replace(
        config,
        rows=config.rows if args.rows is None else int(args.rows),
        cols=config.cols if args.cols is None else int(args.cols),
    )


def build_source_matrix(config: BandwidthLabConfig, device: torch.device) -> torch.Tensor:
    values = torch.arange(config.numel, device=device, dtype=torch.float32)
    values = values.remainder_(1024.0)
    return values.view(config.rows, config.cols).contiguous()


def make_copy_output(config: BandwidthLabConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(config.numel, device=device, dtype=torch.float32)


def make_transpose_output(config: BandwidthLabConfig, device: torch.device) -> torch.Tensor:
    return torch.empty((config.cols, config.rows), device=device, dtype=torch.float32)


def copy_reference(src: torch.Tensor) -> torch.Tensor:
    return src.view(-1).clone()


def transpose_reference(src: torch.Tensor) -> torch.Tensor:
    return src.transpose(0, 1).contiguous()


def effective_bandwidth_gbps(bytes_moved: int, latency_ms: float) -> float:
    if latency_ms <= 0.0:
        return 0.0
    seconds = latency_ms / 1000.0
    return bytes_moved / seconds / 1e9


def measure_cuda_callable(
    fn: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    count = max(iterations, 1)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(count):
        fn()
    end.record(current_stream)
    end.synchronize()
    return float(start.elapsed_time(end) / count)


class BandwidthPatternsBenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Base benchmark for the transpose-focused bandwidth pair."""

    variant_name = ""

    def __init__(self) -> None:
        super().__init__()
        self.config = load_lab_config_from_env()
        self.src: Optional[torch.Tensor] = None
        self.dst: Optional[torch.Tensor] = None
        self.reference: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._extension = None
        self._custom_metrics: dict[str, float] = {}
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(self.config.bytes_per_iteration),
        )

    def _run_variant(self) -> None:
        raise NotImplementedError

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.memory_bandwidth_patterns requires CUDA.")
        self._extension = load_memory_bandwidth_patterns_extension()
        self.src = build_source_matrix(self.config, self.device)
        self.dst = make_transpose_output(self.config, self.device)
        self._verify_output_buffer = torch.empty_like(self.dst)
        self.reference = transpose_reference(self.src)
        self.output = None
        self._custom_metrics = {
            "bandwidth_patterns.rows": float(self.config.rows),
            "bandwidth_patterns.cols": float(self.config.cols),
            "bandwidth_patterns.bytes_per_iteration": float(self.config.bytes_per_iteration),
            "bandwidth_patterns.async_supported": 1.0 if is_async_copy_supported(self.device) else 0.0,
            "bandwidth_patterns.is_tiled": 0.0 if self.variant_name == "transpose_naive" else 1.0,
        }
        self._run_variant()
        torch.cuda.synchronize()
        if self.output is None or self.reference is None:
            raise RuntimeError(f"{self.variant_name} did not produce output during setup verification")
        torch.testing.assert_close(self.output, self.reference, rtol=0.0, atol=0.0)
        self.output = None

    def benchmark_fn(self) -> None:
        self._run_variant()

    def capture_verification_payload(self) -> None:
        if self.src is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"src": self.src},
            output=self._verify_output_buffer,
            batch_size=self.config.rows,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.src = None
        self.dst = None
        self.reference = None
        self.output = None
        self._verify_output_buffer = None
        self._extension = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=10,
            setup_timeout_seconds=180,
            measurement_timeout_seconds=120,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.config.rows,
            dtype="float32",
            rows=self.config.rows,
            cols=self.config.cols,
            precision_flags=PrecisionFlags(tf32=False),
        ).to_dict()

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        expected_shape = (self.config.cols, self.config.rows)
        if tuple(self.output.shape) != expected_shape:
            return f"unexpected output shape: {tuple(self.output.shape)}"
        if not torch.isfinite(self.output).all():
            return "output contains non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.config = apply_workload_overrides(self.config, argv)
        self._refresh_workload_metadata()


class BaselineBandwidthPatternsBenchmark(BandwidthPatternsBenchmarkBase):
    """Naive transpose with strided writes."""

    variant_name = "transpose_naive"

    def _run_variant(self) -> None:
        if self._extension is None or self.src is None or self.dst is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._extension.transpose_naive(self.src, self.dst)
        self.output = self.dst


class OptimizedBandwidthPatternsBenchmark(BandwidthPatternsBenchmarkBase):
    """Shared-memory tiled transpose that restores coalesced writes."""

    variant_name = "transpose_tiled"

    def _run_variant(self) -> None:
        if self._extension is None or self.src is None or self.dst is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        self._extension.transpose_tiled(self.src, self.dst)
        self.output = self.dst
