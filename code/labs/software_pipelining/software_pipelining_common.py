"""Shared benchmark helpers for the software-pipelining lab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.software_pipelining.software_pipelining_extension import (
    load_software_pipelining_extension,
)


@dataclass(frozen=True)
class TilePipelineWorkload:
    length: int = 1 << 25
    repeat_fmas: int = 2
    dtype: torch.dtype = torch.float32
    tile_elems: int = 1024
    block_threads: int = 256


def _workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--repeat-fmas", type=int, default=None)
    return parser


def apply_workload_overrides(
    workload: TilePipelineWorkload,
    argv: list[str],
) -> TilePipelineWorkload:
    args, _ = _workload_parser().parse_known_args(argv)
    updated = TilePipelineWorkload(
        length=args.length if args.length is not None else workload.length,
        repeat_fmas=args.repeat_fmas if args.repeat_fmas is not None else workload.repeat_fmas,
        dtype=workload.dtype,
        tile_elems=workload.tile_elems,
        block_threads=workload.block_threads,
    )
    if updated.length <= 0:
        raise ValueError("length must be positive")
    if updated.repeat_fmas <= 0:
        raise ValueError("repeat_fmas must be positive")
    return updated


def reference_tile_pipeline(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    repeat_fmas: int,
) -> torch.Tensor:
    """Reference implementation matching the CUDA kernels' math."""

    x = lhs.float()
    y = rhs.float()
    for _ in range(repeat_fmas):
        x = x * 1.0001 + y * 0.0003
        y = y * 0.9997 + x * 0.0002
    return x + 0.5 * y


class TilePipelineBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper for the serialized vs pipelined tile-loop kernels."""

    def __init__(
        self,
        *,
        op_name: str,
        label: str,
        pipeline_stage_count: int,
        notes: str,
    ) -> None:
        super().__init__()
        self._op_name = op_name
        self._label = label
        self._notes = notes
        self._pipeline_stage_count = pipeline_stage_count
        self._workload = TilePipelineWorkload()
        self._module: Optional[Any] = None
        self._runner: Optional[Any] = None
        self._lhs: Optional[torch.Tensor] = None
        self._rhs: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._reference: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._config = BenchmarkConfig(
            iterations=20,
            warmup=5,
            timeout_seconds=300,
            deterministic=False,
            enable_nvtx=True,
            enable_profiling=False,
        )
        self._register_workload_metadata()

    @property
    def workload(self) -> TilePipelineWorkload:
        return self._workload

    @property
    def pipeline_stage_count(self) -> int:
        return self._pipeline_stage_count

    def _register_workload_metadata(self) -> None:
        bytes_per_iteration = float(self._workload.length * 3 * 4)
        scalar_ops = float(self._workload.length * self._workload.repeat_fmas * 4)
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            samples_per_iteration=float(self._workload.length),
            bytes_per_iteration=bytes_per_iteration,
            custom_units_per_iteration=scalar_ops,
            custom_unit_name="scalar_ops",
        )

    def apply_target_overrides(self, argv: list[str]) -> None:
        self._workload = apply_workload_overrides(self._workload, argv)
        self._register_workload_metadata()

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: software_pipelining benchmark requires CUDA")

        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            raise RuntimeError(
                f"SKIPPED: software_pipelining benchmark requires SM80+ for cuda::pipeline "
                f"(found SM {major}.{minor})."
            )

        try:
            self._module = load_software_pipelining_extension()
        except Exception as exc:  # pragma: no cover - build/runtime environment dependent
            raise RuntimeError(
                f"SKIPPED: software_pipelining extension unavailable ({type(exc).__name__}: {exc})"
            ) from exc

        self._runner = getattr(self._module, self._op_name, None)
        if self._runner is None:
            raise RuntimeError(f"SKIPPED: extension missing entrypoint {self._op_name}")

        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(4242)
        self._lhs = torch.randn(
            self._workload.length,
            generator=generator,
            dtype=self._workload.dtype,
            device=self.device,
        ).contiguous()
        self._rhs = torch.randn(
            self._workload.length,
            generator=generator,
            dtype=self._workload.dtype,
            device=self.device,
        ).contiguous()
        self._reference = reference_tile_pipeline(
            self._lhs,
            self._rhs,
            self._workload.repeat_fmas,
        ).detach()
        self._verify_output_buffer = torch.empty(
            min(4096, self._workload.length),
            device=self.device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self._runner is None or self._lhs is None or self._rhs is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode(), self._nvtx_range(self._label):
            self._output = self._runner(self._lhs, self._rhs, self._workload.repeat_fmas)

    def capture_verification_payload(self) -> None:
        if self._lhs is None or self._rhs is None or self._output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self._output[: self._verify_output_buffer.shape[0]]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"lhs": self._lhs, "rhs": self._rhs},
            output=self._verify_output_buffer,
            batch_size=1,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-4, 2e-4),
        )

    def validate_result(self) -> Optional[str]:
        if self._output is None or self._reference is None:
            return "benchmark_fn() did not produce output"
        if torch.isnan(self._output).any():
            return "NaNs detected in output tensor"
        max_diff = scalar_tensor_to_float((self._output - self._reference).abs().max())
        if max_diff > 5e-4:
            return f"Max abs diff {max_diff:.6f} exceeds tolerance 5e-4"
        return None

    def teardown(self) -> None:
        self._lhs = None
        self._rhs = None
        self._output = None
        self._reference = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return self._config

    def get_custom_metrics(self) -> Optional[dict[str, float]]:
        return {
            "pipeline_stage_count": float(self._pipeline_stage_count),
            "tile_elems": float(self._workload.tile_elems),
            "repeat_fmas": float(self._workload.repeat_fmas),
            "story.software_pipelining": 1.0,
        }

    def describe(self) -> str:
        return self._notes
