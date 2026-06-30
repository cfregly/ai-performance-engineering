"""Baseline tensor-core stream workload without overlap."""

from __future__ import annotations

from typing import List, Optional

import torch

from ch11.stream_overlap_base import resolve_device
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import canonicalize_nvtx_name


class BaselineTensorCoresStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline tensor-core workload with single-stream staging and compute."""

    declare_all_streams = False

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.label = "baseline_tensor_cores_streams"
        self.num_segments = 24
        self.matrix_dim = 768
        self.num_elements = self.num_segments * self.matrix_dim * self.matrix_dim
        self.stream: torch.cuda.Stream | None = None
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.host_A: torch.Tensor | None = None
        self.host_B: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self.device_A_slot: torch.Tensor | None = None
        self.device_B_slot: torch.Tensor | None = None
        self.device_C_slot: torch.Tensor | None = None
        self.device_C_row: torch.Tensor | None = None
        self.segment_work: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        element_size = float(torch.finfo(self.dtype).bits // 8)
        bytes_transferred = float(self.num_elements * element_size * 3)
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.stream = torch.cuda.Stream()
        self.host_A = torch.randn(
            self.num_segments,
            self.matrix_dim,
            self.matrix_dim,
            device="cpu",
            dtype=self.dtype,
            pin_memory=True,
        )
        self.host_B = torch.randn(
            self.num_segments,
            self.matrix_dim,
            self.matrix_dim,
            device="cpu",
            dtype=self.dtype,
            pin_memory=True,
        )
        self.host_output = torch.empty(
            (self.num_segments, self.matrix_dim),
            device="cpu",
            dtype=self.dtype,
            pin_memory=True,
        )
        self._verify_output_buffer = torch.empty_like(self.host_output, pin_memory=True)
        self.device_A_slot = torch.empty(
            (self.matrix_dim, self.matrix_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.device_B_slot = torch.empty_like(self.device_A_slot)
        self.device_C_slot = torch.empty_like(self.device_A_slot)
        self.device_C_row = self.device_C_slot[0]
        self.segment_work = list(
            zip(
                self.host_A.unbind(0),
                self.host_B.unbind(0),
                self.host_output.unbind(0),
                strict=True,
            )
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self.label):
            assert self.stream is not None
            assert self.host_A is not None
            assert self.host_B is not None
            assert self.host_output is not None
            assert self.device_A_slot is not None
            assert self.device_B_slot is not None
            assert self.device_C_slot is not None
            assert self.device_C_row is not None
            assert self.segment_work is not None

            with torch.inference_mode():
                for host_a, host_b, host_out in self.segment_work:
                    with torch.cuda.stream(self.stream):
                        self.device_A_slot.copy_(host_a, non_blocking=True)
                        self.device_B_slot.copy_(host_b, non_blocking=True)
                        torch.matmul(self.device_A_slot, self.device_B_slot, out=self.device_C_slot)
                        host_out.copy_(self.device_C_row, non_blocking=True)
                current = torch.cuda.current_stream(self.device)
                current.wait_stream(self.stream)

        if self.host_A is None or self.host_B is None or self.host_output is None:
            raise RuntimeError("benchmark_fn() must run after setup() initializes buffers")

    def capture_verification_payload(self) -> None:
        assert self.host_A is not None
        assert self.host_B is not None
        assert self.host_output is not None
        assert self._verify_output_buffer is not None
        self._verify_output_buffer.copy_(self.host_output)
        self._set_verification_payload(
            inputs={"host_A": self.host_A, "host_B": self.host_B},
            output=self._verify_output_buffer,
            batch_size=self.host_output.numel(),
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def teardown(self) -> None:
        self.stream = None
        self.host_A = None
        self.host_B = None
        self.host_output = None
        self._verify_output_buffer = None
        self.device_A_slot = None
        self.device_B_slot = None
        self.device_C_slot = None
        self.device_C_row = None
        self.segment_work = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        nvtx_tag = canonicalize_nvtx_name(self.label)
        return BenchmarkConfig(
            iterations=16,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[nvtx_tag],
        )

    def validate_result(self) -> str | None:
        if self.host_output is None or self.host_A is None or self.host_B is None:
            return "Buffers not initialized"
        if not torch.isfinite(self.host_output).all():
            return "Output contains non-finite values"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        element_size = float(torch.finfo(self.dtype).bits // 8)
        bytes_transferred = float(self.num_elements * element_size * 3)
        return {
            f"{self.label}.elements": float(self.num_elements),
            f"{self.label}.num_segments": float(self.num_segments),
            f"{self.label}.matrix_dim": float(self.matrix_dim),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.num_streams": 1.0,
            f"{self.label}.expected_overlap_pct": 0.0,
            f"{self.label}.dtype": str(self.dtype),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.stream is None:
            return []
        return [self.stream]


def get_benchmark() -> BaselineTensorCoresStreamsBenchmark:
    return BaselineTensorCoresStreamsBenchmark()
