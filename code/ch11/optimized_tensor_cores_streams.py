"""Optimized tensor-core stream workload with overlap."""

from __future__ import annotations

from typing import List, Optional

import torch

from ch11.stream_overlap_base import resolve_device
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import canonicalize_nvtx_name


class OptimizedTensorCoresStreamsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized tensor-core workload with staged H2D/GEMM/D2H overlap."""

    declare_all_streams = False

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.label = "tensor_cores_streams"
        self.nvtx_label = "optimized_tensor_cores_streams"
        self.num_segments = 24
        self.matrix_dim = 768
        self.num_elements = self.num_segments * self.matrix_dim * self.matrix_dim
        self.num_streams = 6
        self.streams: List[torch.cuda.Stream] | None = None
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.host_A: torch.Tensor | None = None
        self.host_B: torch.Tensor | None = None
        self.host_output: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self.device_A_slots: List[torch.Tensor] | None = None
        self.device_B_slots: List[torch.Tensor] | None = None
        self.device_C_slots: List[torch.Tensor] | None = None
        self.device_C_rows: List[torch.Tensor] | None = None
        self.segment_work: List[
            tuple[
                torch.cuda.Stream,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] | None = None
        element_size = float(torch.finfo(self.dtype).bits // 8)
        bytes_transferred = float(self.num_elements * element_size * 3)
        self.register_workload_metadata(bytes_per_iteration=bytes_transferred)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")

        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
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
        self.device_A_slots = [
            torch.empty((self.matrix_dim, self.matrix_dim), device=self.device, dtype=self.dtype)
            for _ in range(self.num_streams)
        ]
        self.device_B_slots = [torch.empty_like(slot) for slot in self.device_A_slots]
        self.device_C_slots = [torch.empty_like(slot) for slot in self.device_A_slots]
        self.device_C_rows = [slot[0] for slot in self.device_C_slots]
        segment_work = []
        for idx, (host_a, host_b, host_out) in enumerate(
            zip(self.host_A.unbind(0), self.host_B.unbind(0), self.host_output.unbind(0), strict=True)
        ):
            slot = idx % self.num_streams
            segment_work.append(
                (
                    self.streams[slot],
                    host_a,
                    host_b,
                    host_out,
                    self.device_A_slots[slot],
                    self.device_B_slots[slot],
                    self.device_C_slots[slot],
                    self.device_C_rows[slot],
                )
            )
        self.segment_work = segment_work
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self.nvtx_label):
            assert self.streams is not None
            assert self.host_A is not None
            assert self.host_B is not None
            assert self.host_output is not None
            assert self.device_A_slots is not None
            assert self.device_B_slots is not None
            assert self.device_C_slots is not None
            assert self.device_C_rows is not None
            assert self.segment_work is not None

            with torch.inference_mode():
                for (
                    stream,
                    host_a,
                    host_b,
                    host_out,
                    device_a,
                    device_b,
                    device_c,
                    device_c_row,
                ) in self.segment_work:
                    with torch.cuda.stream(stream):
                        device_a.copy_(host_a, non_blocking=True)
                        device_b.copy_(host_b, non_blocking=True)
                        torch.matmul(device_a, device_b, out=device_c)
                        host_out.copy_(device_c_row, non_blocking=True)

                current = torch.cuda.current_stream(self.device)
                for stream in self.streams:
                    current.wait_stream(stream)

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
        self.streams = None
        self.host_A = None
        self.host_B = None
        self.host_output = None
        self._verify_output_buffer = None
        self.device_A_slots = None
        self.device_B_slots = None
        self.device_C_slots = None
        self.device_C_rows = None
        self.segment_work = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        nvtx_tag = canonicalize_nvtx_name(self.nvtx_label)
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
            f"{self.label}.num_streams": float(self.num_streams),
            f"{self.label}.matrix_dim": float(self.matrix_dim),
            f"{self.label}.bytes_transferred": bytes_transferred,
            f"{self.label}.expected_overlap_pct": min(100.0, (self.num_streams - 1) / self.num_streams * 100),
            f"{self.label}.dtype": str(self.dtype),
        }

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        if self.streams is None:
            return []
        return list(self.streams)


def get_benchmark() -> OptimizedTensorCoresStreamsBenchmark:
    return OptimizedTensorCoresStreamsBenchmark()
