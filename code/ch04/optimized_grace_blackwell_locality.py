#!/usr/bin/env python3
"""Optimized locality microbench (device-resident compute, no host copy).

Baseline copies from pinned host each iteration; this keeps data resident on GPU
to show placement/locality win. Works on any CUDA GPU; label mirrors Grace-Blackwell locality story.
"""
from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedGb200LocalityBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, size_mb: float = 256.0):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((self.size_mb * 1024 * 1024) / 4)  # float32
        self.device_buf: Optional[torch.Tensor] = None
        self.device_template: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._materialization_buffer: Optional[torch.Tensor] = None
        # Memory copy benchmark - jitter check not applicable
        tokens = float(self.numel)
        self._workload = WorkloadMetadata(tokens_per_iteration=tokens, requests_per_iteration=1.0)

    def setup(self) -> None:
        # Keep data resident on GPU to maximize locality.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.device_template = torch.ones(self.numel, device=self.device, dtype=torch.float32)
        self.device_buf = torch.empty_like(self.device_template)
        self._verify_probe = self.device_template[: 256 * 256].view(256, 256)
        self._output_view = self.device_buf[: 256 * 256].view(256, 256)
        self._materialization_buffer = torch.empty((), device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        assert self.device_buf is not None and self.device_template is not None
        assert self._output_view is not None and self._materialization_buffer is not None
        with torch.inference_mode():
            with self._nvtx_range("device_only_compute"):
                # Reset from a device-resident template to avoid host traffic while keeping outputs identical
                self.device_buf.copy_(self.device_template)
                self.device_buf.add_(1.0)
            torch.sum(self.device_buf, dim=0, out=self._materialization_buffer)
            self.output = self._output_view

    def capture_verification_payload(self) -> None:
        if self._verify_probe is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self._verify_probe
        output = self.output
        self._set_verification_payload(
            inputs={"buf": probe},
            output=output,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.device_buf = None
        self.device_template = None
        self.output = None
        self._verify_probe = None
        self._output_view = None
        self._materialization_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5, enable_memory_tracking=False, enable_profiling=False)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for memory copy benchmark."""
        return (1e-5, 1e-5)



def get_benchmark() -> BaseBenchmark:
    return OptimizedGb200LocalityBenchmark()
