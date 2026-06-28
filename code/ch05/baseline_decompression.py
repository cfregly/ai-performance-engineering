"""baseline_decompression.py - CPU-bound decompression baseline.

This benchmark uses a toy run-length encoding (RLE) format to simulate
CPU-side decompression of a compressed batch.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402


class CPUDecompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.counts: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=float(1024 * 1024 * 4))

    def setup(self) -> None:
        torch.manual_seed(42)
        total_len = 1024 * 1024
        run_len = 256
        if total_len % run_len != 0:
            raise RuntimeError("total_len must be divisible by run_len for this benchmark")
        num_runs = total_len // run_len
        self.counts = torch.full((num_runs,), run_len, dtype=torch.int32)
        self.values = torch.randn((num_runs,), dtype=torch.float32)
        self.output = None

    def benchmark_fn(self) -> Optional[dict]:
        if self.counts is None or self.values is None:
            raise RuntimeError("SKIPPED: missing encoded RLE buffers")

        start = self._record_start()
        with self._nvtx_range("cpu_decompress"):
            decompressed = torch.repeat_interleave(self.values, self.counts)
        latency_ms = self._record_stop(start)
        self.output = decompressed.detach()
        self._payload_counts = self.counts
        self._payload_values = self.values
        return {"latency_ms": latency_ms, "decompressed_len": int(decompressed.numel())}

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        counts = self._payload_counts
        values = self._payload_values
        if counts is None or values is None:
            raise RuntimeError("benchmark_fn() must stash inputs for verification")
        self._set_verification_payload(
            inputs={
                "counts": counts.detach().clone(),
                "values": values.detach().clone(),
            },
            output=self.output[:4096].detach().clone(),
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual decompression workload shape."""
        from ch05.metrics_common import compute_decompression_metrics

        if self.counts is None:
            return None
        run_count = int(self.counts.numel())
        run_length = int(self.counts[0].item()) if run_count > 0 else 0
        return compute_decompression_metrics(
            run_count=run_count,
            run_length=run_length,
            decompressed_elements=run_count * run_length,
            runs_on_device=False,
        )


def get_benchmark() -> BaseBenchmark:
    return CPUDecompressionBenchmark()
