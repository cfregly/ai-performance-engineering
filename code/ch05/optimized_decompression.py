"""optimized_decompression.py - GPU-assisted decompression stand-in.

Decodes the same toy RLE format as `baseline_decompression.py`, but performs the
repeat expansion on the GPU to model offloading decompression work.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class GPUDecompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.counts: Optional[torch.Tensor] = None
        self.counts_i64: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._run_len: int = 0
        self._output_matrix: Optional[torch.Tensor] = None
        self._output_flat: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=float(1024 * 1024 * 4))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for GPU decompression demo")
        torch.manual_seed(42)
        total_len = 1024 * 1024
        run_len = 256
        if total_len % run_len != 0:
            raise RuntimeError("total_len must be divisible by run_len for this benchmark")
        num_runs = total_len // run_len
        counts = torch.full((num_runs,), run_len, dtype=torch.int32)
        values = torch.randn((num_runs,), dtype=torch.float32)
        self.counts = counts.to(self.device)
        self.counts_i64 = self.counts.to(torch.int64)
        self.values = values.to(self.device)
        self._run_len = int(run_len)
        self._output_matrix = torch.empty((num_runs, run_len), device=self.device, dtype=self.values.dtype)
        self._output_flat = self._output_matrix.reshape(-1)

    def benchmark_fn(self) -> Optional[dict]:
        if self.counts_i64 is None or self.values is None or self._output_matrix is None or self._output_flat is None:
            raise RuntimeError("SKIPPED: missing encoded RLE buffers")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("gpu_decompress_rle", enable=enable_nvtx):
            self._output_matrix.copy_(self.values.unsqueeze(1))
            out = self._output_flat
        latency_ms = self._record_stop(start)
        self.output = out.detach()
        self._payload_counts = self.counts
        self._payload_values = self.values
        return {"latency_ms": latency_ms, "decompressed_len": int(out.numel())}

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        counts = self._payload_counts
        values = self._payload_values
        if counts is None or values is None:
            raise RuntimeError("benchmark_fn() must stash inputs for verification")
        self._set_verification_payload(
            inputs={"counts": counts.detach().clone(), "values": values.detach().clone()},
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
        run_length = self._run_len if run_count > 0 else 0
        return compute_decompression_metrics(
            run_count=run_count,
            run_length=run_length,
            decompressed_elements=run_count * run_length,
            runs_on_device=True,
        )


def get_benchmark() -> BaseBenchmark:
    return GPUDecompressionBenchmark()
