"""Baseline NCCL quantization – quantize on CPU and serialize transfers."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineNCCLQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: Simulate per-rank CPU-side quantization with serialized copies."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.tensor = None
        self.num_chunks = 16
        self.chunk_len = 1 << 14
        self._last = 0.0
        tokens = self.num_chunks * self.chunk_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_chunks),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: initialize synthetic gradients."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(self.num_chunks, self.chunk_len, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: CPU quantization + host/device transfers."""
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl_quantization", enable=enable_nvtx):
            if self.tensor is None:
                raise RuntimeError("Tensor not initialized")
            total = 0.0
            for idx in range(self.num_chunks):
                chunk = self.tensor[idx].detach().cpu()
                max_abs = chunk.abs().max().clamp(min=1e-6)
                scale = 127.0 / max_abs
                q = torch.round(chunk * scale).to(torch.int8)
                dq = q.float() / scale
                total += float(dq.sum())
                self.tensor[idx].copy_(dq.to(self.device))
            self._last = total
            self.output = self.tensor.detach()
        if self.output is None or self.tensor is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.tensor.detach().clone()},
            output=self.output.detach().clone(),
            batch_size=self.num_chunks,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.tensor = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="int8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineNCCLQuantizationBenchmark()
