"""baseline_vectorization_memory.py - Naive vectorization baseline."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class VectorizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP32 memory-bandwidth-bound vector add."""

    signature_equivalence_group = "ch19_vectorization_memory_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.output = None
        self.tensor_a: Optional[torch.Tensor] = None
        self.tensor_b: Optional[torch.Tensor] = None
        self._work: Optional[torch.Tensor] = None
        self._verify_probe_a: Optional[torch.Tensor] = None
        self._verify_probe_b: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False

        # Use a large tensor that exceeds L2 so the kernel is HBM bandwidth bound.
        # Keep the iteration count low to avoid kernel-launch overhead dominating.
        self.repeats = 12
        self._repeat_range = range(self.repeats)
        self.N = 67_108_864
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * (self.repeats + 1)),
        )
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * (self.repeats + 1)),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.tensor_a = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.tensor_b = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._work = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._verify_probe_a = torch.empty(1024, dtype=torch.float32, pin_memory=True)
        self._verify_probe_b = torch.empty(1024, dtype=torch.float32, pin_memory=True)
        self._verify_probe_a.copy_(
            self.tensor_a[: self._verify_probe_a.numel()],
            non_blocking=False,
        )
        self._verify_probe_b.copy_(
            self.tensor_b[: self._verify_probe_b.numel()],
            non_blocking=False,
        )
        self._verify_output_buffer = torch.empty(4096, dtype=torch.float32)
        self._repeat_range = range(self.repeats)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.tensor_a is None or self.tensor_b is None or self._work is None:
            raise RuntimeError("setup() must be called before benchmark_fn()")
        with (
            nvtx_range("baseline_vectorization", enable=self._enable_nvtx),
            torch.inference_mode(),
        ):
            for _ in self._repeat_range:
                torch.add(self.tensor_a, self.tensor_b, out=self._work)
            self.output = self._work

    def capture_verification_payload(self) -> None:
        if (
            self.output is None
            or self._verify_probe_a is None
            or self._verify_probe_b is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[: self._verify_output_buffer.numel()].detach()
        self._verify_output_buffer.copy_(output_slice, non_blocking=False)
        self._set_verification_payload(
            inputs={"probe_a": self._verify_probe_a, "probe_b": self._verify_probe_b},
            output=self._verify_output_buffer,
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.tensor_a = None
        self.tensor_b = None
        self._work = None
        self._verify_probe_a = None
        self._verify_probe_b = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp32",
        )

    def validate_result(self) -> Optional[str]:
        if self.tensor_a is None or self.tensor_b is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return VectorizationBenchmark()
