#!/usr/bin/env python3
"""Baseline: coherent-memory transfer path without placement optimizations."""

import time
from typing import Any, Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.grace_platform import host_link_flags
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

_CUDA_SKIP_REASON = "SKIPPED: grace_coherent_memory requires CUDA"
_GRACE_SKIP_REASON = (
    "SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support "
    "(the CUDA driver must report an NVLink-C2C host link to a Grace CPU, as on GB200 and GB300)."
)


class BaselineGraceCoherentMemory:
    """Baseline coherent memory access without optimization."""

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        if not torch.cuda.is_available():
            raise RuntimeError(_CUDA_SKIP_REASON)
        self.size_mb = size_mb
        self.iterations = iterations
        self.device = torch.device("cuda")

        # Check if we're on Grace-Blackwell
        self.is_grace_blackwell = self._detect_grace_blackwell()
        if not self.is_grace_blackwell:
            raise RuntimeError(_GRACE_SKIP_REASON)

    def _detect_grace_blackwell(self) -> bool:
        """Return whether the CUDA driver puts this GPU on an NVLink-C2C host link to Grace.

        The optimized variant needs that link, so the pair runs only where it
        exists. Compute capability cannot decide this. GB200 and GB300 GPUs
        report SM 10.0 and 10.3, the same as B200 and B300 GPUs on PCIe in x86
        servers. Driver query errors propagate.
        """
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        nvlink_c2c, _ = host_link_flags(props, device_index)
        return nvlink_c2c

    def setup(self):
        """Initialize data structures with pageable CPU memory (baseline)."""
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32

        # Baseline: Use regular pageable memory without pinning
        # This will go through explicit H2D transfers
        self.cpu_data = torch.randn(num_elements, dtype=torch.float32)

        # GPU buffer for computation
        self.gpu_data = torch.zeros(num_elements, dtype=torch.float32, device=self.device)

        logger.info(f"Allocated {self.size_mb}MB pageable CPU memory")

    def run_step(self) -> float:
        """Execute one pageable H2D -> compute -> D2H transfer step."""
        torch.cuda.synchronize()
        start = time.perf_counter()

        self.gpu_data.copy_(self.cpu_data, non_blocking=False)
        self.gpu_data.mul_(2.0).add_(1.0)
        self.cpu_data.copy_(self.gpu_data, non_blocking=False)

        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        return elapsed

    def cleanup(self):
        """Clean up resources."""
        del self.cpu_data
        del self.gpu_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GraceCoherentMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Harness-friendly wrapper around the baseline coherent memory example."""

    allowed_benchmark_fn_antipatterns = ("sync", "host_transfer")

    def __init__(self, size_mb: int = 256, iterations: int = 100):
        super().__init__()
        self._impl = BaselineGraceCoherentMemory(
            size_mb=size_mb,
            iterations=iterations,
        )
        bytes_per_iter = size_mb * 1024 * 1024 * 2  # H2D + D2H
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )
        self.elapsed_s: Optional[float] = None
        self.bandwidth_gb_s: Optional[float] = None
        self.size_mb = size_mb
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # Inputs use the RNG already seeded by the harness.
        self._impl.setup()
        self._verify_output_buffer = torch.empty_like(self._impl.cpu_data[:1000])
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

        # Do an initial copy to populate gpu_data with actual values
        self._impl.gpu_data.copy_(self._impl.cpu_data.to(self._impl.device))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        elapsed = self._impl.run_step()
        self.elapsed_s = elapsed
        self.bandwidth_gb_s = self._workload.bytes_per_iteration / elapsed / 1e9
        self.output = None

    def capture_verification_payload(self) -> None:
        # Use the post-transfer host-visible view so verification compares the
        # same observable result across pageable and staged-copy strategies.
        if self.output is None:
            if self._verify_output_buffer is None:
                raise RuntimeError("setup() must be called before verification")
            output_slice = self._impl.cpu_data[: self._verify_output_buffer.numel()].detach()
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={
                "cpu_data": self._impl.cpu_data,
                "gpu_data": self._impl.gpu_data,
            },
            output=self.output,
            batch_size=self._impl.cpu_data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32
                if torch.cuda.is_available()
                else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self._impl.cleanup()
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=max(1, self._impl.iterations),
            warmup=5,
            enable_memory_tracking=False,
            # benchmark_fn mutates the transfer buffers, so adaptive iteration
            # expansion would change the observable output used for verification.
            adaptive_iterations=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return the bytes each step moves across the CPU-GPU link."""
        # core.benchmark.metrics has no NVLink-C2C peak, and its NVLink peak is the
        # GPU-to-GPU link, so this reports achieved bandwidth without an efficiency.
        bytes_moved = self._workload.bytes_per_iteration
        metrics = {"transfer.bytes": bytes_moved}
        if self.elapsed_s is not None:
            metrics["transfer.achieved_gbps"] = bytes_moved / self.elapsed_s / 1e9
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.elapsed_s is None:
            return "Benchmark did not run"
        return None


def get_benchmark() -> BaseBenchmark:
    return GraceCoherentMemoryBenchmark()


def run_benchmark(
    size_mb: int = 256, iterations: int = 100, profile: str = "none", **kwargs
) -> Dict[str, Any]:
    """Run baseline Grace coherent memory benchmark."""

    benchmark = GraceCoherentMemoryBenchmark(size_mb=size_mb, iterations=iterations)
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=max(1, iterations),
            warmup=5,
            profile_mode=profile,
        ),
    )
    result = harness.benchmark(benchmark, name="baseline_grace_coherent_memory")
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    elapsed_s = mean_ms / 1000.0 if mean_ms > 0 else 0.0
    # Decimal GB/s of the H2D plus D2H bytes, the same unit as benchmark_fn.
    bytes_per_iteration = size_mb * 1024 * 1024 * 2
    bandwidth_gb_s = bytes_per_iteration / elapsed_s / 1e9 if elapsed_s > 0 else 0.0

    return {
        "mean_time_ms": mean_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "is_grace_blackwell": getattr(benchmark, "_impl", None).is_grace_blackwell
        if hasattr(benchmark, "_impl")
        else False,
        "size_mb": size_mb,
        "iterations": iterations,
    }
