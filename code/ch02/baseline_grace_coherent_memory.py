#!/usr/bin/env python3
"""Baseline: coherent-memory transfer path without placement optimizations."""

from typing import Any, Dict, Optional
import time

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

_CUDA_SKIP_REASON = "SKIPPED: grace_coherent_memory requires CUDA"
_GRACE_SKIP_REASON = (
    "SKIPPED: grace_coherent_memory requires Grace-Blackwell coherent memory support "
    "(GB200/GB300 on Grace CPU hosts)."
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
        """Detect if running on Grace-Blackwell platform."""
        if not torch.cuda.is_available():
            return False
        
        try:
            import platform
            props = torch.cuda.get_device_properties(0)
            is_arm_host = platform.machine() in ('aarch64', 'arm64')
            # Grace-Blackwell coherent memory needs a Grace (ARM) host paired with
            # a Blackwell-class GPU. GB200/GB300 GPUs report CC 10.x (B200=10.0,
            # B300=10.3); GB10 reports CC 12.x. A non-Grace (x86) B200/B300 host has
            # no NVLink-C2C coherent fabric, so the ARM host check is the
            # discriminator. The old CC==12.1 check wrongly skipped real GB300.
            if is_arm_host and props.major >= 10:
                return True
        except Exception as e:
            logger.debug(f"Grace-Blackwell detection failed: {e}")
        
        return False
    
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
        bandwidth_gb_s = (self.size_mb / 1024) * 2 / elapsed  # H2D + D2H
        
        logger.info(f"Baseline bandwidth: {bandwidth_gb_s:.2f} GB/s")
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
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
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
        self.bandwidth_gb_s = (self._impl.size_mb / 1024) * 2 / elapsed
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
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
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
        """Return memory transfer metrics for grace_coherent_memory."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        bytes_transferred = float(self.size_mb * 1024 * 1024 * 2)
        return compute_memory_transfer_metrics(
            bytes_transferred=bytes_transferred,
            elapsed_ms=(self.elapsed_s or 0.001) * 1000.0,
            transfer_type="nvlink" if self._impl.is_grace_blackwell else "pcie",
        )

    def validate_result(self) -> Optional[str]:
        if self.elapsed_s is None:
            return "Benchmark did not run"
        return None


def get_benchmark() -> BaseBenchmark:
    return GraceCoherentMemoryBenchmark()


def run_benchmark(
    size_mb: int = 256,
    iterations: int = 100,
    profile: str = "none",
    **kwargs
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
    bandwidth_gb_s = (size_mb / 1024) * 2 / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "mean_time_ms": mean_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "is_grace_blackwell": getattr(benchmark, "_impl", None).is_grace_blackwell if hasattr(benchmark, "_impl") else False,
        "size_mb": size_mb,
        "iterations": iterations,
    }
