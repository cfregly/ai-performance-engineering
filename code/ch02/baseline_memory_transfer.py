"""baseline_memory_transfer.py - Traditional PCIe memory transfer (baseline)."""

from __future__ import annotations

from typing import Optional

import torch

from ch02.memory_transfer_common import compute_transfer_digest
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineMemoryTransferBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Traditional PCIe memory transfer - slower path."""
    
    def __init__(self):
        super().__init__()
        self.host_data: Optional[torch.Tensor] = None
        self.device_data: Optional[torch.Tensor] = None
        self._digest_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        # Large enough to saturate PCIe/NVLink H2D paths so pinned DMA wins.
        self.N = 50_000_000
        self._last_elapsed_ms: Optional[float] = None
        self._bytes_transferred = float(self.N * 4)
        bytes_per_iter = self.N * 4  # float32 copy
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and verification output."""
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Baseline uses pageable host memory (slower H2D transfers vs pinned DMA).
        self.host_data = torch.randn(self.N, dtype=torch.float32, pin_memory=False)
        self.device_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        digest_blocks = (self.N + 1_000_000 - 1) // 1_000_000
        self._verify_output_buffer = torch.empty(digest_blocks, dtype=torch.int64, device=self.device)
        
        # Copy data and compute checksum for verification
        self.device_data.copy_(self.host_data, non_blocking=False)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Traditional H2D transfer over PCIe."""
        assert self.host_data is not None and self.device_data is not None
        with torch.inference_mode(), self._nvtx_range("memory_transfer_baseline"):
            self.device_data.copy_(self.host_data, non_blocking=False)

    def capture_verification_payload(self) -> None:
        if self.device_data is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        # Verification: compute a deterministic digest over ALL transferred elements (post-timing).
        self._synchronize()
        digest, self._digest_buffer = compute_transfer_digest(self.device_data, self._digest_buffer)
        self.output = digest.detach()
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"host_data": self.host_data},
            output=self._verify_output_buffer,
            batch_size=self.N,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.host_data = None
        self.device_data = None
        self._digest_buffer = None
        self._verify_output_buffer = None
        self._last_elapsed_ms = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return measured host-to-device transfer metrics."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_elapsed_ms,
            transfer_type="pcie",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.device_data is None:
            return "Device tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMemoryTransferBenchmark()
