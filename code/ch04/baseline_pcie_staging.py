"""baseline_pcie_staging.py - Pageable, blocking PCIe host-staging baseline."""

from __future__ import annotations

import torch

from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselinePcieStagingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Supplementary comparison pair: pageable, blocking host staging over PCIe."""

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "baseline",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "This single-GPU pair isolates pageable versus pinned host staging. "
            "It is not an NVLink fabric benchmark."
        ),
        "comparison_axis": "pageable_blocking_vs_pinned_nonblocking_host_staging",
        "execution_pattern": "single_gpu_host_staging",
        "optimization_mechanism": "pageable host buffer plus blocking copies",
        "chapter_native_targets": ["nvlink_topology_aware", "nvlink_multigpu", "nvlink_topology_aware_multigpu"],
    }

    def __init__(self):
        super().__init__()
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffer = None
        self.output: Optional[torch.Tensor] = None
        self.N = 20_000_000
        # Memory transfer benchmark - jitter check not applicable
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
            bytes_per_iteration=float(self.N * 4),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data_gpu0 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.data_gpu1 = torch.empty_like(self.data_gpu0)
        self.host_buffer = torch.empty(self.N, device="cpu", dtype=torch.float32)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: pageable CPU buffer with blocking staged copies."""
        with torch.inference_mode(), self._nvtx_range("baseline_pcie_staging"):
            self.host_buffer.copy_(self.data_gpu0, non_blocking=False)
            self.data_gpu1.copy_(self.host_buffer, non_blocking=False)

    def capture_verification_payload(self) -> None:
        if self.data_gpu0 is None or self.data_gpu1 is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        probe = self.data_gpu0[: 256 * 256].view(256, 256)
        output = self.data_gpu1[: 256 * 256].view(256, 256)
        self._set_verification_payload(
            inputs={"src": probe},
            output=output,
            batch_size=int(probe.shape[0]),
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
        self.data_gpu0 = None
        self.data_gpu1 = None
        self.host_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="pcie",
        )
        metrics.update(
            {
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
                "pcie.host_buffer_pinned": 0.0,
                "pcie.non_blocking_copy": 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data_gpu0 is None:
            return "Data not initialized"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()
    
    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for memory transfer benchmark."""
        return (0.0, 0.0)



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselinePcieStagingBenchmark()
