"""baseline_work_queue.py - Static work distribution (baseline)."""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# Import CUDA extension
from ch12.cuda_extensions import load_work_queue_extension


class BaselineWorkQueueBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Static work distribution - each thread processes fixed indices (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.output_data = None
        self.N = 1 << 20  # 1M elements
        self.iterations = 5
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_work_queue_extension()
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        self._extension.static_work_distribution(self.input_data, self.output_data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._verify_input = self.input_data.detach().clone()
        self._verify_output_buffer = torch.empty_like(self.output_data)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Static work distribution."""
        with torch.inference_mode(), self._nvtx_range("work_queue"):
            # Call CUDA extension with static work distribution
            self._extension.static_work_distribution(self.input_data, self.output_data, self.iterations)
        if self._verify_input is None or self.output_data is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self.output_data is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._verify_output_buffer.copy_(self.output_data)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
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
        self.input_data = None
        self.output_data = None
        self._verify_input = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
            ncu_replay_mode="application",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return workload metrics for the static work-distribution path."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        metrics = compute_ch12_workload_metrics(
            uses_cuda_graph=False,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
        )
        metrics["cuda_runtime.dynamic_work_queue"] = 0.0
        return metrics
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input_data is None or self.output_data is None:
            return "Data tensors not initialized"
        if self.input_data.shape[0] != self.N or self.output_data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.output_data).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineWorkQueueBenchmark()
