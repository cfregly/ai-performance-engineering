"""optimized_cuda_graphs_router.py - CUDA graphs with a simple branch toggle.

This variant reuses the cuda_graphs extension (same math as the baseline) but
flips a small route flag each iteration to emulate a conditional branch while
keeping verification aligned with the baseline output.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata  # noqa: E402
from ch12.cuda_extensions import load_cuda_graphs_extension  # noqa: E402


class CUDAGraphRouterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUDA Graph routing benchmark with a branch toggle on identical graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.route_flag: int = 0
        self.N = 1 << 10  # Smaller buffers to match baseline
        self.iterations = 32000
        self._extension = None
        self._workload = WorkloadMetadata(tokens_per_iteration=float(self.N * self.iterations))
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for graph capture")
        self._extension = load_cuda_graphs_extension()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        # Warmup capture inside the extension
        self._extension.graph_replay(self.data, self.iterations)
        torch.cuda.synchronize(self.device)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self._verify_input = self.data.detach().clone()
        self._verify_output_buffer = torch.empty_like(self.data)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self._extension is None or self.data is None:
            raise RuntimeError("SKIPPED: graph not initialized")

        with torch.inference_mode(), self._nvtx_range("cuda_graphs_router"):
            # Flip route between iterations to emulate a conditional branch.
            self.route_flag ^= 1
            self._extension.graph_replay(self.data, self.iterations)
        if self.data is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self.data is None
            or self._verify_input is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._verify_output_buffer.copy_(self.data)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self.N,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.data = None
        self._verify_input = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,
            measurement_timeout_seconds=120,
            ncu_replay_mode="application",
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return structural routing metrics without invented launch timings."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        metrics = compute_ch12_workload_metrics(
            uses_cuda_graph=True,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
        )
        metrics["cuda_runtime.route_toggle_enabled"] = 1.0
        return metrics
    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.numel() != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return CUDAGraphRouterBenchmark()
