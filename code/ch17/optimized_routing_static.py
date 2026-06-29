"""optimized_routing_static.py - Vectorized static routing."""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class LargeModel(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return self.output(x)


class OptimizedRoutingStaticBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Static routing optimized: vectorized routing decisions."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24
        self.requests_per_iteration = 512
        self.num_routes = 1024
        tokens = self.batch_size * self.hidden_dim * self.requests_per_iteration
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.requests_per_iteration),
            tokens_per_iteration=float(tokens),
        )
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.route_scores: Optional[torch.Tensor] = None
        self.route_ids: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.requests_per_iteration),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)

        self.model = LargeModel(self.hidden_dim, self.num_layers).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        dtype = next(self.model.parameters()).dtype
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._verify_input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype)
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            10,
            device=self.device,
            dtype=torch.float32,
        )

        self.route_scores = torch.zeros(
            self.requests_per_iteration,
            self.num_routes,
            device=self.device,
            dtype=dtype,
        )
        self.route_ids = torch.empty(
            self.requests_per_iteration,
            device=self.device,
            dtype=torch.int64,
        )
        self.route_scores[:, 0] = 1.0  # always select "large"
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None

        with self._nvtx_range("routing"):
            with torch.inference_mode():
                if self.route_scores is None:
                    raise RuntimeError("Routing scores not initialized")
                if self.route_ids is None:
                    raise RuntimeError("Routing id buffer not initialized")
                # Vectorized routing: compute all argmaxes in one kernel.
                torch.argmax(self.route_scores, dim=1, out=self.route_ids)
            if self._verify_input is not None:
                with torch.inference_mode():
                    self.output = self.model(self._verify_input)
        if self.output is None or self._verify_input is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = torch.float32
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        if self.output is None or self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.route_ids = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics

        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, "total_tokens", 256),
            total_requests=getattr(self, "total_requests", 1),
            batch_size=getattr(self, "batch_size", 1),
            max_batch_size=getattr(self, "max_batch_size", 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> OptimizedRoutingStaticBenchmark:
    return OptimizedRoutingStaticBenchmark()
