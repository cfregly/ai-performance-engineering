"""baseline_moe.py - Eager-mode baseline for the Chapter 20 toy MoE benchmark.

Pairs with `optimized_moe.py`, which uses `torch.compile` to reduce overhead
while preserving the same token outputs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import ch20.arch_config  # noqa: F401 - Apply chapter defaults

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class ToyMoe(nn.Module):
    """Simplified MoE block with two experts and top-1 routing."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 2, bias=False)
        self.expert0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.expert1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_expert = self.gate(x).argmax(dim=-1, keepdim=True)
        out0 = self.expert0(x)
        out1 = self.expert1(x)
        route_expert0 = top_expert == 0
        return torch.where(route_expert0, out0, out1)


class BaselineMoeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Eager MoE forward pass baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 32
        self.hidden_dim = 1024
        tokens = self.batch * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = ToyMoe(self.hidden_dim).to(self.device).half().eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.float16)
        self._verify_input = self.inputs[0:1].clone()
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        for _ in range(2):
            with torch.inference_mode():
                _ = self.model(self.inputs)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model/inputs not initialized")
        with self._nvtx_range("baseline_moe"):
            with torch.inference_mode():
                _ = self.model(self.inputs)

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() must prepare verify input before verification")
        with torch.inference_mode():
            verify_output = self.model(self._verify_input)
            self._verify_output_buffer.copy_(verify_output)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self._verify_input = None
        self._verify_output_buffer = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return BaselineMoeBenchmark()
