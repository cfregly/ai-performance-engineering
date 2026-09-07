"""Reference template for harness-compliant benchmarks.

Copy this file when adding a new benchmark. It demonstrates:
- Caller-owned deterministic seeding
- VerificationPayloadMixin to surface inputs/output/signature/tolerance
- Explicit BenchmarkConfig with warmup/iterations

Run standalone checks from the repo root via ``python -m chXX.your_benchmark``
or ``python -m cli.aisp bench ...``. Do not add local ``sys.path`` surgery to
benchmark files.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class CompliantBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = True

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 8
        self.hidden_dim = 128

    def setup(self) -> None:
        self.model = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.input = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        """Benchmark hot path; avoid syncs, host transfers, and input regeneration here."""
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = self.model(self.input)

    def capture_verification_payload(self) -> None:
        if self.model is None or self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")

        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-5, 1e-8),
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,
        )

    def get_input_signature(self):
        return super().get_input_signature()

    def get_verify_output(self):
        return super().get_verify_output()

    def get_output_tolerance(self):
        return super().get_output_tolerance()

    def teardown(self) -> None:
        self.model = None
        self.input = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_benchmark() -> BaseBenchmark:
    return CompliantBenchmark()
