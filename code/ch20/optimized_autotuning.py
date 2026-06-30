"""optimized_autotuning.py - Compiled variant for the autotuning benchmark.

Pairs with `baseline_autotuning.py`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import ch20.arch_config  # noqa: F401 - Apply chapter defaults


from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import compile_model
from ch20.autotuning_common import AUTOTUNING_SETUP_PREWARM_ITERS, AutotuneModel


class OptimizedAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs a small model with torch.compile to validate autotune plumbing."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.batch = 1024
        self.hidden_dim = 4096
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
        model = AutotuneModel(self.hidden_dim).to(self.device, dtype=torch.bfloat16).eval()
        # Compile once in setup; benchmark_fn measures steady-state execution.
        self.model = compile_model(
            model,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(self.batch, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self._verify_input = self.inputs[0:1].clone()
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        for _ in range(AUTOTUNING_SETUP_PREWARM_ITERS):
            with torch.inference_mode():
                _ = self.model(self.inputs)

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None
        with self._nvtx_range("optimized_autotuning"):
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
            iterations=20,
            warmup=5,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAutotuningBenchmark()
