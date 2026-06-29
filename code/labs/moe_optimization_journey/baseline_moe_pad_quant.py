"""Baseline MoE pad+quant + finalize+slice benchmark (no fusion)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from labs.moe_optimization_journey.moe_pad_quant_common import build_moe_pad_quant_model


class BaselineMoEPadQuantBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: eager pad+quant + finalize+slice path."""

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

        self.vocab_size = 32000
        self.hidden = 512
        self.intermediate = 2048
        self.num_experts = 32
        self.num_experts_per_tok = 2
        self.batch = 8
        self.seq_len = 128
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self._enable_nvtx = False
        self._payload_parameter_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        model, _ = build_moe_pad_quant_model(
            hidden_size=self.hidden,
            intermediate_size=self.intermediate,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            vocab_size=self.vocab_size,
            num_layers=1,
            num_heads=8,
            level=4,
        )
        self.model = model.to(self.device, dtype=torch.bfloat16)
        self.model.eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randint(
            0, self.vocab_size, (self.batch, self.seq_len), device=self.device
        )
        self._verify_output_buffer = torch.empty(
            self.batch,
            min(128, self.seq_len),
            min(256, self.vocab_size),
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Benchmark not initialized")
        with nvtx_range("moe_pad_quant_baseline", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self.inputs is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input_ids": self.inputs.detach()},
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.inputs = None
        self.output = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=6,
            warmup=6,
            # Match the existing MoE journey convention: compile-heavy MoE
            # surfaces can crash during subprocess teardown on some hosts.
            use_subprocess=False,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineMoEPadQuantBenchmark()
