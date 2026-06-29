"""optimized_torchao_quantization.py - torchao INT8 dynamic quantization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import configure_tf32, restore_tf32

try:
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
except Exception as exc:  # pragma: no cover - fail fast if torchao is missing
    quantize_ = None
    Int8DynamicActivationInt8WeightConfig = None
    TORCHAO_IMPORT_ERROR = exc
else:
    TORCHAO_IMPORT_ERROR = None


class OptimizedTorchAOQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: torchao INT8 dynamic activation + INT8 weight quantization."""

    signature_equivalence_group = "ch13_torchao_quantization_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self._tf32_state = (None, None)
        self.batch_size = 8192
        self.in_features = 4096
        self.hidden_features = 4096
        self.out_features = 4096
        tokens = self.batch_size * self.in_features
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for torchao quantization benchmark")
        if TORCHAO_IMPORT_ERROR is not None or quantize_ is None or Int8DynamicActivationInt8WeightConfig is None:
            raise RuntimeError(f"SKIPPED: torchao is required for torchao quantization benchmark: {TORCHAO_IMPORT_ERROR}")
        # Keep the comparison isolated to quantization rather than TF32 fallback behavior.
        self._tf32_state = configure_tf32(
            matmul_precision="highest",
            cudnn_precision="highest",
        )

        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_features, self.out_features),
        ).to(self.device).to(torch.float32)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        quantize_(self.model, Int8DynamicActivationInt8WeightConfig())

        self.data = torch.randn(
            self.batch_size,
            self.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        self._verify_input = self.data.detach().clone()
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.out_features),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self.data)

    def benchmark_fn(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("optimized_torchao_quantization"):
            with torch.inference_mode():
                self.output = self.model(self.data)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.data = None
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        restore_tf32(self._tf32_state)
        self._tf32_state = (None, None)
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedTorchAOQuantizationBenchmark:
    """Factory function for harness discovery."""
    return OptimizedTorchAOQuantizationBenchmark()
