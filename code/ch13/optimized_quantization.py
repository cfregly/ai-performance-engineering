"""optimized_quantization.py - Optimized INT8 quantization via torch._int_mm."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


INT8_MAX = 127.0


class Int8Linear(nn.Module):
    """INT8 dynamic-activation / static-weight linear using torch._int_mm."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("torch._int_mm is required for INT8 quantization benchmark")
        if weight.dim() != 2:
            raise RuntimeError("Expected 2D weight for INT8 linear")

        weight_scale = torch.clamp(weight.abs().amax(dim=1) / INT8_MAX, min=1e-8)
        weight_q = torch.clamp((weight / weight_scale[:, None]).round(), -INT8_MAX, INT8_MAX)
        weight_int8_t = weight_q.to(torch.int8).t().contiguous()

        self.register_buffer("weight_scale", weight_scale.detach())
        self.register_buffer("weight_int8_t", weight_int8_t)
        self.register_buffer("_input_scaled_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_input_int8_buffer", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("_output_float_buffer", torch.empty(0), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def prepare_buffers(self, batch_size: int, device: torch.device) -> None:
        in_features = int(self.weight_int8_t.shape[0])
        out_features = int(self.weight_int8_t.shape[1])
        self._input_scaled_buffer = torch.empty(
            batch_size,
            in_features,
            device=device,
            dtype=torch.float32,
        )
        self._input_int8_buffer = torch.empty(
            batch_size,
            in_features,
            device=device,
            dtype=torch.int8,
        )
        self._output_float_buffer = torch.empty(
            batch_size,
            out_features,
            device=device,
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise RuntimeError("INT8 linear expects 2D input")
        if x.size(0) <= 16:
            raise RuntimeError("torch._int_mm requires M > 16")
        if x.size(1) % 8 != 0 or self.weight_int8_t.size(0) % 8 != 0:
            raise RuntimeError("torch._int_mm requires K and N to be multiples of 8")
        rows = int(x.size(0))
        if (
            self._input_scaled_buffer.dim() != 2
            or self._input_scaled_buffer.size(0) < rows
            or self._input_scaled_buffer.size(1) != x.size(1)
            or self._input_scaled_buffer.device != x.device
        ):
            self.prepare_buffers(rows, x.device)
        input_scaled = self._input_scaled_buffer[:rows]
        input_int8 = self._input_int8_buffer[:rows]
        output = self._output_float_buffer[:rows]
        torch.abs(x, out=input_scaled)
        input_scale = torch.clamp(input_scaled.amax() / INT8_MAX, min=1e-8)
        torch.div(x, input_scale, out=input_scaled)
        torch.round(input_scaled, out=input_scaled)
        torch.clamp(input_scaled, -INT8_MAX, INT8_MAX, out=input_scaled)
        input_int8.copy_(input_scaled)
        out_int32 = torch._int_mm(input_int8, self.weight_int8_t)
        output.copy_(out_int32)
        output.mul_(input_scale * self.weight_scale)
        if self.bias is not None:
            output.add_(self.bias)
        return output


class Int8MLP(nn.Module):
    """Two-layer INT8 MLP for compiled inference."""

    def __init__(
        self,
        weight1: torch.Tensor,
        bias1: Optional[torch.Tensor],
        weight2: torch.Tensor,
        bias2: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.fc1 = Int8Linear(weight1, bias1)
        self.fc2 = Int8Linear(weight2, bias2)

    def prepare_buffers(self, batch_size: int, device: torch.device) -> None:
        self.fc1.prepare_buffers(batch_size, device)
        self.fc2.prepare_buffers(batch_size, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu_(x)
        x = self.fc2(x)
        return x


class OptimizedQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: INT8 quantization with dynamic activation and static weights."""

    signature_equivalence_group = "ch13_quantization_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.int8_model = None
        self.compiled_model = None
        self.data = None
        self.data_fp32 = None
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
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("torch._int_mm is required for INT8 quantization benchmark")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for INT8 quantization benchmark")

        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_features, self.out_features),
        ).to(self.device).to(torch.float32)
        self.model.eval()

        self.data_fp32 = torch.randn(
            self.batch_size,
            self.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        self.data = self.data_fp32
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.out_features),
            device=self.device,
            dtype=torch.float32,
        )

        self.int8_model = Int8MLP(
            self.model[0].weight,
            self.model[0].bias,
            self.model[2].weight,
            self.model[2].bias,
        ).to(self.device)
        self.int8_model.prepare_buffers(self.batch_size, self.device)
        self.compiled_model = torch.compile(self.int8_model, mode="max-autotune")
        for _ in range(2):
            with torch.inference_mode():
                _ = self.compiled_model(self.data_fp32)
    
    def benchmark_fn(self) -> None:
        if self.compiled_model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range("optimized_quantization"):
            with torch.inference_mode():
                self.output = self.compiled_model(self.data)
        if self.data is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.data is None or self.data_fp32 is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("FP32 verification input not initialized")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self.data_fp32},
            output=self._verify_output_buffer,
            batch_size=self.data.shape[0],
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.int8_model = None
        self.compiled_model = None
        self.data = None
        self.data_fp32 = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="int8",
        )

    def validate_result(self) -> Optional[str]:
        if self.compiled_model is None:
            return "Quantized model not initialized"
        return None


def get_benchmark() -> OptimizedQuantizationBenchmark:
    """Factory function for harness discovery."""
    return OptimizedQuantizationBenchmark()
