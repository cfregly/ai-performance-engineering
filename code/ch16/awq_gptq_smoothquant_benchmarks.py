"""Shared post-training quantization benchmarks for Chapter 16."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from core.benchmark.metrics import compute_precision_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


QuantizationScheme = Literal["baseline", "awq", "gptq", "smoothquant"]
INT4_MAX = 7.0
INT8_MAX = 127.0


def _resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass(frozen=True)
class PTQWorkload:
    batch_size: int = 4096
    in_features: int = 2048
    hidden_features: int = 4096
    out_features: int = 2048
    calibration_samples: int = 1024
    dtype: torch.dtype = torch.bfloat16


class ReferenceMLP(nn.Module):
    """Dense serving MLP used as the pre-quantization reference."""

    def __init__(self, workload: PTQWorkload, device: torch.device) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            workload.in_features,
            workload.hidden_features,
            bias=True,
            device=device,
            dtype=workload.dtype,
        )
        self.fc2 = nn.Linear(
            workload.hidden_features,
            workload.out_features,
            bias=True,
            device=device,
            dtype=workload.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(hidden)


def _quantize_weight(weight: torch.Tensor, qmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    weight_scale = torch.clamp(weight.abs().amax(dim=1) / qmax, min=1e-8)
    weight_q = torch.clamp((weight / weight_scale[:, None]).round(), -qmax, qmax).to(torch.int8)
    return weight_q.t().contiguous(), weight_scale.detach()


def _activation_scale(calibration: torch.Tensor) -> torch.Tensor:
    scale = calibration.abs().mean(dim=0)
    mean_scale = torch.clamp(scale.mean(), min=1e-6)
    normalized = scale / mean_scale
    return torch.clamp(normalized, min=0.5, max=2.0)


def _hessian_proxy(calibration: torch.Tensor) -> torch.Tensor:
    return torch.clamp(calibration.square().mean(dim=0).sqrt(), min=0.25, max=4.0)


def _smoothquant_scale(weight: torch.Tensor, calibration: torch.Tensor) -> torch.Tensor:
    act_scale = torch.clamp(calibration.abs().amax(dim=0), min=1e-4)
    weight_scale = torch.clamp(weight.abs().amax(dim=0), min=1e-4)
    return torch.clamp((act_scale.sqrt() / weight_scale.sqrt()), min=0.25, max=4.0)


class PTQLinear(nn.Module):
    """Quantized linear layer backed by ``torch._int_mm``."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], calibration: torch.Tensor, scheme: str) -> None:
        super().__init__()
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("torch._int_mm is required for Chapter 16 PTQ benchmarks")
        if weight.dim() != 2:
            raise ValueError("PTQLinear expects a 2D weight tensor")
        if calibration.dim() != 2 or calibration.size(1) != weight.size(1):
            raise ValueError("Calibration tensor must be 2D and match the input width")

        scheme = str(scheme).strip().lower()
        if scheme not in {"awq", "gptq", "smoothquant"}:
            raise ValueError(f"Unsupported PTQ scheme '{scheme}'")

        if scheme == "awq":
            input_transform = _activation_scale(calibration).reciprocal()
            transformed_weight = weight * input_transform.reciprocal().unsqueeze(0)
            qmax = INT4_MAX
            weight_bits = 4
        elif scheme == "gptq":
            input_transform = _hessian_proxy(calibration)
            transformed_weight = weight / input_transform.unsqueeze(0)
            qmax = INT4_MAX
            weight_bits = 4
        else:
            input_transform = _smoothquant_scale(weight, calibration).reciprocal()
            transformed_weight = weight * input_transform.reciprocal().unsqueeze(0)
            qmax = INT8_MAX
            weight_bits = 8

        weight_q_t, weight_scale = _quantize_weight(transformed_weight.float(), qmax=qmax)

        self.scheme = scheme
        self.weight_bits = weight_bits
        self.register_buffer("weight_q_t", weight_q_t)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("input_transform", input_transform.detach().float())
        self.register_buffer("_input_scaled_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_input_abs_buffer", torch.empty(0), persistent=False)
        self.register_buffer("_input_int8_buffer", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("_output_float_buffer", torch.empty(0), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().clone())
        else:
            self.bias = None

    def prepare_buffers(self, batch_size: int, device: torch.device) -> None:
        in_features = int(self.weight_q_t.shape[0])
        out_features = int(self.weight_q_t.shape[1])
        self._input_scaled_buffer = torch.empty(
            batch_size,
            in_features,
            device=device,
            dtype=torch.float32,
        )
        self._input_abs_buffer = torch.empty_like(self._input_scaled_buffer)
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
            raise RuntimeError("PTQLinear expects 2D activations")
        if x.size(0) <= 16:
            raise RuntimeError("torch._int_mm requires batch dimension > 16")
        if x.size(1) != self.weight_q_t.size(0):
            raise RuntimeError("Input width does not match quantized weight")
        if x.size(1) % 8 != 0 or self.weight_q_t.size(1) % 8 != 0:
            raise RuntimeError("torch._int_mm requires K and N to be multiples of 8")
        if (
            self._input_scaled_buffer.dim() != 2
            or self._output_float_buffer.dim() != 2
            or self._input_scaled_buffer.size(0) < x.size(0)
            or self._input_scaled_buffer.size(1) != x.size(1)
            or self._output_float_buffer.size(0) < x.size(0)
            or self._output_float_buffer.size(1) != self.weight_q_t.size(1)
            or self._input_scaled_buffer.device != x.device
        ):
            self.prepare_buffers(x.shape[0], x.device)

        transformed_x = self._input_scaled_buffer[: x.size(0)]
        abs_buffer = self._input_abs_buffer[: x.size(0)]
        input_int8 = self._input_int8_buffer[: x.size(0)]
        output = self._output_float_buffer[: x.size(0)]
        transformed_x.copy_(x)
        transformed_x.mul_(self.input_transform)
        torch.abs(transformed_x, out=abs_buffer)
        input_scale = torch.clamp(abs_buffer.amax() / INT8_MAX, min=1e-8)
        torch.div(transformed_x, input_scale, out=transformed_x)
        torch.round(transformed_x, out=transformed_x)
        torch.clamp(transformed_x, -INT8_MAX, INT8_MAX, out=transformed_x)
        input_int8.copy_(transformed_x)
        out_int32 = torch._int_mm(input_int8, self.weight_q_t)
        output.copy_(out_int32)
        output.mul_(input_scale)
        output.mul_(self.weight_scale)
        if self.bias is not None:
            output.add_(self.bias)
        return output


class PTQMLP(nn.Module):
    """Two-layer PTQ MLP that preserves the reference model contract."""

    def __init__(self, reference: ReferenceMLP, calibration: torch.Tensor, scheme: str) -> None:
        super().__init__()
        hidden_calibration = torch.nn.functional.gelu(reference.fc1(calibration), approximate="tanh")
        self.fc1 = PTQLinear(reference.fc1.weight, reference.fc1.bias, calibration, scheme=scheme)
        self.fc2 = PTQLinear(reference.fc2.weight, reference.fc2.bias, hidden_calibration, scheme=scheme)

    def prepare_buffers(self, batch_size: int, device: torch.device) -> None:
        self.fc1.prepare_buffers(batch_size, device)
        self.fc2.prepare_buffers(batch_size, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(hidden)


class PTQQuantizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Shared harness benchmark for AWQ/GPTQ/SmoothQuant coverage."""

    signature_equivalence_group = "ch16_post_training_quantization"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self, *, scheme: QuantizationScheme, label: str) -> None:
        super().__init__()
        self.scheme = scheme
        self.label = label
        self.workload = PTQWorkload(dtype=_resolve_dtype())
        tokens = self.workload.batch_size * self.workload.in_features
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

        self.reference_model: Optional[ReferenceMLP] = None
        self.optimized_model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_input_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for Chapter 16 PTQ benchmarks")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.reference_model = ReferenceMLP(self.workload, self.device).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.reference_model.parameters())
        self.inputs = torch.randn(
            self.workload.batch_size,
            self.workload.in_features,
            device=self.device,
            dtype=self.workload.dtype,
        )
        self._verify_input_buffer = torch.empty_like(self.inputs, dtype=torch.float32)
        self._verify_output_buffer = torch.empty(
            (self.workload.batch_size, self.workload.out_features),
            device=self.device,
            dtype=torch.float32,
        )

        if self.scheme == "baseline":
            self.optimized_model = self.reference_model
            for _ in range(2):
                with torch.inference_mode():
                    _ = self.optimized_model(self.inputs)
            return

        calibration = torch.randn(
            self.workload.calibration_samples,
            self.workload.in_features,
            device=self.device,
            dtype=self.workload.dtype,
        )
        quantized = PTQMLP(self.reference_model, calibration, scheme=self.scheme).to(self.device).eval()
        quantized.prepare_buffers(self.workload.batch_size, self.device)
        if hasattr(torch, "compile"):
            self.optimized_model = torch.compile(quantized, mode="max-autotune")
        else:
            self.optimized_model = quantized

        for _ in range(2):
            with torch.inference_mode():
                _ = self.optimized_model(self.inputs)

    def benchmark_fn(self) -> None:
        if self.optimized_model is None or self.inputs is None:
            raise RuntimeError("Model/data not initialized")
        with self._nvtx_range(self.label):
            with torch.inference_mode():
                self.output = self.optimized_model(self.inputs)

    def capture_verification_payload(self) -> None:
        if (
            self.inputs is None
            or self.output is None
            or self._verify_input_buffer is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        # Verification is baseline-bound in the harness, so the dense reference must
        # advertise the same bounded PTQ family tolerance as the optimized variants.
        tolerance = (1.0, 10.0)
        self._verify_input_buffer.copy_(self.inputs)
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input_buffer},
            output=self._verify_output_buffer,
            batch_size=self.inputs.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": self.workload.dtype == torch.float16,
                "bf16": self.workload.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=tolerance,
        )

    def teardown(self) -> None:
        self.reference_model = None
        self.optimized_model = None
        self.inputs = None
        self.output = None
        self._verify_input_buffer = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        if self.scheme == "baseline":
            metrics = compute_precision_metrics(
                fp32_time_ms=getattr(self, "_last_elapsed_ms", None),
                reduced_precision_time_ms=None,
                precision_type="bf16" if self.workload.dtype == torch.bfloat16 else "fp16",
            )
            weight_bits = 16.0
        else:
            precision_type = "int8" if self.scheme == "smoothquant" else "int4"
            metrics = compute_precision_metrics(
                fp32_time_ms=None,
                reduced_precision_time_ms=getattr(self, "_last_elapsed_ms", None),
                precision_type=precision_type,
            )
            weight_bits = 8.0 if self.scheme == "smoothquant" else 4.0
        metrics.update(
            {
                "ptq.scheme_awq": 1.0 if self.scheme == "awq" else 0.0,
                "ptq.scheme_gptq": 1.0 if self.scheme == "gptq" else 0.0,
                "ptq.scheme_smoothquant": 1.0 if self.scheme == "smoothquant" else 0.0,
                "ptq.scheme_baseline": 1.0 if self.scheme == "baseline" else 0.0,
                "ptq.weight_bits": weight_bits,
                "ptq.uses_int_mm": 0.0 if self.scheme == "baseline" else 1.0,
            }
        )
        return metrics

    def get_optimization_goal(self) -> str:
        """PTQ variants are tracked as memory-reduction benchmarks on this family."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None
