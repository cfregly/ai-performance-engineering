"""optimized_fp4_perchannel.py - TE NVFP4 per-channel weight scaling."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ch13.fp4_perchannel_common import build_fp4_perchannel_reference_tensors
from ch13.te_runtime_common import ensure_te_runtime_initialized
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

TELinear: Any | None = None
fp8_autocast: Any | None = None
is_nvfp4_available: Any | None = None
te_recipe: Any | None = None
TE_IMPORT_ERROR: Optional[ImportError] = None


def _load_transformer_engine() -> tuple[Any, Any, Any, Any]:
    global TELinear, fp8_autocast, is_nvfp4_available, te_recipe, TE_IMPORT_ERROR
    if (
        TELinear is not None
        and fp8_autocast is not None
        and is_nvfp4_available is not None
        and te_recipe is not None
    ):
        return TELinear, fp8_autocast, is_nvfp4_available, te_recipe
    ensure_te_runtime_initialized()
    try:
        from transformer_engine.pytorch import (
            Linear as te_linear,
            fp8_autocast as te_fp8_autocast,
            is_nvfp4_available as te_is_nvfp4_available,
        )
        from transformer_engine.common import recipe as te_recipe_module
    except ImportError as exc:  # pragma: no cover
        TE_IMPORT_ERROR = exc
        raise RuntimeError(
            "SKIPPED: Transformer Engine is required for optimized_fp4_perchannel. "
            f"(import error: {exc})"
        ) from exc
    TELinear = te_linear
    fp8_autocast = te_fp8_autocast
    is_nvfp4_available = te_is_nvfp4_available
    te_recipe = te_recipe_module
    return TELinear, fp8_autocast, is_nvfp4_available, te_recipe


def _apply_per_channel_scaling(linear: nn.Module) -> torch.Tensor:
    """Scale weights/bias per output channel and return scale vector for restoration."""
    weight = getattr(linear, "weight", None)
    if weight is None:
        raise RuntimeError("TE Linear weight missing for per-channel scaling")
    scale = weight.abs().amax(dim=1).clamp(min=1e-6).to(weight.dtype)
    with torch.inference_mode():
        weight.div_(scale.unsqueeze(1))
        bias = getattr(linear, "bias", None)
        if bias is not None:
            bias.div_(scale)
    return scale.unsqueeze(0)


class FP4PerChannelMLP(nn.Module):
    """Two-layer MLP with per-channel weight scaling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        te_linear, _, _, _ = _load_transformer_engine()
        self.fc1 = te_linear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = te_linear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()
        self.fc1_scale: Optional[torch.Tensor] = None
        self.fc2_scale: Optional[torch.Tensor] = None

    def prepare_per_channel_scales(self) -> None:
        self.fc1_scale = _apply_per_channel_scaling(self.fc1)
        self.fc2_scale = _apply_per_channel_scaling(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1_scale is None or self.fc2_scale is None:
            raise RuntimeError("Per-channel scales not initialized")
        x = self.fc1(x)
        x.mul_(self.fc1_scale)
        x = self.activation(x)
        x = self.fc2(x)
        x.mul_(self.fc2_scale)
        return x


class OptimizedFP4PerChannelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized NVFP4 path with per-channel weight scaling."""

    signature_equivalence_group = "ch13_fp4_perchannel"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 1024
        self.hidden_dim = 8192
        self.dtype = torch.float32
        self.parameter_count: int = 0
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.fp4_recipe: Optional[object] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        _, fp8_autocast_fn, is_nvfp4_available_fn, te_recipe_module = _load_transformer_engine()
        if not is_nvfp4_available_fn():
            raise RuntimeError("SKIPPED: NVFP4 kernels unavailable on this hardware/driver.")

        self.fp4_recipe = te_recipe_module.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
        )
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = FP4PerChannelMLP(hidden_dim=self.hidden_dim).to(self.device, dtype=self.dtype).eval()
        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        w1, b1, w2, b2, inputs = build_fp4_perchannel_reference_tensors(
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            self.model.fc1.weight.copy_(w1)
            if self.model.fc1.bias is not None:
                self.model.fc1.bias.copy_(b1)
            self.model.fc2.weight.copy_(w2)
            if self.model.fc2.bias is not None:
                self.model.fc2.bias.copy_(b2)

        self.model.prepare_per_channel_scales()

        self.inputs = inputs
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.hidden_dim),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(3):
            with torch.inference_mode(), fp8_autocast_fn(enabled=True, fp8_recipe=self.fp4_recipe):
                _ = self.model(self.inputs)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None or self.fp4_recipe is None:
            raise RuntimeError("Benchmark not initialized")
        _, fp8_autocast_fn, _, _ = _load_transformer_engine()
        with torch.inference_mode(), fp8_autocast_fn(enabled=True, fp8_recipe=self.fp4_recipe):
            self.output = self.model(self.inputs)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

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
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(2.0, 20.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        self.fp4_recipe = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFP4PerChannelBenchmark()
