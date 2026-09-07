"""Shared NVFP4 MLP benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass(frozen=True)
class NVFP4MLPConfig:
    batch_size: int
    d_model: int
    d_ff: int
    num_layers: int = 2
    iterations: int = 20
    warmup: int = 10
    use_bias: bool = True
    activation: str = "gelu"
    output_tolerance: Tuple[float, float] = (1e-1, 1e-1)
    calibrate: bool = True
    name: Optional[str] = None


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _init_weight(
    shape: Tuple[int, int],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = torch.empty(shape, device=device, dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    return weight.to(dtype)


def create_mlp_weights(
    config: NVFP4MLPConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    weights: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]] = []
    for _ in range(config.num_layers):
        w1 = _init_weight((config.d_ff, config.d_model), generator=generator, device=device, dtype=dtype)
        b1 = torch.zeros(config.d_ff, device=device, dtype=dtype) if config.use_bias else None
        w2 = _init_weight((config.d_model, config.d_ff), generator=generator, device=device, dtype=dtype)
        b2 = torch.zeros(config.d_model, device=device, dtype=dtype) if config.use_bias else None
        weights.append((w1, b1, w2, b2))
    return weights


def build_baseline_mlp(
    config: NVFP4MLPConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    weights: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
) -> nn.Module:
    layers: List[nn.Module] = []
    for w1, b1, w2, b2 in weights:
        fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias, device=device, dtype=dtype)
        fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias, device=device, dtype=dtype)
        with torch.no_grad():
            fc1.weight.copy_(w1)
            if config.use_bias and b1 is not None:
                fc1.bias.copy_(b1)
            fc2.weight.copy_(w2)
            if config.use_bias and b2 is not None:
                fc2.bias.copy_(b2)
        layers.extend([fc1, _activation(config.activation), fc2])
    return nn.Sequential(*layers)


def require_nvfp4_support():
    try:
        from transformer_engine.pytorch import Linear as TELinear
        from transformer_engine.pytorch import autocast as te_autocast, is_nvfp4_available
        from transformer_engine.pytorch import quantized_model_init
        from transformer_engine.common import recipe as te_recipe
    except Exception as exc:
        raise RuntimeError("SKIPPED: Transformer Engine not available for NVFP4") from exc
    if not is_nvfp4_available():
        raise RuntimeError("SKIPPED: NVFP4 kernels unavailable on this hardware/driver")
    if not hasattr(te_recipe, "NVFP4BlockScaling"):
        raise RuntimeError("SKIPPED: NVFP4BlockScaling recipe unavailable in Transformer Engine")
    return TELinear, te_autocast, te_recipe.NVFP4BlockScaling(), quantized_model_init


def build_nvfp4_mlp(
    config: NVFP4MLPConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    weights: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]],
) -> Tuple[nn.Module, object, object]:
    TELinear, te_autocast, recipe, quantized_model_init = require_nvfp4_support()
    layers: List[nn.Module] = []
    for w1, b1, w2, b2 in weights:
        with quantized_model_init(enabled=True, recipe=recipe):
            fc1 = TELinear(config.d_model, config.d_ff, bias=config.use_bias, device=device, params_dtype=dtype)
            fc2 = TELinear(config.d_ff, config.d_model, bias=config.use_bias, device=device, params_dtype=dtype)
        with torch.no_grad():
            fc1.weight.copy_(w1)
            if config.use_bias and b1 is not None:
                fc1.bias.copy_(b1)
            fc2.weight.copy_(w2)
            if config.use_bias and b2 is not None:
                fc2.bias.copy_(b2)
        layers.extend([fc1, _activation(config.activation), fc2])
    return nn.Sequential(*layers), te_autocast, recipe


class NVFP4MLPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self, config: Optional[NVFP4MLPConfig] = None, *, use_nvfp4: bool = False) -> None:
        super().__init__()
        self.config = config
        self.config_dict = dict(config.__dict__) if config is not None else {}
        self.use_nvfp4 = use_nvfp4
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._te_autocast = None
        self._nvfp4_recipe = None

    def setup(self) -> None:
        if self.config is None:
            if not self.config_dict:
                raise RuntimeError("NVFP4MLPBenchmark requires an explicit config")
            self.config = NVFP4MLPConfig(**self.config_dict)
        if self.device.type != "cuda":
            raise RuntimeError("SKIPPED: NVFP4 MLP benchmark requires CUDA")
        dtype = torch.bfloat16
        self.input = torch.randn(
            self.config.batch_size,
            self.config.d_model,
            device=self.device,
            dtype=dtype,
        )
        weights = create_mlp_weights(self.config, device=self.device, dtype=dtype)
        if self.use_nvfp4:
            self.model, self._te_autocast, self._nvfp4_recipe = build_nvfp4_mlp(
                self.config,
                device=self.device,
                dtype=dtype,
                weights=weights,
            )
            if self.config.calibrate:
                with torch.inference_mode():
                    with self._te_autocast(enabled=True, recipe=self._nvfp4_recipe, calibrating=True):
                        _ = self.model(self.input)
        else:
            self.model = build_baseline_mlp(
                self.config,
                device=self.device,
                dtype=dtype,
                weights=weights,
            )
        self.model.eval()
        tokens = float(self.config.batch_size * self.config.d_model)
        self.register_workload_metadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=tokens,
        )

    def benchmark_fn(self) -> None:
        if self.model is None or self.input is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            if self.use_nvfp4:
                if self._te_autocast is None or self._nvfp4_recipe is None:
                    raise RuntimeError("NVFP4 autocast not initialized")
                with self._te_autocast(enabled=True, recipe=self._nvfp4_recipe):
                    self.output = self.model(self.input)
            else:
                self.output = self.model(self.input)

    def capture_verification_payload(self) -> None:
        if self.model is None or self.input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=self.config.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=self.config.output_tolerance,
        )

    def get_input_signature(self) -> InputSignature:
        if self.config is None:
            if not self.config_dict:
                raise RuntimeError("NVFP4MLPBenchmark requires an explicit config")
            self.config = NVFP4MLPConfig(**self.config_dict)
        parameter_count = self.config.num_layers * (
            (self.config.d_model * self.config.d_ff)
            + (self.config.d_ff * self.config.d_model)
            + (self.config.d_ff if self.config.use_bias else 0)
            + (self.config.d_model if self.config.use_bias else 0)
        )
        dtype = str(torch.bfloat16)
        return InputSignature(
            shapes={
                "input": (self.config.batch_size, self.config.d_model),
                "output": (self.config.batch_size, self.config.d_model),
            },
            dtypes={"input": dtype, "output": dtype},
            batch_size=self.config.batch_size,
            parameter_count=parameter_count,
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        if torch.isnan(self.output).any():
            return "Output contains NaN"
        if torch.isinf(self.output).any():
            return "Output contains Inf"
        return None

    def get_config(self) -> BenchmarkConfig:
        if self.config is None:
            if not self.config_dict:
                raise RuntimeError("NVFP4MLPBenchmark requires an explicit config")
            self.config = NVFP4MLPConfig(**self.config_dict)
        return BenchmarkConfig(
            iterations=self.config.iterations,
            warmup=self.config.warmup,
        )

    def get_optimization_goal(self) -> str:
        # This family is intentionally kept as a reduced-precision memory tradeoff.
        # On the current validated stack, NVFP4 consistently reduces footprint
        # without producing a durable latency win versus the BF16 baseline.
        return "memory"


__all__ = [
    "NVFP4MLPBenchmark",
    "NVFP4MLPConfig",
    "build_baseline_mlp",
    "build_nvfp4_mlp",
    "create_mlp_weights",
]
