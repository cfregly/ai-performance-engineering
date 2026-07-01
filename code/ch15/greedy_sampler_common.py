"""Shared greedy-sampler benchmark for Chapter 15 serving fast paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class GreedySamplerConfig:
    batch_size: int = 256
    vocab_size: int = 32768
    steps: int = 32
    materialize_probabilities: bool = True
    iterations: int = 20
    warmup: int = 5
    label: str = "greedy_sampler"


class GreedySamplerBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compare full-probability greedy decode with direct logits argmax."""

    def __init__(self, cfg: GreedySamplerConfig):
        super().__init__()
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if cfg.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if cfg.steps < 1:
            raise ValueError("steps must be >= 1")
        self.cfg = cfg
        self.logits: Optional[torch.Tensor] = None
        self.temperature: Optional[torch.Tensor] = None
        self.temperature_column: Optional[torch.Tensor] = None
        self.scaled_logits_buffer: Optional[torch.Tensor] = None
        self.max_values_buffer: Optional[torch.Tensor] = None
        self.output_tokens: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._step_range = range(cfg.steps)
        self._custom_metrics: Dict[str, float] = {}
        tokens = cfg.batch_size * cfg.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(cfg.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(cfg.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        cfg = self.cfg
        g = torch.Generator(device="cpu")
        g.manual_seed(15037)
        self.logits = torch.randn(
            cfg.batch_size,
            cfg.vocab_size,
            generator=g,
            device="cpu",
            dtype=torch.float32,
        ).to(self.device)
        row_ids = torch.arange(cfg.batch_size, device=self.device, dtype=torch.float32)
        self.temperature = 0.7 + 0.6 * (row_ids / float(max(cfg.batch_size - 1, 1)))
        self.temperature_column = self.temperature.view(cfg.batch_size, 1)
        self.scaled_logits_buffer = (
            torch.empty_like(self.logits) if cfg.materialize_probabilities else None
        )
        self.max_values_buffer = torch.empty(
            cfg.batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        self.output_tokens = torch.empty(
            cfg.steps,
            cfg.batch_size,
            device=self.device,
            dtype=torch.long,
        )
        self.output = None
        self._step_range = range(cfg.steps)
        self._refresh_custom_metrics()

    def _refresh_custom_metrics(self) -> None:
        cfg = self.cfg
        probability_elements = (
            cfg.steps * cfg.batch_size * cfg.vocab_size
            if cfg.materialize_probabilities
            else 0
        )
        self._custom_metrics = {
            "greedy_sampler.batch_size": float(cfg.batch_size),
            "greedy_sampler.vocab_size": float(cfg.vocab_size),
            "greedy_sampler.steps": float(cfg.steps),
            "greedy_sampler.materializes_probabilities": float(cfg.materialize_probabilities),
            "greedy_sampler.softmax_calls": float(cfg.steps if cfg.materialize_probabilities else 0),
            "greedy_sampler.argmax_calls": float(cfg.steps),
            "greedy_sampler.probability_elements_materialized": float(probability_elements),
            "greedy_sampler.probability_bytes_materialized": float(
                probability_elements * torch.tensor([], dtype=torch.float32).element_size()
            ),
        }

    def benchmark_fn(self) -> None:
        if (
            self.logits is None
            or self.max_values_buffer is None
            or self.output_tokens is None
        ):
            raise RuntimeError("Greedy sampler buffers are not initialized")

        logits = self.logits
        max_values = self.max_values_buffer
        output = self.output_tokens

        with torch.inference_mode(), self._nvtx_range(self.cfg.label):
            if self.cfg.materialize_probabilities:
                if self.scaled_logits_buffer is None or self.temperature_column is None:
                    raise RuntimeError("Softmax sampler buffers are not initialized")
                scaled = self.scaled_logits_buffer
                temperature = self.temperature_column
                for step in self._step_range:
                    torch.div(logits, temperature, out=scaled)
                    probabilities = torch.softmax(scaled, dim=-1)
                    output_slot = output[step]
                    torch.max(probabilities, dim=-1, out=(max_values, output_slot))
            else:
                for step in self._step_range:
                    output_slot = output[step]
                    torch.max(logits, dim=-1, out=(max_values, output_slot))
        self.output = output

    def capture_verification_payload(self) -> None:
        if self.logits is None or self.temperature is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before verification capture")
        self._set_verification_payload(
            inputs={"logits": self.logits, "temperature": self.temperature},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=self.cfg.iterations, warmup=self.cfg.warmup)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_input_signature(self) -> InputSignature:
        return InputSignature(
            shapes={"logits": (self.cfg.batch_size, self.cfg.vocab_size)},
            dtypes={"logits": "float32"},
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags=PrecisionFlags(
                fp16=False,
                bf16=False,
                fp8=False,
                tf32=torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            ),
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape != (self.cfg.steps, self.cfg.batch_size):
            return f"Unexpected output shape: {tuple(self.output.shape)}"
        return None

    def teardown(self) -> None:
        self.logits = None
        self.temperature = None
        self.temperature_column = None
        self.scaled_logits_buffer = None
        self.max_values_buffer = None
        self.output_tokens = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
