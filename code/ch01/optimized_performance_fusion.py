"""optimized_performance_fusion.py - Batch fusion optimization in FP32 (no precision change)."""

from __future__ import annotations

from typing import Optional

import torch

from core.utils.warning_filters import warn_optional_component_unavailable

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError as exc:
    warn_optional_component_unavailable(
        "ch01.arch_config",
        exc,
        impact="Chapter 1 architecture defaults were not applied; benchmark continues with stock runtime settings",
        context="ch01.optimized_performance_fusion import",
    )

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch01.performance_common import (
    build_training_mlp,
    capture_tf32_state,
    get_environment_custom_metrics,
    preallocate_fused_microbatches,
    restore_tf32_state,
    seed_chapter1,
    set_tf32_state,
)
from ch01.workload_config import WORKLOAD


class OptimizedPerformanceFusionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Batch fusion-only optimization: fuse microbatches while keeping FP32 math."""

    signature_equivalence_group = "ch01_performance_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.workload = WORKLOAD
        self.batch_size = self.workload.microbatch_size
        self.num_microbatches = self.workload.performance_microbatches
        self.fusion = 8
        self.hidden_dim = self.workload.performance_hidden_dim

        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self._verify_input = None
        self._verify_output = None
        self.parameter_count = 0
        self._fused_batches = None
        self._fused_targets = None
        self._fused_pairs = None
        self._tf32_state: tuple[bool, bool | None] | None = None

        samples = float(self.batch_size * self.num_microbatches)
        self.register_workload_metadata(samples_per_iteration=samples)

    def setup(self) -> None:
        seed_chapter1()
        self._tf32_state = capture_tf32_state()
        set_tf32_state(False)

        self.model = build_training_mlp(self.hidden_dim).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.microbatches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32).contiguous()
            for _ in range(self.num_microbatches)
        ]
        self.targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.num_microbatches)
        ]

        self._verify_input = self.microbatches[0].clone()
        self._verify_output = torch.empty(
            self._verify_input.shape[0],
            10,
            device=self.device,
            dtype=torch.float32,
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        self._synchronize()
        self.optimizer.zero_grad(set_to_none=True)

        self._fused_batches, self._fused_targets = preallocate_fused_microbatches(
            self.microbatches,
            self.targets,
            self.fusion,
        )
        self._fused_pairs = tuple(zip(self._fused_batches, self._fused_targets, strict=True))

    def benchmark_fn(self) -> None:
        assert self.model is not None and self._fused_pairs is not None
        with self._nvtx_range("optimized_performance_fusion"):
            for data, target in self._fused_pairs:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self._verify_output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        model_dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            verify_output = self.model(self._verify_input)
            self._verify_output.copy_(verify_output)
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self._verify_output,
            batch_size=self._verify_input.shape[0],
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": model_dtype == torch.float16,
                "bf16": model_dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32),
            },
            output_tolerance=(0.5, 0.5),
        )

    def teardown(self) -> None:
        del self.model, self.microbatches, self.targets, self.optimizer
        self._fused_batches = None
        self._fused_targets = None
        self._fused_pairs = None
        self._verify_input = None
        self._verify_output = None
        restore_tf32_state(self._tf32_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=10)

    def get_custom_metrics(self) -> Optional[dict]:
        return get_environment_custom_metrics()

    def validate_result(self) -> Optional[str]:
        if self.model is None or not self.microbatches:
            return "Model or data not initialized"
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output batch size mismatch: expected {probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as exc:
            return f"Model forward pass failed: {exc}"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedPerformanceFusionBenchmark()
