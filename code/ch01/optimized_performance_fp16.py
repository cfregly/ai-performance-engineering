"""optimized_performance_fp16.py - FP16-only performance benchmark (no batch fusion)."""

from __future__ import annotations

from typing import Optional

import torch

from core.common.device_utils import get_usable_cuda_or_cpu
from core.utils.warning_filters import warn_optional_component_unavailable

try:
    import ch01.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError as exc:
    warn_optional_component_unavailable(
        "ch01.arch_config",
        exc,
        impact="Chapter 1 architecture defaults were not applied; benchmark continues with stock runtime settings",
        context="ch01.optimized_performance_fp16 import",
    )

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch01.performance_common import (
    build_training_mlp,
    capture_tf32_state,
    get_environment_custom_metrics,
    restore_tf32_state,
    seed_chapter1,
    set_tf32_state,
)
from ch01.performance_fp16_common import PERFORMANCE_FP16_WORKLOAD
from ch01.workload_config import WORKLOAD


def _warn_cuda_probe_failure(message: str) -> None:
    print(f"WARNING: {message}")


class OptimizedPerformanceFP16Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP16-only optimization: use tensor cores without changing batch fusion strategy."""

    allow_cpu = True
    signature_equivalence_group = "ch01_performance_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.device = get_usable_cuda_or_cpu(warning_handler=_warn_cuda_probe_failure)
        self.workload = WORKLOAD
        self.batch_size = PERFORMANCE_FP16_WORKLOAD.batch_size
        self.num_microbatches = PERFORMANCE_FP16_WORKLOAD.num_microbatches
        self.fusion = 8
        self.hidden_dim = PERFORMANCE_FP16_WORKLOAD.hidden_dim

        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self._verify_input = None
        self._verify_model_input = None
        self._verify_output = None
        self.parameter_count = 0
        self._microbatch_groups = None
        self._target_groups = None
        self._group_sizes = None
        self._training_groups = None
        self._model_dtype = torch.float32
        self._tf32_state: tuple[bool, bool | None] | None = None
        self.register_workload_metadata(
            samples_per_iteration=PERFORMANCE_FP16_WORKLOAD.samples_per_iteration,
        )

    def setup(self) -> None:
        seed_chapter1()
        self._tf32_state = capture_tf32_state()
        set_tf32_state(False)

        self.model = build_training_mlp(self.hidden_dim)
        if self.device.type == "cuda":
            self.model = self.model.half()
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.model = self.model.to(self.device).eval()
        self._model_dtype = dtype
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.microbatches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype).contiguous()
            for _ in range(self.num_microbatches)
        ]
        self.targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.num_microbatches)
        ]

        # Match baseline input signature: verification inputs are FP32.
        self._verify_input = self.microbatches[0].float().clone()
        self._verify_model_input = self._verify_input.to(dtype=dtype, device=self.device)
        self._verify_output = torch.empty(self._verify_input.shape[0], 10, device=self.device, dtype=torch.float32)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        self._synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        self._microbatch_groups = []
        self._target_groups = []
        self._group_sizes = []
        self._training_groups = []
        for start in range(0, len(self.microbatches), self.fusion):
            data_group = tuple(self.microbatches[start : start + self.fusion])
            target_group = tuple(self.targets[start : start + self.fusion])
            paired_group = tuple(zip(data_group, target_group, strict=True))
            group_size = len(data_group)
            self._microbatch_groups.append(data_group)
            self._target_groups.append(target_group)
            self._group_sizes.append(group_size)
            self._training_groups.append((paired_group, group_size))

    def benchmark_fn(self) -> None:
        assert (
            self.model is not None
            and self._microbatch_groups is not None
            and self._target_groups is not None
            and self._group_sizes is not None
            and self._training_groups is not None
        )
        with self._nvtx_range("optimized_performance_fp16"):
            # Keep the baseline's microbatch grouping intact; the only timed change is precision.
            for paired_group, group_size in self._training_groups:
                self.optimizer.zero_grad(set_to_none=True)
                for data, target in paired_group:
                    logits = self.model(data)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    (loss / group_size).backward()
                self.optimizer.step()

    def capture_verification_payload(self) -> None:
        if (
            self.model is None
            or self._verify_input is None
            or self._verify_model_input is None
            or self._verify_output is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        with torch.inference_mode():
            verify_output = self.model(self._verify_model_input)
            self._verify_output.copy_(verify_output)
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self._verify_output,
            batch_size=self._verify_input.shape[0],
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": self._model_dtype == torch.float16,
                "bf16": False,
                "fp8": False,
                "tf32": torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32),
            },
            output_tolerance=(0.5, 0.5),
        )

    def teardown(self) -> None:
        del self.model, self.microbatches, self.targets, self.optimizer
        self._microbatch_groups = None
        self._target_groups = None
        self._group_sizes = None
        self._training_groups = None
        self._verify_input = None
        self._verify_model_input = None
        self._verify_output = None
        self._model_dtype = torch.float32
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
    return OptimizedPerformanceFP16Benchmark()
