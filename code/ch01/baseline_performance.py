"""baseline_performance.py - Baseline performance benchmark (goodput measurement).

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

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
        context="ch01.baseline_performance import",
    )

from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch01.performance_common import (
    build_training_mlp,
    capture_tf32_state,
    get_environment_custom_metrics,
    restore_tf32_state,
    seed_chapter1,
    set_tf32_state,
)
from ch01.workload_config import WORKLOAD


def _warn_cuda_probe_failure(message: str) -> None:
    print(f"WARNING: {message}")


class BaselinePerformanceBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""

    allow_cpu = True
    signature_equivalence_group = "ch01_performance_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.device = get_usable_cuda_or_cpu(warning_handler=_warn_cuda_probe_failure)
        self.model = None
        self.data = None
        self.target = None
        self.optimizer = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.microbatch_size
        self.num_microbatches = self.workload.performance_microbatches
        self.fusion = 8
        self.hidden_dim = self.workload.performance_hidden_dim
        self.microbatches = None
        self.targets = None
        self._verify_input = None
        self._verify_output = None
        self.parameter_count = 0
        self._microbatch_groups = None
        self._target_groups = None
        self._group_sizes = None
        self._training_groups = None
        self._tf32_state: tuple[bool, bool | None] | None = None
        samples = float(self.batch_size * self.num_microbatches)
        self.register_workload_metadata(samples_per_iteration=samples)
    
    def setup(self) -> None:
        """Setup: initialize model, fixed inputs, and verification output."""
        seed_chapter1()
        self._tf32_state = capture_tf32_state()
        set_tf32_state(False)
        
        self.model = build_training_mlp(self.hidden_dim).to(self.device)
        
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.microbatches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device)
            for _ in range(self.num_microbatches)
        ]
        self.targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.num_microbatches)
        ]
        
        # Create FIXED verification input - output will be captured at END of benchmark_fn()
        self._verify_input = self.microbatches[0].clone()
        self._verify_output = torch.empty(
            self._verify_input.shape[0],
            10,
            device=self.device,
            dtype=torch.float32,
        )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # Warm up: run a few iterations so kernel autotuning/caches are populated
        # before the harness starts timing (and to amortize compile overhead if enabled).
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
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
        """Function to benchmark."""
        assert (
            self._microbatch_groups is not None
            and self._target_groups is not None
            and self._group_sizes is not None
            and self._training_groups is not None
        )
        with self._nvtx_range("baseline_performance"):
            for paired_group, group_size in self._training_groups:
                self.optimizer.zero_grad(set_to_none=True)
                for data, target in paired_group:
                    logits = self.model(data)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    (loss / group_size).backward()
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
        """Cleanup."""
        del self.model, self.microbatches, self.targets, self.optimizer
        self._microbatch_groups = None
        self._target_groups = None
        self._group_sizes = None
        self._training_groups = None
        self._verify_input = None
        self._verify_output = None
        restore_tf32_state(self._tf32_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=5,
            warmup=10,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return real runtime environment metrics for this benchmark host."""
        return get_environment_custom_metrics()

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if not self.microbatches:
            return "Data not initialized"
        # Use the first microbatch as a validation probe so we avoid allocating
        # an additional tensor up front.
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output shape mismatch: expected batch_size={probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselinePerformanceBenchmark()
