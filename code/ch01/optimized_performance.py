"""optimized_performance.py - FP16 + fused-microbatch performance benchmark."""

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
        context="ch01.optimized_performance import",
    )

from typing import Optional

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.verification_mixin import VerificationPayloadMixin
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


def _warn_cuda_probe_failure(message: str) -> None:
    print(f"WARNING: {message}")


class OptimizedPerformanceBatchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Combine FP16 tensor-core math with fused microbatch execution."""

    allow_cpu = True
    signature_equivalence_group = "ch01_performance_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.device = get_usable_cuda_or_cpu(warning_handler=_warn_cuda_probe_failure)
        self.workload = WORKLOAD
        self.batch_size = batch_size if batch_size != 32 else self.workload.microbatch_size
        self.model = None
        self.microbatches = None
        self.targets = None
        self.optimizer = None
        self.fusion = 8
        self.hidden_dim = self.workload.performance_hidden_dim
        self._verify_input = None
        self._verify_model_input = None
        self._verify_output = None
        self.parameter_count = 0
        self._model_dtype = torch.float32
        self._fused_pairs = None
        self._tf32_state: tuple[bool, bool | None] | None = None
        samples = float(self.batch_size * self.workload.performance_microbatches)
        self.register_workload_metadata(samples_per_iteration=samples)
    
    def setup(self) -> None:
        """Setup: initialize model, fixed inputs, and verification output."""
        seed_chapter1()
        self._tf32_state = capture_tf32_state()
        set_tf32_state(False)
        
        self.model = build_training_mlp(self.hidden_dim)
        if self.device.type == "cuda":
            self.model = self.model.half()
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.model = self.model.to(self.device)
        self._model_dtype = dtype
        
        # Match baseline: use eval() mode (baseline has this even though it does backward pass)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        microbatches = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=dtype).contiguous()
            for _ in range(self.workload.performance_microbatches)
        ]
        targets = [
            torch.randint(0, 10, (self.batch_size,), device=self.device)
            for _ in range(self.workload.performance_microbatches)
        ]
        self.microbatches = microbatches
        self.targets = targets
        
        # Keep verification inputs in FP32 so the signature stays stable across variants.
        self._verify_input = self.microbatches[0].float().clone()
        self._verify_model_input = self._verify_input.to(dtype=dtype, device=self.device)
        self._verify_output = torch.empty(self._verify_input.shape[0], 10, device=self.device, dtype=torch.float32)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # Warm up compiled model so the measurement loop only sees steady-state cost.
        for _ in range(3):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(self.microbatches[0])
            loss = torch.nn.functional.cross_entropy(logits, self.targets[0])
            loss.backward()
            self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        # Pre-build fused batches so the benchmark loop can issue fewer, larger kernels.
        self._fused_batches, self._fused_targets = preallocate_fused_microbatches(
            self.microbatches,
            self.targets,
            self.fusion,
        )
        self._fused_pairs = tuple(zip(self._fused_batches, self._fused_targets, strict=True))
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        assert self._fused_pairs is not None
        with self._nvtx_range("optimized_performance"):
            for data, target in self._fused_pairs:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
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
                "bf16": self._model_dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32),
            },
            output_tolerance=(0.5, 0.5),
        )

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.microbatches, self.targets, self.optimizer
        self._fused_batches = None
        self._fused_targets = None
        self._fused_pairs = None
        self._verify_input = None
        self._verify_model_input = None
        self._verify_output = None
        self._model_dtype = torch.float32
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
        # Use first microbatch as validation probe
        probe = self.microbatches[0]
        try:
            with torch.no_grad():
                test_output = self.model(probe)
                if test_output.shape[0] != probe.shape[0]:
                    return f"Output batch size mismatch: expected {probe.shape[0]}, got {test_output.shape[0]}"
                if test_output.shape[1] != 10:
                    return f"Output shape mismatch: expected num_classes=10, got {test_output.shape[1]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedPerformanceBatchBenchmark(batch_size=32)
