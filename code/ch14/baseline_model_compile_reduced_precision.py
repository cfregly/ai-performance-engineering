"""baseline_model_compile_reduced_precision.py - Eager reduced-precision baseline."""

from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from ch14.model_eager_common import (  # noqa: E402
    MODEL_EAGER_WARMUP_ITERS,
    SimpleTransformer,
    model_compile_custom_metrics,
    resolve_model_eager_dtype,
)


class BaselineModelCompileReducedPrecisionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline for the reduced-precision eager-vs-compiled chapter pair.

    This path keeps the same reduced-precision model and workload as the
    optimized sample, but executes it eagerly without compilation.
    """

    signature_equivalence_group = "ch14_model_compile_reduced_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        # Increase work so compile path has enough compute to amortize overhead.
        self.batch_size = 24
        self.seq_len = 1536
        self.vocab_size = 10000
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self.dtype = torch.float16
        self._verification_payload = None
        self._enable_nvtx = False
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.dtype = resolve_model_eager_dtype()
        self.model = SimpleTransformer().to(self.device, dtype=self.dtype).eval()
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Warm eager execution to the same reduced-precision steady state used by the compiled path.
        for _ in range(MODEL_EAGER_WARMUP_ITERS):
            with torch.inference_mode():
                _ = self.model(self.input_ids)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        with nvtx_range("model_compile_reduced_precision_baseline", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.input_ids)
        if self.output is None or self.input_ids is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        return model_compile_custom_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            parameter_count=self.parameter_count,
            dtype=self.dtype,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            compiled=False,
        )

    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineModelCompileReducedPrecisionBenchmark()
