"""optimized_model_compile_reduced_precision.py - Compiled reduced-precision execution."""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import compile_model
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from ch14.model_eager_common import (  # noqa: E402
    MODEL_EAGER_COMPILE_WARMUP_ITERS,
    MODEL_EAGER_WARMUP_ITERS,
    SimpleTransformer,
    model_compile_custom_metrics,
    resolve_model_eager_dtype,
)


class OptimizedModelCompileReducedPrecisionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation for compiled reduced-precision model execution."""

    signature_equivalence_group = "ch14_model_compile_reduced_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.compiled_model = None
        self.input_ids = None
        # Increase work so torch.compile overhead is amortized and speedups are clearer.
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
        """Setup: initialize model and compile it."""

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        self.dtype = resolve_model_eager_dtype()
        model = SimpleTransformer().to(self.device, dtype=self.dtype).eval()
        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        # Use max-autotune for best performance (searches through kernel configs)
        self.compiled_model = compile_model(
            model,
            mode="max-autotune",  # Better than reduce-overhead for sustained workloads
            fullgraph=False,
            dynamic=False,
        )
        
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Extensive warmup for compilation and autotuning
        for _ in range(MODEL_EAGER_COMPILE_WARMUP_ITERS):
            with torch.inference_mode():
                _ = self.compiled_model(self.input_ids)
        torch.cuda.synchronize(self.device)
        
        # Additional warmup after compilation
        for _ in range(MODEL_EAGER_WARMUP_ITERS):
            with torch.inference_mode():
                _ = self.compiled_model(self.input_ids)
        torch.cuda.synchronize()
        self._payload_dtype = self.dtype
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        with nvtx_range("model_compile_reduced_precision_optimized", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.compiled_model(self.input_ids)
        if self.output is None or self.input_ids is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.compiled_model, self.input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            setup_timeout_seconds=1200,  # torch.compile compilation can be lengthy
            measurement_timeout_seconds=1200,
            use_subprocess=True,
        )
    def get_custom_metrics(self) -> Optional[dict]:
        return model_compile_custom_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            parameter_count=self.parameter_count,
            dtype=self.dtype,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            compiled=True,
        )

    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedModelCompileReducedPrecisionBenchmark()
