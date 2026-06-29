"""baseline_regional_compilation.py - Eager whole-model baseline for regional compilation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.utils import warn_benchmark_scaling  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402

MODEL_CANDIDATES: List[Dict[str, Any]] = [
    {"label": "20B (48x7168)", "n_layers": 48, "d_model": 7168, "d_ff": 28672, "seq_len": 2048},
    {"label": "15B (36x6400)", "n_layers": 36, "d_model": 6400, "d_ff": 25600, "seq_len": 2048},
    {"label": "11B (32x5632)", "n_layers": 32, "d_model": 5632, "d_ff": 22528, "seq_len": 1536},
    {"label": "8B (24x5120)", "n_layers": 24, "d_model": 5120, "d_ff": 20480, "seq_len": 1536},
    {"label": "6B (24x4096)", "n_layers": 24, "d_model": 4096, "d_ff": 16384, "seq_len": 1024},
    {"label": "3B (16x3072)", "n_layers": 16, "d_model": 3072, "d_ff": 12288, "seq_len": 1024},
    {"label": "2B (8x2048)", "n_layers": 8, "d_model": 2048, "d_ff": 8192, "seq_len": 1024},
    {"label": "1B (4x1024)", "n_layers": 4, "d_model": 1024, "d_ff": 4096, "seq_len": 768},
]
REGIONAL_COMPILATION_CHOICE_LABEL = "3B (16x3072)"


def select_regional_compilation_choice(
    label: str = REGIONAL_COMPILATION_CHOICE_LABEL,
) -> Dict[str, Any]:
    for candidate in MODEL_CANDIDATES:
        if candidate["label"] == label:
            return candidate
    raise RuntimeError(f"Missing model config for regional compilation: {label}")


class DummyTransformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, d_ff: int):
        super().__init__()
        self.output = None
        self._verify_input = None
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class BaselineRegionalCompilationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Eager whole-model baseline for the regional compilation comparison."""

    def __init__(self):
        super().__init__()
        self.model: Optional[DummyTransformer] = None
        self.inputs: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        # Use a mid-sized config so CUDA-graph replay overhead is amortized.
        self.choice = select_regional_compilation_choice()
        tokens = self.choice["seq_len"] * self.choice["d_model"]
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._enable_nvtx = False

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        n_layers = self.choice["n_layers"]
        d_model = self.choice["d_model"]
        d_ff = self.choice["d_ff"]
        self.model = DummyTransformer(n_layers, d_model, d_ff).to(self.device, dtype=torch.bfloat16).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(
            1,
            self.choice["seq_len"],
            d_model,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        warn_benchmark_scaling(
            scaling_type="Model size",
            original_values={
                "label": self.choice["label"],
                "n_layers": self.choice["n_layers"],
                "d_model": self.choice["d_model"],
                "d_ff": self.choice["d_ff"],
                "seq_len": self.choice["seq_len"],
            },
            scaled_values={
                "label": self.choice["label"],
                "n_layers": self.choice["n_layers"],
                "d_model": self.choice["d_model"],
                "d_ff": self.choice["d_ff"],
                "seq_len": self.choice["seq_len"],
            },
            impact_description="Scaled-down model sizes run on smaller GPUs; speedups are representative but conservative.",
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model/inputs not initialized")
        with nvtx_range("baseline_regional_compilation", enable=self._enable_nvtx):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.output = self.model(self.inputs)
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=10,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=240,
            use_subprocess=True,
            adaptive_iterations=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        metrics = compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )
        metrics.update(
            {
                "regional_compilation.capture_ms": 0.0,
                "regional_compilation.graph_bucket_count": 0.0,
                "regional_compilation.steady_state_only": 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineRegionalCompilationBenchmark()
