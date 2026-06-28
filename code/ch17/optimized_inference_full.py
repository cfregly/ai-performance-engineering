"""optimized_inference_full.py - Early-exit comparison inference benchmark.

This workload is a comparison pair for model-side work reduction. It is not the
chapter's native disaggregated prefill/decode optimization example.
"""

from __future__ import annotations

from typing import Optional

import random

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class FullDepthModel(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_layers: int = 24):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu_(layer(x))
        return self.head(x)


class OptimizedInferenceFullBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Comparison pair that short-circuits identity layers.

    This is useful as an end-to-end inference comparison benchmark, but it is not
    the chapter's main disaggregated prefill/decode example. The optimization
    here is layer skipping, not disaggregation.
    """

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "optimized",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "This pair measures model-side work reduction; the chapter-native "
            "disaggregated serving story lives in the prefill_decode_disagg targets."
        ),
        "comparison_axis": "full_depth_vs_early_exit",
        "execution_pattern": "early_exit_inference",
        "optimization_mechanism": "skip the identity tail layers while preserving outputs",
        "chapter_native_targets": [
            "prefill_decode_disagg_ttft",
            "prefill_decode_disagg_overlap",
            "prefill_decode_disagg_batched",
            "prefill_decode_disagg_tpot_long",
        ],
    }

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self._early_exit_layers: list[nn.Module] = []
        self.inputs: Optional[torch.Tensor] = None
        self.batch_size = 16
        self.hidden_dim = 2048
        self.num_layers = 24
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.identity_start_layer = 6
        self.exit_layer = 6
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)

        self.model = FullDepthModel(self.hidden_dim, self.num_layers).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        with torch.inference_mode():
            dtype = next(self.model.parameters()).dtype
            eye = torch.eye(self.hidden_dim, device=self.device, dtype=dtype)
            for layer in self.model.layers[self.identity_start_layer :]:
                layer.weight.copy_(eye)
                layer.bias.zero_()

        input_dtype = next(self.model.parameters()).dtype
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=input_dtype)
        self._early_exit_layers = list(self.model.layers[: self.exit_layer])

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.inputs is not None

        with self._nvtx_range("inference_full_comparison_early_exit"):
            with torch.inference_mode():
                x = self.inputs
                for layer in self._early_exit_layers:
                    x = torch.relu_(layer(x))
                self.output = self.model.head(x)
        if self.output is None or self.inputs is None:
            raise RuntimeError("benchmark_fn() must produce output")
        dtype = self.output.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        self._set_verification_payload(
            inputs={"input": self.inputs},
            output=self.output,
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
        self.model = None
        self._early_exit_layers = []
        self.inputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics plus explicit comparison-pair work shape."""
        from core.benchmark.metrics import compute_inference_metrics

        metrics = compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, "total_tokens", 256),
            total_requests=getattr(self, "total_requests", 1),
            batch_size=getattr(self, "batch_size", 1),
            max_batch_size=getattr(self, "max_batch_size", 32),
        )
        metrics.update(
            {
                "configured_layers": float(self.num_layers),
                "active_layers": float(self.exit_layer),
                "identity_tail_layers": float(self.num_layers - self.identity_start_layer),
                "identity_layers_skipped": float(self.num_layers - self.exit_layer),
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> OptimizedInferenceFullBenchmark:
    return OptimizedInferenceFullBenchmark()
