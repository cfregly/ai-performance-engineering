"""Sequential microbatch pipeline baseline without overlap."""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

resolve_device = partial(require_cuda_device, "CUDA required for ch20")


class SimpleStage(nn.Module):
    """Heavier pipeline stage to highlight overlap benefits."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffn(x)
        return self.norm(out + x)


class BaselinePipelineSequentialBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sequential pipeline - no overlap."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.stages = None
        self.inputs = None
        self.output = None
        self.microbatches: Optional[list[torch.Tensor]] = None
        self._last_outputs: Optional[list[torch.Tensor]] = None
        self._last_output_count: int = 0
        self.batch_size = 512
        self.hidden_dim = 1536
        self.num_stages = 4
        self.repeats = 6
        self.num_microbatches = 8
        self.register_workload_metadata(requests_per_iteration=float(self.batch_size))
        self._payload_parameter_count = 0
        self._enable_nvtx = False
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        """Describe workload units processed per iteration."""
        return WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size),
            samples_per_iteration=float(self.batch_size),
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=getattr(self, '_last_elapsed_ms', None),
            ai_optimized_time_ms=None,
            suggestions_applied=None,
            suggestions_total=None,
        )

    def setup(self) -> None:
        """Setup: Initialize pipeline stages."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Sequential pipeline stages
        self.stages = nn.ModuleList([
            SimpleStage(self.hidden_dim).to(self.device).half()
            for _ in range(self.num_stages)
        ]).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.stages.parameters())
        
        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.microbatches = [chunk.contiguous() for chunk in self.inputs.chunk(self.num_microbatches, dim=0)]
        self._last_outputs = [
            torch.empty(0, device=self.device, dtype=torch.float16)
            for _ in range(self.num_microbatches)
        ]
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
    
    def _run_pipeline_once(self, microbatches: list[torch.Tensor]) -> list[torch.Tensor]:
        assert self.stages is not None and self._last_outputs is not None
        output_count = 0
        for microbatch in microbatches:
            x = microbatch
            for stage in self.stages:
                x = stage(x)
            self._last_outputs[output_count] = x
            output_count += 1
        self._last_output_count = output_count
        return self._last_outputs

    def benchmark_fn(self) -> None:
        """Benchmark the GPU-native sequential microbatch pipeline."""
        assert self.inputs is not None and self.stages is not None and self.microbatches is not None

        with nvtx_range("baseline_pipeline_sequential", enable=self._enable_nvtx):
            with torch.inference_mode():
                for _ in range(self.repeats):
                    self._run_pipeline_once(self.microbatches)

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self._last_outputs is None or self.stages is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if self._last_output_count != len(self._last_outputs):
            raise RuntimeError("Incomplete pipeline outputs before verification capture")
        self.output = torch.cat(self._last_outputs, dim=0).detach()
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output.float(),
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Cleanup."""
        self.stages = None
        self.inputs = None
        self.output = None
        self.microbatches = None
        self._last_outputs = None
        self._last_output_count = 0
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.stages is None:
            return "Stages not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselinePipelineSequentialBenchmark()
