"""labs.moe_cuda/baseline_router.py - Top-k MoE router baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from core.harness.benchmark_harness import WorkloadMetadata
from labs.moe_cuda.optimized_router_vectorized import VectorizedTopKMoE

class BaselineRouterDenseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline top-k MoE router using per-token weight gathers."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.top_k),
        )
        self._enable_nvtx = False
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda requires CUDA for fair comparison")

        import gc
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state to prevent graph capture errors
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass
        
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        
        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        model = VectorizedTopKMoE(self.hidden_size, self.num_experts, self.top_k, expansion=2)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()
        self._payload_parameter_count = sum(p.numel() for p in model.parameters())
        self.model = model

        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        ).to(self.device)
        torch.cuda.synchronize(self.device)
        self.output = None
        self._verify_output_buffer = torch.empty(
            (self.batch_size, self.hidden_size),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        with nvtx_range("moe_cuda_dense_router", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self.model is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.inputs.detach()},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.inputs = None
        self.output = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=10)  # Match compiled variants warmup

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "router.estimated_flops": flops,
            "router.estimated_bytes": bytes_moved,
            "router.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Top-k MoE model missing"
        if self.inputs is None:
            return "Inputs missing"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineRouterDenseBenchmark()
