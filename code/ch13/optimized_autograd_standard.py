"""optimized_autograd_standard.py - Compiled autograd optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import copy

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SimpleModel(nn.Module):
    """Simple model for autograd comparison."""
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedAutogradCompiledBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Autograd accelerated with CUDA graphs to remove launch overhead."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        # Smaller batch to increase launch overhead share and highlight graph capture.
        self.batch_size = 16
        self.hidden_dim = 1024
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup training step, capture it with CUDA graphs."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        self.inputs = torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
        self.targets = torch.randn_like(self.inputs)
        self.output_buffer = torch.empty_like(self.inputs)
        self._verify_output_buffer = torch.empty_like(self.inputs, dtype=torch.float32)

        saved_model_state = copy.deepcopy(self.model.state_dict())
        saved_opt_state = copy.deepcopy(self.optimizer.state_dict())

        # Warm up a full training step to allocate grads/optimizer buffers needed for capture,
        # then restore state so the first timed iteration matches the baseline.
        for _ in range(3):
            self._train_step(self.inputs, self.targets)
        self._synchronize()
        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)

        self.static_input = self.inputs.clone()
        self.static_target = self.targets.clone()
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(self.capture_stream):
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._train_step(self.static_input, self.static_target, capture_output=True)
        self.capture_stream.synchronize()
        # Restore model/optimizer to post-setup state so capture does not advance training
        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - compiled autograd."""
        if self.graph is None or self.static_input is None or self.static_target is None:
            raise RuntimeError("CUDA graph not initialized")
        if self.capture_stream is None:
            raise RuntimeError("Capture stream not initialized")
        if self.inputs is None or self.targets is None:
            raise RuntimeError("Benchmark inputs not initialized")
        if self.output_buffer is None:
            raise RuntimeError("Output buffer not initialized")

        with self._nvtx_range("autograd_standard"):
            with torch.cuda.stream(self.capture_stream):
                self.static_input.copy_(self.inputs)
                self.static_target.copy_(self.targets)
                self.graph.replay()
            self.output = self.output_buffer
        if self.inputs is None or self.targets is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.inputs, "targets": self.targets},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def _train_step(self, batch: torch.Tensor, target: torch.Tensor, capture_output: bool = False) -> None:
        assert self.model is not None and self.optimizer is not None and self.criterion is not None
        # CUDA graphs do not replay Python-side `set_to_none=True` state changes; use
        # explicit zeroing so gradient reset is captured and replayed.
        self.optimizer.zero_grad(set_to_none=False)
        outputs = self.model(batch)
        if capture_output and self.output_buffer is not None:
            self.output_buffer.copy_(outputs)
        loss = self.criterion(outputs, target)
        loss.backward()
        self.optimizer.step()

    def teardown(self) -> None:
        """Cleanup."""
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.capture_stream = None
        self.output_buffer = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        if self.capture_stream is None:
            return []
        return [self.capture_stream]

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedAutogradCompiledBenchmark:
    """Factory function for harness discovery."""
    return OptimizedAutogradCompiledBenchmark()
