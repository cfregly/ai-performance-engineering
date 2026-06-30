"""optimized_memory.py - Optimized GPU memory management."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

BATCH_SIZE = 1024
INPUT_DIM = 2048
HIDDEN_DIM = 2048
REPETITIONS = 10


class BufferedGeluMlp(nn.Module):
    """Three-layer GELU MLP with reusable inference buffers."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None
        self._fc3_buffer: Optional[torch.Tensor] = None
        self._fc1_weight_t: Optional[torch.Tensor] = None
        self._fc2_weight_t: Optional[torch.Tensor] = None
        self._fc3_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._fc1_weight_t = self.fc1.weight.t()
        self._fc2_weight_t = self.fc2.weight.t()
        self._fc3_weight_t = self.fc3.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, width = (int(shape[0]), int(shape[1]))
        numel = rows * width
        buffer = getattr(self, name)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, buffer)
        return buffer[:numel].view(rows, width)

    def _ensure_hidden_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        fc1_shape = (rows, self.fc1.out_features)
        fc2_shape = (rows, self.fc2.out_features)
        fc1_out = self._workspace("_fc1_buffer", fc1_shape, device=x.device, dtype=x.dtype)
        fc2_out = self._workspace("_fc2_buffer", fc2_shape, device=x.device, dtype=x.dtype)
        return fc1_out, fc2_out

    def _ensure_output_buffer(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (x.shape[0], self.fc3.out_features)
        return self._workspace("_fc3_buffer", out_shape, device=x.device, dtype=x.dtype)

    def forward_into(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self._fc1_weight_t is None or self._fc2_weight_t is None or self._fc3_weight_t is None:
            self.cache_weight_views()
        fc1_out, fc2_out = self._ensure_hidden_buffers(x)
        torch.mm(x, self._fc1_weight_t, out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        F.gelu(fc1_out, out=fc1_out)

        torch.mm(fc1_out, self._fc2_weight_t, out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        F.gelu(fc2_out, out=fc2_out)

        torch.mm(fc2_out, self._fc3_weight_t, out=out)
        if self.fc3.bias is not None:
            out.add_(self.fc3.bias)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = F.gelu(self.fc1(x))
            x = F.gelu(self.fc2(x))
            return self.fc3(x)
        return self.forward_into(x, self._ensure_output_buffer(x))


class OptimizedMemoryBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: GPU memory management with capture and buffer reuse."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size = BATCH_SIZE
        self.input_dim = INPUT_DIM
        self.device_buffer: Optional[torch.Tensor] = None
        self.transform_buffer: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.repetitions = REPETITIONS
        self._repetition_range = range(self.repetitions)
        tokens = self.batch_size * self.input_dim * self.repetitions
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.repetitions),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = BufferedGeluMlp(self.input_dim, HIDDEN_DIM).to(self.device, dtype=torch.float32).eval()
        self.model.cache_weight_views()
        
        self.device_buffer = torch.empty(
            self.batch_size,
            self.input_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.transform_buffer = torch.empty_like(self.device_buffer)
        self.graph_output = torch.empty_like(self.device_buffer)
        self._verify_output_buffer = torch.empty_like(self.device_buffer)
        self._repetition_range = range(self.repetitions)
        self._synchronize()

        with torch.inference_mode():
            _ = self.model(self.device_buffer)
        self._synchronize()

        self.graph = torch.cuda.CUDAGraph()
        # Keep the float buffer on the same discrete 0..255 population as the
        # baseline's uint8 staging path. random_(low, high) already emits
        # integer-valued floats, so no extra floor kernel is needed.
        self.device_buffer.random_(0, 256)
        self._synchronize()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.transform_buffer.copy_(self.device_buffer)
            self.transform_buffer.mul_(1.0 / 255.0)
            self.transform_buffer.add_(-0.5)
            self.transform_buffer.mul_(2.0)
            self.transform_buffer.tanh_()
            self.model.forward_into(self.transform_buffer, self.graph_output)
        self._synchronize()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
    
    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.device_buffer is None
            or self.graph is None
            or self.graph_output is None
        ):
            raise RuntimeError("Optimized memory benchmark not initialized")

        with self._nvtx_range("optimized_memory"):
            with torch.inference_mode():
                for _ in self._repetition_range:
                    # Make the discrete input population explicit on every replay.
                    self.device_buffer.random_(0, 256)
                    self.graph.replay()
                self.output = self.graph_output
        if self.output is None or self.device_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.device_buffer is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.device_buffer},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.device_buffer = None
        self.transform_buffer = None
        self.graph_output = None
        self._verify_output_buffer = None
        self.graph = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedMemoryBenchmark:
    return OptimizedMemoryBenchmark()
