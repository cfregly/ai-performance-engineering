"""optimized_storage_cpu.py - Storage read with mmap + pinned host staging."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedStorageCpuBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Read the same on-disk tensor while reusing pinned host and device buffers."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.filepath: Optional[str] = None
        self.host_buffer: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._host_buffer_view: Optional[np.ndarray] = None
        self._mapped_array: Optional[np.ndarray] = None
        self.size_mb = 64  # Smaller for faster benchmark
        self.size = self.size_mb * 1024 * 1024 // 4  # float32 elements
        bytes_per_iter = self.size * 4
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: materialize a tensor once and prepare reusable staging buffers."""
        torch.manual_seed(42)
        host_template = np.random.default_rng(42).standard_normal(self.size, dtype=np.float32)
        
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.filepath = f.name
        f.close()
        np.save(self.filepath, host_template)
        self._mapped_array = np.load(self.filepath, mmap_mode="r")
        self.host_buffer = torch.empty(self.size, device="cpu", dtype=torch.float32, pin_memory=True)
        self._host_buffer_view = self.host_buffer.numpy()
        self.device_buffer = torch.empty(self.size, device=self.device, dtype=torch.float32)
        self._output_buffer = torch.empty(1, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: storage read with memory mapping and reusable buffers."""
        assert self.filepath is not None
        assert self.host_buffer is not None
        assert self._host_buffer_view is not None
        assert self._mapped_array is not None
        assert self.device_buffer is not None
        assert self._output_buffer is not None
        with torch.inference_mode(), self._nvtx_range("storage_cpu_optimized"):
            np.copyto(self._host_buffer_view, self._mapped_array)
            self.device_buffer.copy_(self.host_buffer, non_blocking=True)
            self.data = self.device_buffer
            torch.sum(self.device_buffer, dim=0, keepdim=True, out=self._output_buffer)
        self.output = self._output_buffer

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self._verify_output_buffer,
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self._mapped_array = None
        if self.filepath and os.path.exists(self.filepath):
            os.unlink(self.filepath)
        self.data = None
        self.filepath = None
        self.host_buffer = None
        self.device_buffer = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self._host_buffer_view = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the CPU-staged mmap plus pinned-buffer data path."""
        from ch05.metrics_common import compute_storage_path_metrics

        bytes_per_tensor = self.size * 4
        return compute_storage_path_metrics(
            bytes_read=bytes_per_tensor,
            bytes_written=0,
            file_count=1,
            uses_cpu_staging=True,
            simulates_gpu_direct=False,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.size:
            return f"Data size mismatch: expected {self.size}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStorageCpuBenchmark()
