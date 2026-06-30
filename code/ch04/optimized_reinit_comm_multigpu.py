"""optimized_reinit_comm_multigpu.py - Initialize NCCL once and reuse (optimized)."""

from __future__ import annotations

import os

from core.common.device_utils import resolve_local_rank

import torch
import torch.distributed as dist

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus

from typing import Optional

from ch04.distributed_helper import setup_single_gpu_env
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

class OptimizedReinitCommBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Initialize NCCL once and reuse - good pattern."""
    multi_gpu_required = True
    
    def __init__(self):
        super().__init__()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.input_tensor = None
        self.tensor = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.initialized = False
        self._enable_nvtx = False
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=4.0,  # single float all-reduce
        )
    
    def setup(self) -> None:
        """Setup: Initialize NCCL once."""
        skip_if_insufficient_gpus()
        setup_single_gpu_env()
        self.local_rank = resolve_local_rank()
        torch.cuda.set_device(self.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                init_method="env://",
                device_id=self.local_rank,
            )
            self.initialized = True
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Intentionally tiny payload: this benchmark isolates communicator reuse vs reinit,
        # not bandwidth. Larger tensors would dilute the init/destroy cost.
        self.input_tensor = torch.randn(1, 1, device=self.device, dtype=torch.float32)
        self.tensor = torch.empty_like(self.input_tensor)
        self._verify_output_buffer = torch.empty_like(self.tensor)
        torch.cuda.synchronize(self.device)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.register_workload_metadata(
            requests_per_iteration=float(self._workload.requests_per_iteration or 1.0),
            bytes_per_iteration=float(self._workload.bytes_per_iteration or 0.0),
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Reuse existing NCCL communicator."""
        if self.tensor is None or self.input_tensor is None:
            raise RuntimeError("Tensor not initialized")

        with nvtx_range("reinit_comm", enable=self._enable_nvtx):
            # Good pattern: reuse existing NCCL communicator
            self.tensor.copy_(self.input_tensor)
            dist.all_reduce(self.tensor)

    def capture_verification_payload(self) -> None:
        if (
            self.input_tensor is None
            or self.tensor is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.tensor)
        self._set_verification_payload(
            inputs={"input": self.input_tensor},
            output=self._verify_output_buffer,
            batch_size=int(self.input_tensor.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-6, 1e-6),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if dist.is_initialized() and self.initialized:
            dist.destroy_process_group()
        self.input_tensor = None
        self.tensor = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            multi_gpu_required=True,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.tensor is None:
            return "Tensor not initialized"
        if not dist.is_initialized():
            return "Distributed process group not initialized"
        # Check tensor shape and values
        if self.tensor.shape != (1, 1):
            return f"Tensor shape mismatch: expected (1, 1), got {self.tensor.shape}"
        if not torch.isfinite(self.tensor).all():
            return "Tensor contains non-finite values"
        return None
    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-6, 1e-6)


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedReinitCommBenchmark()
