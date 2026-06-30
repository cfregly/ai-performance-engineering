"""Double-buffered batch provisioning baseline with blocking data loading.

This benchmark demonstrates inefficient data provisioning with pageable host
batches and blocking H2D copies each iteration. Device staging buffers are
reused so the comparison focuses on transfer behavior instead of allocator
noise. The optimized version pins and prefetches data.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


class BaselineDoubleBufferedBatchProvisioningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Loads each batch with blocking host orchestration and H2D copies."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.host_batches: List[torch.Tensor] = []
        self.target_batches: List[torch.Tensor] = []
        self.device_batch: Optional[torch.Tensor] = None
        self.device_target: Optional[torch.Tensor] = None
        self.batch_idx = 0
        self._batch_count = 0
        self.output: Optional[torch.Tensor] = None
        self._model_parameters: tuple[nn.Parameter, ...] = ()
        self._payload_parameter_count = 0
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        # Two float32 batches per step: inputs + targets (512x1024 elements each)
        elements = 2 * 512 * 1024
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(elements),
            bytes_per_iteration=float(elements * 4),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        ).to(self.device)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self._model_parameters = tuple(self.model.parameters())
        
        # Pre-allocate host batches for deterministic verification
        num_batches = 4
        for _ in range(num_batches):
            self.host_batches.append(torch.randn(512, 1024, dtype=torch.float32))
            self.target_batches.append(torch.randn(512, 1024, dtype=torch.float32))
        self._batch_count = num_batches
        self.device_batch = torch.empty(512, 1024, device=self.device, dtype=torch.float32)
        self.device_target = torch.empty(512, 1024, device=self.device, dtype=torch.float32)
        self.batch_idx = 0
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Training step with blocking data transfer."""
        assert self.model is not None and self.device_batch is not None and self.device_target is not None
        
        # Get next batch (round-robin)
        idx = self.batch_idx % self._batch_count
        self.batch_idx += 1
        
        with self._nvtx_range("baseline_double_buffered_batch_provisioning"):
            # Blocking H2D copy (the slow part), using reusable device staging.
            data = self.device_batch
            target = self.device_target
            data.copy_(self.host_batches[idx], non_blocking=False)
            target.copy_(self.target_batches[idx], non_blocking=False)
            
            # Forward
            out = self.model(data)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            
            # Clear gradients (simulate optimizer step)
            for p in self._model_parameters:
                p.grad = None
        
        self.output = out.detach_()
        self._payload_data = data
        self._payload_target = target

    def capture_verification_payload(self) -> None:
        data = self._payload_data
        target = self._payload_target
        self._set_verification_payload(
            inputs={"data": data, "target": target},
            output=self.output,
            batch_size=data.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.model = None
        self.host_batches = []
        self.target_batches = []
        self.device_batch = None
        self.device_target = None
        self.output = None
        self._batch_count = 0
        self._model_parameters = ()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        # Disable adaptive iterations: this benchmark cycles through batches,
        # so baseline/optimized must execute the same iteration count for
        # post-timing output verification.
        return BenchmarkConfig(iterations=30, warmup=5, adaptive_iterations=False)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineDoubleBufferedBatchProvisioningBenchmark()
