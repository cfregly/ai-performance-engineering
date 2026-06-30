"""Double-buffered batch provisioning optimization with overlapped copies.

This benchmark demonstrates efficient data provisioning - using pinned memory,
prefetching, and async H2D copies to overlap data loading with compute.
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


class OptimizedDoubleBufferedBatchProvisioningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefetches device batches on a side stream for overlapped data loading."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.host_batches: List[torch.Tensor] = []
        self.target_batches: List[torch.Tensor] = []
        self.device_batches: List[torch.Tensor] = []
        self.device_targets: List[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.copy_events: List[torch.cuda.Event] = []
        self.cur_slot = 0
        self.next_slot = 1
        self.batch_idx = 0
        self._batch_count = 0
        self.output: Optional[torch.Tensor] = None
        self._model_parameters: tuple[nn.Parameter, ...] = ()
        self._payload_parameter_count = 0
        # Training benchmarks don't support jitter check - outputs change due to weight updates
        
        elements = 2 * 512 * 1024
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(elements),
            bytes_per_iteration=float(elements * 4),
        )

    def _prefetch_slot(
        self,
        slot: int,
        wait_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Async copy to device buffer on copy stream."""
        batch_idx = (self.batch_idx + 1) % self._batch_count
        producer_stream = wait_stream or torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer_stream)
            self.device_batches[slot].copy_(self.host_batches[batch_idx], non_blocking=True)
            self.device_targets[slot].copy_(self.target_batches[batch_idx], non_blocking=True)
            self.copy_events[slot].record(self.copy_stream)

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Same model as baseline for fair comparison
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        ).to(self.device)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self._model_parameters = tuple(self.model.parameters())
        
        # Pre-allocate host batches with pinned memory (the optimization)
        num_batches = 4
        for _ in range(num_batches):
            self.host_batches.append(torch.randn(512, 1024, dtype=torch.float32, pin_memory=True))
            self.target_batches.append(torch.randn(512, 1024, dtype=torch.float32, pin_memory=True))
        self._batch_count = num_batches
        
        # Device double-buffers for prefetching
        self.device_batches = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
        ]
        self.device_targets = [
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
            torch.empty(512, 1024, device=self.device, dtype=torch.float32),
        ]
        self.copy_events = [torch.cuda.Event() for _ in self.device_batches]
        
        self.batch_idx = 0
        self.cur_slot = 0
        self.next_slot = 1
        
        # Prefetch first (current) batch into slot 0.
        with torch.cuda.stream(self.copy_stream):
            self.device_batches[self.cur_slot].copy_(self.host_batches[0], non_blocking=True)
            self.device_targets[self.cur_slot].copy_(self.target_batches[0], non_blocking=True)
            self.copy_events[self.cur_slot].record(self.copy_stream)
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(self.copy_events[self.cur_slot])

        # Start prefetching the *next* batch into slot 1 (batch 1).
        self._prefetch_slot(self.next_slot, wait_stream=current_stream)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Training step with overlapped data loading."""
        assert self.model is not None
        
        # Wait for current batch to be ready
        current_stream = torch.cuda.current_stream()
        current_stream.wait_event(self.copy_events[self.cur_slot])
        data = self.device_batches[self.cur_slot]
        target = self.device_targets[self.cur_slot]
        
        with self._nvtx_range("optimized_double_buffered_batch_provisioning"):
            # Forward (overlapped with next batch prefetch)
            out = self.model(data)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            
            # Clear gradients (simulate optimizer step)
            for p in self._model_parameters:
                p.grad = None
        
        self.output = out.detach_()
        
        # Swap slots and start next prefetch
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self.batch_idx += 1
        self._prefetch_slot(self.next_slot, wait_stream=current_stream)
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
        self.device_batches = []
        self.device_targets = []
        self.copy_events = []
        self._batch_count = 0
        self.output = None
        self._model_parameters = ()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        # Disable adaptive iterations: this benchmark cycles through batches,
        # so baseline/optimized must execute the same iteration count for
        # post-timing output verification.
        return BenchmarkConfig(iterations=30, warmup=5, adaptive_iterations=False)

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        return [self.copy_stream]

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if not self.device_batches:
            return "Device batches not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDoubleBufferedBatchProvisioningBenchmark()
