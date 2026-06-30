"""baseline_dataparallel_multigpu.py - DataParallel baseline (anti-pattern)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


from typing import Optional

from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class SimpleNet(nn.Module):
    """Lightweight network to expose DataParallel + H2D overhead."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BaselineDataParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """DataParallel baseline - has GIL overhead and single-threaded execution."""
    multi_gpu_required = True
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self._gpu_data: Optional[torch.Tensor] = None
        self._gpu_target: Optional[torch.Tensor] = None
        self.device_ids: list[int] = []
        self._last_input: Optional[torch.Tensor] = None
        self._last_target: Optional[torch.Tensor] = None
        self._verify_state: Optional[dict] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_target: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        # Large input to make per-iteration H2D copies dominate.
        self.input_size = 4096
        self.batch_size = 4096
        self.output: Optional[torch.Tensor] = None
        # Training benchmarks don't support jitter check
        tokens = self.batch_size * self.input_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize model, optimizer, and data."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Keep input tensors on CPU so DataParallel copies every iteration.
        model = SimpleNet(self.input_size).to(self.device)
        self._payload_parameter_count = sum(p.numel() for p in model.parameters())
        # Use all visible GPUs for DataParallel scatter/gather.
        self.device_ids = list(range(torch.cuda.device_count()))
        self.model = nn.DataParallel(model, device_ids=self.device_ids, output_device=self.device_ids[0])
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self._verify_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        data_gen = torch.Generator().manual_seed(1234)
        self.data = torch.randn(self.batch_size, self.input_size, generator=data_gen)
        self.target = torch.randn(self.batch_size, 1, generator=data_gen)
        self._gpu_data = torch.empty(self.batch_size, self.input_size, device=self.device, dtype=torch.float32)
        self._gpu_target = torch.empty(self.batch_size, 1, device=self.device, dtype=torch.float32)
        self._verify_input = self.data.clone()
        self._verify_target = self.target.clone()

        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: DataParallel training step."""
        assert self.data is not None and self.target is not None
        assert self._gpu_data is not None and self._gpu_target is not None
        with self._nvtx_range("dataparallel"):
            gpu_data = self._gpu_data
            gpu_target = self._gpu_target
            gpu_data.copy_(self.data, non_blocking=False)
            gpu_target.copy_(self.target, non_blocking=False)
            self._last_input = gpu_data
            self._last_target = gpu_target
            output = self.model(gpu_data)
            loss = nn.functional.mse_loss(output, gpu_target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.output = output.detach_()

    def capture_verification_payload(self) -> None:
        if (
            self._verify_input is None
            or self._verify_target is None
            or self._verify_state is None
            or self.output is None
        ):
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        verify_model = SimpleNet(self.input_size).to(self.device)
        verify_model.load_state_dict(self._verify_state)
        verify_model.eval()
        with torch.no_grad():
            verify_input = self._verify_input.to(self.device)
            verify_target = self._verify_target.to(self.device)
            output = verify_model(verify_input)
        self._set_verification_payload(
            inputs={"data": verify_input, "target": verify_target},
            output=output,
            batch_size=int(self.batch_size),
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
        """Teardown: Clean up resources."""
        self.model = None
        self.optimizer = None
        self.data = None
        self.target = None
        self._gpu_data = None
        self._gpu_target = None
        self._verify_state = None
        self._verify_input = None
        self._verify_target = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
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
        if self.model is None:
            return "Model not initialized"
        if self.data is None:
            return "Data tensor not initialized"
        if self.target is None:
            return "Target tensor not initialized"
        # Check that model can produce valid output
        try:
            with torch.no_grad():
                test_output = self.model(self.data)
                if test_output.shape[0] != self.batch_size:
                    return f"Output batch size mismatch: expected {self.batch_size}, got {test_output.shape[0]}"
        except Exception as e:
            return f"Model forward pass failed: {e}"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineDataParallelBenchmark()
