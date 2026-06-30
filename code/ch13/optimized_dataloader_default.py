"""optimized_dataloader_default.py - Tuned DataLoader optimization (optimized)."""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class SyntheticDataset(Dataset):
    """Synthetic dataset for DataLoader benchmarking."""
    
    def __init__(self, num_samples: int = 1000, feature_dim: int = 1024, *, preprocess_steps: int = 8):
        if preprocess_steps < 1:
            raise ValueError("preprocess_steps must be >= 1")
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.preprocess_steps = preprocess_steps
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
        self._scratch_buffer: Optional[torch.Tensor] = None

    def _scratch_like(self, sample: torch.Tensor) -> torch.Tensor:
        scratch = self._scratch_buffer
        if (
            scratch is None
            or scratch.shape != sample.shape
            or scratch.device != sample.device
            or scratch.dtype != sample.dtype
        ):
            scratch = torch.empty_like(sample)
            self._scratch_buffer = scratch
        return scratch
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        enriched = sample.clone()
        scratch = self._scratch_like(sample)
        for _ in range(self.preprocess_steps):
            torch.mul(enriched, 0.5, out=scratch)
            torch.sin(scratch, out=scratch)
            enriched.mul_(1.1)
            torch.tanh(enriched, out=enriched)
            enriched.add_(scratch)
        enriched.sub_(enriched.mean())
        return enriched, self.labels[idx]


class SimpleModel(nn.Module):
    """Simple model for training demonstration."""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedDataloaderTunedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized DataLoader - tuned for performance."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataloader: Optional[DataLoader] = None
        self.optimizer = None
        self.criterion = None
        # Heavier preprocessing and a larger resident dataset amplify the
        # benefit of pinned workers, prefetch, and persistent worker reuse.
        self.dataset_size = 4000
        self.batch_size = 64
        self.feature_dim = 1024
        self.preprocess_steps = 16
        self._data_iter: Optional[Iterator] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.feature_dim),
        )
        self.output = None
        self._payload_inputs: dict[str, Optional[torch.Tensor]] = {"data": None, "labels": None}
        self._payload_input_buffers: dict[str, Optional[torch.Tensor]] = {"data": None, "labels": None}
        self._verification_payload_inputs: dict[str, Optional[torch.Tensor]] = {"data": None, "labels": None}
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_inputs_ready = False
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.feature_dim),
        )
    
    def setup(self) -> None:
        # DataLoader workers default to torch.set_num_threads(1) to avoid CPU
        # oversubscription. Set the same in the main process so baseline and
        # optimized variants see identical preprocessing math.
        torch.set_num_threads(1)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.model = SimpleModel(input_dim=self.feature_dim).to(self.device)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        dataset = SyntheticDataset(
            num_samples=self.dataset_size,
            feature_dim=self.feature_dim,
            preprocess_steps=self.preprocess_steps,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        self._data_iter = iter(self.dataloader)
        self._payload_inputs_ready = False
        self._payload_input_buffers["data"] = torch.empty(
            self.batch_size,
            self.feature_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._payload_input_buffers["labels"] = torch.empty(
            self.batch_size,
            device=self.device,
            dtype=torch.long,
        )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            10,
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(2):
            data, labels = self._next_batch()
            _ = self.model(data)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.optimizer is None
            or self.criterion is None
            or self._data_iter is None
        ):
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("optimized_dataloader_default"):
            data, labels = self._next_batch()
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.output = outputs.detach_()
        self._payload_inputs["data"] = data
        self._payload_inputs["labels"] = labels
        self._payload_inputs_ready = True
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if not self._payload_inputs_ready or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        data = self._payload_inputs["data"]
        labels = self._payload_inputs["labels"]
        data_buffer = self._payload_input_buffers["data"]
        labels_buffer = self._payload_input_buffers["labels"]
        if data is None or labels is None or data_buffer is None or labels_buffer is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must stash inputs for verification")
        payload_data = data_buffer[: data.shape[0]]
        payload_labels = labels_buffer[: labels.shape[0]]
        payload_output = self._verify_output_buffer[: self.output.shape[0]]
        payload_data.copy_(data)
        payload_labels.copy_(labels)
        payload_output.copy_(self.output)
        inputs = self._verification_payload_inputs
        inputs["data"] = payload_data
        inputs["labels"] = payload_labels
        self._set_verification_payload(
            inputs=inputs,
            output=payload_output,
            batch_size=int(payload_data.shape[0]),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
        )

    def teardown(self) -> None:
        self.model = None
        self.dataloader = None
        self.optimizer = None
        self.criterion = None
        self._data_iter = None
        self._payload_inputs["data"] = None
        self._payload_inputs["labels"] = None
        self._payload_input_buffers["data"] = None
        self._payload_input_buffers["labels"] = None
        self._verification_payload_inputs["data"] = None
        self._verification_payload_inputs["labels"] = None
        self._payload_inputs_ready = False
        self.output = None
        self._verify_output_buffer = None
        super().teardown()
    
    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.dataloader is not None
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            data, labels = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            data, labels = next(self._data_iter)
        return data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            adaptive_iterations=False,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp32",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> OptimizedDataloaderTunedBenchmark:
    """Factory function for harness discovery."""
    return OptimizedDataloaderTunedBenchmark()
