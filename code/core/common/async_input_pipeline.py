"""Shared helpers for async input pipeline benchmarks and sweeps."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


@dataclass
class PipelineConfig:
    """Configuration knobs for the async input pipeline."""

    batch_size: int = 16
    feature_shape: Tuple[int, int, int] = (3, 64, 64)
    dataset_size: int = 64
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = False
    non_blocking: bool = False
    use_copy_stream: bool = False


class _SyntheticImageDataset(Dataset):
    """Pre-generated CPU dataset to avoid per-sample allocation overhead."""

    def __init__(self, length: int, feature_shape: Tuple[int, int, int]):
        self.length = length
        self.data = torch.randn((length, *feature_shape), dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _build_dataloader(cfg: PipelineConfig) -> Iterable[torch.Tensor]:
    """Construct a DataLoader with the requested overlap knobs."""

    dataset = _SyntheticImageDataset(cfg.dataset_size, cfg.feature_shape)
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": True,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    return DataLoader(dataset, **kwargs)


class AsyncInputPipelineBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark that measures H2D overlap for a simple vision pipeline."""

    def __init__(self, cfg: Optional[PipelineConfig] = None, label: str = "async_input_pipeline"):
        super().__init__()
        self.cfg = cfg or PipelineConfig()
        self.label = label

        self.loader_iter: Optional[Iterable[torch.Tensor]] = None
        self.loader: Optional[DataLoader] = None
        self.model: Optional[nn.Module] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self._prefetched_batch: Optional[torch.Tensor] = None
        self._prefetch_event: Optional[torch.cuda.Event] = None
        self._prefetch_events: List[torch.cuda.Event] = []
        self._prefetch_event_index: int = 0
        self._last_batch: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._parameter_count: int = 0
        self.register_workload_metadata(samples_per_iteration=self.cfg.batch_size)

    def _next_batch_cpu(self) -> torch.Tensor:
        if self.loader_iter is None:
            raise RuntimeError("Loader iterator not initialized")
        try:
            return next(self.loader_iter)
        except StopIteration:
            if self.loader is None:
                raise RuntimeError("Loader not initialized")
            self.loader_iter = iter(self.loader)
            return next(self.loader_iter)

    def _launch_h2d(self, batch_cpu: torch.Tensor) -> torch.Tensor:
        label = "transfer_async:h2d" if (self.copy_stream is not None or self.cfg.non_blocking) else "transfer_sync:h2d"
        if self.copy_stream is None:
            with self._nvtx_range(label):
                return batch_cpu.to(self.device, non_blocking=self.cfg.non_blocking)
        if not self._prefetch_events:
            self._prefetch_events = [
                torch.cuda.Event(enable_timing=False, blocking=False),
                torch.cuda.Event(enable_timing=False, blocking=False),
            ]
            self._prefetch_event_index = 0
        event = self._prefetch_events[self._prefetch_event_index]
        self._prefetch_event_index = (self._prefetch_event_index + 1) % len(self._prefetch_events)
        with torch.cuda.stream(self.copy_stream), self._nvtx_range(label):
            batch_gpu = batch_cpu.to(self.device, non_blocking=self.cfg.non_blocking)
            event.record(self.copy_stream)
            self._prefetch_event = event
        return batch_gpu

    def setup(self) -> None:
        # The harness controls RNG seeding, including fresh-input verification.
        # This is a performance benchmark; allow cuDNN to pick fastest kernels.
        torch.backends.cudnn.benchmark = True

        self.loader = _build_dataloader(self.cfg)
        self.loader_iter = iter(self.loader)
        # Use a small ConvNet so the benchmark is compute+transfer limited (not
        # dominated by Python/DataLoader overhead) and overlap is measurable.
        c, h, w = self.cfg.feature_shape
        if c != 3:
            raise ValueError("feature_shape[0] must be 3 (RGB) for this benchmark")
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
        ).to(self.device)
        self._parameter_count = sum(p.numel() for p in self.model.parameters())

        self.copy_stream = torch.cuda.Stream() if self.cfg.use_copy_stream else None
        self.compute_stream = torch.cuda.current_stream()

        self.register_workload_metadata(samples_per_iteration=self.cfg.batch_size)
        # Lightweight pre-warm to amortize first-call overhead.
        warm = torch.randn((1, *self.cfg.feature_shape), device=self.device)
        with torch.no_grad():
            _ = self.model(warm)
        torch.cuda.synchronize(self.device)

        # Prime the pipeline with a prefetched batch so steady-state overlap is
        # measured during the timed iterations.
        first_cpu = self._next_batch_cpu()
        self._prefetched_batch = self._launch_h2d(first_cpu)

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        if self._prefetched_batch is None:
            raise RuntimeError("Prefetch buffer not initialized (setup() must run)")
        with self._nvtx_range(self.label):
            # The harness may execute on a stream different from setup().
            self.compute_stream = torch.cuda.current_stream(self.device)
            # Current batch is the one prefetched during the previous iteration.
            if self._prefetch_event is not None and self.compute_stream is not None:
                self.compute_stream.wait_event(self._prefetch_event)
            batch_gpu = self._prefetched_batch
            if self.copy_stream is not None:
                # Waiting orders execution; record_stream separately protects the
                # allocation if Python drops it while compute is still queued.
                batch_gpu.record_stream(self.compute_stream)

            # Kick off H2D for the *next* batch on the copy stream while we run compute.
            next_cpu = self._next_batch_cpu()
            next_gpu = self._launch_h2d(next_cpu)

            self._last_batch = batch_gpu.detach()
            with torch.no_grad():
                out = self.model(batch_gpu)
            self.output = out.detach()
            self._prefetched_batch = next_gpu
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._last_batch is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        torch.cuda.synchronize(self.device)
        self._set_verification_payload(
            inputs={"batch": self._last_batch},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=self._parameter_count,
            precision_flags={"fp16": False, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.loader_iter = None
        self.loader = None
        self.model = None
        self.copy_stream = None
        self.compute_stream = None
        self._prefetched_batch = None
        self._prefetch_event = None
        self._prefetch_events = []
        self._prefetch_event_index = 0
        self._last_batch = None
        self.output = None
        self._parameter_count = 0
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Need multiple iterations so the copy/compute overlap reaches steady state.
        return BenchmarkConfig(
            iterations=20,
            warmup=8,
            adaptive_iterations=False,  # Stateful DataLoader iterator must run fixed counts for verification
            timeout_seconds=120,
            measurement_timeout_seconds=120,
            use_subprocess=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return async input pipeline configuration metrics."""
        bytes_per_sample = 4 * self.cfg.feature_shape[0] * self.cfg.feature_shape[1] * self.cfg.feature_shape[2]
        bytes_per_batch = bytes_per_sample * self.cfg.batch_size
        return {
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_workers": float(self.cfg.num_workers),
            f"{self.label}.pin_memory": float(self.cfg.pin_memory),
            f"{self.label}.non_blocking": float(self.cfg.non_blocking),
            f"{self.label}.use_copy_stream": float(self.cfg.use_copy_stream),
            f"{self.label}.bytes_per_batch": float(bytes_per_batch),
        }
