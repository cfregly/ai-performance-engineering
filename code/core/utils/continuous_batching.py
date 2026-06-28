"""Shared continuous batching helpers for single- and multi-GPU benchmarks."""

from __future__ import annotations

import random
from typing import List, Optional

import torch
import torch.nn as nn

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class ContinuousBatchingBase(BaseBenchmark):
    """Shared continuous batching benchmark logic.

    Subclasses should mix in VerificationPayloadMixin and provide the label
    plus dynamic/multi-GPU flags.
    """

    def __init__(
        self,
        *,
        dynamic: bool,
        multi_gpu: bool,
        label: str,
        max_batch_size: int = 12,
        hidden_dim: int = 1024,
        num_batches: int = 12,
        max_decode_steps: int = 32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dynamic = bool(dynamic)
        self.multi_gpu = bool(multi_gpu)
        self.label = label
        self.max_batch_size = int(max_batch_size)
        self.hidden_dim = int(hidden_dim)
        self.num_batches = int(num_batches)
        self.max_decode_steps = int(max_decode_steps)
        self.dtype = dtype

        self.device_ids: List[int] = []
        self.models: List[nn.Module] = []
        self.samples: List[torch.Tensor] = []
        self.state_buffers: List[torch.Tensor] = []
        self.dynamic_batch_buffers: List[torch.Tensor] = []
        self.lengths: List[List[int]] = []
        self.lengths_tensor: List[torch.Tensor] = []
        self.group_indices: List[List[torch.Tensor]] = []
        self.group_ranges: List[List[tuple[int, int]]] = []
        self.group_max_steps: List[List[int]] = []
        self.group_active_masks: List[List[torch.Tensor]] = []
        self.schedules: List[List[torch.Tensor]] = []
        self.outputs: List[torch.Tensor] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.streams: List[torch.cuda.Stream] = []

        self.num_samples_per_device = self.max_batch_size * self.num_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.num_samples_per_device),
            tokens_per_iteration=0.0,
        )

    @staticmethod
    def _build_dynamic_schedule(
        lengths: List[int],
        max_batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Precompute dynamic schedule indices for continuous batching."""
        remaining = lengths.copy()
        active: List[int] = list(range(min(max_batch_size, len(lengths))))
        next_idx = len(active)
        schedule: List[torch.Tensor] = []

        while active:
            schedule.append(torch.tensor(active, device=device, dtype=torch.int64))
            new_active: List[int] = []
            for req_idx in active:
                remaining[req_idx] -= 1
                if remaining[req_idx] > 0:
                    new_active.append(req_idx)
            active = new_active
            while len(active) < max_batch_size and next_idx < len(lengths):
                active.append(next_idx)
                next_idx += 1
        return schedule

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for continuous batching")
        if self.multi_gpu and torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if self.multi_gpu:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            device_index = 0 if self.device.index is None else int(self.device.index)
            self.device_ids = [device_index]

        self.models = []
        self.samples = []
        self.state_buffers = []
        self.dynamic_batch_buffers = []
        self.lengths = []
        self.lengths_tensor = []
        self.group_indices = []
        self.group_ranges = []
        self.group_max_steps = []
        self.group_active_masks = []
        self.schedules = []
        self.outputs = []
        self.streams = []
        total_tokens = 0

        for device_id in self.device_ids:
            device = torch.device(f"cuda:{device_id}")
            model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            ).to(device=device, dtype=self.dtype).eval()
            self.models.append(model)

            samples = torch.randn(
                self.num_samples_per_device,
                self.hidden_dim,
                device=device,
                dtype=self.dtype,
            )
            self.samples.append(samples)
            self.state_buffers.append(torch.empty_like(samples))
            self.dynamic_batch_buffers.append(
                torch.empty(
                    self.max_batch_size,
                    self.hidden_dim,
                    device=device,
                    dtype=self.dtype,
                )
            )

            rng = random.Random(123 + device_id)
            lengths = [
                rng.randint(1, self.max_decode_steps)
                for _ in range(self.num_samples_per_device)
            ]
            total_tokens += int(sum(lengths))
            self.lengths.append(lengths)
            lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.int32)
            self.lengths_tensor.append(lengths_tensor)

            if self.dynamic:
                self.schedules.append(
                    self._build_dynamic_schedule(lengths, self.max_batch_size, device)
                )
                self.group_indices.append([])
                self.group_ranges.append([])
                self.group_max_steps.append([])
                self.group_active_masks.append([])
            else:
                group_ranges = [
                    (i * self.max_batch_size, (i + 1) * self.max_batch_size)
                    for i in range(self.num_batches)
                ]
                group_max_steps = [max(lengths[start:end]) for start, end in group_ranges]
                group_active_masks = []
                for (start, end), group_max in zip(group_ranges, group_max_steps):
                    group_lengths = lengths_tensor[start:end].view(1, -1, 1)
                    group_steps = torch.arange(
                        group_max,
                        device=device,
                        dtype=lengths_tensor.dtype,
                    ).view(-1, 1, 1)
                    group_active_masks.append(group_steps < group_lengths)
                groups = [
                    torch.arange(
                        i * self.max_batch_size,
                        (i + 1) * self.max_batch_size,
                        device=device,
                        dtype=torch.int64,
                    )
                    for i in range(self.num_batches)
                ]
                self.group_indices.append(groups)
                self.group_ranges.append(group_ranges)
                self.group_max_steps.append(group_max_steps)
                self.group_active_masks.append(group_active_masks)
                self.schedules.append([])

        self.outputs = list(self.state_buffers)

        self._verify_input = self.samples[0][:2].detach()
        total_requests = self.num_samples_per_device * len(self.device_ids)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(total_tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(total_requests),
            tokens_per_iteration=float(total_tokens),
        )

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self.label):
            with torch.inference_mode():
                for idx, device_id in enumerate(self.device_ids):
                    model = self.models[idx]
                    samples = self.samples[idx]
                    state = self.state_buffers[idx]
                    dynamic_batch_buffer = self.dynamic_batch_buffers[idx]
                    schedule = self.schedules[idx]
                    group_ranges = self.group_ranges[idx]
                    group_max_steps = self.group_max_steps[idx]
                    group_active_masks = self.group_active_masks[idx]
                    with torch.cuda.device(device_id):
                        state.copy_(samples)
                        if self.dynamic:
                            for active_idx in schedule:
                                batch_state = dynamic_batch_buffer[: active_idx.numel()]
                                torch.index_select(state, 0, active_idx, out=batch_state)
                                y = model(batch_state)
                                state.index_copy_(0, active_idx, y)
                        else:
                            for (group_start, group_end), group_max, active_masks in zip(
                                group_ranges,
                                group_max_steps,
                                group_active_masks,
                            ):
                                group_state = state[group_start:group_end]
                                for step in range(group_max):
                                    y = model(group_state)
                                    torch.where(
                                        active_masks[step],
                                        y,
                                        group_state,
                                        out=group_state,
                                    )

        self.output = self.state_buffers[0]

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None:
            raise RuntimeError(
                "setup() and benchmark_fn() must be called before "
                "capture_verification_payload()"
            )
        param_count = sum(p.numel() for p in self.models[0].parameters())
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self.output,
            batch_size=int(self.num_samples_per_device),
            parameter_count=param_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": (
                    torch.backends.cuda.matmul.allow_tf32
                    if torch.cuda.is_available()
                    else False
                ),
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "world_size": len(self.device_ids),
            },
        )

    def teardown(self) -> None:
        self.models = []
        self.samples = []
        self.state_buffers = []
        self.dynamic_batch_buffers = []
        self.lengths = []
        self.lengths_tensor = []
        self.group_indices = []
        self.group_ranges = []
        self.group_max_steps = []
        self.group_active_masks = []
        self.schedules = []
        self.outputs = []
        self.streams = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5, multi_gpu_required=self.multi_gpu)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_streams(self) -> List[torch.cuda.Stream]:
        return []
