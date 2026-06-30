"""Optimized AI pipeline with staged storage reads and overlapped device copies."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ch05.ai_common import BufferedTinyBlock, TinyBlock, compute_ai_workload_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

__all__ = ["OptimizedAIBenchmark", "TinyBlock", "get_benchmark"]


class OptimizedAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefetches storage-backed activations through pinned buffers for inference."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self):
        super().__init__()
        self.block: Optional[nn.Module] = None
        self.inputs_path: Optional[str] = None
        self.mapped_inputs: Optional[np.memmap] = None
        self.output: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.host_buffers: list[torch.Tensor] = []
        self.device_buffers: list[torch.Tensor] = []
        self.host_views: list[np.ndarray] = []
        self.batch = 64
        self.hidden = 32
        self.num_blocks = 256
        self._block_range = range(self.num_blocks)
        self.parameter_count = 0
        tokens = self.batch * self.hidden * self.num_blocks
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.block = BufferedTinyBlock(self.hidden).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.block.parameters())

        host_batches = np.random.default_rng(42).standard_normal(
            (self.num_blocks, self.batch, self.hidden),
            dtype=np.float32,
        )
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.inputs_path = f.name
        f.close()
        np.save(self.inputs_path, host_batches)
        self.mapped_inputs = np.load(self.inputs_path, mmap_mode="r")
        self._block_range = range(self.num_blocks)

        pin_memory = torch.cuda.is_available()
        self.host_buffers = [
            torch.empty((self.batch, self.hidden), device="cpu", dtype=torch.float32, pin_memory=pin_memory)
            for _ in range(2)
        ]
        self.host_views = [buffer.numpy() for buffer in self.host_buffers]
        self.device_buffers = [
            torch.empty((self.batch, self.hidden), device=self.device, dtype=torch.float32)
            for _ in range(2)
        ]
        if self.device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device=self.device)
        self._synchronize()

    def _stage_to_host(self, slot: int, step: int) -> None:
        assert self.mapped_inputs is not None
        np.copyto(self.host_views[slot], self.mapped_inputs[step])

    def _enqueue_copy(self, slot: int, wait_stream: Optional[torch.cuda.Stream] = None) -> None:
        if self.copy_stream is None:
            self.device_buffers[slot].copy_(self.host_buffers[slot], non_blocking=False)
            return
        producer_stream = wait_stream or torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(producer_stream)
            self.device_buffers[slot].copy_(self.host_buffers[slot], non_blocking=True)

    def _wait_for_copy(self, wait_stream: Optional[torch.cuda.Stream] = None) -> None:
        if self.copy_stream is not None:
            consumer_stream = wait_stream or torch.cuda.current_stream()
            consumer_stream.wait_stream(self.copy_stream)

    def benchmark_fn(self) -> None:
        assert self.block is not None and self.mapped_inputs is not None
        current_stream = torch.cuda.current_stream() if self.copy_stream is not None else None
        self._stage_to_host(0, 0)
        self._enqueue_copy(0, wait_stream=current_stream)

        out: Optional[torch.Tensor] = None
        last_input: Optional[torch.Tensor] = None
        with self._nvtx_range("optimized_ai_storage_pipeline"):
            with torch.inference_mode():
                for step in self._block_range:
                    current = step & 1
                    next_slot = current ^ 1
                    self._wait_for_copy(current_stream)
                    current_input = self.device_buffers[current]
                    if step + 1 < self.num_blocks:
                        self._stage_to_host(next_slot, step + 1)
                        self._enqueue_copy(next_slot, wait_stream=current_stream)
                    out = self.block(current_input)
                    last_input = current_input
        if out is None or last_input is None:
            raise RuntimeError("benchmark_fn() must produce output")
        self._last_input = last_input
        self.output = out

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self._last_input},
            output=self.output,
            batch_size=self.batch,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.block = None
        self.mapped_inputs = None
        self.output = None
        self._last_input = None
        self.copy_stream = None
        self.host_buffers = []
        self.device_buffers = []
        self.host_views = []
        if self.inputs_path and os.path.exists(self.inputs_path):
            os.unlink(self.inputs_path)
        self.inputs_path = None
        self.parameter_count = 0
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return compute_ai_workload_metrics(
            batch_size=self.batch,
            hidden_dim=self.hidden,
            num_blocks=self.num_blocks,
            parameter_count=self.parameter_count,
        )

    def validate_result(self) -> Optional[str]:
        if self.block is None or self.mapped_inputs is None:
            return "Model or storage inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAIBenchmark()
