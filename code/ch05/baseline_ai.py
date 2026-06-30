"""Baseline AI pipeline with blocking storage reads and CPU-mediated transfers."""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ch05.ai_common import TinyBlock, compute_ai_workload_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineAIBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Reads activations from storage synchronously before every tiny inference block."""

    allowed_benchmark_fn_antipatterns = ("host_transfer", "sync")

    def __init__(self):
        super().__init__()
        self.block: Optional[nn.Module] = None
        self.inputs_path: Optional[str] = None
        self.output: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self._device_batch_buffer: Optional[torch.Tensor] = None
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
        self.block = TinyBlock(self.hidden).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.block.parameters())

        host_batches = np.random.default_rng(42).standard_normal(
            (self.num_blocks, self.batch, self.hidden),
            dtype=np.float32,
        )
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.inputs_path = f.name
        f.close()
        np.save(self.inputs_path, host_batches)
        self._block_range = range(self.num_blocks)
        self._device_batch_buffer = torch.empty(
            (self.batch, self.hidden),
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert (
            self.block is not None
            and self.inputs_path is not None
            and self._device_batch_buffer is not None
        )
        host_batches = np.load(self.inputs_path)
        host_tensor = torch.from_numpy(host_batches)
        out: Optional[torch.Tensor] = None
        last_input: Optional[torch.Tensor] = None
        device_batch = self._device_batch_buffer
        with self._nvtx_range("baseline_ai_storage_pipeline"):
            with torch.inference_mode():
                for step in self._block_range:
                    device_batch.copy_(host_tensor[step], non_blocking=False)
                    out = self.block(device_batch)
                    last_input = device_batch
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
        self.output = None
        self._last_input = None
        self._device_batch_buffer = None
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
        if self.block is None or self.inputs_path is None:
            return "Model or storage inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineAIBenchmark()
