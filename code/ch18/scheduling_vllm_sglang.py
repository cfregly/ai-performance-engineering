"""scheduling_vllm_sglang.py - Continuous batching + speculative decode toy."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class SchedulingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Toy scheduler that batches requests and accepts speculative drafts."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: Deque[int] = deque()
        self._workload = WorkloadMetadata(requests_per_iteration=1.0, tokens_per_iteration=64.0)
        self._history: Dict[str, float] = {}
        self.request_lengths: list[int] = []
        self.output: Optional[torch.Tensor] = None
        self._output_values: Optional[list[float]] = None
        self._enable_nvtx = False

    def setup(self) -> None:
        random.seed(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.queue.clear()
        self.request_lengths = [random.randint(4, 32) for _ in range(8)]
        self.output = None

    def _enqueue_requests(self) -> None:
        for tokens in self.request_lengths:
            self.queue.append(tokens)

    def _serve_batch(self, batch_tokens: int) -> int:
        # Simulate speculative accept ratio.
        accepted = int(batch_tokens * 0.8)
        return accepted

    def benchmark_fn(self) -> Optional[dict]:
        with nvtx_range("scheduling_vllm_sglang", enable=self._enable_nvtx):
            if not self.queue:
                self._enqueue_requests()
            batch_tokens = 0
            served = 0
            while self.queue and batch_tokens < 64:
                tokens = self.queue.popleft()
                batch_tokens += tokens
                served += self._serve_batch(tokens)
        self._history["served_tokens"] = served
        self._history["batched_tokens"] = batch_tokens
        self._output_values = [float(served), float(batch_tokens)]
        return {"served_tokens": served, "batched_tokens": batch_tokens}

    def capture_verification_payload(self) -> None:
        if self._output_values is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self.output = torch.tensor(self._output_values, dtype=torch.float32)
        self._set_verification_payload(
            inputs={
                "request_lengths": torch.tensor(self.request_lengths, dtype=torch.int64),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.numel() != 2:
            return "Unexpected output shape"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return scheduling performance metrics."""
        if not self._history:
            return None
        return {
            "scheduling.served_tokens": self._history.get("served_tokens", 0.0),
            "scheduling.speculative_accept_ratio": 0.8,  # Fixed in _serve_batch
        }


def get_benchmark() -> BaseBenchmark:
    return SchedulingBenchmark()
