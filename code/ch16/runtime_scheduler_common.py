"""Shared helpers for async scheduling + stream interval benchmarks (Ch16).

This models vLLM-style runtime improvements:
- Async scheduling overlaps CPU batch prep with GPU compute.
- Stream interval batches token responses to reduce CPU overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class SchedulerScenario:
    name: str
    concurrency: int
    decode_steps: int
    tokens_per_step: int
    stream_interval: int
    matmul_dim: int


class RuntimeSchedulerWorkload:
    """Pre-allocates CPU/GPU buffers to simulate serving workloads deterministically."""

    def __init__(self, device: torch.device, scenarios: Tuple[SchedulerScenario, ...]) -> None:
        self.device = device
        self.scenarios = scenarios

        # CPU prep buffers (simulate prepare_batch).
        self.cpu_a = torch.randn(512, 512)
        self.cpu_b = torch.randn(512, 512)
        self._cpu_product = torch.empty_like(self.cpu_a)

        # CPU send buffers (simulate network serialization).
        self.send_buffer = torch.randn(4096)
        self.send_scale = torch.randn(4096)
        self._send_product = torch.empty_like(self.send_buffer)
        self._send_scalar = torch.empty((), dtype=self.send_buffer.dtype)

        # GPU matmul buffers per scenario.
        self.gpu_buffers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.gpu_outputs: Dict[str, torch.Tensor] = {}
        for scenario in scenarios:
            dim = scenario.matmul_dim
            a = torch.randn(dim, dim, device=self.device, dtype=torch.bfloat16)
            b = torch.randn(dim, dim, device=self.device, dtype=torch.bfloat16)
            self.gpu_buffers[scenario.name] = (a, b)
            self.gpu_outputs[scenario.name] = torch.empty(
                dim,
                dim,
                device=self.device,
                dtype=torch.bfloat16,
            )

    def cpu_prepare(self) -> torch.Tensor:
        """Simulate host-side batch preparation cost."""
        # Matrix multiply uses real CPU work (no sleeps).
        torch.matmul(self.cpu_a, self.cpu_b, out=self._cpu_product)
        return self._cpu_product

    def gpu_compute(self, scenario: SchedulerScenario) -> torch.Tensor:
        """Simulate GPU compute for a decode step."""
        a, b = self.gpu_buffers[scenario.name]
        out = self.gpu_outputs[scenario.name]
        torch.matmul(a, b, out=out)
        return out

    def stream_send(self, tokens: int) -> float:
        """Simulate response serialization cost with fixed overhead + per-token work."""
        # Fixed overhead per send call.
        torch.mul(self.send_buffer, self.send_scale, out=self._send_product)
        torch.sum(self._send_product, dim=0, out=self._send_scalar)
        # Per-token cost scales with tokens.
        chunk = self.send_buffer[: max(1, min(tokens, self.send_buffer.numel()))]
        torch.sum(chunk, dim=0, out=self._send_scalar)
        return float(self._send_scalar)
