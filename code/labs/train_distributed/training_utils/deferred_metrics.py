"""Retain sampled training losses without synchronizing every progress update."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingProgress:
    step: int
    loss: float
    tokens: int


class DeferredTrainingProgress:
    """Copy detached losses on device; transfer recorded samples once at the end.

    Construct before the measured loop and read after its final synchronization.
    Report end-to-end process time separately from the training-loop timer.
    """

    def __init__(self, *, num_steps: int, interval: int, device: torch.device) -> None:
        if num_steps < 1 or interval < 1:
            raise ValueError("num_steps and interval must be positive")
        self.interval = interval
        self.num_steps = num_steps
        self._losses = torch.empty(
            (num_steps + interval - 1) // interval, device=device, dtype=torch.float64
        )
        self._samples: list[tuple[int, int]] = []

    def record(self, *, step: int, loss: torch.Tensor, tokens: int) -> None:
        if step != len(self._samples) * self.interval or step >= self.num_steps:
            raise ValueError("Progress samples must follow the declared step interval")
        if loss.numel() != 1 or loss.device != self._losses.device:
            raise ValueError("Loss must be a scalar on the progress buffer device")
        with torch.no_grad():
            self._losses[len(self._samples)].copy_(loss.detach().reshape(()))
        self._samples.append((step, tokens))

    def read(self) -> list[TrainingProgress]:
        values = self._losses[: len(self._samples)].detach().cpu().tolist()
        return [
            TrainingProgress(step=step, loss=value, tokens=tokens)
            for (step, tokens), value in zip(self._samples, values, strict=True)
        ]
