"""Small reusable buffers for dynamic-router verification payloads."""

from __future__ import annotations

from numbers import Number
from typing import Mapping

import torch


def numeric_metric_values(metrics: Mapping[str, object], out: list[float] | None = None) -> list[float]:
    values = [] if out is None else out
    values.clear()
    for value in metrics.values():
        if isinstance(value, Number):
            values.append(float(value))
    if not values:
        values.append(0.0)
    return values


def metric_row_buffer(owner: object, metric_values: list[float]) -> torch.Tensor:
    width = len(metric_values)
    buffer = getattr(owner, "_metric_output_buffer", None)
    if buffer is None or buffer.shape[1] < width:
        buffer = torch.empty((1, width), dtype=torch.float32)
        setattr(owner, "_metric_output_buffer", buffer)
    for index, value in enumerate(metric_values):
        buffer[0, index] = float(value)
    return buffer[:, :width].detach()


def scalar_int_buffer(owner: object, attr_name: str, value: int) -> torch.Tensor:
    buffer = getattr(owner, attr_name, None)
    if buffer is None:
        buffer = torch.empty(1, dtype=torch.int64)
        setattr(owner, attr_name, buffer)
    buffer[0] = int(value)
    return buffer
