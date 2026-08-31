"""Small reusable buffers for dynamic-router verification payloads."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Number

import torch

VERIFICATION_OUTPUT_KEY = "_verification_output_token_ids"


def numeric_metric_values(metrics: Mapping[str, object], out: list[float] | None = None) -> list[float]:
    """Build the verification row, preferring explicit model outputs when present."""
    values = [] if out is None else out
    values.clear()
    verification_output = metrics.get(VERIFICATION_OUTPUT_KEY)
    if verification_output is not None:
        if not isinstance(verification_output, list | tuple):
            raise TypeError(f"{VERIFICATION_OUTPUT_KEY} must be a list or tuple of token ids")
        for value in verification_output:
            if isinstance(value, bool) or not isinstance(value, Number):
                raise TypeError(f"{VERIFICATION_OUTPUT_KEY} must contain only numeric token ids")
            values.append(float(value))
    else:
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
        owner._metric_output_buffer = buffer
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
