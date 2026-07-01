"""Shared helpers for optimized torchcomms benchmarks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BufferedTorchcommsBlock(nn.Module):
    """Two-layer GELU block with reusable inference buffers."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None
        self._fc1_forward_view: Optional[torch.Tensor] = None
        self._fc2_forward_view: Optional[torch.Tensor] = None
        self._fc1_weight_t: Optional[torch.Tensor] = None
        self._fc2_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._fc1_weight_t = self.fc1.weight.t()
        self._fc2_weight_t = self.fc2.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows, width = (int(shape[0]), int(shape[1]))
        numel = rows * width
        buffer = getattr(self, name)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, buffer)
        return buffer[:numel].view(rows, width)

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.shape[0]
        fc1_shape = (rows, self.fc1.out_features)
        fc2_shape = (rows, self.fc2.out_features)
        fc1_out = self._workspace("_fc1_buffer", fc1_shape, device=x.device, dtype=x.dtype)
        fc2_out = self._workspace("_fc2_buffer", fc2_shape, device=x.device, dtype=x.dtype)
        return fc1_out, fc2_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = F.gelu(self.fc1(x))
            return self.fc2(x)

        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def prepare_forward_buffers(self, x: torch.Tensor) -> None:
        self._fc1_forward_view, self._fc2_forward_view = self._ensure_forward_buffers(x)

    def forward_prepared(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.forward(x)
        if self._fc1_weight_t is None or self._fc2_weight_t is None:
            self.cache_weight_views()
        fc1_out = self._fc1_forward_view
        fc2_out = self._fc2_forward_view
        if fc1_out is None or fc2_out is None:
            raise RuntimeError("forward_prepared() requires prepare_forward_buffers()")
        return self._forward_into_buffers(x, fc1_out, fc2_out)

    def _forward_into_buffers(
        self,
        x: torch.Tensor,
        fc1_out: torch.Tensor,
        fc2_out: torch.Tensor,
    ) -> torch.Tensor:
        torch.mm(x, self._fc1_weight_t, out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        F.gelu(fc1_out, out=fc1_out)
        torch.mm(fc1_out, self._fc2_weight_t, out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out
