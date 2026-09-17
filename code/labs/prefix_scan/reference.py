"""Native inclusive int32 scan reference."""

import torch


def inclusive_scan(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim != 1 or x.dtype != torch.int32 or not x.is_contiguous() or x.numel() == 0:
        raise ValueError("scan requires a nonempty contiguous int32 vector")
    return torch.cumsum(x, dim=0, dtype=torch.int32, out=out)
