"""Zero-copy CUDA views of pinned host memory."""

from __future__ import annotations

import torch


class _PinnedHostArray:
    """CUDA Array Interface description of a pinned float32 host tensor."""

    def __init__(self, host: torch.Tensor) -> None:
        # PyTorch keeps this object alive for the life of the CUDA tensor built
        # from it, which keeps the pinned host pages alive too.
        self._host = host
        self.__cuda_array_interface__ = {
            "shape": tuple(host.shape),
            "typestr": "<f4",
            "data": (host.data_ptr(), False),
            "strides": None,
            "version": 3,
        }


def pinned_host_cuda_view(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return a CUDA tensor on ``device`` that aliases the pinned ``host`` buffer.

    Kernels that use the view read and write ``host`` in place, with no copy.
    Raises RuntimeError unless ``host`` is a pinned, contiguous float32 CPU
    tensor, and when PyTorch returns a device copy instead of a view.
    """
    if host.device.type != "cpu" or not host.is_pinned():
        raise RuntimeError("zero-copy view requires a pinned host tensor")
    if host.dtype != torch.float32 or not host.is_contiguous():
        raise RuntimeError("zero-copy view supports only contiguous float32 tensors")
    view = torch.as_tensor(_PinnedHostArray(host), device=device)
    # Unified virtual addressing maps pinned host memory at the same address on
    # the GPU. Any other address means PyTorch made a device copy instead.
    if view.data_ptr() != host.data_ptr():
        raise RuntimeError(
            "zero-copy view is a device copy, not the pinned host buffer "
            f"(host 0x{host.data_ptr():x}, device 0x{view.data_ptr():x})"
        )
    return view
