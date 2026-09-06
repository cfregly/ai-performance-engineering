"""Helpers for torch.distributed._symmetric_memory without shims."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, cast

try:
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch not available in some tooling environments
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]

_symm_mem: Any
if torch is not None:
    try:  # pragma: no cover - import guarded for CPU-only builds
        from torch.distributed import _symmetric_memory as _symm_mem
    except ImportError:  # pragma: no cover
        _symm_mem = None
else:  # pragma: no cover
    _symm_mem = None

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup as _ProcessGroup
else:
    _ProcessGroup = object

ProcessGroup = _ProcessGroup
SymmetricMemoryBackend = Literal["NVSHMEM", "CUDA"]
_SUPPORTED_BACKENDS = frozenset({"NVSHMEM", "CUDA"})


def _normalize_backend(backend: str) -> SymmetricMemoryBackend:
    normalized = backend.upper()
    if normalized not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "SymmetricMemory backend must be explicitly set to NVSHMEM or CUDA"
        )
    return cast(SymmetricMemoryBackend, normalized)


def symmetric_memory_backend_supported(backend: str = "NVSHMEM") -> bool:
    """Return whether the runtime exposes the requested backend's required API.

    This is a non-mutating capability check. The handle constructor still selects
    and verifies the requested backend in the worker before allocating memory.
    """
    requested_backend = _normalize_backend(backend)
    if os.environ.get("AISP_DISABLE_SYMMETRIC_MEMORY", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return False
    if torch is None or dist is None or _symm_mem is None:
        return False
    if not torch.cuda.is_available():
        return False
    required = ("empty", "rendezvous", "set_backend", "get_backend")
    if not all(callable(getattr(_symm_mem, name, None)) for name in required):
        return False
    if requested_backend == "CUDA":
        return True
    try:
        return bool(_symm_mem.is_nvshmem_available())
    except Exception:
        return False


def symmetric_memory_available() -> bool:
    """Return True when NVSHMEM-backed symmetric memory is available."""
    return symmetric_memory_backend_supported("NVSHMEM")


class SymmetricMemoryHandle:
    """Direct symmetric memory handle backed by torch.distributed._symmetric_memory."""

    __slots__ = (
        "buffer",
        "_symm_tensor",
        "_handle",
        "_shape",
        "_dtype",
        "_group",
        "_rank",
        "_world_size",
        "backend",
    )

    def __init__(
        self,
        tensor: torch.Tensor,
        group: str | ProcessGroup | None = None,
        *,
        backend: SymmetricMemoryBackend = "NVSHMEM",
    ) -> None:
        if torch is None or dist is None or _symm_mem is None:
            raise RuntimeError("torch.distributed._symmetric_memory is not available")
        if not tensor.is_cuda:
            raise ValueError("SymmetricMemory tensors must live on CUDA devices")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SymmetricMemory")
        if group is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before using SymmetricMemory"
                )
            group = dist.group.WORLD

        requested_backend = _normalize_backend(backend)
        if not symmetric_memory_backend_supported(requested_backend):
            raise RuntimeError(
                f"{requested_backend} symmetric memory is not available on this system"
            )

        try:
            # Backend choice is process-global and cannot be changed after the
            # first symmetric allocation. Avoid invoking the global setter when
            # a previous handle already selected the requested backend.
            observed_backend = _symm_mem.get_backend(tensor.device)
            if observed_backend != requested_backend:
                _symm_mem.set_backend(requested_backend)
                observed_backend = _symm_mem.get_backend(tensor.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to select {requested_backend} symmetric-memory backend"
            ) from exc
        if observed_backend != requested_backend:
            raise RuntimeError(
                "Symmetric-memory backend identity mismatch: "
                f"requested {requested_backend}, observed {observed_backend!r}"
            )

        try:
            if hasattr(_symm_mem, "is_symm_mem_enabled_for_group") and hasattr(
                _symm_mem, "enable_symm_mem_for_group"
            ):
                from torch.distributed import distributed_c10d as c10d

                group_name = c10d._get_process_group_name(group)
                if not _symm_mem.is_symm_mem_enabled_for_group(group_name):
                    _symm_mem.enable_symm_mem_for_group(group_name)
        except Exception as exc:
            raise RuntimeError("Failed to enable symmetric memory for process group") from exc

        self._group = group
        self._rank = dist.get_rank(group=group)
        self._world_size = dist.get_world_size(group=group)
        self._shape = tuple(tensor.shape)
        self._dtype = tensor.dtype

        self._symm_tensor = _symm_mem.empty(
            self._shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if tensor.numel() > 0:
            self._symm_tensor.copy_(tensor)

        self._handle = _symm_mem.rendezvous(self._symm_tensor, group)
        if requested_backend == "CUDA" and not callable(
            getattr(self._handle, "barrier", None)
        ):
            raise RuntimeError("CUDA symmetric-memory handle has no device barrier")
        self.buffer = cast(
            torch.Tensor,
            self._handle.get_buffer(self._rank, self._shape, self._dtype),
        )
        self.backend = observed_backend

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def get_buffer(self, rank: int) -> torch.Tensor:
        """Return a view of the symmetric buffer for ``rank``."""
        return cast(
            torch.Tensor,
            self._handle.get_buffer(rank, self._shape, self._dtype),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def create_symmetric_memory_handle(
    tensor: torch.Tensor,
    group: str | ProcessGroup | None = None,
    *,
    backend: SymmetricMemoryBackend = "NVSHMEM",
) -> SymmetricMemoryHandle:
    """Create a symmetric memory handle or raise if unavailable."""
    return SymmetricMemoryHandle(tensor, group=group, backend=backend)


def maybe_create_symmetric_memory_handle(
    tensor: torch.Tensor,
    group: str | ProcessGroup | None = None,
    *,
    backend: SymmetricMemoryBackend = "NVSHMEM",
) -> SymmetricMemoryHandle | None:
    """Return a symmetric memory handle when available; otherwise None."""
    if not symmetric_memory_backend_supported(backend):
        return None
    try:
        return SymmetricMemoryHandle(tensor, group=group, backend=backend)
    except Exception:
        return None
