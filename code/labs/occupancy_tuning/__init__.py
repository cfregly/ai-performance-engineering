"""Public helpers for the Triton Proton occupancy lab."""

from __future__ import annotations

from importlib import import_module

__all__ = ["matmul_kernel", "run_one", "describe_schedule", "triton_matmul"]


def __getattr__(name: str):
    """Load the actual Triton implementation only when a kernel API is requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".triton_matmul", __name__)
    value = module if name == "triton_matmul" else getattr(module, name)
    globals()[name] = value
    return value
