"""Reject active in-process PyTorch profilers before benchmark execution."""

import torch


def assert_no_active_profiler() -> None:
    """Keep measurement and profiler collection in separate phases.

    This checks PyTorch's current-thread recorder state, including both Kineto
    and the legacy autograd profiler. It does not inspect external Nsight
    sessions or detect a profiler that starts and stops inside benchmark_fn.
    Calling it before dispatch also protects the public threaded entrypoint.
    """
    if torch.autograd._profiler_enabled():
        raise RuntimeError(
            "Active PyTorch profiler detected before benchmark execution. "
            "Exit the enclosing profiler before measuring or collecting another "
            "profile; use the harness's separate profiling phase."
        )
