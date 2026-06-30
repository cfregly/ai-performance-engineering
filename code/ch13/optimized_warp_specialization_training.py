"""optimized_warp_specialization_training.py - torch.compile fused epilogue.

Optimized: compile the elementwise epilogue chain so Inductor can fuse ops,
reducing intermediate memory traffic and kernel launches.
"""

from __future__ import annotations

from typing import Callable

import torch

from ch13.baseline_warp_specialization_training import (
    BaselineWarpSpecializationTrainingBenchmark,
    _epilogue_chain,
)
from core.optimization.inductor_guard import (
    InductorCudagraphState,
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
)
from core.utils.compile_utils import compile_callable


class OptimizedWarpSpecializationTrainingBenchmark(BaselineWarpSpecializationTrainingBenchmark):
    """Optimized: torch.compile the epilogue chain for fusion."""

    def setup(self) -> None:
        self._inductor_cfg_state: InductorCudagraphState = disable_inductor_cudagraph_features()
        try:
            super().setup()
            self._compiled_chain: Callable[..., torch.Tensor] = compile_callable(
                _epilogue_chain,
                mode="max-autotune",
            )
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise

    def benchmark_fn(self) -> None:
        if (
            self.x is None
            or self.scale0 is None
            or self.bias0 is None
            or self.scale1 is None
            or self.bias1 is None
            or self.scale2 is None
            or self.bias2 is None
        ):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("optimized_warp_specialization_training"):
            with torch.inference_mode():
                self.output = self._compiled_chain(
                    self.x,
                    self.scale0,
                    self.bias0,
                    self.scale1,
                    self.bias1,
                    self.scale2,
                    self.bias2,
                )
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def teardown(self) -> None:
        restore_inductor_cudagraph_features(getattr(self, "_inductor_cfg_state", None))
        self._inductor_cfg_state = None
        super().teardown()


def get_benchmark() -> OptimizedWarpSpecializationTrainingBenchmark:
    return OptimizedWarpSpecializationTrainingBenchmark()
