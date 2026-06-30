"""Optimized training-speed benchmark using CUDA graph replay."""

from __future__ import annotations

import copy
from typing import Optional

import torch

from ch13.baseline_training_speed import BaselineTrainingSpeedBenchmark
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class OptimizedTrainingSpeedBenchmark(BaselineTrainingSpeedBenchmark):
    """Replay the fixed-shape training step from a captured CUDA graph."""

    def __init__(self) -> None:
        super().__init__()
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.capture_stream: Optional[torch.cuda.Stream] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.static_target_flat: Optional[torch.Tensor] = None

    def setup(self) -> None:
        super().setup()
        if any(v is None for v in (self.model, self.input_ids, self.targets, self.optimizer, self.criterion)):
            raise RuntimeError("Benchmark not configured")

        saved_model_state = copy.deepcopy(self.model.state_dict())
        saved_opt_state = copy.deepcopy(self.optimizer.state_dict())

        if self.targets_flat is None:
            raise RuntimeError("Benchmark target view not configured")

        for _ in range(3):
            self._train_step(self.input_ids, self.targets_flat)
        self._synchronize()
        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)

        self.static_input = self.input_ids.clone()
        self.static_target = self.targets.clone()
        self.static_target_flat = self.static_target.view(-1)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()

        def _captured_step() -> None:
            self._train_step(self.static_input, self.static_target_flat)

        with torch.cuda.stream(self.capture_stream):
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                _captured_step()
        self.capture_stream.synchronize()

        self.model.load_state_dict(saved_model_state)
        self.optimizer.load_state_dict(saved_opt_state)
        self.output = None

    def benchmark_fn(self) -> None:
        if (
            self.graph is None
            or self.capture_stream is None
            or self.input_ids is None
            or self.targets is None
            or self.static_input is None
            or self.static_target is None
            or self.static_target_flat is None
        ):
            raise RuntimeError("CUDA graph not initialized")

        with self._nvtx_range("optimized_training_speed"):
            with torch.cuda.stream(self.capture_stream):
                self.static_input.copy_(self.input_ids)
                self.static_target.copy_(self.targets)
                self.graph.replay()
            self.output = None

    def teardown(self) -> None:
        self.graph = None
        self.capture_stream = None
        self.static_input = None
        self.static_target = None
        self.static_target_flat = None
        super().teardown()

    def get_custom_streams(self) -> list["torch.cuda.Stream"]:
        if self.capture_stream is None:
            return []
        return [self.capture_stream]

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=10,
            enable_memory_tracking=True,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "graph_replay": 1.0,
            "autocast_bf16": 1.0,
        }


def get_benchmark() -> BaseBenchmark:
    return OptimizedTrainingSpeedBenchmark()
