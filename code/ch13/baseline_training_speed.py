"""Baseline end-to-end transformer training loop focused on throughput."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ch13.training_speed_common import TrainingSpeedConfig, TrainingSpeedModel, make_training_batch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineTrainingSpeedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Eager BF16-autocast training step with static shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = TrainingSpeedConfig()
        self.model: Optional[nn.Module] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.targets_flat: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.autocast_dtype = torch.bfloat16

        tokens = self.cfg.batch_size * self.cfg.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = TrainingSpeedModel(self.cfg).to(self.device).train()
        self.input_ids, self.targets = make_training_batch(self.cfg, self.device)
        self.targets_flat = self.targets.view(-1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        self.criterion = nn.CrossEntropyLoss()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self._verify_output_buffer = torch.empty(
            (1, 1, min(8, self.cfg.vocab_size)),
            device=self.device,
            dtype=torch.float32,
        )
        self.output = None
        self._synchronize()

    def _train_step(self, input_ids: torch.Tensor, targets_flat: torch.Tensor) -> torch.Tensor:
        assert self.model is not None and self.optimizer is not None and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=False)
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            logits = self.model(input_ids)
        loss = self.criterion(logits.float().view(-1, self.cfg.vocab_size), targets_flat)
        loss.backward()
        self.optimizer.step()
        return logits

    def benchmark_fn(self) -> None:
        if (
            self.model is None
            or self.input_ids is None
            or self.targets is None
            or self.targets_flat is None
            or self.optimizer is None
            or self.criterion is None
        ):
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_training_speed"):
            self._train_step(self.input_ids, self.targets_flat)
            self.output = None
        if self.input_ids is None:
            raise RuntimeError("benchmark_fn() requires input_ids for verification")

    def capture_verification_payload(self) -> None:
        if self.model is None or self.input_ids is None:
            raise RuntimeError("capture_verification_payload() requires model and inputs")
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            verify_logits = self.model(self.input_ids)
            output_slice = verify_logits[
                : self._verify_output_buffer.shape[0],
                : self._verify_output_buffer.shape[1],
                : self._verify_output_buffer.shape[2],
            ]
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.2, 2.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.input_ids = None
        self.targets = None
        self.targets_flat = None
        self.optimizer = None
        self.criterion = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=30,
            warmup=10,
            enable_memory_tracking=True,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "graph_replay": 0.0,
            "autocast_bf16": 1.0,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.input_ids is None:
            return "Model or inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTrainingSpeedBenchmark()
