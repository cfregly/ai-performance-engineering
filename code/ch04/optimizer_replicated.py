"""optimizer_replicated.py

Baseline training step that replicates optimizer state on every GPU.
Each device keeps its own momentum buffers and updates locally (no sharing),
which inflates memory and increases communication later.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineOptimizerReplicatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Replicated optimizer state per GPU."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.momentum: List[torch.Tensor] = []
        self.inputs: List[torch.Tensor] = []
        self._update_groups: List[Tuple[nn.Linear, torch.Tensor, torch.Tensor]] = []
        self._model_count = 0
        self._update_group_count = 0
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.batch_size = 8
        self.hidden = 512
        self._payload_parameter_count = 0
        tokens = self.batch_size * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        skip_if_insufficient_gpus(2)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        num_gpus = torch.cuda.device_count()
        for rank in range(num_gpus):
            device = f"cuda:{rank}"
            model = nn.Linear(self.hidden, self.hidden).to(device)
            buf = torch.zeros_like(model.weight, dtype=torch.float32, device=device)
            self.models.append(model)
            self.momentum.append(buf)
            self.inputs.append(torch.randn(self.batch_size, self.hidden, device=device, dtype=torch.float32))
        self._model_count = len(self.models)
        self._update_groups = list(zip(self.models, self.momentum, self.inputs, strict=True))
        self._update_group_count = self._model_count
        self._verify_output_buffer = torch.empty(
            (32, 32),
            device=self.models[0].weight.device,
            dtype=self.models[0].weight.dtype,
        )
        self._payload_parameter_count = sum(p.numel() for model in self.models for p in model.parameters())
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._update_group_count <= 0 or self._update_group_count != self._model_count:
            raise RuntimeError("setup() must initialize optimizer update groups")
        with self._nvtx_range("baseline_optimizer_replicated"):
            for model, mom, x in self._update_groups:
                y = model(x)
                loss = y.square().mean()
                loss.backward()
                # Local momentum update (per-GPU state)
                with torch.no_grad():
                    mom.mul_(0.9).add_(model.weight.grad)
                    model.weight.add_(-1e-3, mom)
                    model.weight.grad.zero_()
                    model.bias.grad.zero_()

    def capture_verification_payload(self) -> None:
        if not self.models or not self.momentum or not self.inputs:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        model0 = self.models[0]
        x0 = self.inputs[0]
        if self._verify_output_buffer is None:
            raise RuntimeError("setup() must initialize verification output buffer")
        weight_probe = model0.weight[:32, :32].detach()
        self._verify_output_buffer.copy_(weight_probe)
        self._set_verification_payload(
            inputs={"x": x0},
            output=self._verify_output_buffer,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-6, 1e-6),
        )

    def teardown(self) -> None:
        self.models.clear()
        self.momentum.clear()
        self.inputs.clear()
        self._update_groups = []
        self._model_count = 0
        self._update_group_count = 0
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if not self.models or not self.momentum:
            return "Models or optimizer state not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (1e-6, 1e-6)


def get_benchmark() -> BaseBenchmark:
    return BaselineOptimizerReplicatedBenchmark()
