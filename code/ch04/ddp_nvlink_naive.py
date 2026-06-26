"""ddp_nvlink_naive.py

Simplified DDP-style training loop that blocks on gradient exchange and does
not bucket or overlap communication. Uses two microbatches to show the cost
of sequential reduce + compute. Falls back to single-GPU execution when
additional GPUs are unavailable.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineDdpNvlinkNaiveBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """No overlap, naive gradient sync."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self._inputs: List[List[torch.Tensor]] = []
        self._allreduce_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.microbatches = 2
        self.batch_size = 8
        self.hidden = 512
        tokens = self.batch_size * self.hidden * self.microbatches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size * self.microbatches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        num = torch.cuda.device_count()
        skip_if_insufficient_gpus(2)
        for rank in range(num):
            device = f"cuda:{rank}"
            self.models.append(nn.Linear(self.hidden, self.hidden).to(device))
        self._inputs = []
        for _micro in range(self.microbatches):
            micro_inputs: List[torch.Tensor] = []
            for model in self.models:
                micro_inputs.append(torch.randn(self.batch_size, self.hidden, device=model.weight.device))
            self._inputs.append(micro_inputs)
        self._allreduce_buffer = torch.empty_like(self.models[0].weight, device=self.models[0].weight.device)
        self._synchronize()

    def _simulate_allreduce(self, grads: List[torch.Tensor]) -> None:
        """Simple blocking allreduce (sum + scatter) across model gradients."""
        if len(grads) == 1:
            return
        root = grads[0].device
        buf = self._allreduce_buffer
        if buf is None:
            raise RuntimeError("Allreduce buffer not initialized")
        buf.copy_(grads[0].to(root))
        for g in grads[1:]:
            buf.add_(g.to(root))
        buf.mul_(1.0 / len(grads))
        for g in grads:
            g.copy_(buf.to(g.device))

    def benchmark_fn(self) -> None:
        assert self.models
        with self._nvtx_range("baseline_ddp_multigpu_nvlink_naive"):
            for micro in range(self.microbatches):
                grads = []
                for model_idx, model in enumerate(self.models):
                    x = self._inputs[micro][model_idx]
                    y = model(x)
                    loss = y.pow(2).mean()
                    loss.backward()
                    grads.append(model.weight.grad)
                # Blocking gradient sync
                self._simulate_allreduce(grads)
                for model in self.models:
                    with torch.no_grad():
                        model.weight.add_(-1e-3, model.weight.grad)
                        model.weight.grad.zero_()
                        model.bias.grad.zero_()
            self.output = self.models[0].weight.detach()

    def capture_verification_payload(self) -> None:
        if self.output is None or not self._inputs:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        x_probe = self._inputs[0][0]
        param_count = sum(p.numel() for m in self.models for p in m.parameters())
        weight_slice = self.output[:8, :8].to(dtype=torch.float32).clone()
        self._set_verification_payload(
            inputs={"x": x_probe},
            output=weight_slice,
            batch_size=int(x_probe.shape[0]),
            parameter_count=param_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.models.clear()
        self._inputs = []
        self._allreduce_buffer = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

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
        if not self.models:
            return "Models not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineDdpNvlinkNaiveBenchmark()
