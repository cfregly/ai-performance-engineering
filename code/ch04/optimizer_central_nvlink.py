"""optimizer_central_nvlink.py

Centralized optimizer state on a single GPU (typically within the same NVSwitch
island) with peer access enabled. Remote GPUs ship gradients to the central
GPU over NVLink; updated weights are multicast back.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import skip_if_insufficient_gpus
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedOptimizerCentralNvlinkBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Centralized optimizer state (one shard per switch island)."""

    def __init__(self):
        super().__init__()
        self.models: List[nn.Linear] = []
        self.master_weights: List[torch.Tensor] = []
        self.momentum: List[torch.Tensor] = []
        self.grad_root_buffers: List[torch.Tensor] = []
        self.inputs: List[torch.Tensor] = []
        self._update_groups: List[Tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._model_count = 0
        self._update_group_count = 0
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.batch_size = 8
        self.hidden = 512
        self.root_device = torch.device("cuda:0")
        self._payload_parameter_count = 0
        tokens = self.batch_size * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def _enable_peer_access(self) -> None:
        num = torch.cuda.device_count()
        skip_if_insufficient_gpus(2)
        for src in range(num):
            for dst in range(num):
                if src == dst:
                    continue
                if torch.cuda.can_device_access_peer(src, dst):
                    try:
                        torch.cuda.device(src).enable_peer_access(dst)
                    except RuntimeError:
                        # Already enabled or unsupported; ignore
                        pass

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._enable_peer_access()
        num_gpus = max(1, torch.cuda.device_count())
        skip_if_insufficient_gpus(2)

        for rank in range(num_gpus):
            device = f"cuda:{rank}"
            model = nn.Linear(self.hidden, self.hidden).to(device)
            self.models.append(model)
            # Master copies live on the root device
            master_w = model.weight.detach().to(self.root_device)
            master_m = torch.zeros_like(master_w, dtype=torch.float32, device=self.root_device)
            self.master_weights.append(master_w)
            self.momentum.append(master_m)
            self.grad_root_buffers.append(torch.empty_like(master_w, device=self.root_device))
            self.inputs.append(torch.randn(self.batch_size, self.hidden, device=device, dtype=torch.float32))
        self._model_count = len(self.models)
        self._update_groups = list(
            zip(self.models, self.master_weights, self.momentum, self.grad_root_buffers, self.inputs, strict=True)
        )
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
        with self._nvtx_range("optimized_optimizer_central_nvlink"):
            for model, master_w, mom, grad_root_buf, x in self._update_groups:
                y = model(x)
                loss = y.square().mean()
                loss.backward()

                # Ship gradient to root over NVLink (non-blocking if available)
                grad = model.weight.grad
                if grad is None:
                    raise RuntimeError("Expected weight gradient after backward")
                if grad.device == self.root_device:
                    grad_root = grad
                else:
                    grad_root_buf.copy_(grad, non_blocking=True)
                    grad_root = grad_root_buf
                with torch.no_grad():
                    mom.mul_(0.9).add_(grad_root)
                    master_w.add_(-1e-3, mom)
                # Multicast updated weights back
                model.weight.data.copy_(master_w, non_blocking=True)
                model.bias.grad.zero_()
                model.weight.grad.zero_()

    def capture_verification_payload(self) -> None:
        if not self.models or not self.master_weights or not self.momentum or not self.inputs:
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
        self.master_weights.clear()
        self.momentum.clear()
        self.grad_root_buffers.clear()
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
        if not self.models or not self.master_weights:
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
    return OptimizedOptimizerCentralNvlinkBenchmark()
