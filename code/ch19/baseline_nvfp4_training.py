"""BF16 reference training run to compare against NVFP4/Transformer Engine paths."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class _BF16Trainer(nn.Module):
    """Simple transformer-style feed-forward stack to stress activations."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_layers: int) -> None:
        super().__init__()
        self.output = None
        self._verify_input = None
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.LayerNorm(hidden_dim, eps=1e-5),
                    nn.Linear(hidden_dim, intermediate_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(intermediate_dim, hidden_dim, bias=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard forward
        return self.net(x)


class BaselineNVFP4TrainingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Reference BF16 training loop (no NVFP4 compression)."""

    def __init__(self) -> None:
        super().__init__()
        # Larger workload to amortize TE overhead and show NVFP4 benefits
        self.hidden_dim = 4096
        self.intermediate_dim = self.hidden_dim * 4
        self.num_layers = 8
        self.batch_size = 32
        self.seq_len = 1024
        self.micro_batches = 4
        self.model: Optional[_BF16Trainer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self._micro_batch_pairs: List[tuple[torch.Tensor, torch.Tensor]] = []
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.seq_len * self.micro_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self._verification_payload = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = _BF16Trainer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
        ).to(self.device, dtype=torch.bfloat16)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=True)
        self.inputs = [
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.hidden_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            for _ in range(self.micro_batches)
        ]
        self.targets = [torch.randn_like(self.inputs[0]) for _ in range(self.micro_batches)]
        self._micro_batch_pairs = list(zip(self.inputs, self.targets, strict=True))
        self._verify_input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def _train_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        assert self.model is not None and self.optimizer is not None

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(inp)
            loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        with nvtx_range("nvfp4_training_baseline", enable=self._enable_nvtx):
            for inp, target in self._micro_batch_pairs:
                self._train_step(inp, target)
        self.output = None

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None:
            raise RuntimeError("Verification input/model missing")
        with torch.inference_mode():
            verify_output = self.model(self._verify_input)
            if self._verify_output_buffer is None:
                raise RuntimeError("Verification output buffer missing")
            self._verify_output_buffer.copy_(verify_output)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={"fp16": False, "bf16": True, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.5, 5.0),
        )

    def get_input_signature(self) -> InputSignature:
        parameter_count = self.num_layers * (
            (self.hidden_dim * self.intermediate_dim)
            + (self.intermediate_dim * self.hidden_dim)
            + self.hidden_dim * 3
            + self.intermediate_dim
            + self.hidden_dim
        )
        return InputSignature(
            shapes={
                "verify_input": (self.batch_size, self.seq_len, self.hidden_dim),
                "output": (self.batch_size, self.seq_len, self.hidden_dim),
            },
            dtypes={
                "verify_input": str(torch.bfloat16),
                "output": str(torch.float32),
            },
            batch_size=self.batch_size,
            parameter_count=parameter_count,
            precision_flags=PrecisionFlags(bf16=True, tf32=torch.backends.cuda.matmul.allow_tf32),
        )

    def teardown(self) -> None:
        self.model = None
        self.optimizer = None
        self.inputs = []
        self.targets = []
        self._micro_batch_pairs = []
        self._verify_input = None
        self._verify_output_buffer = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            enable_memory_tracking=False,
            deterministic=False,  # allow fastest kernels
            seed=None,  # avoid global seeding that can trip TE/cuRAND
            measurement_timeout_seconds=60,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if not self.inputs:
            return None
        return {
            "nvfp4_baseline.micro_batches": float(self.micro_batches),
            "nvfp4_baseline.seq_len": float(self.seq_len),
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.optimizer is None:
            return "Trainer not initialized"
        if not self.inputs:
            return "Input tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNVFP4TrainingBenchmark()
