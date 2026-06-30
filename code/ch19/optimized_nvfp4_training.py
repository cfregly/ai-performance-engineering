"""NVFP4 training benchmark that exercises Transformer Engine block scaling.

Chapter 19 demonstrates NVFP4 (4-bit floating point) quantization for training,
which provides memory savings and potential speedups through reduced memory bandwidth.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

try:
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import LayerNorm as TELayerNorm
    from transformer_engine.pytorch import autocast as te_autocast
    from transformer_engine.pytorch import quantized_model_init, is_nvfp4_available
    from transformer_engine.common import recipe as te_recipe

    TE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc
    TELinear = TELayerNorm = te_autocast = quantized_model_init = te_recipe = None  # type: ignore[assignment]
    is_nvfp4_available = lambda: False  # type: ignore[assignment]
else:
    TE_IMPORT_ERROR = None

resolve_device = partial(require_cuda_device, "CUDA required for NVFP4 benchmarks")


class _NVFP4Block(nn.Module):
    """Feed-forward block composed of Transformer Engine modules for NVFP4 quantization."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.output = None
        self._verify_input = None
        self.ln = TELayerNorm(hidden_dim)
        self.fc1 = TELinear(hidden_dim, intermediate_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = TELinear(intermediate_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        orig_shape = x.shape
        y = x.reshape(-1, orig_shape[-1])
        y = self.ln(y)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return y.reshape(*orig_shape)


class OptimizedNVFP4TrainingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """NVFP4 quantized training using Transformer Engine.
    
    This demonstrates the memory and compute benefits of NVFP4 (4-bit) quantization
    compared to the BF16 baseline. NVFP4 provides:
    - 4x memory compression for activations
    - Reduced memory bandwidth requirements
    - Potential speedup from smaller data transfers
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        # Larger workload to amortize TE overhead and show NVFP4 benefits
        self.hidden_dim = 4096
        self.intermediate_dim = self.hidden_dim * 4
        self.num_layers = 8
        self.batch_size = 32
        self.seq_len = 1024
        self.micro_batches = 4
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.inputs: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self._micro_batch_pairs: List[tuple[torch.Tensor, torch.Tensor]] = []
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        
        # NVFP4 recipe with calibration
        self.nvfp4_recipe = (
            te_recipe.NVFP4BlockScaling(calibration_steps=20, amax_history_len=16, fp4_tensor_block=16)
            if TE_AVAILABLE
            else None
        )
        self.active_recipe = None
        self.use_nvfp4 = False
        self._verification_payload = None
        tokens = self.batch_size * self.seq_len * self.micro_batches
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.micro_batches),
            tokens_per_iteration=float(tokens),
        )
        self._enable_nvtx = False
        self._payload_parameter_count = 0

    def setup(self) -> None:
        if not TE_AVAILABLE:
            raise RuntimeError(f"Transformer Engine not available: {TE_IMPORT_ERROR}")

        torch.manual_seed(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

        if not is_nvfp4_available() or self.nvfp4_recipe is None:
            raise RuntimeError("NVFP4 not available: ensure Blackwell GPU + Transformer Engine NVFP4 support")

        self.active_recipe = self.nvfp4_recipe
        self.use_nvfp4 = True
        
        # Build model with TE modules
        layers = [
            _NVFP4Block(self.hidden_dim, self.intermediate_dim)
            for _ in range(self.num_layers)
        ]
        
        with quantized_model_init(enabled=True, recipe=self.active_recipe):
            self.model = nn.Sequential(*layers).to(self.device, dtype=torch.bfloat16)
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
        self.targets = [
            torch.randn_like(self.inputs[0]) for _ in range(self.micro_batches)
        ]
        self._micro_batch_pairs = list(zip(self.inputs, self.targets, strict=True))
        self._verify_input = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        
        # Calibration warmup (important for quantization)
        self._calibration_warmup()
        torch.cuda.synchronize()

    def _calibration_warmup(self) -> None:
        """Run calibration steps to collect scaling factors."""
        if self.model is None or self.active_recipe is None:
            return
        
        # Run several forward passes to calibrate quantization scales
        for _ in range(5):
            for inp, target in self._micro_batch_pairs:
                self._train_step(inp, target)
        torch.cuda.synchronize()

    def _train_step(self, inp: torch.Tensor, target: torch.Tensor) -> None:
        assert self.model is not None and self.optimizer is not None

        self.optimizer.zero_grad(set_to_none=True)
        with te_autocast(enabled=True, recipe=self.active_recipe):
            out = self.model(inp)
            loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        with nvtx_range("nvfp4_training", enable=self._enable_nvtx):
            for inp, target in self._micro_batch_pairs:
                self._train_step(inp, target)
        self.output = None

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.model is None:
            raise RuntimeError("Verification input/model missing")
        with torch.inference_mode():
            with te_autocast(enabled=True, recipe=self.active_recipe):
                out = self.model(self._verify_input)
            if self._verify_output_buffer is None:
                raise RuntimeError("Verification output buffer missing")
            self._verify_output_buffer.copy_(out)
            self.output = self._verify_output_buffer
        precision_flags = {
            "fp16": False,
            "bf16": True,
            "fp8": False,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        self._set_verification_payload(
            inputs={"verify_input": self._verify_input},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags=precision_flags,
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
            deterministic=False,
            seed=None,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return NVFP4-specific metrics."""
        return {
            "nvfp4.active": 1.0,
            "nvfp4.compression_ratio": 4.0,
            "nvfp4.micro_batches": float(self.micro_batches),
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.optimizer is None:
            return "Transformer Engine model not initialized"
        if not self.inputs:
            return "Input tensors missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNVFP4TrainingBenchmark()
