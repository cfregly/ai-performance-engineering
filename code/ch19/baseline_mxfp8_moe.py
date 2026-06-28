"""Baseline MXFP8 MoE microbenchmark with naive Python-side block quantization."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch19.mxfp8_moe_common import (  # noqa: E402
    MX_BLOCK_SIZE,
    balanced_assignments,
    block_dequantize_mxfp8,
    block_quantize_mxfp8,
    bucket_by_expert,
    require_blackwell,
    restore_bucketed,
)


class _NaiveMXFP8Matmul:
    """Reference MXFP8 GEMM that quantizes on the host side each iteration."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        weights: torch.Tensor,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.weights = weights  # [E, N, K]

    def __call__(self, expert_input: torch.Tensor, expert_idx: int) -> torch.Tensor:
        weight = self.weights[expert_idx]  # [N, K]
        q_act, act_scale = block_quantize_mxfp8(expert_input, block_size=MX_BLOCK_SIZE)
        q_wt, wt_scale = block_quantize_mxfp8(weight, block_size=MX_BLOCK_SIZE)
        act = block_dequantize_mxfp8(q_act, act_scale, block_size=MX_BLOCK_SIZE, dtype=torch.float16)
        wt = block_dequantize_mxfp8(q_wt, wt_scale, block_size=MX_BLOCK_SIZE, dtype=torch.float16)
        return act @ wt.transpose(0, 1)


class BaselineMXFP8MoEBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """MXFP8 MoE forward path with unfused quantization and per-expert GEMMs."""

    def __init__(self) -> None:
        super().__init__()
        self.output = None
        self.num_tokens = 4096
        self.hidden_dim = 4096
        self.ffn_dim = 14336
        self.num_experts = 8
        self.inputs: Optional[torch.Tensor] = None
        self.assignments: Optional[torch.Tensor] = None
        self.bucketed_inputs: Optional[torch.Tensor] = None
        self.bucket_indices: Optional[torch.Tensor] = None
        self.expert_order: Optional[torch.Tensor] = None
        self.expert_order_list: List[int] = []
        self.m_splits: List[int] = []
        self.weights: Optional[torch.Tensor] = None
        self.matmul_ref: Optional[_NaiveMXFP8Matmul] = None
        self._bucketed_out: Optional[torch.Tensor] = None
        self._restored_out: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        require_blackwell("ch19 baseline_mxfp8_moe")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.inputs = torch.randn(
            self.num_tokens, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        # Store weights as [E, N, K] to quantize along K (reduction) blocks.
        self.weights = torch.randn(
            self.num_experts, self.ffn_dim, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        self._payload_parameter_count = self.weights.numel()
        self.assignments = balanced_assignments(
            num_tokens=self.num_tokens, num_experts=self.num_experts, device=self.device
        )
        bucketed, m_splits, bucket_indices, expert_order, _ = bucket_by_expert(
            self.inputs, self.assignments, num_experts=self.num_experts
        )
        self.bucketed_inputs = bucketed
        self.bucket_indices = bucket_indices
        self.expert_order = expert_order
        self.expert_order_list = [int(idx) for idx in expert_order.detach().cpu()]
        self.m_splits = m_splits
        self.matmul_ref = _NaiveMXFP8Matmul(
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            weights=self.weights,
        )
        self._restored_out = torch.empty(
            (self.num_tokens, self.ffn_dim), device=self.device, dtype=torch.float16
        )
        self._bucketed_out = torch.empty_like(self._restored_out)
        tokens_per_iteration = float(self.num_tokens)
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration)

    def _run_naive(self) -> torch.Tensor:
        assert (
            self.bucketed_inputs is not None
            and self.bucket_indices is not None
            and self.expert_order is not None
            and self.weights is not None
            and self.matmul_ref is not None
            and self._bucketed_out is not None
        )
        offset = 0
        for expert_idx, m in zip(self.expert_order_list, self.m_splits):
            expert_slice = self.bucketed_inputs.narrow(0, offset, m)
            self._bucketed_out.narrow(0, offset, m).copy_(
                self.matmul_ref(expert_slice, expert_idx)
            )
            offset += m
        return restore_bucketed(
            self._bucketed_out,
            self.bucket_indices,
            num_tokens=self.num_tokens,
            out=self._restored_out,
        )

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), nvtx_range("mxfp8_moe_baseline", enable=self._enable_nvtx):
            self.output = self._run_naive()
        if self.output is None or self.inputs is None or self.weights is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output,
            batch_size=self.num_tokens,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.5, 20.0),
            precision_flags={"fp16": False, "bf16": True, "fp8": True, "tf32": False},
        )

    def get_input_signature(self) -> InputSignature:
        parameter_count = self.num_experts * self.ffn_dim * self.hidden_dim
        return InputSignature(
            shapes={
                "inputs": (self.num_tokens, self.hidden_dim),
                "weights": (self.num_experts, self.ffn_dim, self.hidden_dim),
                "output": (self.num_tokens, self.ffn_dim),
            },
            dtypes={
                "inputs": str(torch.bfloat16),
                "weights": str(torch.bfloat16),
                "output": str(torch.float16),
            },
            batch_size=self.num_tokens,
            parameter_count=parameter_count,
            precision_flags=PrecisionFlags(bf16=True, fp8=True, tf32=False),
        )

    def teardown(self) -> None:
        self.inputs = None
        self.weights = None
        self.assignments = None
        self.bucketed_inputs = None
        self.bucket_indices = None
        self.expert_order = None
        self.expert_order_list = []
        self.m_splits = []
        self.matmul_ref = None
        self._bucketed_out = None
        self._restored_out = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.weights is None:
            return "Inputs not initialized"
        if any(m == 0 for m in self.m_splits):
            return "Empty expert bucket detected"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=5, deterministic=False, enable_nvtx=True)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp8",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineMXFP8MoEBenchmark()
