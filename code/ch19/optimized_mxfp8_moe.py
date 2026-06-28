"""Optimized MXFP8 MoE microbenchmark using Transformer Engine grouped GEMMs."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import torch

from core.utils import logger  # noqa: E402
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402
from ch19 import arch_config  # noqa: E402
from ch19.mxfp8_moe_common import (  # noqa: E402
    balanced_assignments,
    bucket_by_expert,
    require_blackwell,
    restore_bucketed_reduce,
)

try:
    from transformer_engine.pytorch.module import GroupedLinear  # type: ignore
    from transformer_engine.pytorch import autocast as te_autocast  # type: ignore
    from transformer_engine.pytorch import quantized_model_init  # type: ignore
    from transformer_engine.common import recipe as te_recipe  # type: ignore

    TE_AVAILABLE = True
    TE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc
    GroupedLinear = te_autocast = quantized_model_init = te_recipe = None  # type: ignore

_log = logger.get_logger(__name__)


def _flat_topk_token_ids(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.int64)
    if top_k > 1:
        token_ids.div_(top_k, rounding_mode="floor")
    return token_ids


class OptimizedMXFP8MoEBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """MXFP8 MoE forward path with grouped GEMMs and fused quantization in TE."""

    def __init__(self) -> None:
        super().__init__()
        self.output = None
        self.num_tokens = 4096
        self.hidden_dim = 4096
        self.ffn_dim = 14336
        self.num_experts = 8
        flags = self._parse_flags()
        self.top_k = max(1, flags.top_k)
        self.use_cuda_graphs = bool(flags.cuda_graphs)
        self.inputs: Optional[torch.Tensor] = None
        self.assignments: Optional[torch.Tensor] = None
        self.bucketed_inputs: Optional[torch.Tensor] = None
        self.bucket_indices: Optional[torch.Tensor] = None
        self.expert_order: Optional[torch.Tensor] = None
        self.bucket_token_ids: Optional[torch.Tensor] = None
        self._bucket_token_scatter_index: Optional[torch.Tensor] = None
        self.gating_weights: Optional[torch.Tensor] = None
        self._gating_weight_factors: Optional[torch.Tensor] = None
        self.m_splits: List[int] = []
        self.weights: Optional[torch.Tensor] = None
        self.layer: Optional[GroupedLinear] = None
        self.recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3) if TE_AVAILABLE else None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_out: Optional[torch.Tensor] = None
        self._graph_weight: Optional[torch.Tensor] = None
        self._graph_weight_factors: Optional[torch.Tensor] = None
        self._graph_weighted_out: Optional[torch.Tensor] = None
        self._restored_out: Optional[torch.Tensor] = None
        self._restored_weight: Optional[torch.Tensor] = None
        self._restored_weight_factors: Optional[torch.Tensor] = None
        self._weighted_out: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(requests_per_iteration=1.0)

    @staticmethod
    def _parse_flags(argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--top-k", type=int, default=1, help="Top-k experts to route per token.")
        parser.add_argument(
            "--cuda-graphs",
            action="store_true",
            help="Enable CUDA Graph capture/replay for the grouped GEMM path.",
        )
        args, _ = parser.parse_known_args(argv)
        return args

    def _supergroup_tokens(
        self,
        bucketed: torch.Tensor,
        m_splits: List[int],
        bucket_indices: torch.Tensor,
        expert_order: torch.Tensor,
        bucket_token_ids: torch.Tensor,
        gating_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reorder experts by bucket size to improve L2 reuse."""
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for m in m_splits:
            offsets.append((cursor, cursor + m))
            cursor += m
        order = sorted(range(len(m_splits)), key=lambda i: m_splits[i], reverse=True)
        reordered_splits = [m_splits[idx] for idx in order]
        order_tensor = torch.tensor(order, device=bucketed.device, dtype=torch.int64)
        base_rows = torch.arange(bucketed.shape[0], device=bucketed.device, dtype=torch.int64)
        row_order = torch.empty_like(base_rows)
        row_cursor = 0
        for idx in order:
            start, end = offsets[idx]
            width = end - start
            row_order.narrow(0, row_cursor, width).copy_(base_rows.narrow(0, start, width))
            row_cursor += width
        new_bucketed = bucketed.index_select(0, row_order)
        new_indices = bucket_indices.index_select(0, row_order)
        new_order = expert_order.index_select(0, order_tensor)
        new_token_ids = bucket_token_ids.index_select(0, row_order)
        new_weights = gating_weights.index_select(0, row_order)
        return new_bucketed, reordered_splits, new_indices, new_order, new_token_ids, new_weights

    def _maybe_log_missing_te(self) -> None:
        if TE_AVAILABLE:
            return
        raise RuntimeError(
            f"SKIPPED: Transformer Engine is required for optimized MXFP8 benchmarks: {TE_IMPORT_ERROR}"
        )

    def setup(self) -> None:
        require_blackwell("ch19 optimized_mxfp8_moe")
        self._maybe_log_missing_te()
        if not arch_config.USE_TE_FP8:
            raise RuntimeError("SKIPPED: MXFP8 path disabled via arch_config.USE_TE_FP8.")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.inputs = torch.randn(
            self.num_tokens, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        self.weights = torch.randn(
            self.num_experts, self.ffn_dim, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        self._payload_parameter_count = self.weights.numel()
        base_assign = balanced_assignments(
            num_tokens=self.num_tokens, num_experts=self.num_experts, device=self.device
        )
        top_k = max(1, int(self.top_k))
        token_ids = _flat_topk_token_ids(self.num_tokens, top_k, self.device)
        if top_k == 1:
            assignments = base_assign
            expanded_inputs = self.inputs
            gating_weights = torch.ones(self.num_tokens, device=self.device, dtype=torch.float16)
        else:
            offsets = torch.arange(top_k, device=self.device, dtype=base_assign.dtype)
            assignment_matrix = torch.empty(
                self.num_tokens,
                top_k,
                device=self.device,
                dtype=base_assign.dtype,
            )
            torch.add(base_assign.unsqueeze(1), offsets, out=assignment_matrix)
            torch.remainder(assignment_matrix, self.num_experts, out=assignment_matrix)
            assignments = assignment_matrix.reshape(-1)
            expanded_inputs = self.inputs.index_select(0, token_ids)
            gating_weights = torch.full(
                (self.num_tokens * top_k,),
                1.0 / float(top_k),
                device=self.device,
                dtype=torch.float16,
            )
        self.assignments = assignments
        bucketed, m_splits, bucket_indices, expert_order, bucket_token_ids = bucket_by_expert(
            expanded_inputs,
            assignments,
            num_experts=self.num_experts,
            token_ids=token_ids,
        )
        bucketed, m_splits, bucket_indices, expert_order, bucket_token_ids, gating_weights = self._supergroup_tokens(
            bucketed, m_splits, bucket_indices, expert_order, bucket_token_ids, gating_weights
        )
        self.bucketed_inputs = bucketed
        self.bucket_indices = bucket_indices
        self.expert_order = expert_order
        self.bucket_token_ids = bucket_token_ids
        self._bucket_token_scatter_index = bucket_token_ids.unsqueeze(-1).expand(
            bucketed.shape[0],
            self.ffn_dim,
        )
        self.gating_weights = gating_weights
        self._gating_weight_factors = gating_weights.unsqueeze(-1)
        self.m_splits = m_splits

        ordered_weights = self.weights.index_select(0, self.expert_order)
        with quantized_model_init(enabled=True, recipe=self.recipe):
            self.layer = GroupedLinear(
                num_gemms=len(self.m_splits),
                in_features=self.hidden_dim,
                out_features=self.ffn_dim,
                bias=False,
                params_dtype=torch.bfloat16,
            ).to(self.device)
            with torch.inference_mode():
                for idx in range(len(self.m_splits)):
                    weight_param = getattr(self.layer, f"weight{idx}")
                    weight_param.copy_(ordered_weights[idx])

        self._calibrate_fp8()
        self._restored_out = torch.empty(
            (self.num_tokens, self.ffn_dim), device=self.device, dtype=torch.float16
        )
        self._restored_weight = torch.empty((self.num_tokens,), device=self.device, dtype=torch.float16)
        self._restored_weight_factors = self._restored_weight.unsqueeze(-1)
        self._weighted_out = torch.empty((bucketed.shape[0], self.ffn_dim), device=self.device, dtype=torch.float16)
        if self.use_cuda_graphs:
            self._capture_graph()
        self.register_workload_metadata(tokens_per_iteration=float(self.num_tokens))
        torch.cuda.synchronize(self.device)

    def _calibrate_fp8(self) -> None:
        if self.layer is None or self.bucketed_inputs is None:
            return
        with te_autocast(enabled=True, recipe=self.recipe, calibrating=True):
            _ = self.layer(
                self.bucketed_inputs,
                self.m_splits,
                is_first_microbatch=True,
            )
        torch.cuda.synchronize(self.device)

    def _forward_grouped(self) -> torch.Tensor:
        assert (
            self.layer is not None
            and self.bucketed_inputs is not None
            and self.bucket_indices is not None
            and self.bucket_token_ids is not None
            and self._bucket_token_scatter_index is not None
            and self.gating_weights is not None
            and self._gating_weight_factors is not None
            and self._restored_out is not None
            and self._restored_weight is not None
            and self._restored_weight_factors is not None
            and self._weighted_out is not None
        )
        with te_autocast(enabled=True, recipe=self.recipe):
            bucketed_out = self.layer(
                self.bucketed_inputs,
                self.m_splits,
                is_first_microbatch=False,
            )
        return restore_bucketed_reduce(
            bucketed_out,
            self.bucket_token_ids,
            num_tokens=self.num_tokens,
            weights=self.gating_weights,
            out=self._restored_out,
            weight_out=self._restored_weight,
            weighted_out=self._weighted_out,
            bucket_token_ids_expanded=self._bucket_token_scatter_index,
            weights_expanded=self._gating_weight_factors,
            weight_out_expanded=self._restored_weight_factors,
        )

    def _capture_graph(self) -> None:
        assert (
            self.bucketed_inputs is not None
            and self.bucket_token_ids is not None
            and self._bucket_token_scatter_index is not None
            and self.gating_weights is not None
            and self._gating_weight_factors is not None
        )
        self._graph = torch.cuda.CUDAGraph()
        self._graph_out = torch.empty(
            (self.num_tokens, self.ffn_dim), device=self.device, dtype=torch.float16
        )
        self._graph_weight = torch.empty((self.num_tokens,), device=self.device, dtype=torch.float16)
        self._graph_weight_factors = self._graph_weight.unsqueeze(-1)
        self._graph_weighted_out = torch.empty(
            (self.bucketed_inputs.shape[0], self.ffn_dim), device=self.device, dtype=torch.float16
        )
        torch.cuda.synchronize(self.device)
        with torch.cuda.graph(self._graph):
            with te_autocast(enabled=True, recipe=self.recipe):
                bucketed_out = self.layer(  # type: ignore[arg-type]
                    self.bucketed_inputs,  # type: ignore[arg-type]
                    self.m_splits,
                    is_first_microbatch=False,
                )
            restore_bucketed_reduce(
                bucketed_out,
                self.bucket_token_ids,
                num_tokens=self.num_tokens,
                weights=self.gating_weights,
                out=self._graph_out,
                weight_out=self._graph_weight,
                weighted_out=self._graph_weighted_out,
                bucket_token_ids_expanded=self._bucket_token_scatter_index,
                weights_expanded=self._gating_weight_factors,
                weight_out_expanded=self._graph_weight_factors,
            )

    def benchmark_fn(self) -> None:
        with torch.inference_mode(), nvtx_range("mxfp8_moe_optimized", enable=self._enable_nvtx):
            if self.use_cuda_graphs and self._graph is not None and self._graph_out is not None:
                self._graph.replay()
                self.output = self._graph_out
            else:
                self.output = self._forward_grouped()
        if self.output is None or self.inputs is None or self.weights is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"inputs": self.inputs},
            output=self.output.to(torch.float16) if self.output is not None else self.output,
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
        self.bucket_token_ids = None
        self._bucket_token_scatter_index = None
        self.gating_weights = None
        self._gating_weight_factors = None
        self.m_splits = []
        self.layer = None
        self._graph = None
        self._graph_out = None
        self._graph_weight = None
        self._graph_weight_factors = None
        self._graph_weighted_out = None
        self._restored_out = None
        self._restored_weight = None
        self._restored_weight_factors = None
        self._weighted_out = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.layer is None or self.bucketed_inputs is None:
            return "Layer not initialized"
        if any(m == 0 for m in self.m_splits):
            return "Empty expert bucket detected"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            deterministic=False,
            enable_nvtx=True,
            measurement_timeout_seconds=90,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp8",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedMXFP8MoEBenchmark()
