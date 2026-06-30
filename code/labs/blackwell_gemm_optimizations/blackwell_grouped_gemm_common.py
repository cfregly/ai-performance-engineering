"""Shared workload and benchmark logic for the Blackwell grouped-GEMM lab."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_autotune import (
    KernelSchedule,
    resolve_schedule,
)
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_kernel import (
    launch_grouped_gemm_autotuned,
    launch_grouped_gemm_persistent,
    launch_grouped_gemm_standard,
)


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for Blackwell grouped GEMM lab")
    return torch.device("cuda")


@dataclass
class BlackwellGroupedGemmWorkload:
    num_tokens: int = 8192
    num_experts: int = 4
    hidden_dim: int = 2048
    expert_ffn_dim: int = 3072
    dtype: torch.dtype = torch.float16
    histogram: str = "balanced"

    def validate(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be > 1")
        if self.hidden_dim <= 0 or self.expert_ffn_dim <= 0:
            raise ValueError("hidden_dim and expert_ffn_dim must be > 0")
        if self.hidden_dim % 32 != 0:
            raise ValueError("hidden_dim must be divisible by 32")
        if self.expert_ffn_dim % 64 != 0:
            raise ValueError("expert_ffn_dim must be divisible by 64")
        if self.histogram not in {"balanced", "skewed"}:
            raise ValueError("histogram must be 'balanced' or 'skewed'")
        if self.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("dtype must be fp16 or bf16")

    @property
    def dtype_name(self) -> str:
        return "fp16" if self.dtype == torch.float16 else "bf16"


@dataclass
class GroupedGemmState:
    x: torch.Tensor
    x_with_padding: torch.Tensor
    expert_weights: torch.Tensor
    flat_padded_indices: torch.Tensor
    padded_route_weights: torch.Tensor
    padded_route_weight_factors: torch.Tensor
    counts: torch.Tensor
    counts_cpu: tuple[int, ...]
    max_count: int
    reference_output: torch.Tensor


@dataclass(frozen=True)
class VariantResult:
    output: torch.Tensor
    schedule: KernelSchedule


def _workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--expert-ffn-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--histogram", choices=("balanced", "skewed"), default=None)
    return parser


def apply_workload_overrides(
    workload: BlackwellGroupedGemmWorkload, argv: list[str]
) -> BlackwellGroupedGemmWorkload:
    args, _ = _workload_parser().parse_known_args(argv)
    dtype = workload.dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    updated = BlackwellGroupedGemmWorkload(
        num_tokens=args.num_tokens or workload.num_tokens,
        num_experts=args.num_experts or workload.num_experts,
        hidden_dim=args.hidden_dim or workload.hidden_dim,
        expert_ffn_dim=args.expert_ffn_dim or workload.expert_ffn_dim,
        dtype=dtype,
        histogram=args.histogram or workload.histogram,
    )
    updated.validate()
    return updated


def _build_assignments(
    workload: BlackwellGroupedGemmWorkload, device: torch.device
) -> torch.Tensor:
    token_ids = torch.arange(workload.num_tokens, device=device, dtype=torch.long)
    if workload.histogram == "balanced":
        return token_ids % workload.num_experts

    counts_cpu = _assignment_counts_cpu(workload)
    expert_ids = torch.arange(workload.num_experts, device=device, dtype=torch.long)
    repeats = torch.tensor(counts_cpu, device=device, dtype=torch.long)
    return torch.repeat_interleave(
        expert_ids,
        repeats,
        output_size=workload.num_tokens,
    )


def _assignment_counts_cpu(workload: BlackwellGroupedGemmWorkload) -> tuple[int, ...]:
    if workload.histogram == "balanced":
        tokens_per_expert, remainder = divmod(
            workload.num_tokens,
            workload.num_experts,
        )
        return tuple(
            tokens_per_expert + int(expert_id < remainder)
            for expert_id in range(workload.num_experts)
        )

    step = (0.35 - 1.85) / (workload.num_experts - 1)
    weights = [1.85 + step * expert_id for expert_id in range(workload.num_experts)]
    weight_sum = sum(weights)
    counts = [math.floor(weight / weight_sum * workload.num_tokens) for weight in weights]
    remainder = workload.num_tokens - sum(counts)
    if remainder > 0:
        for expert_id in range(remainder):
            counts[expert_id] += 1
    return tuple(counts)


def _build_route_weights(
    workload: BlackwellGroupedGemmWorkload, assignments: torch.Tensor
) -> torch.Tensor:
    token_positions = torch.arange(
        workload.num_tokens,
        device=assignments.device,
        dtype=torch.float32,
    )
    expert_term = assignments.to(torch.float32) * 0.173
    weights = 0.72 + 0.24 * torch.sigmoid(torch.sin(token_positions * 0.013 + expert_term))
    return weights.clamp_(0.5, 1.0)


def grouped_work_unit_count(
    counts_cpu: tuple[int, ...],
    *,
    output_dim: int,
    reduction_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> int:
    total = 0
    col_blocks = math.ceil(output_dim / tile_n)
    red_blocks = math.ceil(reduction_dim / tile_k)
    for count in counts_cpu:
        row_blocks = math.ceil(count / tile_m) if count > 0 else 0
        total += row_blocks * col_blocks * red_blocks
    return total


def build_state(
    workload: BlackwellGroupedGemmWorkload, device: torch.device
) -> GroupedGemmState:
    workload.validate()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260319)

    x = torch.randn(
        workload.num_tokens,
        workload.hidden_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=workload.dtype)
    x = x.contiguous()

    expert_weights = (
        torch.randn(
            workload.num_experts,
            workload.hidden_dim,
            workload.expert_ffn_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(device=device, dtype=workload.dtype)
    expert_weights = expert_weights.contiguous()

    assignments = _build_assignments(workload, device)
    route_weights = _build_route_weights(workload, assignments)

    counts_cpu = _assignment_counts_cpu(workload)
    counts = torch.tensor(counts_cpu, device=device, dtype=torch.int32)
    max_count = int(max(counts_cpu, default=0))
    sentinel = workload.num_tokens

    padded_indices = torch.full(
        (workload.num_experts, max_count),
        sentinel,
        device=device,
        dtype=torch.long,
    )
    padded_route_weights = torch.zeros(
        (workload.num_experts, max_count),
        device=device,
        dtype=torch.float32,
    )
    sorted_token_ids = torch.argsort(assignments, stable=True)
    sorted_experts = assignments.index_select(0, sorted_token_ids)
    counts_long = counts.to(torch.long)
    cumulative_counts = torch.cumsum(counts_long, dim=0)
    expert_starts = torch.empty_like(cumulative_counts)
    expert_starts[0] = 0
    expert_starts[1:] = cumulative_counts[:-1]
    packed_offsets = torch.arange(
        workload.num_tokens,
        device=device,
        dtype=torch.long,
    ) - expert_starts.index_select(0, sorted_experts)
    padded_indices[sorted_experts, packed_offsets] = sorted_token_ids
    padded_route_weights[sorted_experts, packed_offsets] = route_weights.index_select(
        0,
        sorted_token_ids,
    )

    x_with_padding = torch.empty(
        workload.num_tokens + 1,
        workload.hidden_dim,
        device=device,
        dtype=workload.dtype,
    )
    x_with_padding[: workload.num_tokens].copy_(x)
    x_with_padding[workload.num_tokens].zero_()
    flat_padded_indices = padded_indices.reshape(-1).contiguous()
    gathered = torch.index_select(x_with_padding, 0, flat_padded_indices).view(
        workload.num_experts,
        max_count,
        workload.hidden_dim,
    )
    reference = torch.bmm(
        gathered.to(torch.float32),
        expert_weights.to(torch.float32),
    )
    padded_route_weight_factors = padded_route_weights.unsqueeze(-1).to(workload.dtype)
    reference *= padded_route_weights.unsqueeze(-1)
    reference = reference.to(workload.dtype).contiguous()

    return GroupedGemmState(
        x=x,
        x_with_padding=x_with_padding,
        expert_weights=expert_weights,
        flat_padded_indices=flat_padded_indices,
        padded_route_weights=padded_route_weights.contiguous(),
        padded_route_weight_factors=padded_route_weight_factors.contiguous(),
        counts=counts.contiguous(),
        counts_cpu=counts_cpu,
        max_count=max_count,
        reference_output=reference,
    )


def require_blackwell_grouped_gemm_support(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("SKIPPED: CUDA required for Blackwell grouped GEMM lab")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        raise RuntimeError(
            "SKIPPED: Blackwell grouped GEMM lab requires compute capability 10.0+."
        )


def _gather_packed_tokens(
    state: GroupedGemmState,
    out_flat: torch.Tensor,
    out_view: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch.index_select(state.x_with_padding, 0, state.flat_padded_indices, out=out_flat)
    if out_view is not None:
        return out_view
    return out_flat.view(
        state.counts.shape[0],
        state.max_count,
        state.x.shape[1],
    )


def run_variant(
    state: GroupedGemmState,
    *,
    variant: str,
    packed_tokens_flat: torch.Tensor,
    output_buffer: torch.Tensor,
    packed_tokens_view: torch.Tensor | None = None,
    experimental: str | None = None,
) -> VariantResult:
    packed_tokens = _gather_packed_tokens(state, packed_tokens_flat, packed_tokens_view)
    schedule_name = experimental if experimental is not None else variant
    schedule = resolve_schedule(schedule_name)

    if schedule_name == "full_stack":
        output = launch_grouped_gemm_autotuned(
            packed_tokens,
            state.expert_weights,
            state.padded_route_weights,
            state.counts,
            output_buffer,
        )
    elif schedule.persistent:
        output = launch_grouped_gemm_persistent(
            packed_tokens,
            state.expert_weights,
            state.padded_route_weights,
            state.counts,
            output_buffer,
            schedule,
        )
    else:
        output = launch_grouped_gemm_standard(
            packed_tokens,
            state.expert_weights,
            state.padded_route_weights,
            state.counts,
            output_buffer,
            schedule,
            state.padded_route_weight_factors,
        )
    return VariantResult(output=output, schedule=schedule)


class BlackwellGroupedGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Grouped MoE-style GEMM journey for Blackwell-class GPUs."""

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, variant: str, label: str) -> None:
        super().__init__()
        self.variant = variant
        self.label = label
        self.workload = BlackwellGroupedGemmWorkload()
        self.state: Optional[GroupedGemmState] = None
        self.output: Optional[torch.Tensor] = None
        self._flat_packed_tokens: Optional[torch.Tensor] = None
        self._packed_tokens_view: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verification_shape_tensor: Optional[torch.Tensor] = None
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.workload.num_tokens),
            tokens_per_iteration=float(self.workload.num_tokens),
            custom_units_per_iteration=float(
                2 * self.workload.num_tokens * self.workload.hidden_dim * self.workload.expert_ffn_dim
            ),
            custom_unit_name="FLOPs",
        )
        self._custom_metrics: dict[str, float] = {}

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_workload_overrides(self.workload, argv)
        self._workload_metadata = WorkloadMetadata(
            requests_per_iteration=float(self.workload.num_tokens),
            tokens_per_iteration=float(self.workload.num_tokens),
            custom_units_per_iteration=float(
                2 * self.workload.num_tokens * self.workload.hidden_dim * self.workload.expert_ffn_dim
            ),
            custom_unit_name="FLOPs",
        )

    def setup(self) -> None:
        device = _resolve_device()
        require_blackwell_grouped_gemm_support(device)
        self.state = build_state(self.workload, device)
        self.output = None
        self._flat_packed_tokens = torch.empty(
            self.workload.num_experts * self.state.max_count,
            self.workload.hidden_dim,
            device=device,
            dtype=self.workload.dtype,
        )
        self._packed_tokens_view = self._flat_packed_tokens.view(
            self.workload.num_experts,
            self.state.max_count,
            self.workload.hidden_dim,
        )
        self._output_buffer = torch.empty(
            self.workload.num_experts,
            self.state.max_count,
            self.workload.expert_ffn_dim,
            device=device,
            dtype=self.workload.dtype,
        )
        self._verify_output_buffer = torch.empty(
            min(2, self.workload.num_experts),
            min(16, self.state.max_count),
            min(32, self.workload.expert_ffn_dim),
            device=device,
            dtype=self.workload.dtype,
        )
        self._verification_shape_tensor = torch.tensor(
            [
                self.workload.num_tokens,
                self.workload.num_experts,
                self.workload.hidden_dim,
                self.workload.expert_ffn_dim,
            ],
            device="cpu",
            dtype=torch.int64,
        )
        schedule = resolve_schedule(self.variant)
        grouped_units = grouped_work_unit_count(
            self.state.counts_cpu,
            output_dim=self.workload.expert_ffn_dim,
            reduction_dim=self.workload.hidden_dim,
            tile_m=schedule.block_m,
            tile_n=schedule.block_n,
            tile_k=schedule.block_k,
        )
        self._custom_metrics = {
            "blackwell_grouped_gemm.variant.baseline": 1.0 if self.variant == "baseline" else 0.0,
            "blackwell_grouped_gemm.variant.large_tiles": 1.0 if self.variant == "large_tiles" else 0.0,
            "blackwell_grouped_gemm.variant.full_stack": 1.0 if self.variant == "full_stack" else 0.0,
            "blackwell_grouped_gemm.variant.persistent": 1.0 if self.variant == "persistent" else 0.0,
            "blackwell_grouped_gemm.tile_m": float(schedule.block_m),
            "blackwell_grouped_gemm.tile_n": float(schedule.block_n),
            "blackwell_grouped_gemm.tile_k": float(schedule.block_k),
            "blackwell_grouped_gemm.group_m": float(schedule.group_m),
            "blackwell_grouped_gemm.num_warps": float(schedule.num_warps),
            "blackwell_grouped_gemm.num_stages": float(schedule.num_stages),
            "blackwell_grouped_gemm.fused_weights": 1.0 if schedule.fused_weights else 0.0,
            "blackwell_grouped_gemm.persistent": 1.0 if schedule.persistent else 0.0,
            "blackwell_grouped_gemm.num_experts": float(self.workload.num_experts),
            "blackwell_grouped_gemm.max_tokens_per_expert": float(self.state.max_count),
            "blackwell_grouped_gemm.histogram.skewed": 1.0 if self.workload.histogram == "skewed" else 0.0,
            "blackwell_grouped_gemm.histogram.balanced": 1.0 if self.workload.histogram == "balanced" else 0.0,
            "blackwell_grouped_gemm.grouped_work_units": float(grouped_units),
            "story.chapter_native_exemplar": 1.0,
        }
        torch.cuda.synchronize(device)

    def benchmark_fn(self) -> None:
        if (
            self.state is None
            or self._flat_packed_tokens is None
            or self._packed_tokens_view is None
            or self._output_buffer is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode(), self._nvtx_range(self.label):
            result = run_variant(
                self.state,
                variant=self.variant,
                packed_tokens_flat=self._flat_packed_tokens,
                packed_tokens_view=self._packed_tokens_view,
                output_buffer=self._output_buffer,
            )
            self.output = result.output

    def capture_verification_payload(self) -> None:
        if self.state is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output before verification capture")
        if self._verify_output_buffer is None or self._verification_shape_tensor is None:
            raise RuntimeError("setup() must initialize verification buffers")
        verification_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(verification_slice)
        self._set_verification_payload(
            inputs={"shape": self._verification_shape_tensor},
            output=self._verify_output_buffer,
            batch_size=self.workload.num_tokens,
            parameter_count=int(self.state.expert_weights.numel()),
            precision_flags={
                "fp16": self.workload.dtype == torch.float16,
                "bf16": self.workload.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.state = None
        self.output = None
        self._flat_packed_tokens = None
        self._packed_tokens_view = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self._verification_shape_tensor = None
        torch.cuda.empty_cache()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            setup_timeout_seconds=900,
            measurement_timeout_seconds=600,
            profiling_timeout_seconds=900,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload_metadata

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def get_problem_shape(self) -> tuple[int, int, int]:
        return (self.state.max_count if self.state is not None else 0, self.workload.expert_ffn_dim, self.workload.hidden_dim)

    @property
    def tensor_dtype(self) -> torch.dtype:
        return self.workload.dtype

    def validate_result(self) -> Optional[str]:
        if self.state is None or self.output is None:
            return "benchmark_fn() did not produce an output tensor"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        diff = (self.output.float() - self.state.reference_output.float()).abs()
        max_diff = scalar_tensor_to_float(diff.max())
        if max_diff > 0.35:
            return f"Grouped GEMM output drifted from reference (max_abs_diff={max_diff:.4f})"
        return None
