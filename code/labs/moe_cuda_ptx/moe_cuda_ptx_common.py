"""Shared workload generation and benchmark logic for the MoE CUDA/PTX lab."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from labs.moe_cuda_ptx.moe_cuda_ptx_extension import ensure_moe_ptx_supported, load_moe_ptx_extension

resolve_device = lambda: require_cuda_device("MoE CUDA/PTX lab requires CUDA.")

_MXFP8_BLOCK_SIZE = 32
_MXFP8_E4M3_MAX = 448.0
_MXFP8_E8M0_MIN = float(2.0 ** -127)
_QUANT_VERIFY_MAX_NUMEL = 2 * (4 * 32 + 4 * 8)


@dataclass(frozen=True)
class GroupedMatmulWorkUnit:
    """Logical grouped matmul tile descriptor used for schedule metadata."""

    expert_idx: int
    row_block_idx: int
    col_block_idx: int
    reduction_block_start_idx: int
    reduction_block_end_idx: int


@dataclass
class MoECudaPtxWorkload:
    num_tokens: int = 32768
    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 7168
    expert_ffn_dim: int = 2048
    capacity_factor: float = 1.25
    mode: str = "forward"
    dtype: torch.dtype = torch.bfloat16
    histogram: str = "balanced"

    @property
    def routed_tokens(self) -> int:
        return int(self.num_tokens * self.top_k)

    @property
    def capacity_tokens_per_expert(self) -> int:
        average = self.routed_tokens / max(1, self.num_experts)
        return int(math.ceil(self.capacity_factor * average))

    @property
    def dtype_name(self) -> str:
        return "bf16" if self.dtype == torch.bfloat16 else "fp16"

    def validate(self) -> None:
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be > 1")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.hidden_dim <= 0 or self.expert_ffn_dim <= 0:
            raise ValueError("hidden_dim and expert_ffn_dim must be > 0")
        if self.capacity_factor < 1.0:
            raise ValueError("capacity_factor must be >= 1.0")
        if self.mode not in {"forward", "fwd_bwd"}:
            raise ValueError("mode must be 'forward' or 'fwd_bwd'")
        if self.histogram not in {"balanced", "skewed"}:
            raise ValueError("histogram must be 'balanced' or 'skewed'")


@dataclass
class MoELabState:
    x: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    loss_grad: torch.Tensor
    route_counts_cpu: tuple[int, ...] = ()


@dataclass
class PackedRoutes:
    packed_tokens: torch.Tensor
    packed_weights: torch.Tensor
    packed_weight_column: torch.Tensor
    token_indices: torch.Tensor
    combine_index: torch.Tensor
    expert_indices: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    padded_indices: torch.Tensor
    counts_cpu: tuple[int, ...]
    max_count: int
    uniform_count: int


@dataclass
class QuantizedMatrix:
    quantized: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]


@dataclass
class QuantizedBundle:
    forward: QuantizedMatrix
    transpose: Optional[QuantizedMatrix] = None


def _flat_topk_token_ids(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.long)
    if top_k > 1:
        token_ids.div_(top_k, rounding_mode="floor")
    return token_ids


def _workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--expert-ffn-dim", type=int, default=None)
    parser.add_argument("--capacity-factor", type=float, default=None)
    parser.add_argument("--mode", choices=("forward", "fwd_bwd"), default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--histogram", choices=("balanced", "skewed"), default=None)
    return parser


def apply_workload_overrides(workload: MoECudaPtxWorkload, argv: list[str]) -> MoECudaPtxWorkload:
    args, _ = _workload_parser().parse_known_args(argv)
    dtype = workload.dtype
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    updated = MoECudaPtxWorkload(
        num_tokens=args.num_tokens or workload.num_tokens,
        num_experts=args.num_experts or workload.num_experts,
        top_k=args.top_k or workload.top_k,
        hidden_dim=args.hidden_dim or workload.hidden_dim,
        expert_ffn_dim=args.expert_ffn_dim or workload.expert_ffn_dim,
        capacity_factor=args.capacity_factor or workload.capacity_factor,
        mode=args.mode or workload.mode,
        dtype=dtype,
        histogram=args.histogram or workload.histogram,
    )
    updated.validate()
    return updated


def _counts_from_weights(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    weight_sum = sum(weights)
    normalized = [weight / weight_sum * float(total) for weight in weights]
    base = [math.floor(value) for value in normalized]
    remainder = total - sum(base)
    if remainder > 0:
        fractional_order = sorted(
            range(len(base)),
            key=lambda expert_idx: (-(normalized[expert_idx] - base[expert_idx]), expert_idx),
        )
        for expert_idx in fractional_order[:remainder]:
            base[expert_idx] += 1
    return tuple(base)


def _primary_route_counts_cpu(workload: MoECudaPtxWorkload) -> tuple[int, ...]:
    if workload.histogram == "balanced":
        base, remainder = divmod(workload.num_tokens, workload.num_experts)
        return tuple(
            base + int(expert_idx < remainder)
            for expert_idx in range(workload.num_experts)
        )

    first_weight = workload.capacity_factor
    last_weight = max(0.25, 2.0 - workload.capacity_factor)
    step = (last_weight - first_weight) / (workload.num_experts - 1)
    weights = [first_weight + step * expert_idx for expert_idx in range(workload.num_experts)]
    return _counts_from_weights(workload.num_tokens, weights)


def _primary_routes_cpu(workload: MoECudaPtxWorkload) -> torch.Tensor:
    if workload.histogram == "balanced":
        return torch.arange(workload.num_tokens, dtype=torch.long) % workload.num_experts

    counts_cpu = _primary_route_counts_cpu(workload)
    expert_ids = torch.arange(workload.num_experts, dtype=torch.long)
    repeats = torch.tensor(counts_cpu, dtype=torch.long)
    return torch.repeat_interleave(
        expert_ids,
        repeats,
        output_size=workload.num_tokens,
    )


def _route_counts_cpu(workload: MoECudaPtxWorkload) -> tuple[int, ...]:
    primary_counts = _primary_route_counts_cpu(workload)
    counts = list(primary_counts)
    if workload.histogram == "balanced":
        for token_idx in range(workload.num_tokens):
            primary_expert = token_idx % workload.num_experts
            secondary_expert = (token_idx * 3 + 1) % workload.num_experts
            if secondary_expert == primary_expert:
                secondary_expert = (secondary_expert + 1) % workload.num_experts
            counts[secondary_expert] += 1
    else:
        token_idx = 0
        for primary_expert, primary_count in enumerate(primary_counts):
            for _ in range(primary_count):
                secondary_expert = (token_idx * 3 + 1) % workload.num_experts
                if secondary_expert == primary_expert:
                    secondary_expert = (secondary_expert + 1) % workload.num_experts
                counts[secondary_expert] += 1
                token_idx += 1
    return tuple(counts)


def _build_primary_routes(workload: MoECudaPtxWorkload, device: torch.device) -> torch.Tensor:
    if workload.histogram == "balanced":
        return torch.arange(workload.num_tokens, device=device, dtype=torch.long) % workload.num_experts

    counts_cpu = _primary_route_counts_cpu(workload)
    expert_ids = torch.arange(workload.num_experts, device=device, dtype=torch.long)
    repeats = torch.tensor(counts_cpu, device=device, dtype=torch.long)
    return torch.repeat_interleave(
        expert_ids,
        repeats,
        output_size=workload.num_tokens,
    )


def _build_routes_with_counts(
    workload: MoECudaPtxWorkload,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    primary = _build_primary_routes(workload, device)
    secondary = (torch.arange(workload.num_tokens, device=device, dtype=torch.long) * 3 + 1) % workload.num_experts
    collision = secondary == primary
    secondary[collision] = (secondary[collision] + 1) % workload.num_experts

    expert_indices = torch.stack([primary, secondary], dim=1)

    token_positions = torch.arange(workload.num_tokens, device=device, dtype=torch.float32)
    logits = torch.stack(
        [
            1.25 + 0.05 * torch.sin(token_positions * 0.013),
            0.90 + 0.05 * torch.cos(token_positions * 0.017),
        ],
        dim=1,
    )
    expert_weights = torch.softmax(logits, dim=1).to(dtype=workload.dtype)

    route_counts_cpu = _route_counts_cpu(workload)
    max_count = max(route_counts_cpu, default=0)
    if max_count > workload.capacity_tokens_per_expert:
        raise RuntimeError(
            "Deterministic routing exceeded the configured capacity factor; "
            f"max_count={max_count}, capacity={workload.capacity_tokens_per_expert}"
        )
    return expert_indices, expert_weights, route_counts_cpu


def build_routes(workload: MoECudaPtxWorkload, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    expert_indices, expert_weights, _ = _build_routes_with_counts(workload, device)
    return expert_indices, expert_weights


def build_state(workload: MoECudaPtxWorkload, device: torch.device) -> MoELabState:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(314159)

    x = torch.randn(workload.num_tokens, workload.hidden_dim, generator=generator, dtype=torch.float32)
    x += torch.linspace(0.0, 1e-3, steps=workload.hidden_dim, dtype=torch.float32).view(1, -1)
    x = x.to(device=device, dtype=workload.dtype).contiguous()

    expert_indices, expert_weights, route_counts_cpu = _build_routes_with_counts(workload, device)

    gate_proj = (
        torch.randn(
            workload.num_experts,
            workload.hidden_dim,
            workload.expert_ffn_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(device=device, dtype=workload.dtype)
    up_proj = (
        torch.randn(
            workload.num_experts,
            workload.hidden_dim,
            workload.expert_ffn_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(device=device, dtype=workload.dtype)
    down_proj = (
        torch.randn(
            workload.num_experts,
            workload.expert_ffn_dim,
            workload.hidden_dim,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
    ).to(device=device, dtype=workload.dtype)

    loss_grad = torch.randn(
        workload.num_tokens,
        workload.hidden_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=workload.dtype)

    return MoELabState(
        x=x,
        expert_indices=expert_indices.contiguous(),
        expert_weights=expert_weights.contiguous(),
        gate_proj=gate_proj.contiguous(),
        up_proj=up_proj.contiguous(),
        down_proj=down_proj.contiguous(),
        loss_grad=loss_grad.contiguous(),
        route_counts_cpu=route_counts_cpu,
    )


def pack_topk_routes(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    num_experts: int,
    counts_cpu: Optional[Sequence[int]] = None,
) -> PackedRoutes:
    top_k = expert_indices.shape[1]
    flat_experts = expert_indices.reshape(-1)
    flat_weights = expert_weights.reshape(-1)
    flat_token_ids = _flat_topk_token_ids(x.shape[0], top_k, x.device)
    sort_order = torch.argsort(flat_experts, stable=True)

    sorted_token_ids = flat_token_ids.index_select(0, sort_order)
    sorted_expert_ids = flat_experts.index_select(0, sort_order)
    sorted_weights = flat_weights.index_select(0, sort_order)
    packed_tokens = x.index_select(0, sorted_token_ids).contiguous()

    if counts_cpu is None:
        counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
        counts_host = counts.detach().cpu()
        counts_cpu = tuple(int(counts_host[idx]) for idx in range(counts_host.numel()))
    else:
        counts_cpu = tuple(int(count) for count in counts_cpu)
        counts = torch.tensor(counts_cpu, device=x.device, dtype=torch.long)
    cumsum = counts.cumsum(dim=0)
    starts = torch.empty_like(counts)
    starts[0] = 0
    if starts.numel() > 1:
        starts[1:].copy_(cumsum[:-1])
    positions = torch.arange(sorted_expert_ids.numel(), device=x.device, dtype=torch.long) - starts.index_select(
        0, sorted_expert_ids
    )
    max_count = max(counts_cpu, default=0)
    uniform_count = counts_cpu[0] if counts_cpu and all(count == counts_cpu[0] for count in counts_cpu) else 0
    padded_indices = sorted_expert_ids * max_count + positions

    packed_weights = sorted_weights.contiguous()
    token_indices = sorted_token_ids.contiguous()

    return PackedRoutes(
        packed_tokens=packed_tokens,
        packed_weights=packed_weights,
        packed_weight_column=packed_weights.unsqueeze(-1),
        token_indices=token_indices,
        combine_index=token_indices.unsqueeze(-1).expand(-1, x.shape[1]),
        expert_indices=sorted_expert_ids.contiguous(),
        counts=counts.contiguous(),
        starts=starts.contiguous(),
        padded_indices=padded_indices.contiguous(),
        counts_cpu=counts_cpu,
        max_count=max_count,
        uniform_count=uniform_count,
    )


def grouped_ffn_reference(
    packed_tokens: torch.Tensor,
    counts_cpu: Sequence[int],
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = output_buffer
    if output is None:
        output = torch.empty(packed_tokens.shape[0], down_proj.shape[-1], device=packed_tokens.device, dtype=packed_tokens.dtype)
    offset = 0
    for expert_idx, count in enumerate(counts_cpu):
        if count == 0:
            continue
        tokens_e = packed_tokens[offset : offset + count]
        gate = tokens_e @ gate_proj[expert_idx]
        up = tokens_e @ up_proj[expert_idx]
        hidden = _silu_mul_in_place_if_safe(gate, up)
        output[offset : offset + count] = hidden @ down_proj[expert_idx]
        offset += count
    return output


def _silu_mul_in_place_if_safe(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and gate.requires_grad:
        return F.silu(gate) * up
    F.silu(gate, inplace=True)
    gate.mul_(up)
    return gate


def _weight_routes_in_place_if_safe(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and (out.requires_grad or weights.requires_grad):
        return out * weights
    out.mul_(weights)
    return out


def grouped_ffn_cuda(
    packed_tokens: torch.Tensor,
    packed: PackedRoutes,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    padded_tokens_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if packed.max_count == 0:
        return torch.empty_like(packed_tokens)

    device = packed_tokens.device
    num_experts = gate_proj.shape[0]
    hidden_dim = packed_tokens.shape[1]
    output_dim = down_proj.shape[-1]

    if packed.uniform_count == packed.max_count and packed.max_count > 0:
        grouped_tokens = packed_tokens.view(num_experts, packed.max_count, hidden_dim)
        gate = torch.bmm(grouped_tokens, gate_proj)
        up = torch.bmm(grouped_tokens, up_proj)
        hidden = _silu_mul_in_place_if_safe(gate, up)
        return torch.bmm(hidden, down_proj).reshape(-1, output_dim)

    flat_slots = num_experts * packed.max_count

    padded_tokens = padded_tokens_buffer
    padded_numel = flat_slots * hidden_dim
    if (
        padded_tokens is None
        or padded_tokens.device != device
        or padded_tokens.dtype != packed_tokens.dtype
        or not padded_tokens.is_contiguous()
        or padded_tokens.numel() < padded_numel
    ):
        padded_tokens = torch.empty(padded_numel, device=device, dtype=packed_tokens.dtype)
    padded_tokens = padded_tokens.view(-1)[:padded_numel].view(flat_slots, hidden_dim)
    padded_tokens.index_copy_(0, packed.padded_indices, packed_tokens)
    padded_tokens = padded_tokens.view(num_experts, packed.max_count, hidden_dim)

    gate = torch.bmm(padded_tokens, gate_proj)
    up = torch.bmm(padded_tokens, up_proj)
    hidden = _silu_mul_in_place_if_safe(gate, up)
    out = torch.bmm(hidden, down_proj)
    flat_out = out.reshape(flat_slots, output_dim)
    return flat_out.index_select(0, packed.padded_indices).contiguous()


def combine_weighted_outputs(
    sorted_outputs: torch.Tensor,
    packed: PackedRoutes,
    num_tokens: int,
    *,
    output_buffer: Optional[torch.Tensor] = None,
    consume_sorted_outputs: bool = False,
) -> torch.Tensor:
    combined = output_buffer
    output_shape = (int(num_tokens), int(sorted_outputs.shape[1]))
    output_numel = output_shape[0] * output_shape[1]
    if (
        combined is None
        or combined.device != sorted_outputs.device
        or combined.dtype != sorted_outputs.dtype
        or not combined.is_contiguous()
        or combined.numel() < output_numel
    ):
        combined = torch.empty(output_numel, device=sorted_outputs.device, dtype=sorted_outputs.dtype)
    combined = combined.view(-1)[:output_numel].view(output_shape)
    weighted_outputs = sorted_outputs
    weights = getattr(packed, "packed_weight_column", None)
    if weights is None:
        weights = packed.packed_weights.unsqueeze(-1)
    if consume_sorted_outputs:
        weighted_outputs.mul_(weights)
    else:
        weighted_outputs = sorted_outputs * weights
    combine_index = getattr(packed, "combine_index", None)
    if combine_index is None or tuple(combine_index.shape) != tuple(weighted_outputs.shape):
        combine_index = packed.token_indices.unsqueeze(-1).expand_as(weighted_outputs)
    combined.scatter_reduce_(0, combine_index, weighted_outputs, reduce="sum", include_self=False)
    return combined


def run_layer_baseline(
    state: MoELabState,
    workload: MoECudaPtxWorkload,
    *,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output = output_buffer if output_buffer is not None else torch.empty_like(state.x)
    output.zero_()
    for expert_idx in range(workload.num_experts):
        for slot_idx in range(workload.top_k):
            token_ids = (state.expert_indices[:, slot_idx] == expert_idx).nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue
            tokens_e = state.x[token_ids]
            gate = tokens_e @ state.gate_proj[expert_idx]
            up = tokens_e @ state.up_proj[expert_idx]
            hidden = _silu_mul_in_place_if_safe(gate, up)
            expert_out = hidden @ state.down_proj[expert_idx]
            route_weights = state.expert_weights[token_ids, slot_idx].unsqueeze(-1)
            expert_out = _weight_routes_in_place_if_safe(expert_out, route_weights)
            output[token_ids] += expert_out
    return output


def run_layer_cuda(
    state: MoELabState,
    workload: MoECudaPtxWorkload,
    *,
    packed: Optional[PackedRoutes] = None,
    combined_buffer: Optional[torch.Tensor] = None,
    padded_tokens_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    route_counts_cpu = getattr(state, "route_counts_cpu", None) or None
    packed = packed or pack_topk_routes(
        state.x,
        state.expert_indices,
        state.expert_weights,
        num_experts=workload.num_experts,
        counts_cpu=route_counts_cpu,
    )
    # Keep the standalone quantization surface on `moe_quant` until the layer path
    # has a real low-precision kernel that benefits from quantized activations.
    grouped_tokens = packed.packed_tokens
    sorted_outputs = grouped_ffn_cuda(
        grouped_tokens,
        packed,
        state.gate_proj,
        state.up_proj,
        state.down_proj,
        padded_tokens_buffer=padded_tokens_buffer,
    )
    return combine_weighted_outputs(
        sorted_outputs,
        packed,
        workload.num_tokens,
        output_buffer=combined_buffer,
        consume_sorted_outputs=True,
    )


def _compute_scale_blocks(matrix: torch.Tensor) -> tuple[torch.Tensor, int]:
    pad = (-matrix.shape[1]) % _MXFP8_BLOCK_SIZE
    if pad:
        matrix = F.pad(matrix, (0, pad))
    return matrix, pad


def _pow2_scales(blocks: torch.Tensor) -> torch.Tensor:
    amax = blocks.abs().amax(dim=-1)
    unclamped = (amax / _MXFP8_E4M3_MAX).clamp_min(_MXFP8_E8M0_MIN)
    scales = torch.pow(2.0, torch.ceil(torch.log2(unclamped)))
    scales.masked_fill_(amax <= 0, _MXFP8_E8M0_MIN)
    return scales


def _quantize_matrix(matrix: torch.Tensor) -> QuantizedMatrix:
    padded, _ = _compute_scale_blocks(matrix)
    rows, padded_cols = padded.shape
    blocks = padded.to(torch.float32).reshape(rows, padded_cols // _MXFP8_BLOCK_SIZE, _MXFP8_BLOCK_SIZE)
    scales_fp32 = _pow2_scales(blocks)
    normalized = (blocks / scales_fp32.unsqueeze(-1)).clamp(-_MXFP8_E4M3_MAX, _MXFP8_E4M3_MAX)
    quantized = normalized.to(torch.float8_e4m3fn).reshape(rows, padded_cols).contiguous()
    scales = scales_fp32.to(torch.float8_e8m0fnu).contiguous()
    return QuantizedMatrix(quantized=quantized, scales=scales, original_shape=tuple(matrix.shape))


def _expand_mxfp8_scales(scales: torch.Tensor) -> torch.Tensor:
    scales_fp32 = scales.to(torch.float32)
    return scales_fp32.unsqueeze(-1).expand(*scales_fp32.shape, _MXFP8_BLOCK_SIZE).reshape(
        scales_fp32.shape[0],
        scales_fp32.shape[1] * _MXFP8_BLOCK_SIZE,
    )


def quantize_mxfp8_reference(matrix: torch.Tensor, *, include_transpose: bool) -> QuantizedBundle:
    forward = _quantize_matrix(matrix)
    # Reference path pays the reshape tax explicitly to reflect the cost of
    # materializing a tcgen05-style scale layout from a generic quantizer.
    _ = _expand_mxfp8_scales(forward.scales)
    transpose = None
    if include_transpose:
        transpose = _quantize_matrix(matrix.t().contiguous())
        _ = _expand_mxfp8_scales(transpose.scales)
    return QuantizedBundle(forward=forward, transpose=transpose)


def quantize_mxfp8_optimized(matrix: torch.Tensor, *, include_transpose: bool) -> QuantizedBundle:
    forward = _quantize_matrix(matrix)
    transpose = _quantize_matrix(matrix.t().contiguous()) if include_transpose else None
    return QuantizedBundle(forward=forward, transpose=transpose)


def dequantize_mxfp8(qmat: QuantizedMatrix, *, dtype: torch.dtype) -> torch.Tensor:
    scales = _expand_mxfp8_scales(qmat.scales)
    values = qmat.quantized.to(torch.float32) * scales[:, : qmat.quantized.shape[1]]
    rows, cols = qmat.original_shape
    return values[:rows, :cols].to(dtype=dtype)


def _quant_verification_pieces(bundle: QuantizedBundle) -> tuple[torch.Tensor, ...]:
    pieces: list[torch.Tensor] = [
        bundle.forward.quantized[:4, :32],
        bundle.forward.scales[:4, :8],
    ]
    if bundle.transpose is not None:
        pieces.append(bundle.transpose.quantized[:4, :32])
        pieces.append(bundle.transpose.scales[:4, :8])
    return tuple(pieces)


def quant_verification_numel(bundle: QuantizedBundle) -> int:
    return sum(piece.numel() for piece in _quant_verification_pieces(bundle))


def build_quant_verification_tensor(bundle: QuantizedBundle, out: torch.Tensor) -> torch.Tensor:
    verification = out[: quant_verification_numel(bundle)]
    offset = 0
    for piece in _quant_verification_pieces(bundle):
        flat = piece.reshape(-1)
        next_offset = offset + flat.numel()
        verification[offset:next_offset].copy_(flat)
        offset = next_offset
    return verification


def build_tensor_slice_verification(
    output: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    rows = min(32, output.shape[0])
    cols = min(32, output.shape[1])
    output_slice = output[:rows, :cols]
    verification = out[: rows * cols]
    verification.view(rows, cols).copy_(output_slice)
    return verification


def build_backward_verification(
    output: torch.Tensor,
    x_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    down_grad: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    verification = out[: backward_verification_numel(output, x_grad, gate_grad, down_grad)]
    offset = 0
    for piece in (
        output[: min(8, output.shape[0]), : min(16, output.shape[1])],
        x_grad[: min(8, x_grad.shape[0]), : min(16, x_grad.shape[1])],
        gate_grad[0, : min(8, gate_grad.shape[1]), : min(8, gate_grad.shape[2])],
        down_grad[0, : min(8, down_grad.shape[1]), : min(8, down_grad.shape[2])],
    ):
        flat = piece.reshape(-1)
        next_offset = offset + flat.numel()
        verification[offset:next_offset].copy_(flat)
        offset = next_offset
    return verification


def backward_verification_numel(
    output: torch.Tensor,
    x_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    down_grad: torch.Tensor,
) -> int:
    return (
        min(8, output.shape[0]) * min(16, output.shape[1])
        + min(8, x_grad.shape[0]) * min(16, x_grad.shape[1])
        + min(8, gate_grad.shape[1]) * min(8, gate_grad.shape[2])
        + min(8, down_grad.shape[1]) * min(8, down_grad.shape[2])
    )


def grouped_work_unit_count(
    counts_cpu: Sequence[int],
    *,
    output_dim: int,
    reduction_dim: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
) -> int:
    total = 0
    col_blocks = math.ceil(output_dim / tile_n)
    red_blocks = math.ceil(reduction_dim / tile_k)
    for count in counts_cpu:
        row_blocks = math.ceil(count / tile_m) if count > 0 else 0
        total += row_blocks * col_blocks * red_blocks
    return int(total)


class MoECudaPtxBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark a routed top-2 SwiGLU MoE FFN across multiple surfaces."""

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, target: str, backend: str, label: str) -> None:
        super().__init__()
        if target not in {"moe_quant", "moe_grouped_gemm_fwd", "moe_grouped_gemm_bwd", "moe_layer"}:
            raise ValueError(f"Unsupported target: {target}")
        if backend not in {"baseline", "cuda", "ptx"}:
            raise ValueError(f"Unsupported backend: {backend}")
        self.target = target
        self.backend = backend
        self.label = label
        self.workload = MoECudaPtxWorkload()
        self.state: Optional[MoELabState] = None
        self.packed: Optional[PackedRoutes] = None
        self.outputs: Optional[torch.Tensor] = None
        self.quantized: Optional[QuantizedBundle] = None
        self.x_grad: Optional[torch.Tensor] = None
        self.gate_grad: Optional[torch.Tensor] = None
        self.down_grad: Optional[torch.Tensor] = None
        self._bwd_x: Optional[torch.Tensor] = None
        self._bwd_tokens: Optional[torch.Tensor] = None
        self._bwd_gate_proj: Optional[torch.Tensor] = None
        self._bwd_up_proj: Optional[torch.Tensor] = None
        self._bwd_down_proj: Optional[torch.Tensor] = None
        self._packed_loss_grad: Optional[torch.Tensor] = None
        self._combined_buffer: Optional[torch.Tensor] = None
        self._grouped_output_buffer: Optional[torch.Tensor] = None
        self._padded_tokens_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._quant_verify_output_buffer: Optional[torch.Tensor] = None
        self._backward_verify_output_buffer: Optional[torch.Tensor] = None
        self._quant_shape_tensor: Optional[torch.Tensor] = None
        self._verification_shape_tensor: Optional[torch.Tensor] = None
        self._routing_verification_tensor: Optional[torch.Tensor] = None
        self._benchmark_impl: Optional[Callable[[], None]] = None
        self._custom_metrics: dict[str, float] = {}
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        if self.target == "moe_quant":
            self._workload = WorkloadMetadata(
                requests_per_iteration=float(self.workload.num_tokens),
                tokens_per_iteration=float(self.workload.routed_tokens),
                bytes_per_iteration=float(self.workload.routed_tokens * self.workload.hidden_dim * 2),
            )
        elif self.target == "moe_layer":
            self._workload = WorkloadMetadata(
                requests_per_iteration=float(self.workload.num_tokens),
                tokens_per_iteration=float(self.workload.num_tokens),
                custom_units_per_iteration=float(
                    3 * self.workload.routed_tokens * self.workload.hidden_dim * self.workload.expert_ffn_dim * 2
                ),
                custom_unit_name="FLOPs",
            )
        else:
            self._workload = WorkloadMetadata(
                requests_per_iteration=float(self.workload.routed_tokens),
                tokens_per_iteration=float(self.workload.routed_tokens),
                custom_units_per_iteration=float(
                    3 * self.workload.routed_tokens * self.workload.hidden_dim * self.workload.expert_ffn_dim * 2
                ),
                custom_unit_name="FLOPs",
            )

    def _force_target_mode(self) -> None:
        if self.target == "moe_grouped_gemm_fwd":
            self.workload.mode = "forward"
        elif self.target == "moe_grouped_gemm_bwd":
            self.workload.mode = "fwd_bwd"

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_workload_overrides(self.workload, argv)
        self._force_target_mode()
        self._refresh_workload_metadata()

    def _populate_metrics(self, counts_cpu: Sequence[int]) -> None:
        gate_units = grouped_work_unit_count(
            counts_cpu,
            output_dim=self.workload.expert_ffn_dim,
            reduction_dim=self.workload.hidden_dim,
        )
        down_units = grouped_work_unit_count(
            counts_cpu,
            output_dim=self.workload.hidden_dim,
            reduction_dim=self.workload.expert_ffn_dim,
        )
        max_count = float(max(counts_cpu, default=0))
        min_count = float(min(counts_cpu, default=0))
        mean_count = float(sum(counts_cpu) / len(counts_cpu)) if counts_cpu else 0.0
        self._custom_metrics = {
            "moe.backend.baseline": 1.0 if self.backend == "baseline" else 0.0,
            "moe.backend.cuda": 1.0 if self.backend == "cuda" else 0.0,
            "moe.backend.ptx": 1.0 if self.backend == "ptx" else 0.0,
            "moe.histogram.balanced": 1.0 if self.workload.histogram == "balanced" else 0.0,
            "moe.histogram.skewed": 1.0 if self.workload.histogram == "skewed" else 0.0,
            "moe.mode.forward_only": 1.0 if self.workload.mode == "forward" else 0.0,
            "moe.mode.fwd_bwd": 1.0 if self.workload.mode == "fwd_bwd" else 0.0,
            "moe.route.max_tokens_per_expert": max_count,
            "moe.route.min_tokens_per_expert": min_count,
            "moe.route.mean_tokens_per_expert": mean_count,
            "moe.route.capacity_limit": float(self.workload.capacity_tokens_per_expert),
            "moe.grouped_work_units.gate": float(gate_units),
            "moe.grouped_work_units.down": float(down_units),
        }

    def setup(self) -> None:
        self.workload.validate()
        self._force_target_mode()
        self.state = build_state(self.workload, self.device)
        route_counts_cpu = self.state.route_counts_cpu
        self._populate_metrics(route_counts_cpu)

        self.packed = None
        prepack_routes = self.target != "moe_layer" or (
            self.backend == "cuda" and self.workload.mode == "forward"
        )
        if prepack_routes:
            self.packed = pack_topk_routes(
                self.state.x,
                self.state.expert_indices,
                self.state.expert_weights,
                num_experts=self.workload.num_experts,
                counts_cpu=route_counts_cpu,
            )

        self.outputs = None
        self.quantized = None
        self.x_grad = None
        self.gate_grad = None
        self.down_grad = None
        self._bwd_x = None
        self._bwd_tokens = None
        self._bwd_gate_proj = None
        self._bwd_up_proj = None
        self._bwd_down_proj = None
        self._packed_loss_grad = None
        self._combined_buffer = torch.empty_like(self.state.x)
        self._grouped_output_buffer = None
        self._padded_tokens_buffer = None
        verify_rows = min(32, max(self.workload.num_tokens, self.workload.routed_tokens))
        verify_cols = min(32, self.workload.hidden_dim)
        self._verify_output_buffer = torch.empty(
            verify_rows * verify_cols,
            device=self.device,
            dtype=torch.float32,
        )
        self._quant_verify_output_buffer = torch.empty(
            _QUANT_VERIFY_MAX_NUMEL,
            device=self.device,
            dtype=torch.float32,
        )
        self._backward_verify_output_buffer = torch.empty(
            backward_verification_numel(
                self.state.x,
                self.state.x,
                self.state.gate_proj,
                self.state.down_proj,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        self._quant_shape_tensor = torch.tensor(
            [
                self.workload.num_tokens,
                self.workload.hidden_dim,
                self.workload.top_k,
                self.workload.num_experts,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._verification_shape_tensor = torch.tensor(
            [
                self.workload.num_tokens,
                self.workload.hidden_dim,
                self.workload.expert_ffn_dim,
                self.workload.num_experts,
                self.workload.top_k,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._routing_verification_tensor = self.state.expert_indices[
            : min(32, self.state.expert_indices.shape[0])
        ].detach().cpu()

        if self.packed is not None:
            self._grouped_output_buffer = torch.empty(
                self.packed.packed_tokens.shape[0],
                self.workload.hidden_dim,
                device=self.device,
                dtype=self.workload.dtype,
            )
            if self.packed.max_count > 0:
                flat_slots = self.workload.num_experts * self.packed.max_count
                self._padded_tokens_buffer = torch.empty(
                    flat_slots,
                    self.workload.hidden_dim,
                    device=self.device,
                    dtype=self.workload.dtype,
                )

        if self.target == "moe_grouped_gemm_bwd" or (
            self.target == "moe_layer" and self.workload.mode == "fwd_bwd"
        ):
            self._prepare_backward_tensors()

        if self.backend == "ptx":
            ensure_moe_ptx_supported()
            load_moe_ptx_extension()

        # Warm the selected execution path outside the measured region.
        if self.target == "moe_quant" and self.packed is not None:
            if self.backend == "baseline":
                _ = quantize_mxfp8_reference(self.packed.packed_tokens, include_transpose=self.workload.mode == "fwd_bwd")
            elif self.backend == "cuda":
                _ = quantize_mxfp8_optimized(self.packed.packed_tokens, include_transpose=self.workload.mode == "fwd_bwd")
            else:
                raise RuntimeError("SKIPPED: PTX quant backend scaffold exists, but kernels are not implemented yet.")
        elif self.target == "moe_grouped_gemm_fwd" and self.packed is not None and self.state is not None:
            if self.backend == "baseline":
                _ = grouped_ffn_reference(
                    self.packed.packed_tokens,
                    self.packed.counts_cpu,
                    self.state.gate_proj,
                    self.state.up_proj,
                    self.state.down_proj,
                    output_buffer=self._grouped_output_buffer,
                )
            elif self.backend == "cuda":
                _ = grouped_ffn_cuda(
                    self.packed.packed_tokens,
                    self.packed,
                    self.state.gate_proj,
                    self.state.up_proj,
                    self.state.down_proj,
                    padded_tokens_buffer=self._padded_tokens_buffer,
                )
            else:
                raise RuntimeError("SKIPPED: PTX grouped GEMM backend scaffold exists, but kernels are not implemented yet.")
        elif self.target == "moe_grouped_gemm_bwd" and self.packed is not None and self.state is not None:
            _ = self._run_grouped_gemm_backward()
        elif self.target == "moe_layer" and self.state is not None:
            if self.backend == "baseline":
                if self.workload.mode == "fwd_bwd":
                    _ = self._run_layer_backward()
                else:
                    _ = run_layer_baseline(self.state, self.workload, output_buffer=self._combined_buffer)
            elif self.backend == "cuda":
                if self.workload.mode == "fwd_bwd":
                    _ = self._run_layer_backward()
                else:
                    _ = run_layer_cuda(
                        self.state,
                        self.workload,
                        packed=self.packed,
                        combined_buffer=self._combined_buffer,
                        padded_tokens_buffer=self._padded_tokens_buffer,
                    )
            else:
                raise RuntimeError("SKIPPED: PTX layer backend scaffold exists, but kernels are not implemented yet.")
        self._synchronize()
        self._benchmark_impl = self._select_benchmark_impl()

    def _prepare_backward_tensors(self) -> None:
        if self.state is None:
            raise RuntimeError("setup() must build state before preparing backward tensors")
        self._bwd_gate_proj = self.state.gate_proj.detach().clone().requires_grad_(True)
        self._bwd_up_proj = self.state.up_proj.detach().clone().requires_grad_(True)
        self._bwd_down_proj = self.state.down_proj.detach().clone().requires_grad_(True)
        if self.target == "moe_grouped_gemm_bwd":
            if self.packed is None:
                raise RuntimeError("Packed routes must be initialized before preparing grouped backward tensors")
            self._bwd_tokens = self.packed.packed_tokens.detach().clone().requires_grad_(True)
            self._packed_loss_grad = self.state.loss_grad.index_select(0, self.packed.token_indices).contiguous()
        elif self.target == "moe_layer":
            self._bwd_x = self.state.x.detach().clone().requires_grad_(True)

    def _clear_backward_grads(self) -> None:
        for tensor in (
            self._bwd_x,
            self._bwd_tokens,
            self._bwd_gate_proj,
            self._bwd_up_proj,
            self._bwd_down_proj,
        ):
            if tensor is not None:
                tensor.grad = None

    def _run_grouped_gemm_backward(self) -> torch.Tensor:
        if (
            self.packed is None
            or self.state is None
            or self._bwd_tokens is None
            or self._bwd_gate_proj is None
            or self._bwd_up_proj is None
            or self._bwd_down_proj is None
            or self._packed_loss_grad is None
        ):
            raise RuntimeError("Grouped GEMM backward tensors are not initialized")
        self._clear_backward_grads()
        if self.backend == "baseline":
            sorted_out = grouped_ffn_reference(
                self._bwd_tokens,
                self.packed.counts_cpu,
                self._bwd_gate_proj,
                self._bwd_up_proj,
                self._bwd_down_proj,
            )
        elif self.backend == "cuda":
            sorted_out = grouped_ffn_cuda(
                self._bwd_tokens,
                self.packed,
                self._bwd_gate_proj,
                self._bwd_up_proj,
                self._bwd_down_proj,
                padded_tokens_buffer=self._padded_tokens_buffer,
            )
        else:
            raise RuntimeError("SKIPPED: PTX grouped GEMM backend scaffold exists, but kernels are not implemented yet.")
        (sorted_out * self._packed_loss_grad).sum().backward()
        return sorted_out

    def _run_layer_backward(self) -> torch.Tensor:
        if (
            self.state is None
            or self._bwd_x is None
            or self._bwd_gate_proj is None
            or self._bwd_up_proj is None
            or self._bwd_down_proj is None
        ):
            raise RuntimeError("Layer backward tensors are not initialized")
        self._clear_backward_grads()
        state = MoELabState(
            x=self._bwd_x,
            expert_indices=self.state.expert_indices,
            expert_weights=self.state.expert_weights,
            gate_proj=self._bwd_gate_proj,
            up_proj=self._bwd_up_proj,
            down_proj=self._bwd_down_proj,
            loss_grad=self.state.loss_grad,
            route_counts_cpu=self.state.route_counts_cpu,
        )
        if self.backend == "baseline":
            output = run_layer_baseline(state, self.workload)
        elif self.backend == "cuda":
            output = run_layer_cuda(state, self.workload)
        else:
            raise RuntimeError("SKIPPED: PTX layer backend scaffold exists, but kernels are not implemented yet.")
        (output * self.state.loss_grad).sum().backward()
        return output

    def _select_benchmark_impl(self) -> Callable[[], None]:
        if self.target == "moe_quant":
            return self._benchmark_quant
        if self.target == "moe_grouped_gemm_fwd":
            return self._benchmark_grouped_gemm_fwd
        if self.target == "moe_grouped_gemm_bwd":
            return self._benchmark_grouped_gemm_bwd
        if self.workload.mode == "forward":
            return self._benchmark_layer_forward
        return self._benchmark_layer_fwd_bwd

    def _reset_outputs(self) -> None:
        self.outputs = None
        self.quantized = None
        self.x_grad = None
        self.gate_grad = None
        self.down_grad = None

    def _benchmark_quant(self) -> None:
        if self.packed is None:
            raise RuntimeError("Packed routes not initialized for quant target")
        if self.backend == "baseline":
            self.quantized = quantize_mxfp8_reference(
                self.packed.packed_tokens,
                include_transpose=self.workload.mode == "fwd_bwd",
            )
        elif self.backend == "cuda":
            self.quantized = quantize_mxfp8_optimized(
                self.packed.packed_tokens,
                include_transpose=self.workload.mode == "fwd_bwd",
            )
        else:
            raise RuntimeError("SKIPPED: PTX quant backend scaffold exists, but kernels are not implemented yet.")

    def _benchmark_grouped_gemm_fwd(self) -> None:
        if self.packed is None or self.state is None:
            raise RuntimeError("Grouped forward benchmark state is not initialized")
        if self.backend == "baseline":
            self.outputs = grouped_ffn_reference(
                self.packed.packed_tokens,
                self.packed.counts_cpu,
                self.state.gate_proj,
                self.state.up_proj,
                self.state.down_proj,
                output_buffer=self._grouped_output_buffer,
            )
        elif self.backend == "cuda":
            self.outputs = grouped_ffn_cuda(
                self.packed.packed_tokens,
                self.packed,
                self.state.gate_proj,
                self.state.up_proj,
                self.state.down_proj,
                padded_tokens_buffer=self._padded_tokens_buffer,
            )
        else:
            raise RuntimeError("SKIPPED: PTX grouped GEMM backend scaffold exists, but kernels are not implemented yet.")

    def _benchmark_grouped_gemm_bwd(self) -> None:
        if self.packed is None or self.state is None:
            raise RuntimeError("Grouped backward benchmark state is not initialized")
        sorted_out = self._run_grouped_gemm_backward()
        self.outputs = sorted_out.detach()
        if (
            self._bwd_tokens is None
            or self._bwd_tokens.grad is None
            or self._bwd_gate_proj is None
            or self._bwd_gate_proj.grad is None
            or self._bwd_down_proj is None
            or self._bwd_down_proj.grad is None
        ):
            raise RuntimeError("Grouped GEMM backward did not produce expected gradients")
        self.x_grad = self._bwd_tokens.grad.detach()
        self.gate_grad = self._bwd_gate_proj.grad.detach()
        self.down_grad = self._bwd_down_proj.grad.detach()

    def _benchmark_layer_forward(self) -> None:
        if self.state is None:
            raise RuntimeError("Layer benchmark state is not initialized")
        if self.backend == "baseline":
            self.outputs = run_layer_baseline(self.state, self.workload, output_buffer=self._combined_buffer)
        elif self.backend == "cuda":
            self.outputs = run_layer_cuda(
                self.state,
                self.workload,
                packed=self.packed,
                combined_buffer=self._combined_buffer,
                padded_tokens_buffer=self._padded_tokens_buffer,
            )
        else:
            raise RuntimeError("SKIPPED: PTX layer backend scaffold exists, but kernels are not implemented yet.")

    def _benchmark_layer_fwd_bwd(self) -> None:
        if self.state is None:
            raise RuntimeError("Layer benchmark state is not initialized")
        output = self._run_layer_backward()
        self.outputs = output.detach()
        if (
            self._bwd_x is None
            or self._bwd_x.grad is None
            or self._bwd_gate_proj is None
            or self._bwd_gate_proj.grad is None
            or self._bwd_down_proj is None
            or self._bwd_down_proj.grad is None
        ):
            raise RuntimeError("Layer backward did not produce expected gradients")
        self.x_grad = self._bwd_x.grad.detach()
        self.gate_grad = self._bwd_gate_proj.grad.detach()
        self.down_grad = self._bwd_down_proj.grad.detach()

    def benchmark_fn(self) -> None:
        if self._benchmark_impl is None:
            raise RuntimeError("Benchmark implementation was not initialized")
        self._reset_outputs()

        with self._nvtx_range(self.label):
            self._benchmark_impl()

    def capture_verification_payload(self) -> None:
        if self.state is None:
            raise RuntimeError("setup() must run before verification capture")

        mode = self.workload.mode
        if self.target == "moe_quant":
            if self.quantized is None:
                raise RuntimeError("benchmark_fn() did not produce quantized outputs")
            if self._quant_shape_tensor is None:
                raise RuntimeError("setup() must initialize quant verification shape tensor")
            if self._quant_verify_output_buffer is None:
                raise RuntimeError("setup() must initialize quant verification output buffer")
            self._set_verification_payload(
                inputs={"shape": self._quant_shape_tensor},
                output=build_quant_verification_tensor(
                    self.quantized,
                    self._quant_verify_output_buffer,
                ),
                batch_size=self.workload.num_tokens,
                parameter_count=0,
                precision_flags={"bf16": self.workload.dtype == torch.bfloat16, "fp16": self.workload.dtype == torch.float16},
                output_tolerance=(0.0, 0.0),
                signature_overrides={"quantization_mode": "mxfp8_block32"},
            )
            return

        if self.outputs is None:
            raise RuntimeError("benchmark_fn() did not produce outputs")
        if self._verification_shape_tensor is None or self._routing_verification_tensor is None:
            raise RuntimeError("setup() must initialize verification input tensors")

        inputs = {
            "shape": self._verification_shape_tensor,
            "routing": self._routing_verification_tensor,
        }

        if self.target == "moe_grouped_gemm_bwd" or mode == "fwd_bwd":
            if self.x_grad is None or self.gate_grad is None or self.down_grad is None:
                raise RuntimeError("Backward mode did not capture gradients")
            if self._backward_verify_output_buffer is None:
                raise RuntimeError("setup() must initialize backward verification buffer")
            verification = build_backward_verification(
                self.outputs,
                self.x_grad,
                self.gate_grad,
                self.down_grad,
                self._backward_verify_output_buffer,
            )
        else:
            verification = build_tensor_slice_verification(self.outputs, self._verify_output_buffer)

        tolerance = (2e-2, 2e-2)
        if self.target == "moe_layer" and mode == "forward":
            # The end-to-end layer surface intentionally quantizes routed
            # activations before grouped expert compute, so verification needs
            # to allow the expected FP8 roundtrip drift.
            tolerance = (5e-2, 2e-1)

        self._set_verification_payload(
            inputs=inputs,
            output=verification,
            batch_size=self.workload.num_tokens,
            parameter_count=int(
                self.state.gate_proj.numel() + self.state.up_proj.numel() + self.state.down_proj.numel()
            ),
            precision_flags={
                "bf16": self.workload.dtype == torch.bfloat16,
                "fp16": self.workload.dtype == torch.float16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=tolerance,
        )

    def teardown(self) -> None:
        self.state = None
        self.packed = None
        self.outputs = None
        self.quantized = None
        self.x_grad = None
        self.gate_grad = None
        self.down_grad = None
        self._bwd_x = None
        self._bwd_tokens = None
        self._bwd_gate_proj = None
        self._bwd_up_proj = None
        self._bwd_down_proj = None
        self._packed_loss_grad = None
        self._combined_buffer = None
        self._grouped_output_buffer = None
        self._padded_tokens_buffer = None
        self._verify_output_buffer = None
        self._quant_verify_output_buffer = None
        self._backward_verify_output_buffer = None
        self._quant_shape_tensor = None
        self._verification_shape_tensor = None
        self._routing_verification_tensor = None
        self._benchmark_impl = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        if self.target == "moe_quant":
            iterations = 10 if self.workload.mode == "forward" else 6
            warmup = 5
        elif self.target == "moe_grouped_gemm_fwd":
            iterations = 8
            warmup = 5
        elif self.target == "moe_grouped_gemm_bwd":
            iterations = 4
            warmup = 5
        else:
            iterations = 4 if self.workload.mode == "forward" else 2
            warmup = 5
        return BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            use_subprocess=self.target == "moe_grouped_gemm_bwd" or self.workload.mode == "fwd_bwd",
            setup_timeout_seconds=900,
            measurement_timeout_seconds=600,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def validate_result(self) -> Optional[str]:
        if self.target == "moe_quant":
            if self.quantized is None:
                return "Quantization target did not produce outputs"
            if self.quantized.forward.quantized.dtype != torch.float8_e4m3fn:
                return "Forward quantized tensor is not float8_e4m3fn"
            if self.quantized.forward.scales.dtype != torch.float8_e8m0fnu:
                return "Forward scale tensor is not float8_e8m0fnu"
            return None

        if self.outputs is None:
            return "benchmark_fn() did not produce output"
        if not torch.isfinite(self.outputs).all():
            return "Outputs contain non-finite values"
        if self.target == "moe_grouped_gemm_bwd" or self.workload.mode == "fwd_bwd":
            if self.x_grad is None or self.gate_grad is None or self.down_grad is None:
                return "Backward path did not produce gradients"
            if not torch.isfinite(self.x_grad).all():
                return "Input gradients contain non-finite values"
            if not torch.isfinite(self.gate_grad).all():
                return "Gate gradients contain non-finite values"
            if not torch.isfinite(self.down_grad).all():
                return "Down projection gradients contain non-finite values"
        return None
