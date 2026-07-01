"""Shared benchmarks and kernels for the GQA Top-K selection lab."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guard
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

resolve_device = lambda: require_cuda_device("Top-K selection lab requires CUDA.")


@dataclass
class TopKKernelWorkload:
    batch_size: int = 4
    heads: int = 8
    kv_heads: int = 1
    q_len: int = 32768
    compressed_k_len: int = 32768
    head_dim: int = 128
    top_k: int = 16
    selection_block_size: int = 64
    compress_stride: int = 1
    mode: str = "forward"
    dtype: torch.dtype = torch.float16

    @property
    def gqa_size(self) -> int:
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.heads % self.kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        return self.heads // self.kv_heads

    @property
    def positions_per_block(self) -> int:
        if self.compress_stride <= 0:
            raise ValueError("compress_stride must be positive")
        if self.selection_block_size % self.compress_stride != 0:
            raise ValueError(
                "selection_block_size must be divisible by compress_stride"
            )
        return self.selection_block_size // self.compress_stride

    @property
    def num_blocks(self) -> int:
        d = self.positions_per_block
        if self.compressed_k_len % d != 0:
            raise ValueError(
                "compressed_k_len must be divisible by selection_block_size // compress_stride"
            )
        return self.compressed_k_len // d

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(float(self.head_dim))

    @property
    def packed_q_tiles(self) -> int:
        return self.gqa_size

    @property
    def packed_kv_blocks(self) -> int:
        # The current Top-K-only refactor aggregates each selection block into one
        # block-key vector, so no multi-block K/V packing is materialized yet.
        return 1

    @property
    def cuda_q_tile(self) -> int:
        return 1024 if self.q_len >= 1024 else max(64, self.q_len)

    @property
    def cuda_tensorop_compatible(self) -> bool:
        return (
            self.dtype == torch.float16
            and self.head_dim % 8 == 0
            and self.num_blocks % 8 == 0
        )


@dataclass
class TopKKernelInputs:
    q: torch.Tensor
    k: torch.Tensor
    loss_weights: torch.Tensor


@dataclass
class TopKKernelOutputs:
    probs: torch.Tensor
    indices: torch.Tensor
    q_grad: Optional[torch.Tensor] = None
    k_grad: Optional[torch.Tensor] = None


def _workload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--q-len", type=int, default=None)
    parser.add_argument("--compressed-k-len", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--selection-block-size", type=int, default=None)
    parser.add_argument("--compress-stride", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("forward", "fwd_bwd"),
        default=None,
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default=None,
    )
    return parser


def apply_workload_overrides(
    workload: TopKKernelWorkload,
    argv: list[str],
) -> TopKKernelWorkload:
    args, _ = _workload_parser().parse_known_args(argv)
    dtype = workload.dtype
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    updated = TopKKernelWorkload(
        batch_size=args.batch_size or workload.batch_size,
        heads=args.heads or workload.heads,
        kv_heads=args.kv_heads or workload.kv_heads,
        q_len=args.q_len or workload.q_len,
        compressed_k_len=args.compressed_k_len or workload.compressed_k_len,
        head_dim=args.head_dim or workload.head_dim,
        top_k=args.top_k or workload.top_k,
        selection_block_size=args.selection_block_size or workload.selection_block_size,
        compress_stride=args.compress_stride or workload.compress_stride,
        mode=args.mode or workload.mode,
        dtype=dtype,
    )
    if updated.top_k > updated.num_blocks:
        raise ValueError("top_k must be <= number of selection blocks")
    _ = updated.gqa_size
    return updated


def build_inputs(workload: TopKKernelWorkload, device: torch.device) -> TopKKernelInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4242)

    q = torch.randn(
        workload.batch_size,
        workload.heads,
        workload.q_len,
        workload.head_dim,
        generator=generator,
        dtype=workload.dtype,
    )
    k = torch.randn(
        workload.batch_size,
        workload.kv_heads,
        workload.compressed_k_len,
        workload.head_dim,
        generator=generator,
        dtype=workload.dtype,
    )

    q += torch.linspace(
        0.0,
        1e-3,
        steps=workload.q_len,
        dtype=workload.dtype,
    ).view(1, 1, workload.q_len, 1)
    k += torch.linspace(
        0.0,
        1e-3,
        steps=workload.compressed_k_len,
        dtype=workload.dtype,
    ).view(1, 1, workload.compressed_k_len, 1)

    # Add a deterministic routing feature so Top-K cutoffs are not dominated by
    # near-ties across numerically different backends.
    q[..., 0] += 1.0
    block_bias = (
        torch.arange(workload.compressed_k_len, dtype=torch.int64)
        .div_(workload.positions_per_block, rounding_mode="floor")
        .to(torch.float32)
        .mul_(5e-3)
        .to(dtype=workload.dtype)
    )
    k[..., 0] += block_bias.view(1, 1, workload.compressed_k_len)

    q = q.to(device=device).contiguous()
    k = k.to(device=device).contiguous()
    loss_weights = torch.randn(
        workload.batch_size,
        workload.kv_heads,
        workload.q_len,
        workload.top_k,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device).contiguous()
    return TopKKernelInputs(q=q, k=k, loss_weights=loss_weights)


def _group_q(q: torch.Tensor, workload: TopKKernelWorkload) -> torch.Tensor:
    batch_size, _, q_len, head_dim = q.shape
    return (
        q.view(batch_size, workload.kv_heads, workload.gqa_size, q_len, head_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def _repeat_k_over_query_heads(k: torch.Tensor, workload: TopKKernelWorkload) -> torch.Tensor:
    batch_size, kv_heads, k_len, head_dim = k.shape
    return (
        k.view(batch_size, kv_heads, 1, k_len, head_dim)
        .expand(batch_size, kv_heads, workload.gqa_size, k_len, head_dim)
        .reshape(batch_size, kv_heads * workload.gqa_size, k_len, head_dim)
        .contiguous()
    )


def _build_block_k(k: torch.Tensor, workload: TopKKernelWorkload) -> torch.Tensor:
    batch, kv_heads, _, head_dim = k.shape
    return (
        k.float()
        .view(
            batch,
            kv_heads,
            workload.num_blocks,
            workload.positions_per_block,
            head_dim,
        )
        .sum(dim=3)
        .contiguous()
    )


def _build_block_k_reduced(
    k: torch.Tensor,
    workload: TopKKernelWorkload,
) -> torch.Tensor:
    batch, kv_heads, _, head_dim = k.shape
    return (
        k.view(
            batch,
            kv_heads,
            workload.num_blocks,
            workload.positions_per_block,
            head_dim,
        )
        .sum(dim=3, dtype=torch.float32)
        .contiguous()
    )


def _finalize_topk_from_block_scores(
    block_scores: torch.Tensor,
    workload: TopKKernelWorkload,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_values, topk_indices = torch.topk(block_scores, workload.top_k, dim=-1)
    probs = torch.softmax(topk_values, dim=-1)
    return probs, topk_indices


def baseline_top_k_select(
    q: torch.Tensor,
    k: torch.Tensor,
    workload: TopKKernelWorkload,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_scores = (
        torch.einsum(
            "bhqd,bhkd->bhqk",
            q.float(),
            _repeat_k_over_query_heads(k, workload).float(),
        )
        .view(
            workload.batch_size,
            workload.kv_heads,
            workload.gqa_size,
            workload.q_len,
            workload.num_blocks,
            workload.positions_per_block,
        )
        .sum(dim=5)
        .sum(dim=2)
        * workload.scale
    )
    return _finalize_topk_from_block_scores(dense_scores, workload)


def _softmax_topk_backward(
    probs: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_probs: torch.Tensor,
    workload: TopKKernelWorkload,
) -> torch.Tensor:
    grad_probs_float = grad_probs.float()
    grad_topk_values = probs * (
        grad_probs_float - (grad_probs_float * probs).sum(dim=-1, keepdim=True)
    )
    grad_block_scores = torch.zeros(
        workload.batch_size,
        workload.kv_heads,
        workload.q_len,
        workload.num_blocks,
        device=grad_probs.device,
        dtype=torch.float32,
    )
    grad_block_scores.scatter_add_(-1, topk_indices, grad_topk_values)
    return grad_block_scores * workload.scale


def _reshape_group_rows(
    q_group: torch.Tensor,
    workload: TopKKernelWorkload,
) -> torch.Tensor:
    return q_group.reshape(
        workload.batch_size * workload.kv_heads,
        workload.q_len * workload.gqa_size,
        workload.head_dim,
    ).contiguous()


def _expand_query_grads(
    grad_q_group_rows: torch.Tensor,
    workload: TopKKernelWorkload,
    q_dtype: torch.dtype,
) -> torch.Tensor:
    return (
        grad_q_group_rows.view(
            workload.batch_size,
            workload.kv_heads,
            workload.q_len,
            workload.gqa_size,
            workload.head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            workload.batch_size,
            workload.heads,
            workload.q_len,
            workload.head_dim,
        )
        .to(dtype=q_dtype)
    )


def _broadcast_group_query_grads(
    grad_q_single: torch.Tensor,
    workload: TopKKernelWorkload,
    q_dtype: torch.dtype,
) -> torch.Tensor:
    return (
        grad_q_single[:, :, :, None, :]
        .expand(
            workload.batch_size,
            workload.kv_heads,
            workload.q_len,
            workload.gqa_size,
            workload.head_dim,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            workload.batch_size,
            workload.heads,
            workload.q_len,
            workload.head_dim,
        )
        .to(dtype=q_dtype)
    )


def _expand_block_k_grads(
    grad_block_k: torch.Tensor,
    workload: TopKKernelWorkload,
    k_dtype: torch.dtype,
) -> torch.Tensor:
    return (
        grad_block_k[:, :, :, None, :]
        .expand(
            workload.batch_size,
            workload.kv_heads,
            workload.num_blocks,
            workload.positions_per_block,
            workload.head_dim,
        )
        .reshape(
            workload.batch_size,
            workload.kv_heads,
            workload.compressed_k_len,
            workload.head_dim,
        )
        .to(dtype=k_dtype)
    )


if triton is not None:

    @triton.jit
    def topk_group_block_score_kernel(
        q_ptr,
        block_k_ptr,
        out_ptr,
        q_len,
        num_blocks,
        head_dim,
        stride_q_row,
        stride_q_group,
        stride_q_d,
        stride_block_group,
        stride_block_n,
        stride_block_d,
        stride_out_row,
        stride_out_n,
        GQA_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        block_tile_id = tl.program_id(1)

        block_ids = block_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        group_id = row_id // q_len

        for g in range(0, GQA_SIZE):
            for d_start in range(0, head_dim, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)
                q = tl.load(
                    q_ptr
                    + row_id * stride_q_row
                    + g * stride_q_group
                    + d_offsets * stride_q_d,
                    mask=d_offsets < head_dim,
                    other=0.0,
                )
                block_ptrs = (
                    block_k_ptr
                    + group_id * stride_block_group
                    + block_ids[:, None] * stride_block_n
                    + d_offsets[None, :] * stride_block_d
                )
                block_k = tl.load(
                    block_ptrs,
                    mask=(block_ids[:, None] < num_blocks)
                    & (d_offsets[None, :] < head_dim),
                    other=0.0,
                )
                acc += tl.sum(block_k * q[None, :], axis=1)

        tl.store(
            out_ptr + row_id * stride_out_row + block_ids * stride_out_n,
            acc,
            mask=block_ids < num_blocks,
        )


def _run_triton_group_block_scores(
    q_group: torch.Tensor,
    block_k: torch.Tensor,
    workload: TopKKernelWorkload,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError(f"SKIPPED: Triton not available ({_TRITON_IMPORT_ERROR})")

    q_group_flat = q_group.reshape(
        workload.batch_size * workload.kv_heads * workload.q_len,
        workload.gqa_size,
        workload.head_dim,
    ).contiguous()
    block_k_flat = block_k.reshape(
        workload.batch_size * workload.kv_heads,
        workload.num_blocks,
        workload.head_dim,
    ).contiguous()
    block_scores = torch.empty(
        workload.batch_size * workload.kv_heads * workload.q_len,
        workload.num_blocks,
        device=q_group.device,
        dtype=torch.float32,
    )
    grid = (
        workload.batch_size * workload.kv_heads * workload.q_len,
        triton.cdiv(workload.num_blocks, 16),
    )
    topk_group_block_score_kernel[grid](
        q_group_flat,
        block_k_flat,
        block_scores,
        workload.q_len,
        workload.num_blocks,
        workload.head_dim,
        q_group_flat.stride(0),
        q_group_flat.stride(1),
        q_group_flat.stride(2),
        block_k_flat.stride(0),
        block_k_flat.stride(1),
        block_k_flat.stride(2),
        block_scores.stride(0),
        block_scores.stride(1),
        GQA_SIZE=workload.gqa_size,
        BLOCK_N=16,
        BLOCK_D=32,
        num_warps=4,
        num_stages=2,
    )
    return block_scores.view(
        workload.batch_size,
        workload.kv_heads,
        workload.q_len,
        workload.num_blocks,
    ) * workload.scale


def _run_cutlass_reduced_group_block_scores(
    q_sum: torch.Tensor,
    block_k: torch.Tensor,
    workload: TopKKernelWorkload,
) -> torch.Tensor:
    from labs.top_k_kernel.top_k_kernel_extension import load_top_k_kernel_extension

    extension = load_top_k_kernel_extension()

    q_sum_half = q_sum.to(dtype=torch.float16)
    block_k_half = block_k.to(dtype=torch.float16)
    scores = torch.empty(
        workload.batch_size,
        workload.kv_heads,
        workload.q_len,
        workload.num_blocks,
        device=q_sum.device,
        dtype=torch.float32,
    )

    q_groups = q_sum_half.reshape(
        workload.batch_size * workload.kv_heads,
        workload.q_len,
        workload.head_dim,
    )
    block_groups = block_k_half.reshape(
        workload.batch_size * workload.kv_heads,
        workload.num_blocks,
        workload.head_dim,
    )
    score_groups = scores.reshape(
        workload.batch_size * workload.kv_heads,
        workload.q_len,
        workload.num_blocks,
    )

    for group_idx in range(q_groups.shape[0]):
        q_group_tile = q_groups[group_idx]
        block_k_tile = block_groups[group_idx].contiguous()
        group_scores = score_groups[group_idx]
        for q_start in range(0, workload.q_len, workload.cuda_q_tile):
            q_end = min(q_start + workload.cuda_q_tile, workload.q_len)
            dense_chunk = extension.matmul_cutlass_topk(
                q_group_tile[q_start:q_end].contiguous(),
                block_k_tile,
            )
            torch.mul(dense_chunk, workload.scale, out=group_scores[q_start:q_end])
    return scores


def _run_cuda_reduced_group_block_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    workload: TopKKernelWorkload,
    *,
    use_cutlass: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The grouped Top-K score reduces query heads before selection, so the
    # backward benchmark can score the reduced Q tile directly instead of
    # materializing all GQA rows and summing them after GEMM.
    q_sum = _group_q(q, workload).sum(dim=3, dtype=torch.float32).contiguous()
    block_k = _build_block_k_reduced(k, workload)
    if use_cutlass:
        block_scores = _run_cutlass_reduced_group_block_scores(q_sum, block_k, workload)
    else:
        block_scores = torch.matmul(q_sum, block_k.transpose(-1, -2)) * workload.scale
    return block_scores, q_sum, block_k


def _validate_cuda_backend_workload(workload: TopKKernelWorkload) -> None:
    if workload.dtype != torch.float16:
        raise ValueError(
            "CUDA Top-K kernel currently supports fp16 only; "
            f"got dtype={workload.dtype}."
        )
    if workload.head_dim % 8 != 0:
        raise ValueError(
            "CUDA Top-K kernel requires head_dim to be divisible by 8 for the "
            f"CUTLASS tensor-op path; got head_dim={workload.head_dim}."
        )
    if workload.num_blocks % 8 != 0:
        raise ValueError(
            "CUDA Top-K kernel requires num_blocks to be divisible by 8 for the "
            "CUTLASS tensor-op path; "
            f"got num_blocks={workload.num_blocks}. "
            "Increase compressed_k_len or adjust selection_block_size/compress_stride."
        )


def _triton_group_backward(
    q_saved: torch.Tensor,
    k_saved: torch.Tensor,
    probs: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_probs: torch.Tensor,
    workload: TopKKernelWorkload,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_group_rows = _reshape_group_rows(_group_q(q_saved, workload), workload)
    block_k = _build_block_k(k_saved, workload).reshape(
        workload.batch_size * workload.kv_heads,
        workload.num_blocks,
        workload.head_dim,
    )
    grad_block_scores = _softmax_topk_backward(
        probs,
        topk_indices,
        grad_probs,
        workload,
    )
    grad_row_scores = (
        grad_block_scores[:, :, :, None, :]
        .expand(
            workload.batch_size,
            workload.kv_heads,
            workload.q_len,
            workload.gqa_size,
            workload.num_blocks,
        )
        .reshape(
            workload.batch_size * workload.kv_heads,
            workload.q_len * workload.gqa_size,
            workload.num_blocks,
        )
    )
    grad_q_group_rows = torch.matmul(grad_row_scores, block_k)
    grad_block_k = torch.matmul(grad_row_scores.transpose(1, 2), q_group_rows).view(
        workload.batch_size,
        workload.kv_heads,
        workload.num_blocks,
        workload.head_dim,
    )
    return (
        _expand_query_grads(grad_q_group_rows, workload, q_dtype),
        _expand_block_k_grads(grad_block_k, workload, k_dtype),
    )


def _cutlass_group_backward(
    q_sum_saved: torch.Tensor,
    block_k_saved: torch.Tensor,
    probs: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_probs: torch.Tensor,
    workload: TopKKernelWorkload,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_k = block_k_saved.to(dtype=torch.float16)
    grad_block_scores = _softmax_topk_backward(
        probs,
        topk_indices,
        grad_probs,
        workload,
    ).to(dtype=torch.float16)
    q_sum = q_sum_saved.to(dtype=torch.float16)

    grad_q_single = torch.matmul(grad_block_scores, block_k).float()
    grad_block_k = torch.matmul(grad_block_scores.transpose(-1, -2), q_sum).float()

    return (
        _broadcast_group_query_grads(grad_q_single, workload, q_dtype),
        _expand_block_k_grads(grad_block_k, workload, k_dtype),
    )


class TritonTopKSelectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, workload: TopKKernelWorkload):
        q_float = q.float().contiguous()
        k_float = k.float().contiguous()
        q_group = _group_q(q_float, workload)
        block_k = _build_block_k(k_float, workload)
        block_scores = _run_triton_group_block_scores(q_group, block_k, workload)
        probs, topk_indices = _finalize_topk_from_block_scores(block_scores, workload)
        ctx.save_for_backward(q_float, k_float, probs, topk_indices)
        ctx.workload = workload
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.mark_non_differentiable(topk_indices)
        return probs, topk_indices

    @staticmethod
    def backward(ctx, grad_probs: torch.Tensor, grad_indices: Optional[torch.Tensor]):
        del grad_indices
        q_saved, k_saved, probs, topk_indices = ctx.saved_tensors
        grad_q, grad_k = _triton_group_backward(
            q_saved=q_saved,
            k_saved=k_saved,
            probs=probs,
            topk_indices=topk_indices,
            grad_probs=grad_probs,
            workload=ctx.workload,
            q_dtype=ctx.q_dtype,
            k_dtype=ctx.k_dtype,
        )
        return grad_q, grad_k, None


class CudaTopKSelectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, workload: TopKKernelWorkload):
        needs_backward = bool(ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
        block_scores, q_sum, block_k = _run_cuda_reduced_group_block_scores(
            q,
            k,
            workload,
            use_cutlass=workload.mode == "forward",
        )
        probs, topk_indices = _finalize_topk_from_block_scores(block_scores, workload)
        if needs_backward:
            ctx.save_for_backward(q_sum, block_k, probs, topk_indices)
        else:
            ctx.save_for_backward()
        ctx.workload = workload
        ctx.q_dtype = q.dtype
        ctx.k_dtype = k.dtype
        ctx.mark_non_differentiable(topk_indices)
        return probs, topk_indices

    @staticmethod
    def backward(ctx, grad_probs: torch.Tensor, grad_indices: Optional[torch.Tensor]):
        del grad_indices
        q_sum_saved, block_k_saved, probs, topk_indices = ctx.saved_tensors
        grad_q, grad_k = _cutlass_group_backward(
            q_sum_saved=q_sum_saved,
            block_k_saved=block_k_saved,
            probs=probs,
            topk_indices=topk_indices,
            grad_probs=grad_probs,
            workload=ctx.workload,
            q_dtype=ctx.q_dtype,
            k_dtype=ctx.k_dtype,
        )
        return grad_q, grad_k, None


def run_optimized_top_k_select(
    q: torch.Tensor,
    k: torch.Tensor,
    workload: TopKKernelWorkload,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "triton":
        return TritonTopKSelectionFunction.apply(q, k, workload)
    if backend == "cuda":
        return CudaTopKSelectionFunction.apply(q, k, workload)
    raise ValueError(f"Unsupported optimized backend: {backend}")


def _verification_tensor_size(outputs: TopKKernelOutputs) -> int:
    size = 32
    if outputs.q_grad is not None:
        size += 16
    if outputs.k_grad is not None:
        size += 16
    return size


def _copy_verification_piece(out: torch.Tensor, offset: int, tensor: torch.Tensor) -> int:
    flat = tensor.reshape(-1)
    next_offset = offset + int(flat.numel())
    out[offset:next_offset].copy_(flat)
    return next_offset


def build_verification_tensor(
    outputs: TopKKernelOutputs,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    required_size = _verification_tensor_size(outputs)
    if out is None:
        out = torch.empty(required_size, device=outputs.probs.device, dtype=torch.float32)
    elif out.numel() < required_size or out.dtype != torch.float32 or out.device != outputs.probs.device:
        raise RuntimeError("verification output buffer has an incompatible shape, dtype, or device")

    sorted_indices = outputs.indices[:1, :1, :4, :4].sort(dim=-1).values
    offset = 0
    offset = _copy_verification_piece(out, offset, outputs.probs[:1, :1, :4, :4])
    offset = _copy_verification_piece(out, offset, sorted_indices)
    if outputs.q_grad is not None:
        offset = _copy_verification_piece(out, offset, outputs.q_grad[:1, :1, :2, :8])
    if outputs.k_grad is not None:
        offset = _copy_verification_piece(out, offset, outputs.k_grad[:1, :1, :2, :8])
    return out[:offset]


class TopKKernelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark grouped-query Top-K selection kernels."""

    preferred_ncu_replay_mode = "application"

    def __init__(self, *, backend: str, label: str) -> None:
        super().__init__()
        self.backend = backend
        self.label = label
        self.workload = TopKKernelWorkload()
        self.inputs: Optional[TopKKernelInputs] = None
        self.outputs: Optional[TopKKernelOutputs] = None
        self.device = resolve_device()
        self._custom_metrics: dict[str, float] = {}
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._loss_weights_flat: Optional[torch.Tensor] = None
        self._refresh_workload_metadata()

    def _refresh_workload_metadata(self) -> None:
        total_scores = (
            self.workload.batch_size
            * self.workload.heads
            * self.workload.q_len
            * self.workload.num_blocks
        )
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.workload.batch_size * self.workload.kv_heads),
            tokens_per_iteration=float(total_scores),
        )

    def setup(self) -> None:
        self.inputs = build_inputs(self.workload, self.device)
        self._loss_weights_flat = self.inputs.loss_weights.reshape(-1)
        self.outputs = None
        verify_size = 64 if self.workload.mode == "fwd_bwd" else 32
        self._verify_output_buffer = torch.empty(verify_size, device=self.device, dtype=torch.float32)
        if self.workload.mode == "fwd_bwd":
            self.inputs.q.requires_grad_(True)
            self.inputs.k.requires_grad_(True)

        if self.backend == "cuda":
            _validate_cuda_backend_workload(self.workload)
            if self.workload.mode == "forward":
                from labs.top_k_kernel.top_k_kernel_extension import load_top_k_kernel_extension

                load_top_k_kernel_extension()

        if self.backend in {"triton", "cuda"}:
            if self.workload.mode == "forward":
                with torch.inference_mode():
                    run_optimized_top_k_select(
                        self.inputs.q.detach(),
                        self.inputs.k.detach(),
                        self.workload,
                        self.backend,
                    )
            else:
                warm_q = self.inputs.q.detach().clone().requires_grad_(True)
                warm_k = self.inputs.k.detach().clone().requires_grad_(True)
                warm_probs, _ = run_optimized_top_k_select(
                    warm_q,
                    warm_k,
                    self.workload,
                    self.backend,
                )
                torch.dot(warm_probs.reshape(-1), self._loss_weights_flat).backward()

        self._custom_metrics = {
            "topk.backend.baseline": 1.0 if self.backend == "baseline" else 0.0,
            "topk.backend.triton": 1.0 if self.backend == "triton" else 0.0,
            "topk.backend.cuda": 1.0 if self.backend == "cuda" else 0.0,
            "topk.mode.forward_only": 1.0 if self.workload.mode == "forward" else 0.0,
            "topk.mode.fwd_bwd": 1.0 if self.workload.mode == "fwd_bwd" else 0.0,
            "topk.positions_per_block": float(self.workload.positions_per_block),
            "topk.num_blocks": float(self.workload.num_blocks),
            "topk.top_k": float(self.workload.top_k),
            "topk.head_dim": float(self.workload.head_dim),
            "topk.gqa_size": float(self.workload.gqa_size),
            "topk.kv_heads": float(self.workload.kv_heads),
            "topk.packed_q_tiles": float(
                self.workload.packed_q_tiles if self.backend in {"triton", "cuda"} else 1
            ),
            "topk.packed_kv_blocks": float(
                self.workload.packed_kv_blocks if self.backend in {"triton", "cuda"} else 1
            ),
            "topk.cuda_q_tile": float(
                self.workload.cuda_q_tile if self.backend == "cuda" else 0
            ),
            "topk.cuda_tensorop_compatible": float(self.workload.cuda_tensorop_compatible),
        }
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Top-K inputs not initialized")
        if self.workload.mode == "fwd_bwd" and self._loss_weights_flat is None:
            raise RuntimeError("Top-K loss weights not initialized")

        if self.workload.mode == "fwd_bwd":
            self.inputs.q.grad = None
            self.inputs.k.grad = None

        with self._nvtx_range(self.label):
            if self.workload.mode == "forward":
                with torch.inference_mode():
                    if self.backend == "baseline":
                        probs, indices = baseline_top_k_select(
                            self.inputs.q,
                            self.inputs.k,
                            self.workload,
                        )
                    else:
                        probs, indices = run_optimized_top_k_select(
                            self.inputs.q,
                            self.inputs.k,
                            self.workload,
                            self.backend,
                        )
            elif self.backend == "baseline":
                probs, indices = baseline_top_k_select(
                    self.inputs.q,
                    self.inputs.k,
                    self.workload,
                )
            else:
                probs, indices = run_optimized_top_k_select(
                    self.inputs.q,
                    self.inputs.k,
                    self.workload,
                    self.backend,
                )

            q_grad = None
            k_grad = None
            if self.workload.mode == "fwd_bwd":
                loss = torch.dot(probs.reshape(-1), self._loss_weights_flat)
                loss.backward()
                q_grad = self.inputs.q.grad
                k_grad = self.inputs.k.grad
                probs = probs.detach_()

        self.outputs = TopKKernelOutputs(
            probs=probs,
            indices=indices,
            q_grad=q_grad,
            k_grad=k_grad,
        )

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.outputs is None:
            raise RuntimeError("setup() and benchmark_fn() must run before verification capture")
        output_tolerance = (
            (2e-2, 2e-2)
            if self.backend == "cuda" and self.workload.mode == "fwd_bwd"
            else (1e-2, 1e-2)
        )
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
            },
            output=build_verification_tensor(self.outputs, self._verify_output_buffer),
            batch_size=self.workload.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.workload.dtype == torch.float16,
                "bf16": self.workload.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=output_tolerance,
        )

    def teardown(self) -> None:
        self.inputs = None
        self.outputs = None
        self._verify_output_buffer = None
        self._loss_weights_flat = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=20 if self.workload.mode == "forward" else 10,
            warmup=5,
            setup_timeout_seconds=240,
            measurement_timeout_seconds=180,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._custom_metrics)

    def validate_result(self) -> Optional[str]:
        if self.outputs is None:
            return "benchmark_fn() did not produce output"
        if not torch.isfinite(self.outputs.probs).all():
            return "Top-k probabilities contain non-finite values"
        if self.outputs.indices.min() < 0 or self.outputs.indices.max() >= self.workload.num_blocks:
            return "Top-k indices are out of range"
        if self.workload.mode == "fwd_bwd":
            if self.outputs.q_grad is None or self.outputs.k_grad is None:
                return "Backward mode did not produce gradients"
            if not torch.isfinite(self.outputs.q_grad).all():
                return "Query gradients contain non-finite values"
            if not torch.isfinite(self.outputs.k_grad).all():
                return "Key gradients contain non-finite values"
        return None

    def apply_target_overrides(self, argv: list[str]) -> None:
        self.workload = apply_workload_overrides(self.workload, argv)
        self._refresh_workload_metadata()
