"""Shared workload and Triton helpers for the RecSys sequence-ranking lab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from core.benchmark.triton_compat import ensure_triton_compat

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - Triton is optional at import time
    triton = None
    tl = None
    TRITON_AVAILABLE = False


_SCORE_BACKEND_CHOICES = ("auto", "triton", "torch")


@dataclass
class SequenceRankingWorkload:
    """Synthetic session-ranking workload parameters."""

    batch_size: int = 64
    seq_len: int = 32
    num_tables: int = 8
    embedding_dim: int = 128
    hidden_dim: int = 192
    num_candidates: int = 128
    item_vocab_size: int = 20000
    context_vocab_size: int = 4096
    min_history_len: int = 8
    zipf_alpha: float = 1.15
    seed: int = 1234
    dtype: torch.dtype = torch.float32
    use_compile: bool = True
    score_backend: str = "auto"


@dataclass
class RankingInputs:
    """Synthetic sparse-input batch for session ranking."""

    sequence_ids: torch.Tensor
    sequence_mask: torch.Tensor
    sequence_lengths: torch.Tensor
    context_ids: torch.Tensor
    candidate_ids: torch.Tensor
    sequence_ids_1d: torch.Tensor
    candidate_ids_1d: torch.Tensor
    avg_sequence_length: float
    hot_candidate_share_pct: float


@dataclass
class RankingModelState:
    """Model tensors and modules shared by baseline and optimized paths."""

    item_embeddings: torch.Tensor
    context_embeddings: torch.Tensor
    tower: SequenceRankingTower
    parameter_count: int


@dataclass
class RankingWorkspace:
    """Reusable scratch buffers to keep allocator noise out of benchmark_fn()."""

    sequence_accum: torch.Tensor
    context_accum: torch.Tensor
    score_output: torch.Tensor
    sequence_embedding_flat: torch.Tensor
    context_embedding_flat: torch.Tensor
    candidate_embedding_flat: torch.Tensor
    sequence_mask_float: torch.Tensor
    sequence_length_recip: torch.Tensor
    context_table_index: torch.Tensor
    context_flat_ids: torch.Tensor
    context_flat_ids_1d: torch.Tensor
    candidate_embedding_f32: torch.Tensor | None = None
    user_vec_f32: torch.Tensor | None = None
    sequence_metadata_key: tuple[int, int] | None = None
    context_metadata_key: tuple[int, int] | None = None


class SequenceRankingTower(nn.Module):
    """Small MLP tower that turns sparse features into a user vector."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, user_input: torch.Tensor) -> torch.Tensor:
        hidden = self.in_proj(user_input)
        hidden = F.gelu(hidden, approximate="tanh")
        user_vec = self.out_proj(hidden)
        return self.norm(user_vec)


def default_workload() -> SequenceRankingWorkload:
    """Return the default synthetic ranking workload."""

    return SequenceRankingWorkload()


def apply_cli_overrides(
    workload: SequenceRankingWorkload, argv: list[str]
) -> SequenceRankingWorkload:
    """Apply per-target CLI overrides without mutating global process state."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--num-tables", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--item-vocab-size", type=int, default=None)
    parser.add_argument("--context-vocab-size", type=int, default=None)
    parser.add_argument("--min-history-len", type=int, default=None)
    parser.add_argument("--zipf-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--score-backend", choices=_SCORE_BACKEND_CHOICES, default=None)
    args, _ = parser.parse_known_args(argv)

    updates: dict[str, Any] = {}
    for field_name in (
        "batch_size",
        "seq_len",
        "num_tables",
        "embedding_dim",
        "hidden_dim",
        "num_candidates",
        "item_vocab_size",
        "context_vocab_size",
        "min_history_len",
        "zipf_alpha",
        "seed",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value
    if args.disable_compile:
        updates["use_compile"] = False
    if args.score_backend is not None:
        updates["score_backend"] = args.score_backend

    merged = SequenceRankingWorkload(**{**workload.__dict__, **updates})
    if merged.min_history_len > merged.seq_len:
        merged.min_history_len = merged.seq_len
    return merged


def requests_per_iteration(workload: SequenceRankingWorkload) -> float:
    return float(workload.batch_size)


def tokens_per_iteration(workload: SequenceRankingWorkload) -> float:
    return float(workload.batch_size * workload.seq_len)


def _zipf_probs(cardinality: int, alpha: float) -> torch.Tensor:
    ranks = torch.arange(1, cardinality + 1, dtype=torch.float64)
    weights = ranks.pow(-alpha)
    return weights / weights.sum()


def _sample_zipf(
    count: int,
    *,
    cardinality: int,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    probs = _zipf_probs(cardinality, alpha)
    return torch.multinomial(probs, count, replacement=True, generator=generator)


def _randn(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, scale: float = 0.02
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).mul_(scale).to(dtype=dtype)


def build_inputs(workload: SequenceRankingWorkload, device: torch.device) -> RankingInputs:
    """Create a deterministic synthetic clickstream batch."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(workload.seed)

    lengths = torch.randint(
        low=workload.min_history_len,
        high=workload.seq_len + 1,
        size=(workload.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    sequence_ids = _sample_zipf(
        workload.batch_size * workload.seq_len,
        cardinality=workload.item_vocab_size,
        alpha=workload.zipf_alpha,
        generator=generator,
    ).view(workload.batch_size, workload.seq_len)
    context_ids = _sample_zipf(
        workload.batch_size * workload.num_tables,
        cardinality=workload.context_vocab_size,
        alpha=max(workload.zipf_alpha - 0.1, 1.01),
        generator=generator,
    ).view(workload.batch_size, workload.num_tables)
    candidate_ids = _sample_zipf(
        workload.batch_size * workload.num_candidates,
        cardinality=workload.item_vocab_size,
        alpha=workload.zipf_alpha,
        generator=generator,
    ).view(workload.batch_size, workload.num_candidates)

    last_positions = lengths.sub(1).clamp_min(0)
    positives = sequence_ids.gather(1, last_positions.view(-1, 1))
    candidate_ids[:, 0:1] = positives

    time_index = torch.arange(workload.seq_len, dtype=torch.int64).view(1, workload.seq_len)
    sequence_mask = time_index < lengths.view(-1, 1)
    hot_threshold = max(workload.item_vocab_size // 100, 1)
    avg_sequence_length = float(lengths.to(torch.float32).mean())
    hot_candidate_share_pct = float((candidate_ids < hot_threshold).to(torch.float32).mean() * 100.0)

    sequence_ids_device = sequence_ids.to(device=device, dtype=torch.int64)
    candidate_ids_device = candidate_ids.to(device=device, dtype=torch.int64)
    return RankingInputs(
        sequence_ids=sequence_ids_device,
        sequence_mask=sequence_mask.to(device=device),
        sequence_lengths=lengths.to(device=device, dtype=torch.int64),
        context_ids=context_ids.to(device=device, dtype=torch.int64),
        candidate_ids=candidate_ids_device,
        sequence_ids_1d=sequence_ids_device.view(-1),
        candidate_ids_1d=candidate_ids_device.view(-1),
        avg_sequence_length=avg_sequence_length,
        hot_candidate_share_pct=hot_candidate_share_pct,
    )


def build_model_state(workload: SequenceRankingWorkload, device: torch.device) -> RankingModelState:
    """Create deterministic embedding tables and tower weights."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(workload.seed + 17)

    item_embeddings = _randn(
        (workload.item_vocab_size, workload.embedding_dim),
        generator=generator,
        dtype=workload.dtype,
    ).to(device=device)
    context_embeddings = _randn(
        (workload.num_tables, workload.context_vocab_size, workload.embedding_dim),
        generator=generator,
        dtype=workload.dtype,
    ).to(device=device)

    tower = SequenceRankingTower(workload.embedding_dim, workload.hidden_dim).to(
        device=device, dtype=workload.dtype
    )
    with torch.inference_mode():
        tower.in_proj.weight.copy_(
            _randn(
                (workload.hidden_dim, workload.embedding_dim),
                generator=generator,
                dtype=workload.dtype,
            ).to(device=device)
        )
        tower.in_proj.bias.copy_(
            _randn((workload.hidden_dim,), generator=generator, dtype=workload.dtype).to(
                device=device
            )
        )
        tower.out_proj.weight.copy_(
            _randn(
                (workload.embedding_dim, workload.hidden_dim),
                generator=generator,
                dtype=workload.dtype,
            ).to(device=device)
        )
        tower.out_proj.bias.copy_(
            _randn((workload.embedding_dim,), generator=generator, dtype=workload.dtype).to(
                device=device
            )
        )
        tower.norm.weight.copy_(
            torch.ones(workload.embedding_dim, device=device, dtype=workload.dtype)
        )
        tower.norm.bias.zero_()

    parameter_count = int(
        item_embeddings.numel()
        + context_embeddings.numel()
        + sum(p.numel() for p in tower.parameters())
    )
    return RankingModelState(
        item_embeddings=item_embeddings,
        context_embeddings=context_embeddings,
        tower=tower.eval(),
        parameter_count=parameter_count,
    )


def build_workspace(workload: SequenceRankingWorkload, device: torch.device) -> RankingWorkspace:
    """Allocate reusable scratch buffers in setup()."""

    context_table_index = torch.arange(workload.num_tables, device=device, dtype=torch.int64)
    context_table_index = context_table_index.view(1, workload.num_tables)
    context_flat_ids = torch.empty(
        workload.batch_size,
        workload.num_tables,
        device=device,
        dtype=torch.int64,
    )
    needs_score_cast_workspace = workload.dtype != torch.float32
    return RankingWorkspace(
        sequence_accum=torch.empty(
            workload.batch_size,
            workload.embedding_dim,
            device=device,
            dtype=workload.dtype,
        ),
        context_accum=torch.empty(
            workload.batch_size,
            workload.embedding_dim,
            device=device,
            dtype=workload.dtype,
        ),
        score_output=torch.empty(
            workload.batch_size,
            workload.num_candidates,
            device=device,
            dtype=torch.float32,
        ),
        sequence_embedding_flat=torch.empty(
            workload.batch_size * workload.seq_len,
            workload.embedding_dim,
            device=device,
            dtype=workload.dtype,
        ),
        context_embedding_flat=torch.empty(
            workload.batch_size * workload.num_tables,
            workload.embedding_dim,
            device=device,
            dtype=workload.dtype,
        ),
        candidate_embedding_flat=torch.empty(
            workload.batch_size * workload.num_candidates,
            workload.embedding_dim,
            device=device,
            dtype=workload.dtype,
        ),
        sequence_mask_float=torch.empty(
            workload.batch_size,
            workload.seq_len,
            1,
            device=device,
            dtype=workload.dtype,
        ),
        sequence_length_recip=torch.empty(
            workload.batch_size,
            1,
            device=device,
            dtype=workload.dtype,
        ),
        context_table_index=context_table_index,
        context_flat_ids=context_flat_ids,
        context_flat_ids_1d=context_flat_ids.view(-1),
        candidate_embedding_f32=(
            torch.empty(
                workload.batch_size * workload.num_candidates,
                workload.embedding_dim,
                device=device,
                dtype=torch.float32,
            )
            if needs_score_cast_workspace
            else None
        ),
        user_vec_f32=(
            torch.empty(
                workload.batch_size,
                workload.embedding_dim,
                device=device,
                dtype=torch.float32,
            )
            if needs_score_cast_workspace
            else None
        ),
    )


def _sequence_metadata_key(inputs: RankingInputs) -> tuple[int, int]:
    return (inputs.sequence_mask.data_ptr(), inputs.sequence_lengths.data_ptr())


def _context_metadata_key(inputs: RankingInputs, state: RankingModelState) -> tuple[int, int]:
    return (inputs.context_ids.data_ptr(), int(state.context_embeddings.shape[1]))


def prepare_workspace_for_inputs(inputs: RankingInputs, workspace: RankingWorkspace) -> None:
    """Cache immutable sequence metadata derived from the benchmark inputs."""

    workspace.sequence_mask_float.copy_(inputs.sequence_mask.unsqueeze(-1))
    workspace.sequence_length_recip.copy_(inputs.sequence_lengths.unsqueeze(1))
    workspace.sequence_length_recip.clamp_min_(1).reciprocal_()
    workspace.sequence_metadata_key = _sequence_metadata_key(inputs)


def prepare_context_workspace_for_inputs(
    inputs: RankingInputs,
    state: RankingModelState,
    workspace: RankingWorkspace,
) -> None:
    """Cache flattened context lookup ids derived from immutable benchmark inputs."""

    context_vocab_size = int(state.context_embeddings.shape[1])
    torch.add(
        inputs.context_ids,
        workspace.context_table_index,
        alpha=context_vocab_size,
        out=workspace.context_flat_ids,
    )
    workspace.context_metadata_key = _context_metadata_key(inputs, state)


def sequence_mean_baseline(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
    workspace: RankingWorkspace,
) -> torch.Tensor:
    """Conservative sequence pooling using one embedding lookup per time step."""

    if workspace.sequence_metadata_key != _sequence_metadata_key(inputs):
        prepare_workspace_for_inputs(inputs, workspace)
    mask = workspace.sequence_mask_float.squeeze(-1)
    if inputs.sequence_ids.shape[1] == 0:
        out.zero_()
        return out
    token_vec = state.item_embeddings[inputs.sequence_ids[:, 0]]
    torch.mul(token_vec, mask[:, 0:1], out=out)
    for t in range(1, inputs.sequence_ids.shape[1]):
        token_vec = state.item_embeddings[inputs.sequence_ids[:, t]]
        token_vec.mul_(mask[:, t : t + 1])
        out.add_(token_vec)
    out.mul_(workspace.sequence_length_recip)
    return out


def context_sum_baseline(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
) -> torch.Tensor:
    """Conservative context lookup using one table at a time."""

    if inputs.context_ids.shape[1] == 0:
        out.zero_()
        return out
    out.copy_(state.context_embeddings[0, inputs.context_ids[:, 0]])
    for table_idx in range(1, inputs.context_ids.shape[1]):
        out.add_(state.context_embeddings[table_idx, inputs.context_ids[:, table_idx]])
    return out


def candidate_scores_baseline(
    user_vec: torch.Tensor,
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
) -> torch.Tensor:
    """Score each candidate in a Python loop to expose launch overhead."""

    candidate_emb = F.embedding(inputs.candidate_ids, state.item_embeddings)
    for idx in range(inputs.candidate_ids.shape[1]):
        out[:, idx] = (candidate_emb[:, idx, :] * user_vec).sum(dim=-1, dtype=torch.float32)
    return out


def sequence_mean_vectorized(
    inputs: RankingInputs,
    state: RankingModelState,
    workspace: RankingWorkspace | None = None,
) -> torch.Tensor:
    if workspace is None:
        seq_emb = F.embedding(inputs.sequence_ids, state.item_embeddings)
        mask = inputs.sequence_mask.to(dtype=seq_emb.dtype).unsqueeze(-1)
        lengths = inputs.sequence_lengths.to(dtype=seq_emb.dtype).clamp_min_(1).unsqueeze(1)
        return (seq_emb * mask).sum(dim=1) / lengths

    if workspace.sequence_metadata_key != _sequence_metadata_key(inputs):
        prepare_workspace_for_inputs(inputs, workspace)
    if torch.is_grad_enabled() and state.item_embeddings.requires_grad:
        seq_emb = F.embedding(inputs.sequence_ids, state.item_embeddings)
        return (seq_emb * workspace.sequence_mask_float).sum(dim=1) * workspace.sequence_length_recip
    if TRITON_AVAILABLE and inputs.sequence_ids.is_cuda and state.item_embeddings.is_cuda:
        return sequence_mean_triton(inputs, state, workspace.sequence_accum, workspace)

    flat_sequence_ids = inputs.sequence_ids_1d
    sequence_rows = int(flat_sequence_ids.numel())
    embedding_dim = int(state.item_embeddings.shape[1])
    sequence_embedding_flat = workspace.sequence_embedding_flat[:sequence_rows]
    torch.index_select(
        state.item_embeddings,
        0,
        flat_sequence_ids,
        out=sequence_embedding_flat,
    )
    seq_emb = sequence_embedding_flat.view(
        inputs.sequence_ids.shape[0],
        inputs.sequence_ids.shape[1],
        embedding_dim,
    )
    seq_emb.mul_(workspace.sequence_mask_float)
    torch.sum(seq_emb, dim=1, out=workspace.sequence_accum)
    workspace.sequence_accum.mul_(workspace.sequence_length_recip)
    return workspace.sequence_accum


def context_sum_vectorized(
    inputs: RankingInputs,
    state: RankingModelState,
    workspace: RankingWorkspace | None = None,
) -> torch.Tensor:
    batch_size, num_tables = inputs.context_ids.shape
    if workspace is None:
        table_index = torch.arange(num_tables, device=inputs.context_ids.device, dtype=torch.int64)
        table_index = table_index.view(1, num_tables).expand(batch_size, -1)
        context_vecs = state.context_embeddings[table_index, inputs.context_ids]
        return context_vecs.sum(dim=1)

    if torch.is_grad_enabled() and state.context_embeddings.requires_grad:
        context_vecs = state.context_embeddings[workspace.context_table_index, inputs.context_ids]
        return context_vecs.sum(dim=1)

    embedding_dim = int(state.context_embeddings.shape[2])
    context_rows = int(inputs.context_ids.numel())
    if workspace.context_metadata_key != _context_metadata_key(inputs, state):
        prepare_context_workspace_for_inputs(inputs, state, workspace)
    if TRITON_AVAILABLE and inputs.context_ids.is_cuda and state.context_embeddings.is_cuda:
        return context_sum_triton(inputs, state, workspace.context_accum)

    flat_context_embeddings = state.context_embeddings.view(-1, embedding_dim)
    context_embedding_flat = workspace.context_embedding_flat[:context_rows]
    torch.index_select(
        flat_context_embeddings,
        0,
        workspace.context_flat_ids_1d[:context_rows],
        out=context_embedding_flat,
    )
    context_vecs = context_embedding_flat.view(batch_size, num_tables, embedding_dim)
    torch.sum(context_vecs, dim=1, out=workspace.context_accum)
    return workspace.context_accum


def candidate_scores_torch(
    user_vec: torch.Tensor,
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor | None = None,
    candidate_buffer: torch.Tensor | None = None,
    candidate_f32_buffer: torch.Tensor | None = None,
    user_vec_f32_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    flat_candidate_ids = inputs.candidate_ids_1d
    candidate_rows = int(flat_candidate_ids.numel())
    embedding_dim = int(state.item_embeddings.shape[1])
    can_reuse_candidate_buffer = (
        candidate_buffer is not None
        and candidate_buffer.dim() == 2
        and candidate_buffer.device == state.item_embeddings.device
        and candidate_buffer.dtype == state.item_embeddings.dtype
        and candidate_buffer.size(0) >= candidate_rows
        and candidate_buffer.size(1) == embedding_dim
        and not (torch.is_grad_enabled() and state.item_embeddings.requires_grad)
    )
    if can_reuse_candidate_buffer:
        candidate_buffer_view = candidate_buffer[:candidate_rows]
        torch.index_select(
            state.item_embeddings,
            0,
            flat_candidate_ids,
            out=candidate_buffer_view,
        )
        candidate_emb = candidate_buffer_view.view(
            inputs.candidate_ids.shape[0],
            inputs.candidate_ids.shape[1],
            embedding_dim,
        )
    else:
        candidate_emb = F.embedding(inputs.candidate_ids, state.item_embeddings)
    can_reuse_candidate_f32_buffer = (
        candidate_f32_buffer is not None
        and candidate_f32_buffer.dim() == 2
        and candidate_f32_buffer.device == candidate_emb.device
        and candidate_f32_buffer.dtype == torch.float32
        and candidate_f32_buffer.size(0) >= candidate_rows
        and candidate_f32_buffer.size(1) == embedding_dim
        and not (torch.is_grad_enabled() and candidate_emb.requires_grad)
    )
    if candidate_emb.dtype == torch.float32:
        candidate_emb_f32 = candidate_emb
    elif can_reuse_candidate_f32_buffer:
        candidate_f32_view = candidate_f32_buffer[:candidate_rows]
        candidate_f32_view.copy_(candidate_emb.view(candidate_rows, embedding_dim))
        candidate_emb_f32 = candidate_f32_view.view(
            inputs.candidate_ids.shape[0],
            inputs.candidate_ids.shape[1],
            embedding_dim,
        )
    else:
        candidate_emb_f32 = candidate_emb.to(torch.float32)

    can_reuse_user_f32_buffer = (
        user_vec_f32_buffer is not None
        and user_vec_f32_buffer.dim() == 2
        and user_vec_f32_buffer.device == user_vec.device
        and user_vec_f32_buffer.dtype == torch.float32
        and user_vec_f32_buffer.size(0) >= user_vec.size(0)
        and user_vec_f32_buffer.size(1) == user_vec.size(1)
        and not (torch.is_grad_enabled() and user_vec.requires_grad)
    )
    if user_vec.dtype == torch.float32:
        user_vec_f32 = user_vec.unsqueeze(2)
    elif can_reuse_user_f32_buffer:
        user_vec_f32_view = user_vec_f32_buffer[: user_vec.size(0), : user_vec.size(1)]
        user_vec_f32_view.copy_(user_vec)
        user_vec_f32 = user_vec_f32_view.unsqueeze(2)
    else:
        user_vec_f32 = user_vec.to(torch.float32).unsqueeze(2)
    if out is None or (
        torch.is_grad_enabled() and (candidate_emb_f32.requires_grad or user_vec_f32.requires_grad)
    ):
        return torch.bmm(candidate_emb_f32, user_vec_f32).squeeze(2)
    torch.bmm(candidate_emb_f32, user_vec_f32, out=out.unsqueeze(2))
    return out


if TRITON_AVAILABLE:

    @triton.jit
    def _sequence_mean_kernel(
        sequence_ids_ptr,
        item_embedding_ptr,
        sequence_mask_ptr,
        length_recip_ptr,
        out_ptr,
        embedding_dim,
        stride_ids_b,
        stride_ids_t,
        stride_item_vocab,
        stride_item_d,
        stride_mask_b,
        stride_mask_t,
        stride_recip_b,
        stride_out_b,
        stride_out_d,
        SEQ_LEN: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_idx = tl.program_id(0)
        dim_block_idx = tl.program_id(1)

        offs_d = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        dim_mask = offs_d < embedding_dim
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t in range(0, SEQ_LEN):
            token_id = tl.load(sequence_ids_ptr + batch_idx * stride_ids_b + t * stride_ids_t)
            token_mask = tl.load(
                sequence_mask_ptr + batch_idx * stride_mask_b + t * stride_mask_t
            )
            token_vec = tl.load(
                item_embedding_ptr + token_id * stride_item_vocab + offs_d * stride_item_d,
                mask=dim_mask,
                other=0.0,
            )
            acc += token_vec.to(tl.float32) * token_mask.to(tl.float32)

        length_recip = tl.load(length_recip_ptr + batch_idx * stride_recip_b).to(tl.float32)
        tl.store(
            out_ptr + batch_idx * stride_out_b + offs_d * stride_out_d,
            acc * length_recip,
            mask=dim_mask,
        )

    @triton.jit
    def _context_sum_kernel(
        context_ids_ptr,
        context_embedding_ptr,
        out_ptr,
        embedding_dim,
        stride_ids_b,
        stride_ids_t,
        stride_context_table,
        stride_context_vocab,
        stride_context_d,
        stride_out_b,
        stride_out_d,
        NUM_TABLES: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_idx = tl.program_id(0)
        dim_block_idx = tl.program_id(1)

        offs_d = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        dim_mask = offs_d < embedding_dim
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for table_idx in range(0, NUM_TABLES):
            context_id = tl.load(
                context_ids_ptr + batch_idx * stride_ids_b + table_idx * stride_ids_t
            )
            context_vec = tl.load(
                context_embedding_ptr
                + table_idx * stride_context_table
                + context_id * stride_context_vocab
                + offs_d * stride_context_d,
                mask=dim_mask,
                other=0.0,
            )
            acc += context_vec.to(tl.float32)

        tl.store(
            out_ptr + batch_idx * stride_out_b + offs_d * stride_out_d,
            acc,
            mask=dim_mask,
        )

    @triton.jit
    def _sequence_context_user_input_kernel(
        sequence_ids_ptr,
        item_embedding_ptr,
        sequence_mask_ptr,
        length_recip_ptr,
        context_ids_ptr,
        context_embedding_ptr,
        out_ptr,
        embedding_dim,
        stride_seq_ids_b,
        stride_seq_ids_t,
        stride_item_vocab,
        stride_item_d,
        stride_mask_b,
        stride_mask_t,
        stride_recip_b,
        stride_context_ids_b,
        stride_context_ids_t,
        stride_context_table,
        stride_context_vocab,
        stride_context_d,
        stride_out_b,
        stride_out_d,
        SEQ_LEN: tl.constexpr,  # noqa: N803
        NUM_TABLES: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_idx = tl.program_id(0)
        dim_block_idx = tl.program_id(1)

        offs_d = dim_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        dim_mask = offs_d < embedding_dim
        seq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        context_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for t in range(0, SEQ_LEN):
            token_id = tl.load(
                sequence_ids_ptr + batch_idx * stride_seq_ids_b + t * stride_seq_ids_t
            )
            token_mask = tl.load(
                sequence_mask_ptr + batch_idx * stride_mask_b + t * stride_mask_t
            )
            token_vec = tl.load(
                item_embedding_ptr + token_id * stride_item_vocab + offs_d * stride_item_d,
                mask=dim_mask,
                other=0.0,
            )
            seq_acc += token_vec.to(tl.float32) * token_mask.to(tl.float32)

        for table_idx in range(0, NUM_TABLES):
            context_id = tl.load(
                context_ids_ptr
                + batch_idx * stride_context_ids_b
                + table_idx * stride_context_ids_t
            )
            context_vec = tl.load(
                context_embedding_ptr
                + table_idx * stride_context_table
                + context_id * stride_context_vocab
                + offs_d * stride_context_d,
                mask=dim_mask,
                other=0.0,
            )
            context_acc += context_vec.to(tl.float32)

        length_recip = tl.load(length_recip_ptr + batch_idx * stride_recip_b).to(tl.float32)
        tl.store(
            out_ptr + batch_idx * stride_out_b + offs_d * stride_out_d,
            seq_acc * length_recip + context_acc,
            mask=dim_mask,
        )

    @triton.jit
    def _candidate_dot_kernel(
        user_ptr,
        item_embedding_ptr,
        candidate_ids_ptr,
        out_ptr,
        batch_size,
        num_candidates,
        embedding_dim,
        stride_user_b,
        stride_user_d,
        stride_item_vocab,
        stride_item_d,
        stride_candidate_ids_b,
        stride_candidate_ids_c,
        stride_out_b,
        stride_out_c,
        BLOCK_C: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_idx = tl.program_id(0)
        candidate_block_idx = tl.program_id(1)

        offs_c = candidate_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        candidate_ids = tl.load(
            candidate_ids_ptr
            + batch_idx * stride_candidate_ids_b
            + offs_c * stride_candidate_ids_c,
            mask=offs_c < num_candidates,
            other=0,
        )

        for d_start in range(0, embedding_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < embedding_dim

            user = tl.load(
                user_ptr + batch_idx * stride_user_b + offs_d * stride_user_d,
                mask=mask_d,
                other=0.0,
            )
            cand = tl.load(
                item_embedding_ptr
                + candidate_ids[:, None] * stride_item_vocab
                + offs_d[None, :] * stride_item_d,
                mask=(offs_c[:, None] < num_candidates) & mask_d[None, :],
                other=0.0,
            )
            acc += tl.sum(cand.to(tl.float32) * user[None, :].to(tl.float32), axis=1)

        tl.store(
            out_ptr + batch_idx * stride_out_b + offs_c * stride_out_c,
            acc,
            mask=offs_c < num_candidates,
        )


def sequence_mean_triton(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
    workspace: RankingWorkspace,
) -> torch.Tensor:
    """Pool item history with one fused Triton gather-mask-reduce kernel."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not inputs.sequence_ids.is_cuda:
        raise RuntimeError("Triton sequence pooling requires CUDA tensors")

    ensure_triton_compat()
    grid = (
        inputs.sequence_ids.shape[0],
        triton.cdiv(state.item_embeddings.shape[1], 64),
    )
    _sequence_mean_kernel[grid](
        inputs.sequence_ids,
        state.item_embeddings,
        workspace.sequence_mask_float,
        workspace.sequence_length_recip,
        out,
        state.item_embeddings.shape[1],
        inputs.sequence_ids.stride(0),
        inputs.sequence_ids.stride(1),
        state.item_embeddings.stride(0),
        state.item_embeddings.stride(1),
        workspace.sequence_mask_float.stride(0),
        workspace.sequence_mask_float.stride(1),
        workspace.sequence_length_recip.stride(0),
        out.stride(0),
        out.stride(1),
        SEQ_LEN=inputs.sequence_ids.shape[1],
        BLOCK_D=64,
    )
    return out


def context_sum_triton(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pool context tables with one fused Triton gather-reduce kernel."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not inputs.context_ids.is_cuda:
        raise RuntimeError("Triton context pooling requires CUDA tensors")

    ensure_triton_compat()
    grid = (
        inputs.context_ids.shape[0],
        triton.cdiv(state.context_embeddings.shape[2], 64),
    )
    _context_sum_kernel[grid](
        inputs.context_ids,
        state.context_embeddings,
        out,
        state.context_embeddings.shape[2],
        inputs.context_ids.stride(0),
        inputs.context_ids.stride(1),
        state.context_embeddings.stride(0),
        state.context_embeddings.stride(1),
        state.context_embeddings.stride(2),
        out.stride(0),
        out.stride(1),
        NUM_TABLES=inputs.context_ids.shape[1],
        BLOCK_D=64,
    )
    return out


def sequence_context_user_input_triton(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
    workspace: RankingWorkspace,
) -> torch.Tensor:
    """Fuse sequence pooling and context pooling into the tower input buffer."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not inputs.sequence_ids.is_cuda or not inputs.context_ids.is_cuda:
        raise RuntimeError("Triton user-input pooling requires CUDA tensors")

    ensure_triton_compat()
    grid = (
        inputs.sequence_ids.shape[0],
        triton.cdiv(state.item_embeddings.shape[1], 64),
    )
    _sequence_context_user_input_kernel[grid](
        inputs.sequence_ids,
        state.item_embeddings,
        workspace.sequence_mask_float,
        workspace.sequence_length_recip,
        inputs.context_ids,
        state.context_embeddings,
        out,
        state.item_embeddings.shape[1],
        inputs.sequence_ids.stride(0),
        inputs.sequence_ids.stride(1),
        state.item_embeddings.stride(0),
        state.item_embeddings.stride(1),
        workspace.sequence_mask_float.stride(0),
        workspace.sequence_mask_float.stride(1),
        workspace.sequence_length_recip.stride(0),
        inputs.context_ids.stride(0),
        inputs.context_ids.stride(1),
        state.context_embeddings.stride(0),
        state.context_embeddings.stride(1),
        state.context_embeddings.stride(2),
        out.stride(0),
        out.stride(1),
        SEQ_LEN=inputs.sequence_ids.shape[1],
        NUM_TABLES=inputs.context_ids.shape[1],
        BLOCK_D=64,
    )
    return out


def candidate_scores_triton(
    user_vec: torch.Tensor,
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
) -> torch.Tensor:
    """Score candidates with a Triton kernel."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not user_vec.is_cuda:
        raise RuntimeError("Triton candidate scoring requires CUDA tensors")

    ensure_triton_compat()
    user_vec = user_vec.contiguous()
    grid = (inputs.candidate_ids.shape[0], triton.cdiv(inputs.candidate_ids.shape[1], 64))
    _candidate_dot_kernel[grid](
        user_vec,
        state.item_embeddings,
        inputs.candidate_ids,
        out,
        inputs.candidate_ids.shape[0],
        inputs.candidate_ids.shape[1],
        user_vec.shape[1],
        user_vec.stride(0),
        user_vec.stride(1),
        state.item_embeddings.stride(0),
        state.item_embeddings.stride(1),
        inputs.candidate_ids.stride(0),
        inputs.candidate_ids.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_C=64,
        BLOCK_D=32,
    )
    return out


def user_input_vectorized(
    inputs: RankingInputs,
    state: RankingModelState,
    workspace: RankingWorkspace | None = None,
) -> torch.Tensor:
    if workspace is not None and workspace.sequence_metadata_key != _sequence_metadata_key(inputs):
        prepare_workspace_for_inputs(inputs, workspace)
    if (
        workspace is not None
        and workspace.context_metadata_key != _context_metadata_key(inputs, state)
    ):
        prepare_context_workspace_for_inputs(inputs, state, workspace)
    if (
        workspace is not None
        and TRITON_AVAILABLE
        and inputs.sequence_ids.is_cuda
        and inputs.context_ids.is_cuda
        and state.item_embeddings.is_cuda
        and state.context_embeddings.is_cuda
        and inputs.sequence_ids.shape[1] > 0
        and inputs.context_ids.shape[1] > 0
        and not (
            torch.is_grad_enabled()
            and (state.item_embeddings.requires_grad or state.context_embeddings.requires_grad)
        )
    ):
        return sequence_context_user_input_triton(
            inputs,
            state,
            workspace.sequence_accum,
            workspace,
        )

    seq_vec = sequence_mean_vectorized(inputs, state, workspace)
    context_vec = context_sum_vectorized(inputs, state, workspace)
    return seq_vec.add_(context_vec)


def resolve_score_backend(requested: str) -> str:
    """Pick a score backend while keeping the selection explicit."""

    if requested == "auto":
        return "triton" if TRITON_AVAILABLE else "torch"
    return requested


def baseline_forward(
    inputs: RankingInputs,
    state: RankingModelState,
    workspace: RankingWorkspace,
) -> torch.Tensor:
    """Execute the conservative sparse-ranking path."""

    seq_vec = sequence_mean_baseline(inputs, state, workspace.sequence_accum, workspace)
    context_vec = context_sum_baseline(inputs, state, workspace.context_accum)
    user_vec = state.tower(seq_vec + context_vec)
    return candidate_scores_baseline(user_vec, inputs, state, workspace.score_output)


def optimized_forward(
    inputs: RankingInputs,
    state: RankingModelState,
    *,
    compiled_tower: nn.Module | None = None,
    score_backend: str,
    workspace: RankingWorkspace | None = None,
) -> torch.Tensor:
    """Execute the vectorized sparse-ranking path."""

    user_input = user_input_vectorized(inputs, state, workspace)
    tower = compiled_tower if compiled_tower is not None else state.tower
    user_vec = tower(user_input)
    if score_backend == "triton":
        if workspace is None:
            raise RuntimeError("Triton scoring requires a RankingWorkspace")
        return candidate_scores_triton(user_vec, inputs, state, workspace.score_output)
    if workspace is None:
        return candidate_scores_torch(user_vec, inputs, state)
    return candidate_scores_torch(
        user_vec,
        inputs,
        state,
        workspace.score_output,
        workspace.candidate_embedding_flat,
        workspace.candidate_embedding_f32,
        workspace.user_vec_f32,
    )


def ranking_metrics(
    workload: SequenceRankingWorkload,
    inputs: RankingInputs,
    *,
    score_backend: str,
    compile_enabled: bool,
) -> dict:
    return {
        "ranking.avg_sequence_length": inputs.avg_sequence_length,
        "ranking.num_tables": float(workload.num_tables),
        "ranking.num_candidates": float(workload.num_candidates),
        "ranking.hot_candidate_share_pct": inputs.hot_candidate_share_pct,
        "ranking.compile_enabled": 1.0 if compile_enabled else 0.0,
        "ranking.score_backend_triton": 1.0 if score_backend == "triton" else 0.0,
    }


def warm_optimized_path(
    workload: SequenceRankingWorkload,
    inputs: RankingInputs,
    state: RankingModelState,
    *,
    compiled_tower: nn.Module | None,
    score_backend: str,
    workspace: RankingWorkspace | None = None,
) -> None:
    """Pay one-time compile/autotune costs before the measured loop."""

    with torch.inference_mode():
        user_input = user_input_vectorized(inputs, state, workspace)
        tower = compiled_tower if compiled_tower is not None else state.tower
        user_vec = tower(user_input)
        if score_backend == "triton":
            if workspace is None:
                raise RuntimeError("Triton scoring requires a RankingWorkspace")
            _ = candidate_scores_triton(user_vec, inputs, state, workspace.score_output)
        elif workspace is None:
            _ = candidate_scores_torch(user_vec, inputs, state)
        else:
            _ = candidate_scores_torch(
                user_vec,
                inputs,
                state,
                workspace.score_output,
                workspace.candidate_embedding_flat,
                workspace.candidate_embedding_f32,
                workspace.user_vec_f32,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


__all__ = [
    "RankingInputs",
    "RankingModelState",
    "SequenceRankingTower",
    "SequenceRankingWorkload",
    "TRITON_AVAILABLE",
    "apply_cli_overrides",
    "build_workspace",
    "build_inputs",
    "build_model_state",
    "baseline_forward",
    "candidate_scores_baseline",
    "candidate_scores_torch",
    "candidate_scores_triton",
    "context_sum_baseline",
    "context_sum_triton",
    "context_sum_vectorized",
    "default_workload",
    "optimized_forward",
    "prepare_workspace_for_inputs",
    "prepare_context_workspace_for_inputs",
    "ranking_metrics",
    "RankingWorkspace",
    "requests_per_iteration",
    "resolve_score_backend",
    "sequence_mean_baseline",
    "sequence_context_user_input_triton",
    "sequence_mean_triton",
    "sequence_mean_vectorized",
    "tokens_per_iteration",
    "user_input_vectorized",
    "warm_optimized_path",
]
