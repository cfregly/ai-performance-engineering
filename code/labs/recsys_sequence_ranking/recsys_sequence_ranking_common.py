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
    sequence_mask_float: torch.Tensor
    sequence_length_recip: torch.Tensor
    context_table_index: torch.Tensor
    sequence_metadata_key: tuple[int, int] | None = None


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
    input_stats = torch.stack(
        (
            lengths.to(torch.float32).mean(),
            (candidate_ids < hot_threshold).to(torch.float32).mean() * 100.0,
        )
    ).tolist()
    avg_sequence_length = float(input_stats[0])
    hot_candidate_share_pct = float(input_stats[1])

    return RankingInputs(
        sequence_ids=sequence_ids.to(device=device, dtype=torch.int64),
        sequence_mask=sequence_mask.to(device=device),
        sequence_lengths=lengths.to(device=device, dtype=torch.int64),
        context_ids=context_ids.to(device=device, dtype=torch.int64),
        candidate_ids=candidate_ids.to(device=device, dtype=torch.int64),
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
    with torch.no_grad():
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
    context_table_index = (
        context_table_index.view(1, workload.num_tables).expand(workload.batch_size, -1).clone()
    )
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
    )


def _sequence_metadata_key(inputs: RankingInputs) -> tuple[int, int]:
    return (inputs.sequence_mask.data_ptr(), inputs.sequence_lengths.data_ptr())


def prepare_workspace_for_inputs(inputs: RankingInputs, workspace: RankingWorkspace) -> None:
    """Cache immutable sequence metadata derived from the benchmark inputs."""

    workspace.sequence_mask_float.copy_(inputs.sequence_mask.unsqueeze(-1))
    workspace.sequence_length_recip.copy_(inputs.sequence_lengths.unsqueeze(1))
    workspace.sequence_length_recip.clamp_min_(1).reciprocal_()
    workspace.sequence_metadata_key = _sequence_metadata_key(inputs)


def sequence_mean_baseline(
    inputs: RankingInputs,
    state: RankingModelState,
    out: torch.Tensor,
) -> torch.Tensor:
    """Conservative sequence pooling using one embedding lookup per time step."""

    mask = inputs.sequence_mask.to(dtype=state.item_embeddings.dtype)
    if inputs.sequence_ids.shape[1] == 0:
        out.zero_()
        return out
    token_vec = state.item_embeddings[inputs.sequence_ids[:, 0]]
    out.copy_(token_vec * mask[:, 0:1])
    for t in range(1, inputs.sequence_ids.shape[1]):
        token_vec = state.item_embeddings[inputs.sequence_ids[:, t]]
        out.add_(token_vec * mask[:, t : t + 1])
    lengths = inputs.sequence_lengths.to(dtype=state.item_embeddings.dtype).clamp_min(1)
    out.div_(lengths.unsqueeze(1))
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
        out += state.context_embeddings[table_idx, inputs.context_ids[:, table_idx]]
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
    seq_emb = F.embedding(inputs.sequence_ids, state.item_embeddings)
    if workspace is None:
        mask = inputs.sequence_mask.to(dtype=seq_emb.dtype).unsqueeze(-1)
        lengths = inputs.sequence_lengths.to(dtype=seq_emb.dtype).clamp_min_(1).unsqueeze(1)
        return (seq_emb * mask).sum(dim=1) / lengths

    if workspace.sequence_metadata_key != _sequence_metadata_key(inputs):
        prepare_workspace_for_inputs(inputs, workspace)
    if torch.is_grad_enabled() and seq_emb.requires_grad:
        return (seq_emb * workspace.sequence_mask_float).sum(dim=1) * workspace.sequence_length_recip
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
    else:
        table_index = workspace.context_table_index
    context_vecs = state.context_embeddings[table_index, inputs.context_ids]
    if workspace is None or (torch.is_grad_enabled() and context_vecs.requires_grad):
        return context_vecs.sum(dim=1)
    torch.sum(context_vecs, dim=1, out=workspace.context_accum)
    return workspace.context_accum


def candidate_scores_torch(
    user_vec: torch.Tensor, inputs: RankingInputs, state: RankingModelState
) -> torch.Tensor:
    candidate_emb = F.embedding(inputs.candidate_ids, state.item_embeddings)
    return torch.bmm(
        candidate_emb.to(torch.float32), user_vec.to(torch.float32).unsqueeze(2)
    ).squeeze(2)


if TRITON_AVAILABLE:

    @triton.jit
    def _candidate_dot_kernel(
        user_ptr,
        candidate_ptr,
        out_ptr,
        batch_size,
        num_candidates,
        embedding_dim,
        stride_user_b,
        stride_user_d,
        stride_candidate_b,
        stride_candidate_c,
        stride_candidate_d,
        stride_out_b,
        stride_out_c,
        BLOCK_C: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        batch_idx = tl.program_id(0)
        candidate_block_idx = tl.program_id(1)

        offs_c = candidate_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for d_start in range(0, embedding_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < embedding_dim

            user = tl.load(
                user_ptr + batch_idx * stride_user_b + offs_d * stride_user_d,
                mask=mask_d,
                other=0.0,
            )
            cand = tl.load(
                candidate_ptr
                + batch_idx * stride_candidate_b
                + offs_c[:, None] * stride_candidate_c
                + offs_d[None, :] * stride_candidate_d,
                mask=(offs_c[:, None] < num_candidates) & mask_d[None, :],
                other=0.0,
            )
            acc += tl.sum(cand.to(tl.float32) * user[None, :].to(tl.float32), axis=1)

        tl.store(
            out_ptr + batch_idx * stride_out_b + offs_c * stride_out_c,
            acc,
            mask=offs_c < num_candidates,
        )


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
    candidate_emb = F.embedding(inputs.candidate_ids, state.item_embeddings).contiguous()
    user_vec = user_vec.contiguous()
    grid = (inputs.candidate_ids.shape[0], triton.cdiv(inputs.candidate_ids.shape[1], 64))
    _candidate_dot_kernel[grid](
        user_vec,
        candidate_emb,
        out,
        inputs.candidate_ids.shape[0],
        inputs.candidate_ids.shape[1],
        user_vec.shape[1],
        user_vec.stride(0),
        user_vec.stride(1),
        candidate_emb.stride(0),
        candidate_emb.stride(1),
        candidate_emb.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_C=64,
        BLOCK_D=32,
    )
    return out


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

    seq_vec = sequence_mean_baseline(inputs, state, workspace.sequence_accum)
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

    seq_vec = sequence_mean_vectorized(inputs, state, workspace)
    context_vec = context_sum_vectorized(inputs, state, workspace)
    tower = compiled_tower if compiled_tower is not None else state.tower
    user_vec = tower(seq_vec.add_(context_vec))
    if score_backend == "triton":
        if workspace is None:
            raise RuntimeError("Triton scoring requires a RankingWorkspace")
        return candidate_scores_triton(user_vec, inputs, state, workspace.score_output)
    return candidate_scores_torch(user_vec, inputs, state)


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

    with torch.no_grad():
        seq_vec = sequence_mean_vectorized(inputs, state, workspace)
        context_vec = context_sum_vectorized(inputs, state, workspace)
        user_input = seq_vec.add_(context_vec)
        tower = compiled_tower if compiled_tower is not None else state.tower
        user_vec = tower(user_input)
        if score_backend == "triton":
            if workspace is None:
                raise RuntimeError("Triton scoring requires a RankingWorkspace")
            _ = candidate_scores_triton(user_vec, inputs, state, workspace.score_output)
        else:
            _ = candidate_scores_torch(user_vec, inputs, state)
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
    "context_sum_vectorized",
    "default_workload",
    "optimized_forward",
    "prepare_workspace_for_inputs",
    "ranking_metrics",
    "RankingWorkspace",
    "requests_per_iteration",
    "resolve_score_backend",
    "sequence_mean_baseline",
    "sequence_mean_vectorized",
    "tokens_per_iteration",
    "warm_optimized_path",
]
