"""Shared helpers for context-parallel attention benchmarks (multi-GPU)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

from core.common.device_utils import resolve_local_rank
import time
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class ContextParallelConfig:
    batch_size: int = 1
    seq_len: int = 8192  # total sequence length across ranks
    hidden_size: int = 1024
    num_heads: int = 8
    num_layers: int = 2
    dtype: torch.dtype = torch.bfloat16
    causal: bool = True


@dataclass
class AttentionWorkspace:
    gather_k: Optional[list[torch.Tensor]] = None
    gather_v: Optional[list[torch.Tensor]] = None
    k_full: Optional[torch.Tensor] = None
    v_full: Optional[torch.Tensor] = None
    recv_k: Optional[torch.Tensor] = None
    recv_v: Optional[torch.Tensor] = None
    recv_k_alt: Optional[torch.Tensor] = None
    recv_v_alt: Optional[torch.Tensor] = None


_CAUSAL_MASK_POSITION_CACHE: dict[tuple[int, int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_CAUSAL_MASK_CACHE: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
_RING_POSITION_VIEW_CACHE: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_RING_CAUSAL_MASK_CACHE: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def align_seq_len(seq_len: int, world_size: int) -> int:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if seq_len % world_size == 0:
        return seq_len
    return world_size * ((seq_len + world_size - 1) // world_size)


def build_attention_workspace(
    *,
    batch_size: int,
    num_heads: int,
    seq_shard: int,
    head_dim: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> AttentionWorkspace:
    shape = (batch_size, num_heads, seq_shard, head_dim)
    full_shape = (batch_size, num_heads, seq_shard * world_size, head_dim)
    if world_size <= 1:
        return AttentionWorkspace()
    return AttentionWorkspace(
        gather_k=[torch.empty(shape, device=device, dtype=dtype) for _ in range(world_size)],
        gather_v=[torch.empty(shape, device=device, dtype=dtype) for _ in range(world_size)],
        k_full=torch.empty(full_shape, device=device, dtype=dtype),
        v_full=torch.empty(full_shape, device=device, dtype=dtype),
        recv_k=torch.empty(shape, device=device, dtype=dtype),
        recv_v=torch.empty(shape, device=device, dtype=dtype),
        recv_k_alt=torch.empty(shape, device=device, dtype=dtype),
        recv_v_alt=torch.empty(shape, device=device, dtype=dtype),
    )


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Context-parallel benchmarks require torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


class ContextParallelLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False, dtype=dtype)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def split_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return q, k, v

    def merge_heads(self, attn: torch.Tensor) -> torch.Tensor:
        return attn.transpose(1, 2).contiguous().view(attn.shape[0], attn.shape[2], self.hidden_size)


def _apply_causal_mask(
    scores: torch.Tensor,
    *,
    rank: int,
    seq_shard: int,
    world_size: int,
) -> torch.Tensor:
    mask = _causal_mask_for(rank, seq_shard, world_size, scores.device)
    return scores.masked_fill_(mask, float("-inf"))


def _causal_mask_position_views(
    rank: int,
    seq_shard: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(rank), int(seq_shard), int(world_size), torch.device(device))
    cached = _CAUSAL_MASK_POSITION_CACHE.get(key)
    if cached is None:
        seq_total = int(seq_shard) * int(world_size)
        local_positions = torch.arange(seq_shard, device=device)
        global_q = (int(rank) * int(seq_shard) + local_positions).view(1, 1, seq_shard, 1)
        global_k = torch.arange(seq_total, device=device).view(1, 1, 1, seq_total)
        cached = (global_q, global_k)
        _CAUSAL_MASK_POSITION_CACHE[key] = cached
    return cached


def _causal_mask_for(
    rank: int,
    seq_shard: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    key = (int(rank), int(seq_shard), int(world_size), torch.device(device))
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        global_q, global_k = _causal_mask_position_views(rank, seq_shard, world_size, device)
        mask = global_k > global_q
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _ring_position_views(
    rank: int,
    seq_shard: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(rank), int(seq_shard), torch.device(device))
    cached = _RING_POSITION_VIEW_CACHE.get(key)
    if cached is None:
        local_positions = torch.arange(seq_shard, device=device)
        global_q = (int(rank) * int(seq_shard) + local_positions).view(1, 1, seq_shard, 1)
        k_indices = local_positions.view(1, 1, 1, seq_shard)
        cached = (global_q, k_indices)
        _RING_POSITION_VIEW_CACHE[key] = cached
    return cached


def _ring_causal_mask_for(
    rank: int,
    target_rank: int,
    seq_shard: int,
    device: torch.device,
) -> torch.Tensor:
    key = (int(rank), int(target_rank), int(seq_shard), torch.device(device))
    mask = _RING_CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        global_q, k_indices = _ring_position_views(rank, seq_shard, device)
        global_k = int(target_rank) * int(seq_shard) + k_indices
        mask = global_k > global_q
        _RING_CAUSAL_MASK_CACHE[key] = mask
    return mask


def all_gather_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
    causal: bool,
    seq_shard: int,
    scale: float,
    workspace: Optional[AttentionWorkspace] = None,
) -> torch.Tensor:
    if world_size > 1 and dist.is_initialized():
        if (
            workspace is None
            or workspace.gather_k is None
            or workspace.gather_v is None
            or workspace.k_full is None
            or workspace.v_full is None
        ):
            raise RuntimeError("all_gather_attention() requires preallocated gather buffers when world_size > 1")
        gather_k = workspace.gather_k
        gather_v = workspace.gather_v
        dist.all_gather(gather_k, k, group=process_group)
        dist.all_gather(gather_v, v, group=process_group)
        torch.cat(gather_k, dim=2, out=workspace.k_full)
        torch.cat(gather_v, dim=2, out=workspace.v_full)
        k_full = workspace.k_full
        v_full = workspace.v_full
    else:
        k_full = k
        v_full = v

    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    if causal:
        scores = _apply_causal_mask(scores, rank=rank, seq_shard=seq_shard, world_size=world_size)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_full)


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    process_group: Optional[dist.ProcessGroup],
    causal: bool,
    seq_shard: int,
    scale: float,
    workspace: Optional[AttentionWorkspace] = None,
) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return all_gather_attention(
            q,
            k,
            v,
            rank=rank,
            world_size=1,
            process_group=None,
            causal=causal,
            seq_shard=seq_shard,
            scale=scale,
        )

    k_current = k
    v_current = v
    attn_num: Optional[torch.Tensor] = None
    global_max: Optional[torch.Tensor] = None
    global_sum: Optional[torch.Tensor] = None

    for step in range(world_size):
        target_rank = (rank - step) % world_size
        scores = torch.matmul(q, k_current.transpose(-2, -1)) * scale

        if causal:
            mask = _ring_causal_mask_for(rank, target_rank, seq_shard, q.device)
            scores.masked_fill_(mask, float("-inf"))

        local_max = scores.amax(dim=-1, keepdim=True)
        exp_scores = torch.exp(scores - local_max)
        local_sum = exp_scores.sum(dim=-1, keepdim=True)
        local_num = torch.matmul(exp_scores, v_current)

        if global_max is None:
            global_max = local_max
            global_sum = local_sum
            attn_num = local_num
        else:
            new_max = torch.maximum(global_max, local_max)
            scale_prev = torch.exp(global_max - new_max)
            scale_local = torch.exp(local_max - new_max)
            attn_num = attn_num * scale_prev + local_num * scale_local  # type: ignore[assignment]
            global_sum = global_sum * scale_prev + local_sum * scale_local
            global_max = new_max

        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            if (
                workspace is None
                or workspace.recv_k is None
                or workspace.recv_v is None
                or workspace.recv_k_alt is None
                or workspace.recv_v_alt is None
            ):
                raise RuntimeError("ring_attention() requires preallocated recv buffers when world_size > 1")
            if step & 1:
                k_recv = workspace.recv_k_alt
                v_recv = workspace.recv_v_alt
            else:
                k_recv = workspace.recv_k
                v_recv = workspace.recv_v

            ops = [
                dist.P2POp(dist.isend, k_current, next_rank, group=process_group),
                dist.P2POp(dist.irecv, k_recv, prev_rank, group=process_group),
                dist.P2POp(dist.isend, v_current, next_rank, group=process_group),
                dist.P2POp(dist.irecv, v_recv, prev_rank, group=process_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            k_current = k_recv
            v_current = v_recv

    if attn_num is None or global_sum is None:
        raise RuntimeError("Ring attention accumulation failed")
    return attn_num / (global_sum + 1e-8)


def build_layers(config: ContextParallelConfig, device: torch.device) -> nn.ModuleList:
    return nn.ModuleList(
        [ContextParallelLayer(config.hidden_size, config.num_heads, config.dtype) for _ in range(config.num_layers)]
    ).to(device=device)


def run_context_parallel(
    *,
    config: ContextParallelConfig,
    iters: int,
    warmup: int,
    attention_fn: Callable[..., torch.Tensor],
) -> None:
    rank, world_size, local_rank = init_distributed()
    seq_len = align_seq_len(config.seq_len, world_size)
    seq_shard = seq_len // world_size

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    layers = build_layers(config, device)
    workspace = build_attention_workspace(
        batch_size=config.batch_size,
        num_heads=config.num_heads,
        seq_shard=seq_shard,
        head_dim=config.hidden_size // config.num_heads,
        world_size=world_size,
        dtype=config.dtype,
        device=device,
    )
    inputs = torch.randn(
        config.batch_size,
        seq_shard,
        config.hidden_size,
        device=device,
        dtype=config.dtype,
    )

    def _step() -> torch.Tensor:
        x = inputs
        for layer in layers:
            q, k, v = layer.split_qkv(x)
            attn_out = attention_fn(
                q,
                k,
                v,
                rank=rank,
                world_size=world_size,
                process_group=dist.group.WORLD,
                causal=config.causal,
                seq_shard=seq_shard,
                scale=layer.scale,
                workspace=workspace,
            )
            x = layer.proj(layer.merge_heads(attn_out))
        return x

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _step()
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(max(iters, 1)):
            _step()
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if rank == 0:
        tokens_per_iter = config.batch_size * seq_len
        tokens_per_s = tokens_per_iter * (max(iters, 1) / max(elapsed, 1e-9))
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters, 1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()
