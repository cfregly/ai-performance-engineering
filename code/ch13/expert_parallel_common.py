"""Shared helpers for expert-parallel all-to-all benchmarks (multi-GPU)."""

from __future__ import annotations

from dataclasses import dataclass
import os

from core.common.device_utils import resolve_local_rank
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class ExpertParallelConfig:
    batch_size: int = 8
    seq_len: int = 2048
    hidden_size: int = 1024
    dtype: torch.dtype = torch.bfloat16


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Expert-parallel benchmarks require torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def build_routing(
    *,
    tokens_per_rank: int,
    world_size: int,
    rank: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[int]]:
    indices = torch.arange(tokens_per_rank, device=device)
    dest = (indices * 13 + rank) % world_size
    send_indices = [torch.nonzero(dest == dst, as_tuple=False).squeeze(-1) for dst in range(world_size)]
    send_splits = [int(idx.numel()) for idx in send_indices]
    if sum(send_splits) != tokens_per_rank:
        raise RuntimeError("Routing split sizes do not sum to tokens_per_rank")
    return send_indices, send_splits


def gather_recv_splits(send_splits: List[int], world_size: int, rank: int) -> List[int]:
    if world_size <= 1 or not dist.is_initialized():
        return [send_splits[0]]
    send_counts = torch.tensor(send_splits, dtype=torch.int64, device="cuda")
    gathered = [torch.empty_like(send_counts) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts)
    recv_counts = torch.empty(world_size, dtype=send_counts.dtype, device=send_counts.device)
    for src, counts in enumerate(gathered):
        recv_counts[src].copy_(counts[rank])
    recv_counts_host = recv_counts.detach().cpu()
    return [int(recv_counts_host[src]) for src in range(recv_counts_host.numel())]


def pack_tokens(
    *,
    tokens: torch.Tensor,
    send_indices: List[torch.Tensor],
    send_splits: List[int],
    send_buf: torch.Tensor,
) -> None:
    offset = 0
    for idx, count in zip(send_indices, send_splits):
        if count:
            send_buf[offset:offset + count].copy_(tokens[idx])
        offset += count


def split_buffer_views(buffer: torch.Tensor, splits: List[int]) -> List[torch.Tensor]:
    views: List[torch.Tensor] = []
    offset = 0
    for count in splits:
        next_offset = offset + int(count)
        views.append(buffer[offset:next_offset])
        offset = next_offset
    return views


def run_expert_parallel(
    *,
    config: ExpertParallelConfig,
    iters: int,
    warmup: int,
    impl: str,
) -> None:
    rank, world_size, local_rank = init_distributed()
    if world_size < 2:
        raise RuntimeError("SKIPPED: Requires >= 2 GPUs (found 1 GPU)")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    tokens_per_rank = config.batch_size * config.seq_len
    tokens = torch.randn(tokens_per_rank, config.hidden_size, device=device, dtype=config.dtype)
    send_indices, send_splits = build_routing(
        tokens_per_rank=tokens_per_rank,
        world_size=world_size,
        rank=rank,
        device=device,
    )
    recv_splits = gather_recv_splits(send_splits, world_size, rank)
    recv_total = int(sum(recv_splits))
    expert_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=config.dtype).to(device)

    send_buf = torch.empty((tokens_per_rank, config.hidden_size), device=device, dtype=config.dtype)
    recv_buf = torch.empty((recv_total, config.hidden_size), device=device, dtype=config.dtype)
    send_views = split_buffer_views(send_buf, send_splits)
    recv_views = split_buffer_views(recv_buf, recv_splits)
    if impl == "list":
        pack_tokens(tokens=tokens, send_indices=send_indices, send_splits=send_splits, send_buf=send_buf)

    def _all_to_all_list() -> torch.Tensor:
        dist.all_to_all(recv_views, send_views)
        return recv_buf

    def _all_to_all_single() -> torch.Tensor:
        pack_tokens(tokens=tokens, send_indices=send_indices, send_splits=send_splits, send_buf=send_buf)
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        return recv_buf

    def _step() -> torch.Tensor:
        if impl == "list":
            recv = _all_to_all_list()
        elif impl == "single":
            recv = _all_to_all_single()
        else:
            raise ValueError(f"Unsupported impl '{impl}'")
        return expert_proj(recv)

    for _ in range(max(warmup, 0)):
        _step()
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(iters, 1)):
        _step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    if rank == 0:
        tokens_global = tokens_per_rank * world_size
        tokens_per_s = tokens_global * (max(iters, 1) / max(elapsed, 1e-9))
        print(f"rank0 tokens/s: {tokens_per_s:.2f} tokens/s")
        print(f"rank0 time_per_iter_ms: {(elapsed / max(iters, 1)) * 1000.0:.3f}")

    dist.barrier()
    dist.destroy_process_group()
