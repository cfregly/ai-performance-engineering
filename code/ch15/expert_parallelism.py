#!/usr/bin/env python3
"""ch15/expert_parallelism.py - MoE expert-parallelism demo (tool).

This is a *tool* (not a comparable baseline/optimized benchmark pair).

Single-GPU local-overlap demo:
  python ch15/expert_parallelism.py --mode local

Distributed all-to-all demo (requires torchrun):
  torchrun --nproc_per_node <num_gpus> ch15/expert_parallelism.py --mode distributed

Or via the CLI:
  python -m cli.aisp demos ch15-expert-parallel -- --mode local
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def _init_distributed() -> DistributedContext:
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for distributed mode")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        raise RuntimeError("Missing LOCAL_RANK. Run distributed mode via torchrun.")
    local_rank = int(local_rank_str)
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for cuda.device_count()={torch.cuda.device_count()}"
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return DistributedContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def _expert_to_rank(expert_id: int, experts_per_rank: int) -> int:
    return expert_id // experts_per_rank


class Top2MoE(nn.Module):
    """Top-2 MoE with capacity factor and explicit local/distributed modes."""

    def __init__(self, hidden_dim: int, num_experts: int, capacity_factor: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self._local_streams: Optional[list[torch.cuda.Stream]] = None

    def init_local_streams(self, device: torch.device) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for local overlap streams")
        self._local_streams = [torch.cuda.Stream(device=device) for _ in range(2)]

    def forward_local(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._local_streams is None:
            raise RuntimeError("init_local_streams() must be called before forward_local()")
        batch, seq, hidden = tokens.shape
        probs = F.softmax(self.gate(tokens), dim=-1)
        top2_w, top2_idx = torch.topk(probs, k=2, dim=-1)
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        cap = int(self.capacity_factor * (batch * seq) / self.num_experts)
        counts = torch.bincount(flat_idx.view(-1), minlength=self.num_experts)
        mask_overflow = counts > cap
        overflow_flags = [bool(flag) for flag in mask_overflow.detach().cpu().tolist()]

        partials: list[torch.Tensor] = []
        for slot, stream in enumerate(self._local_streams):
            expert_ids = flat_idx[:, slot]
            local_out = torch.zeros_like(flat_tokens)
            unique_expert_ids = [int(eid) for eid in torch.unique(expert_ids).detach().cpu().tolist()]
            with torch.cuda.stream(stream):
                for eid_int in unique_expert_ids:
                    if overflow_flags[eid_int]:
                        continue
                    mask = expert_ids == eid_int
                    contrib = self.experts[eid_int](flat_tokens[mask]) * flat_w[mask, slot:slot + 1]
                    local_out[mask] += contrib
            partials.append(local_out)

        current = torch.cuda.current_stream(tokens.device)
        for stream in self._local_streams:
            current.wait_stream(stream)
        return sum(partials).view(batch, seq, hidden)

    def forward_distributed(self, tokens: torch.Tensor, *, ctx: DistributedContext) -> torch.Tensor:
        batch, seq, hidden = tokens.shape
        probs = F.softmax(self.gate(tokens), dim=-1)
        top2_w, top2_idx = torch.topk(probs, k=2, dim=-1)
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        cap = int(self.capacity_factor * (batch * seq) / self.num_experts)
        counts = torch.bincount(flat_idx.view(-1), minlength=self.num_experts)
        mask_overflow = counts > cap
        overflow_flags = [bool(flag) for flag in mask_overflow.detach().cpu().tolist()]

        world_size = ctx.world_size
        rank = ctx.rank
        experts_per_rank = max(1, self.num_experts // world_size)

        top1 = flat_idx[:, 0]
        dest_ranks = top1 // experts_per_rank

        send_tokens: list[torch.Tensor] = []
        send_indices: list[torch.Tensor] = []
        send_expert_ids: list[torch.Tensor] = []
        for r in range(world_size):
            mask = dest_ranks == r
            send_tokens.append(flat_tokens[mask])
            send_indices.append(torch.nonzero(mask, as_tuple=False).view(-1))
            send_expert_ids.append(top1[mask])

        send_splits = [int(t.size(0)) for t in send_tokens]
        splits_all: list[list[int]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(splits_all, send_splits)
        recv_splits = [splits_all[r][rank] for r in range(world_size)]

        send_buf = torch.cat(send_tokens, dim=0) if send_tokens else flat_tokens[:0]
        send_ids = torch.cat(send_expert_ids, dim=0) if send_expert_ids else top1[:0]
        send_pos = (
            torch.cat(send_indices, dim=0)
            if send_indices
            else torch.empty(0, device=tokens.device, dtype=torch.int64)
        )

        total_recv = int(sum(recv_splits))
        recv_buf = torch.empty(total_recv, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_ids = torch.empty(total_recv, device=tokens.device, dtype=torch.int64)
        recv_pos = torch.empty(total_recv, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(recv_buf, send_buf, out_split_sizes=recv_splits, in_split_sizes=send_splits)
        dist.all_to_all_single(recv_ids, send_ids, out_split_sizes=recv_splits, in_split_sizes=send_splits)
        dist.all_to_all_single(recv_pos, send_pos, out_split_sizes=recv_splits, in_split_sizes=send_splits)

        local_out = torch.zeros_like(recv_buf)
        for eid_int in [int(eid) for eid in torch.unique(recv_ids).detach().cpu().tolist()]:
            if overflow_flags[eid_int]:
                continue
            if _expert_to_rank(eid_int, experts_per_rank) != rank:
                continue
            mask = recv_ids == eid_int
            local_out[mask] = self.experts[eid_int](recv_buf[mask])

        send_back_splits = recv_splits
        recv_back_splits = send_splits
        total_back = int(sum(recv_back_splits))
        recv_back_buf = torch.empty(total_back, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_back_pos = torch.empty(total_back, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(
            recv_back_buf, local_out, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits
        )
        dist.all_to_all_single(
            recv_back_pos, recv_pos, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits
        )

        out = torch.zeros_like(flat_tokens)
        out[recv_back_pos] = recv_back_buf
        return out.view(batch, seq, hidden)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE expert parallelism demo")
    parser.add_argument("--mode", type=str, required=True, choices=("local", "distributed"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def _dtype_from_string(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype '{name}'")


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_string(args.dtype)

    if args.mode == "distributed":
        ctx = _init_distributed()
        if ctx.world_size < 2:
            raise RuntimeError("distributed mode requires world_size >= 2")
        device = ctx.device
        is_rank0 = ctx.rank == 0
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for local mode")
        ctx = None
        device = torch.device("cuda")
        is_rank0 = True

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = Top2MoE(args.hidden_dim, args.num_experts, args.capacity_factor).to(device=device, dtype=dtype).eval()
    if args.mode == "local":
        model.init_local_streams(device)

    tokens = torch.randn(args.batch, args.seq, args.hidden_dim, device=device, dtype=dtype)
    torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            if args.mode == "distributed":
                out = model.forward_distributed(tokens, ctx=ctx)  # type: ignore[arg-type]
            else:
                out = model.forward_local(tokens)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            if args.mode == "distributed":
                out = model.forward_distributed(tokens, ctx=ctx)  # type: ignore[arg-type]
            else:
                out = model.forward_local(tokens)
        end.record()
        torch.cuda.synchronize(device)

    per_iter_ms = start.elapsed_time(end) / max(args.iters, 1)

    if args.mode == "distributed":
        t = torch.tensor([per_iter_ms], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        per_iter_ms = float(t.item())

    if is_rank0:
        total_tokens = args.batch * args.seq
        tokens_per_s = (total_tokens / (per_iter_ms / 1000.0)) if per_iter_ms > 0 else 0.0
        print(
            f"expert_parallelism:{args.mode} batch={args.batch} seq={args.seq} hidden={args.hidden_dim} "
            f"experts={args.num_experts} dtype={args.dtype} -> {per_iter_ms:.3f} ms/iter, {tokens_per_s:,.0f} tok/s"
        )


if __name__ == "__main__":
    main()
