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
import math
import os
from dataclasses import dataclass
from typing import Optional

from core.benchmark.utils import scalar_tensor_to_float

import torch
import torch.distributed as dist
import torch.nn as nn


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


def _accumulate_route_outputs(
    output: torch.Tensor,
    route_positions: torch.Tensor,
    weighted_route_outputs: torch.Tensor,
) -> torch.Tensor:
    """Sum every weighted expert route back into its originating token."""
    if route_positions.ndim != 1 or weighted_route_outputs.ndim != 2:
        raise ValueError("route positions must be 1D and route outputs must be 2D")
    if route_positions.numel() != weighted_route_outputs.size(0):
        raise ValueError("route positions and route outputs must contain the same number of routes")
    output.zero_()
    output.index_add_(0, route_positions, weighted_route_outputs)
    return output


def _weight_route_outputs(
    expert_outputs: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply one gate weight to each routed expert output."""
    if route_weights.ndim != 1 or route_weights.numel() != expert_outputs.size(0):
        raise ValueError("route weights must provide one scalar per expert output")
    return expert_outputs * route_weights.unsqueeze(-1)


def _expert_capacity(
    token_count: int,
    routes_per_token: int,
    num_experts: int,
    capacity_factor: float,
) -> int:
    """Return per-expert capacity for all routed token/expert pairs."""
    if token_count <= 0 or routes_per_token <= 0 or num_experts <= 0:
        raise ValueError("token, route, and expert counts must be positive")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    return max(
        1,
        math.ceil(capacity_factor * token_count * routes_per_token / num_experts),
    )


def _route_overflow_mask(expert_ids: torch.Tensor, capacity: int) -> torch.Tensor:
    """Mark only routes beyond each expert's capacity, preserving route order."""
    if expert_ids.ndim != 1:
        raise ValueError("expert_ids must be one-dimensional")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    overflow = torch.empty_like(expert_ids, dtype=torch.bool)
    route_count = expert_ids.numel()
    if route_count == 0:
        return overflow

    order = torch.argsort(expert_ids, stable=True)
    sorted_expert_ids = expert_ids[order]
    sorted_positions = torch.arange(route_count, device=expert_ids.device)
    new_expert = torch.ones(route_count, device=expert_ids.device, dtype=torch.bool)
    new_expert[1:] = sorted_expert_ids[1:] != sorted_expert_ids[:-1]
    group_starts = torch.where(
        new_expert,
        sorted_positions,
        torch.zeros_like(sorted_positions),
    ).cummax(dim=0).values
    sorted_overflow = (sorted_positions - group_starts) >= capacity
    overflow[order] = sorted_overflow
    return overflow


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
        self._local_out_buffers: list[torch.Tensor] = []
        self._local_accum_buffer: Optional[torch.Tensor] = None
        self._distributed_workspaces: dict[str, torch.Tensor] = {}

    def init_local_streams(self, device: torch.device) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for local overlap streams")
        self._local_streams = [torch.cuda.Stream(device=device) for _ in range(2)]

    def _local_buffers(self, flat_tokens: torch.Tensor, slots: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        needs_partials = (
            len(self._local_out_buffers) != slots
            or any(
                buf.shape != flat_tokens.shape
                or buf.device != flat_tokens.device
                or buf.dtype != flat_tokens.dtype
                for buf in self._local_out_buffers
            )
        )
        if needs_partials:
            self._local_out_buffers = [torch.empty_like(flat_tokens) for _ in range(slots)]
        if (
            self._local_accum_buffer is None
            or self._local_accum_buffer.shape != flat_tokens.shape
            or self._local_accum_buffer.device != flat_tokens.device
            or self._local_accum_buffer.dtype != flat_tokens.dtype
        ):
            self._local_accum_buffer = torch.empty_like(flat_tokens)
        return self._local_out_buffers, self._local_accum_buffer

    def _distributed_workspace(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        workspace = self._distributed_workspaces.get(name)
        if (
            workspace is None
            or workspace.shape != shape
            or workspace.device != device
            or workspace.dtype != dtype
        ):
            workspace = torch.empty(shape, device=device, dtype=dtype)
            self._distributed_workspaces[name] = workspace
        return workspace

    def forward_local(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._local_streams is None:
            raise RuntimeError("init_local_streams() must be called before forward_local()")
        batch, seq, hidden = tokens.shape
        logits = self.gate(tokens)
        top2_logits, top2_idx = torch.topk(logits, k=2, dim=-1)
        top2_w = torch.exp(top2_logits - torch.logsumexp(logits, dim=-1, keepdim=True))
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        cap = _expert_capacity(batch * seq, 2, self.num_experts, self.capacity_factor)
        route_overflow = _route_overflow_mask(flat_idx.reshape(-1), cap).view(batch * seq, 2)

        local_buffers, local_accum = self._local_buffers(flat_tokens, len(self._local_streams))
        current = torch.cuda.current_stream(tokens.device)
        for stream in self._local_streams:
            stream.wait_stream(current)
        for slot, stream in enumerate(self._local_streams):
            expert_ids = flat_idx[:, slot]
            local_out = local_buffers[slot]
            unique_expert_ids_host = torch.unique(expert_ids).detach().cpu()
            with torch.cuda.stream(stream):
                local_out.zero_()
                for unique_idx in range(unique_expert_ids_host.numel()):
                    eid_int = int(unique_expert_ids_host[unique_idx])
                    mask = (expert_ids == eid_int) & ~route_overflow[:, slot]
                    contrib = self.experts[eid_int](flat_tokens[mask]) * flat_w[mask, slot:slot + 1]
                    local_out[mask] += contrib

        for stream in self._local_streams:
            current.wait_stream(stream)
        local_accum.copy_(local_buffers[0])
        for local_out in local_buffers[1:]:
            local_accum.add_(local_out)
        return local_accum.view(batch, seq, hidden)

    def forward_distributed(self, tokens: torch.Tensor, *, ctx: DistributedContext) -> torch.Tensor:
        batch, seq, hidden = tokens.shape
        logits = self.gate(tokens)
        top2_logits, top2_idx = torch.topk(logits, k=2, dim=-1)
        top2_w = torch.exp(top2_logits - torch.logsumexp(logits, dim=-1, keepdim=True))
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        world_size = ctx.world_size
        rank = ctx.rank
        if self.num_experts < world_size or self.num_experts % world_size != 0:
            raise ValueError(
                "distributed mode requires num_experts to be evenly divisible by world_size"
            )
        experts_per_rank = self.num_experts // world_size

        route_expert_ids = flat_idx.reshape(-1)
        route_weights = flat_w.reshape(-1)
        total_send = route_expert_ids.numel()
        cap = _expert_capacity(batch * seq, 2, self.num_experts, self.capacity_factor)
        route_overflow = _route_overflow_mask(route_expert_ids, cap)
        route_positions = self._distributed_workspace(
            "route_positions",
            (total_send,),
            device=tokens.device,
            dtype=torch.int64,
        )
        torch.arange(total_send, out=route_positions)
        route_positions.div_(2, rounding_mode="floor")
        effective_route_weights = self._distributed_workspace(
            "effective_route_weights",
            (total_send,),
            device=tokens.device,
            dtype=route_weights.dtype,
        )
        effective_route_weights.copy_(route_weights)
        effective_route_weights.masked_fill_(route_overflow, 0.0)
        dest_ranks = route_expert_ids // experts_per_rank

        send_buf = self._distributed_workspace(
            "send_buf",
            (total_send, hidden),
            device=tokens.device,
            dtype=flat_tokens.dtype,
        )
        send_ids = self._distributed_workspace(
            "send_ids",
            (total_send,),
            device=tokens.device,
            dtype=torch.int64,
        )
        send_pos = self._distributed_workspace(
            "send_pos",
            (total_send,),
            device=tokens.device,
            dtype=torch.int64,
        )
        send_weights = self._distributed_workspace(
            "send_weights",
            (total_send,),
            device=tokens.device,
            dtype=route_weights.dtype,
        )
        send_splits: list[int] = []
        send_offset = 0
        for r in range(world_size):
            mask = dest_ranks == r
            send_indices = torch.nonzero(mask, as_tuple=False).view(-1)
            count = int(send_indices.numel())
            send_splits.append(count)
            if count:
                end = send_offset + count
                torch.index_select(
                    route_positions,
                    0,
                    send_indices,
                    out=send_pos[send_offset:end],
                )
                torch.index_select(
                    flat_tokens,
                    0,
                    send_pos[send_offset:end],
                    out=send_buf[send_offset:end],
                )
                torch.index_select(
                    route_expert_ids,
                    0,
                    send_indices,
                    out=send_ids[send_offset:end],
                )
                torch.index_select(
                    effective_route_weights,
                    0,
                    send_indices,
                    out=send_weights[send_offset:end],
                )
                send_offset = end
        splits_all: list[list[int]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(splits_all, send_splits)
        recv_splits = [splits_all[r][rank] for r in range(world_size)]

        total_recv = int(sum(recv_splits))
        recv_buf = self._distributed_workspace(
            "recv_buf",
            (total_recv, hidden),
            device=tokens.device,
            dtype=flat_tokens.dtype,
        )
        recv_ids = self._distributed_workspace(
            "recv_ids",
            (total_recv,),
            device=tokens.device,
            dtype=torch.int64,
        )
        recv_pos = self._distributed_workspace(
            "recv_pos",
            (total_recv,),
            device=tokens.device,
            dtype=torch.int64,
        )
        recv_weights = self._distributed_workspace(
            "recv_weights",
            (total_recv,),
            device=tokens.device,
            dtype=route_weights.dtype,
        )

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_ids,
            send_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_pos,
            send_pos,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        dist.all_to_all_single(
            recv_weights,
            send_weights,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        local_out = self._distributed_workspace(
            "local_out",
            (total_recv, hidden),
            device=tokens.device,
            dtype=flat_tokens.dtype,
        )
        local_out.zero_()
        recv_unique_ids_host = torch.unique(recv_ids).detach().cpu()
        for unique_idx in range(recv_unique_ids_host.numel()):
            eid_int = int(recv_unique_ids_host[unique_idx])
            if _expert_to_rank(eid_int, experts_per_rank) != rank:
                continue
            mask = (recv_ids == eid_int) & recv_weights.ne(0)
            local_out[mask] = _weight_route_outputs(
                self.experts[eid_int](recv_buf[mask]),
                recv_weights[mask],
            )

        send_back_splits = recv_splits
        recv_back_splits = send_splits
        total_back = int(sum(recv_back_splits))
        recv_back_buf = self._distributed_workspace(
            "recv_back_buf",
            (total_back, hidden),
            device=tokens.device,
            dtype=flat_tokens.dtype,
        )
        recv_back_pos = self._distributed_workspace(
            "recv_back_pos",
            (total_back,),
            device=tokens.device,
            dtype=torch.int64,
        )

        dist.all_to_all_single(
            recv_back_buf,
            local_out,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )
        dist.all_to_all_single(
            recv_back_pos,
            recv_pos,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )

        out = self._distributed_workspace(
            "out",
            (flat_tokens.shape[0], hidden),
            device=tokens.device,
            dtype=flat_tokens.dtype,
        )
        _accumulate_route_outputs(out, recv_back_pos, recv_back_buf)
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
        current_stream = torch.cuda.current_stream(device)
        start.record(current_stream)
        for _ in range(args.iters):
            if args.mode == "distributed":
                out = model.forward_distributed(tokens, ctx=ctx)  # type: ignore[arg-type]
            else:
                out = model.forward_local(tokens)
        end.record(current_stream)
        torch.cuda.synchronize(device)

    per_iter_ms = start.elapsed_time(end) / max(args.iters, 1)

    if args.mode == "distributed":
        t = torch.tensor(per_iter_ms, device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        per_iter_ms = scalar_tensor_to_float(t)

    if is_rank0:
        total_tokens = args.batch * args.seq
        tokens_per_s = (total_tokens / (per_iter_ms / 1000.0)) if per_iter_ms > 0 else 0.0
        print(
            f"expert_parallelism:{args.mode} batch={args.batch} seq={args.seq} hidden={args.hidden_dim} "
            f"experts={args.num_experts} dtype={args.dtype} -> {per_iter_ms:.3f} ms/iter, {tokens_per_s:,.0f} tok/s"
        )


if __name__ == "__main__":
    main()
