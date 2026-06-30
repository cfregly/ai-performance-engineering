#!/usr/bin/env python3
"""ch13/context_parallelism.py - Context parallelism (ring attention) demo.

This is a *tool* (not a comparable baseline/optimized benchmark pair).

Run with torchrun (multi-GPU required):

  torchrun --nproc_per_node 4 ch13/context_parallelism.py --sequence-length 131072
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

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
        raise RuntimeError("torch.distributed is required for this tool")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this tool")

    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        raise RuntimeError(
            "Missing LOCAL_RANK. Run via torchrun, e.g. "
            "`torchrun --nproc_per_node 4 ch13/context_parallelism.py ...`"
        )
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


class RingAttention(nn.Module):
    """Ring attention across ranks with streaming log-sum-exp normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        process_group: dist.ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.process_group = process_group
        self.rank = rank
        self.world_size = world_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._position_view_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._causal_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        self._recv_k_buffers: list[torch.Tensor] = []
        self._recv_v_buffers: list[torch.Tensor] = []

    def _position_views_for(self, seq_shard: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(seq_shard), torch.device(device))
        cached = self._position_view_cache.get(key)
        if cached is None:
            local_positions = torch.arange(seq_shard, device=device)
            global_q = (self.rank * seq_shard + local_positions).view(1, 1, seq_shard, 1)
            k_indices = local_positions.view(1, 1, 1, seq_shard)
            cached = (global_q, k_indices)
            self._position_view_cache[key] = cached
        return cached

    def _causal_mask_for(
        self,
        target_rank: int,
        seq_shard: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (int(target_rank), int(seq_shard), torch.device(device))
        cached = self._causal_mask_cache.get(key)
        if cached is None:
            global_q, k_indices = self._position_views_for(seq_shard, device)
            global_k = int(target_rank) * int(seq_shard) + k_indices
            cached = global_k > global_q
            self._causal_mask_cache[key] = cached
        return cached

    def _recv_buffers_for(
        self,
        k_template: torch.Tensor,
        v_template: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            len(self._recv_k_buffers) != 2
            or len(self._recv_v_buffers) != 2
            or self._recv_k_buffers[0].shape != k_template.shape
            or self._recv_v_buffers[0].shape != v_template.shape
            or self._recv_k_buffers[0].device != k_template.device
            or self._recv_v_buffers[0].device != v_template.device
            or self._recv_k_buffers[0].dtype != k_template.dtype
            or self._recv_v_buffers[0].dtype != v_template.dtype
        ):
            self._recv_k_buffers = [
                torch.empty(k_template.shape, device=k_template.device, dtype=k_template.dtype),
                torch.empty(k_template.shape, device=k_template.device, dtype=k_template.dtype),
            ]
            self._recv_v_buffers = [
                torch.empty(v_template.shape, device=v_template.device, dtype=v_template.dtype),
                torch.empty(v_template.shape, device=v_template.device, dtype=v_template.dtype),
            ]
        recv_slot = int(slot) & 1
        return self._recv_k_buffers[recv_slot], self._recv_v_buffers[recv_slot]

    def _ring_pass(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        *,
        seq_shard: int,
        causal: bool,
    ) -> torch.Tensor:
        k_current = k_local
        v_current = v_local
        attn_num: Optional[torch.Tensor] = None
        global_max: Optional[torch.Tensor] = None
        global_sum: Optional[torch.Tensor] = None

        for step in range(self.world_size):
            target_rank = (self.rank - step) % self.world_size
            scores = torch.matmul(q, k_current.transpose(-2, -1)) * self.scale

            if causal:
                mask = self._causal_mask_for(target_rank, seq_shard, q.device)
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

            if step < self.world_size - 1:
                next_rank = (self.rank + 1) % self.world_size
                prev_rank = (self.rank - 1) % self.world_size

                k_recv, v_recv = self._recv_buffers_for(k_current, v_current, step & 1)

                send_k = dist.isend(k_current.contiguous(), next_rank, group=self.process_group)
                recv_k = dist.irecv(k_recv, prev_rank, group=self.process_group)
                send_v = dist.isend(v_current.contiguous(), next_rank, group=self.process_group)
                recv_v = dist.irecv(v_recv, prev_rank, group=self.process_group)

                send_k.wait()
                recv_k.wait()
                send_v.wait()
                recv_v.wait()

                k_current = k_recv
                v_current = v_recv

        if attn_num is None or global_sum is None:
            raise RuntimeError("Ring attention accumulation failed")
        return attn_num / (global_sum + 1e-8)

    def forward(self, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        batch_size, seq_shard, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self._ring_pass(q, k, v, seq_shard=seq_shard, causal=causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_shard, self.hidden_size)
        return self.o_proj(attn_output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context parallelism (ring attention) demo")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
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
    ctx = _init_distributed()
    if ctx.world_size < 2:
        raise RuntimeError("Context parallelism requires world_size >= 2")
    if args.sequence_length % ctx.world_size != 0:
        raise ValueError(f"--sequence-length must be divisible by world_size={ctx.world_size}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    seq_shard = args.sequence_length // ctx.world_size
    dtype = _dtype_from_string(args.dtype)

    layers = nn.ModuleList(
        [
            RingAttention(
                args.hidden_size,
                args.num_heads,
                process_group=dist.group.WORLD,
                rank=ctx.rank,
                world_size=ctx.world_size,
            ).to(ctx.device, dtype=dtype)
            for _ in range(args.num_layers)
        ]
    )

    x = torch.randn(args.batch_size, seq_shard, args.hidden_size, device=ctx.device, dtype=dtype)
    torch.cuda.synchronize(ctx.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            y = x
            for layer in layers:
                y = layer(y, causal=True)
        torch.cuda.synchronize(ctx.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            y = x
            for layer in layers:
                y = layer(y, causal=True)
        end.record()
        torch.cuda.synchronize(ctx.device)

    per_iter_ms = start.elapsed_time(end) / max(args.iters, 1)
    total_tokens = args.batch_size * args.sequence_length

    worst = torch.tensor([per_iter_ms], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst_ms = float(worst.item())

    if ctx.rank == 0:
        tokens_per_s = (total_tokens / (worst_ms / 1000.0)) if worst_ms > 0 else 0.0
        print(
            f"context_parallelism: world={ctx.world_size} seq={args.sequence_length} "
            f"hidden={args.hidden_size} heads={args.num_heads} layers={args.num_layers} "
            f"dtype={args.dtype} -> {worst_ms:.3f} ms/iter, {tokens_per_s:,.0f} tokens/s"
        )


if __name__ == "__main__":
    main()
