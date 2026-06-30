"""Shared helpers for TP-only versus TP+SP hybrid benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from core.common.device_utils import resolve_local_rank


@dataclass(frozen=True)
class SequenceParallelConfig:
    batch_size: int = 8
    seq_len: int = 8192
    hidden_size: int = 2048
    ffn_hidden_size: int = 4096
    num_layers: int = 2
    dtype: torch.dtype = torch.bfloat16


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def align_seq_len(seq_len: int, world_size: int) -> int:
    if seq_len % world_size == 0:
        return seq_len
    return world_size * ((seq_len + world_size - 1) // world_size)


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Sequence-parallel benchmarks require torchrun (RANK/WORLD_SIZE missing).")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def build_layers(config: SequenceParallelConfig, world_size: int, device: torch.device) -> tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList]:
    if config.ffn_hidden_size % world_size != 0:
        raise ValueError("ffn_hidden_size must be divisible by world_size")
    shard_hidden = config.ffn_hidden_size // world_size
    up_proj = nn.ModuleList(
        [
            nn.Linear(config.hidden_size, shard_hidden, bias=False, device=device, dtype=config.dtype)
            for _ in range(config.num_layers)
        ]
    )
    down_proj = nn.ModuleList(
        [
            nn.Linear(shard_hidden, config.hidden_size, bias=False, device=device, dtype=config.dtype)
            for _ in range(config.num_layers)
        ]
    )
    norms = nn.ModuleList(
        [
            nn.LayerNorm(config.hidden_size, device=device, dtype=config.dtype)
            for _ in range(config.num_layers)
        ]
    )
    return up_proj, down_proj, norms


def run_sequence_parallel(
    *,
    config: SequenceParallelConfig,
    iters: int,
    warmup: int,
    sequence_parallel: bool,
) -> None:
    rank, world_size, local_rank = init_distributed()
    if world_size < 2:
        raise RuntimeError("SKIPPED: Requires >= 2 GPUs (found 1 GPU)")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(f"cuda:{local_rank}")
    seq_len = align_seq_len(config.seq_len, world_size)
    seq_per_rank = seq_len // world_size
    up_proj, down_proj, norms = build_layers(config, world_size, device)
    x_local = torch.randn(
        config.batch_size,
        seq_per_rank,
        config.hidden_size,
        device=device,
        dtype=config.dtype,
    )
    full_sequence_gather_buf = None
    if not sequence_parallel:
        full_sequence_gather_buf = torch.empty(
            world_size * config.batch_size,
            seq_per_rank,
            config.hidden_size,
            device=device,
            dtype=config.dtype,
        )
    layer_triples = tuple(zip(up_proj, down_proj, norms, strict=True))

    def _step() -> torch.Tensor:
        x = x_local
        for up_layer, down_layer, norm in layer_triples:
            hidden_local = torch.nn.functional.gelu(up_layer(x), approximate="tanh")
            out_partial = down_layer(hidden_local)
            dist.all_reduce(out_partial)
            if sequence_parallel:
                x = norm(out_partial)
            else:
                if full_sequence_gather_buf is None:
                    raise RuntimeError("full_sequence gather buffer missing for TP-only path")
                dist.all_gather_into_tensor(full_sequence_gather_buf, out_partial)
                full_sequence_by_rank = full_sequence_gather_buf.view(
                    world_size,
                    config.batch_size,
                    seq_per_rank,
                    config.hidden_size,
                )
                full_sequence = norm(full_sequence_by_rank)
                x = full_sequence[rank]
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
