#!/usr/bin/env python3
"""Symmetric Memory Inference Patterns for multi-GPU B200.

Chapter 16: FP8 KV Cache and Symmetric Memory

Extends the inference serving stack with PyTorch 2.10 symmetric memory
primitives (backed by NVSHMEM on Blackwell) to achieve <5 µs cross-GPU
access for latency-sensitive workloads.

Patterns:
1. Distributed KV cache pages backed by symmetric memory buffers
2. Multi-model serving with shared weight snapshots and hot-swapping
3. Speculative decoding coordinator with low-latency cross-model sync

NOTE: Pattern 3 (speculative decoding) is a forward reference to Chapter 18.
This demonstrates the symmetric memory infrastructure for cross-GPU coordination;
for complete speculative decoding implementation details, see:
- ch18/optimized_speculative_decode.py (draft model + verification logic)
- ch18/optimized_vllm_decode_graphs.py (CUDA graph optimization)

All demos degrade to NCCL-based fallbacks when symmetric memory is not
available so the file can be executed on developer machines.

Usage:
    torchrun --nproc_per_node=<num_gpus> -m ch16.symmetric_memory_inference --demo kv
    torchrun --nproc_per_node=<num_gpus> -m ch16.symmetric_memory_inference --demo multi
    torchrun --nproc_per_node=<num_gpus> -m ch16.symmetric_memory_inference --demo speculative
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import string
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from core.common.device_utils import resolve_local_rank
from core.optimization.symmetric_memory_patch import (
    SymmetricMemoryHandle,
    maybe_create_symmetric_memory_handle,
)
from core.optimization.symmetric_memory_patch import (
    symmetric_memory_available as _symmetric_memory_available,
)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8


# ============================================================================
# Setup
# ============================================================================


def symmetric_memory_available() -> bool:
    """Return True when symmetric memory APIs are accessible."""
    return _symmetric_memory_available()


def init_distributed() -> Tuple[int, int, torch.device]:
    """Initialize process group for inference scenarios."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
        local_rank = resolve_local_rank()
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=30),
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    return rank, world_size, device


# ============================================================================
# Pattern 1: Distributed KV Cache
# ============================================================================


@dataclass
class SymmetricMemoryCacheShard:
    """Shard of the KV cache that can be accessed from any peer GPU."""

    max_tokens: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    world_size: int
    handle: Optional[SymmetricMemoryHandle] = None
    buffer: torch.Tensor = field(init=False)
    meta: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens_per_rank = max(1, self.max_tokens // max(1, self.world_size))
        shape = (
            tokens_per_rank,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            2,
        )
        local = torch.zeros(shape, dtype=self.dtype, device=self.device)
        handle = maybe_create_symmetric_memory_handle(local)
        if handle is not None:
            self.handle = handle
            self.buffer = handle.buffer
        else:
            self.buffer = local
        self._tokens_per_rank = tokens_per_rank

    def append(self, request_id: str, k: torch.Tensor, v: torch.Tensor) -> None:
        tokens = k.size(0)
        if tokens == 0:
            return

        if request_id not in self.meta:
            start = 0
            self.meta[request_id] = (start, 0)
        else:
            start, _ = self.meta[request_id]

        start, current = self.meta[request_id]
        if current + tokens > self._tokens_per_rank:
            raise RuntimeError(
                f"Symmetric cache full for request {request_id}: "
                f"{current + tokens}/{self._tokens_per_rank} tokens"
            )

        dest_slice = slice(start + current, start + current + tokens)
        self.buffer[dest_slice, :, :, :, 0].copy_(k)
        self.buffer[dest_slice, :, :, :, 1].copy_(v)
        self.meta[request_id] = (start, current + tokens)

    def fetch(self, request_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if request_id not in self.meta:
            raise KeyError(f"KV cache miss for request {request_id}")
        start, length = self.meta[request_id]
        slice_k = self.buffer[start : start + length, :, :, :, 0]
        slice_v = self.buffer[start : start + length, :, :, :, 1]
        return slice_k.contiguous(), slice_v.contiguous()

    def release(self, request_id: str) -> None:
        if request_id in self.meta:
            del self.meta[request_id]


def demo_kv_cache(batch_size: int = 8, seq_len: int = 2048) -> None:
    """Simulate distributed KV cache Updates and reads."""
    rank, world_size, device = init_distributed()
    cache = SymmetricMemoryCacheShard(
        max_tokens=batch_size * seq_len,
        num_layers=40,
        num_heads=32,
        head_dim=128,
        dtype=torch.float16,
        device=device,
        world_size=world_size,
    )

    generator = torch.Generator(device=device.type)
    generator.manual_seed(0)
    k = torch.randn(seq_len, cache.num_layers, cache.num_heads, cache.head_dim, device=device, dtype=cache.dtype, generator=generator)
    v = torch.randn_like(k)
    request_id = f"req-{rank}"
    cache.append(request_id, k, v)

    # Any peer can read the cache slice
    target_rank = (rank + 1) % world_size
    if symmetric_memory_available() and cache.handle is not None:
        remote = cache.handle.get_buffer(target_rank)
        dist.barrier()
        if rank == target_rank:
            remote_k = remote[:, :, :, :, 0]
            _ = remote_k.norm()
    else:
        dist.barrier()
        if rank == target_rank:
            remote_k, _ = cache.fetch(request_id)
            _ = remote_k.norm()

    if rank == 0:
        print(f"[kv] symmetric memory available={symmetric_memory_available()}, slots={cache.buffer.shape[0]}")


# ============================================================================
# Pattern 2: Multi-Model Serving
# ============================================================================


def _random_model_id(prefix: str = "model") -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}-{suffix}"


@dataclass
class ModelWeightsSnapshot:
    name: str
    tensor: torch.Tensor
    handle: Optional[SymmetricMemoryHandle] = None


class MultiModelSymmetricPool:
    """
    Maintain multiple model weights in symmetric memory for barge-in serving.

    Hot model swap happens by pointing routers to a new symmetric buffer,
    avoiding redundant device<->device copies when switching tenants.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, world_size: int):
        self.device = device
        self.dtype = dtype
        self.world_size = world_size
        self.snapshots: Dict[str, ModelWeightsSnapshot] = {}

    def register(self, name: str, size_mb: int = 512) -> None:
        elements = max(1, (size_mb * 1024 * 1024) // _dtype_bytes(self.dtype))
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(abs(hash(name)) % (2**31))
        tensor = torch.randn(elements, device=self.device, dtype=self.dtype, generator=generator)
        handle = maybe_create_symmetric_memory_handle(tensor)
        if handle is not None:
            tensor = handle.buffer
        self.snapshots[name] = ModelWeightsSnapshot(name=name, tensor=tensor, handle=handle)

    def route_to(self, name: str) -> torch.Tensor:
        if name not in self.snapshots:
            raise KeyError(f"Model {name} not registered")
        return self.snapshots[name].tensor


def demo_multi_model(size_mb: int = 256) -> None:
    rank, world_size, device = init_distributed()
    pool = MultiModelSymmetricPool(device=device, dtype=torch.float16, world_size=world_size)
    model_a = _random_model_id("llama")
    model_b = _random_model_id("mistral")
    pool.register(model_a, size_mb=size_mb)
    pool.register(model_b, size_mb=max(1, size_mb // 2))

    active = model_a if rank % 2 == 0 else model_b
    weights = pool.route_to(active)
    checksum = weights[:1024].float().sum()
    dist.all_reduce(checksum)

    if rank == 0:
        print(f"[multi] routed to {active}, pool_size={len(pool.snapshots)}")


# ============================================================================
# Pattern 3: Speculative Decoding Sync
# ============================================================================


class SpeculativeDecodingCoordinator:
    """
    Coordinates draft/target model speculation with symmetric memory sync.

    Draft ranks produce token candidates that are written into shared buffers.
    Target ranks consume and validate them with microsecond latency.
    """

    def __init__(self, world_size: int, device: torch.device, max_tokens: int = 512):
        self.world_size = world_size
        self.device = device
        self.max_tokens = max_tokens
        self.buffer = torch.zeros(max_tokens, 256, device=device, dtype=torch.float16)
        handle = maybe_create_symmetric_memory_handle(self.buffer)
        if handle is not None:
            self.handle = handle
            self.buffer = handle.buffer
        else:
            self.handle = None

    def publish(self, token_scores: torch.Tensor, step: int) -> None:
        self.buffer[step, : token_scores.size(0)].copy_(token_scores)

    def consume(self, step: int) -> torch.Tensor:
        return self.buffer[step].clone()


def demo_speculative(num_steps: int = 8) -> None:
    rank, world_size, device = init_distributed()
    coordinator = SpeculativeDecodingCoordinator(world_size, device)

    role = "draft" if rank < world_size // 2 else "target"
    torch.manual_seed(rank)
    topk_values = None
    topk_indices = None
    topk_host = None
    topk_display = [0] * 4
    if role == "target":
        topk_values = torch.empty(4, device=device, dtype=torch.float16)
        topk_indices = torch.empty(4, device=device, dtype=torch.long)
        topk_host = torch.empty(
            4,
            dtype=torch.long,
            device="cpu",
            pin_memory=device.type == "cuda",
        )

    for step in range(num_steps):
        if role == "draft":
            scores = torch.randn(256, device=device, dtype=torch.float16)
            coordinator.publish(scores, step)
        dist.barrier()
        if role == "target":
            scores = coordinator.consume(step)
            assert topk_values is not None
            assert topk_indices is not None
            assert topk_host is not None
            torch.topk(scores, k=4, out=(topk_values, topk_indices))
            topk_host.copy_(topk_indices, non_blocking=False)
            for topk_idx in range(4):
                topk_display[topk_idx] = int(topk_host[topk_idx])
            if rank == world_size // 2:
                print(f"[speculative] step={step} topk={topk_display}")
        dist.barrier()


# ============================================================================
# Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetric memory inference patterns")
    parser.add_argument(
        "--demo",
        choices=("kv", "multi", "speculative"),
        default="kv",
        help="Which scenario to run",
    )
    args = parser.parse_args()

    if args.demo == "kv":
        demo_kv_cache()
    elif args.demo == "multi":
        demo_multi_model()
    else:
        demo_speculative()

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Symmetric memory available: {symmetric_memory_available()}")


if __name__ == "__main__":
    main()
