#!/usr/bin/env python3
"""Production NVSHMEM Training Patterns for multi-GPU B200.

Demonstrates how to extend PyTorch training workloads with NVSHMEM and
torch.distributed.nn.SymmetricMemory for latency-critical paths on multi-GPU
Blackwell clusters.

Patterns:
1. Gradient buckets backed by symmetric memory for custom AllReduce
2. Hybrid sharding (FSDP parameters + NVSHMEM parameter server buffers)
3. Pipeline parallel microbatch handoff using one-sided NVSHMEM puts

Hardware assumptions:
- >=2 Blackwell B200 GPUs (NVLink 5.0 @ 1800 GB/s per pair)
- CUDA 13.0+, PyTorch 2.10+, NVSHMEM 3.4+

Usage (conceptual):
    torchrun --nproc_per_node=<num_gpus> nvshmem_training_example.py --demo gradient
    torchrun --nproc_per_node=<num_gpus> nvshmem_training_example.py --demo hybrid
    torchrun --nproc_per_node=<num_gpus> nvshmem_training_example.py --demo pipeline

Each demo now fails fast with `SKIPPED:` when the required multi-GPU launch or
NVSHMEM/symmetric-memory support is unavailable.
"""

from __future__ import annotations

import argparse
import datetime
import os

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.optimization.symmetric_memory_patch import (
    SymmetricMemoryHandle,
    maybe_create_symmetric_memory_handle,
    symmetric_memory_available,
)

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)


# ============================================================================
# Helpers
# ============================================================================


def nvshmem_available() -> bool:
    """Detect whether NVSHMEM-like symmetric memory APIs are exposed."""
    return symmetric_memory_available()


def init_process_group() -> Tuple[int, int, torch.device]:
    """Initialize a real NCCL process group for the NVSHMEM demos."""
    rank, world_size, local_rank = setup_single_gpu_env(
        "nvshmem_training_example",
        min_world_size=2,
    )

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
            device_id=local_rank,
        )

    if not nvshmem_available():
        raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")

    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def pipeline_reuse_buffers_enabled() -> bool:
    return os.environ.get("AISP_NVSHMEM_PIPELINE_REUSE_BUFFERS", "").lower() not in {"0", "false", "no"}


# ============================================================================
# Demo 1: Gradient Buckets with Symmetric Memory
# ============================================================================


@dataclass
class GradientBucket:
    """
    Gradient bucket backed by symmetric memory for one-sided NVSHMEM puts.

    Each GPU writes gradients into its local chunk, then performs a ring
    reduction using remote puts for latency-critical microbatches.
    """

    numel: int
    dtype: torch.dtype
    device: torch.device
    world_size: int
    handle: Optional[SymmetricMemoryHandle] = None
    tensor: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        local = torch.zeros(self.numel, device=self.device, dtype=self.dtype)
        handle = maybe_create_symmetric_memory_handle(local)
        if handle is not None:
            self.handle = handle
            self.tensor = handle.buffer
        else:
            self.tensor = local

    def allreduce_ring(self, rank: int) -> None:
        """
        Conceptual ring AllReduce using one-sided puts/gets.
        """
        if self.tensor is None:
            raise RuntimeError("GradientBucket tensor not initialized")

        if self.handle is None or not nvshmem_available():
            raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")

        # Conceptual one-sided ring: each rank writes into next rank slot
        chunk = self.tensor.view(self.world_size, -1)[rank].clone()
        for step in range(self.world_size - 1):
            target = (rank + 1 + step) % self.world_size
            remote_buf = self.handle.get_buffer(target)
            remote_buf.view(self.world_size, -1)[rank].copy_(chunk)
            dist.barrier()
            chunk.add_(
                self.handle.get_buffer((rank - 1 - step) % self.world_size)
                .view(self.world_size, -1)[rank]
            )
        self.tensor.view(self.world_size, -1)[rank].copy_(chunk)

    def close(self) -> None:
        if self.handle is None:
            self.tensor = None
            return
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.synchronize(self.device)
        self.tensor = None
        self.handle = None


def demo_gradient_buckets(batch: torch.Tensor, model: nn.Module, steps: int = 1) -> None:
    """Attach hooks that fuse gradients into symmetric memory buckets."""
    rank, world_size, device = init_process_group()

    model = model.to(device)
    batch = batch.to(device)

    bucket = GradientBucket(
        numel=sum(p.numel() for p in model.parameters()),
        dtype=batch.dtype,
        device=device,
        world_size=world_size,
    )

    offset = 0

    def _hook(param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
        nonlocal offset
        if grad is None:
            return grad
        next_offset = offset + grad.numel()
        bucket.tensor.flatten()[offset:next_offset].copy_(grad.flatten())
        offset = next_offset
        return grad

    for p in model.parameters():
        p.register_hook(lambda grad, p=p: _hook(p, grad))

    for _ in range(steps):
        offset = 0
        output = model(batch)
        loss = output.float().sum()
        loss.backward()
        bucket.allreduce_ring(rank)
        model.zero_grad(set_to_none=True)
    dist.barrier()

    if rank == 0:
        reduced_norm = float(bucket.tensor.norm())
        print(f"[gradient_bucket] Reduced norm={reduced_norm:.3f}")
    bucket.close()


# ============================================================================
# Demo 2: Hybrid Sharding (FSDP + NVSHMEM Parameter Server)
# ============================================================================


class TransformerBlock(nn.Module):
    """Simple transformer block used across demos."""

    def __init__(self, d_model: int = 2048, n_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.linear2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = resid + attn_out
        resid = x
        x = self.ln2(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return x + resid


class NVSHMEMParameterServer:
    """
    Minimal parameter server that mirrors FSDP weights into symmetric memory.

    FSDP handles parameter sharding + optimizer states; NVSHMEM enables
    low-latency lookups for layers that need dense access (e.g. embeddings).
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.buffers: Dict[str, GradientBucket] = {}

    def materialize_parameter(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in self.buffers:
            bucket = GradientBucket(
                numel=tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
                world_size=dist.get_world_size(),
            )
            bucket.tensor.copy_(tensor.flatten())
            self.buffers[name] = bucket
        return self.buffers[name].tensor.view_as(tensor)

    def pull(self, name: str) -> torch.Tensor:
        if name not in self.buffers:
            raise KeyError(f"Parameter {name} not backed by NVSHMEM buffer")
        return self.buffers[name].tensor

    def close(self) -> None:
        for bucket in self.buffers.values():
            bucket.close()
        self.buffers.clear()


def demo_hybrid_sharding(microbatch: torch.Tensor, steps: int = 1) -> None:
    """Showcase FSDP sharding with NVSHMEM-backed parameter server tensors."""
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid sharding demo requires CUDA-enabled environment")

    rank, world_size, device = init_process_group()
    generator = torch.Generator(device=device)
    generator.manual_seed(17 + rank)
    microbatch = torch.randn(
        microbatch.shape,
        device=device,
        dtype=microbatch.dtype,
        generator=generator,
    )

    model = TransformerBlock().to(device)
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float32
        ),
    )

    server = NVSHMEMParameterServer(fsdp_model)

    with FSDP.state_dict_type(fsdp_model, StateDictType.LOCAL_STATE_DICT):
        state_dict = fsdp_model.state_dict()

    for name, tensor in state_dict.items():
        mirror = server.materialize_parameter(name, tensor)
        tensor.copy_(mirror.view_as(tensor))

    buffer_count = 0
    for _ in range(steps):
        out = fsdp_model(microbatch.to(device))
        loss = out.sum()
        loss.backward()
        buffer_count = len(server.buffers)
        dist.barrier()
        fsdp_model.zero_grad(set_to_none=True)
    server.close()

    if rank == 0:
        print(f"[hybrid_shard] gradients ready, NVSHMEM buffers={buffer_count}")


# ============================================================================
# Demo 3: Pipeline Parallel with NVSHMEM Handoff
# ============================================================================


class PipelineStage(nn.Module):
    """Lightweight pipeline stage used for the NVSHMEM pipeline demo."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo_pipeline_parallel(microbatch: torch.Tensor, steps: int = 1) -> None:
    """
    Pipeline two stages using symmetric memory buffers for microbatch handoff.

    Stage 0 (ranks 0-3) writes outputs into symmetric buffers that stage 1
    (ranks 4-7) reads without host synchronization.
    """

    rank, world_size, device = init_process_group()
    assert world_size >= 2, "Pipeline demo requires at least 2 ranks"

    dim = microbatch.shape[-1]
    stage0 = PipelineStage(dim, dim * 4).to(device)
    stage1 = PipelineStage(dim, dim * 4).to(device)

    if not nvshmem_available():
        raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")
    reuse_buffers = pipeline_reuse_buffers_enabled()
    bucket = None
    if reuse_buffers:
        bucket = GradientBucket(
            numel=microbatch.numel(),
            dtype=microbatch.dtype,
            device=device,
            world_size=world_size,
        )

    partner_offset = world_size // 2
    for _ in range(steps):
        if not reuse_buffers:
            bucket = GradientBucket(
                numel=microbatch.numel(),
                dtype=microbatch.dtype,
                device=device,
                world_size=world_size,
            )
        if rank < partner_offset:
            microbatch = stage0(microbatch.to(device))
            bucket.tensor.copy_(microbatch.flatten())
            if bucket.handle is None:
                raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")
            next_rank = rank + partner_offset
            remote = bucket.handle.get_buffer(next_rank)
            remote.copy_(bucket.tensor, non_blocking=True)
            torch.cuda.synchronize(device)
            dist.barrier()
        else:
            if bucket.handle is None:
                raise RuntimeError("SKIPPED: nvshmem_training_example requires NVSHMEM or SymmetricMemory support")
            dist.barrier()
            remote = bucket.handle.get_buffer(rank)
            microbatch = remote.view_as(microbatch)
            microbatch = stage1(microbatch)
        if not reuse_buffers and bucket is not None:
            bucket.close()

    dist.barrier()
    if bucket is not None:
        bucket.close()
    if rank == 0:
        print("[pipeline] completed forward pass with symmetric-memory handoff")


# ============================================================================
# CLI Entrypoint
# ============================================================================


def _build_sample_batch(batch_size: int = 8, seq_len: int = 512, dim: int = 2048) -> torch.Tensor:
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.randn(batch_size, seq_len, dim, device=device, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="NVSHMEM training patterns")
    parser.add_argument(
        "--demo",
        choices=("gradient", "hybrid", "pipeline"),
        default="gradient",
        help="Which demonstration to run",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the sample workload.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for the sample workload.")
    parser.add_argument("--dim", type=int, default=2048, help="Model hidden dimension.")
    parser.add_argument("--steps", type=int, default=8, help="Iterations per demo run.")
    args = parser.parse_args()

    demo_map = {
        "gradient": demo_gradient_buckets,
        "hybrid": demo_hybrid_sharding,
        "pipeline": demo_pipeline_parallel,
    }

    init_process_group()
    batch = _build_sample_batch(batch_size=args.batch_size, seq_len=args.seq_len, dim=args.dim)
    model = TransformerBlock(d_model=args.dim)

    if args.demo == "gradient":
        demo_map[args.demo](batch, model, steps=args.steps)
    elif args.demo == "hybrid":
        demo_map[args.demo](batch, steps=args.steps)
    else:
        demo_map[args.demo](batch, steps=args.steps)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"NVSHMEM available: {nvshmem_available()}")


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
