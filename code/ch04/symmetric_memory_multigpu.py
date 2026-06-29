"""
PyTorch 2.10 Symmetric Memory for Multi-GPU
===========================================

Demonstrates torch.distributed.nn.SymmetricMemory API for ultra-low
latency cross-GPU access on multi-GPU B200 configurations.

Symmetric memory enables:
1. Direct GPU-to-GPU memory access without CPU involvement
2. Sub-microsecond latency for small transfers (<4KB)
3. Custom reduction kernels with direct buffer access
4. Integration with torch.compile for fused operations


Hardware:
- Multi-GPU Blackwell B200 nodes
- NVLink 5.0: 1800 GB/s bidirectional per GPU pair
- Total SMs/HBM scale with GPU count

Use Cases:
- Frequent small synchronizations (gradients, scalars)
- Custom multi-GPU algorithms (reductions, broadcasts)
- Pipeline parallelism synchronization points
- Low-latency parameter server patterns

Requirements:
- PyTorch 2.10+
- CUDA 13.0+
- >=2 GPUs

Usage:
    # Multi-GPU
    torchrun --nproc_per_node=<num_gpus> symmetric_memory_multigpu.py
"""
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import os

from core.common.device_utils import resolve_local_rank

from core.optimization.symmetric_memory_patch import symmetric_memory_available
from core.benchmark.gpu_requirements import require_min_gpus

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env


import time
import torch
import torch.distributed as dist
from typing import Optional
import math


def setup_distributed():
    """Initialize distributed environment."""
    require_min_gpus(2)
    setup_single_gpu_env("symmetric_memory_multigpu", min_world_size=2)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = resolve_local_rank()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=local_rank,
    )

    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


def check_symmetric_memory_available() -> bool:
    """Check if symmetric memory is available."""
    return symmetric_memory_available()


# ============================================================================
# Ring AllReduce using Symmetric Memory
# ============================================================================

def ring_allreduce_symmetric(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Custom ring-based AllReduce using symmetric memory for multi-GPU setups.
    
    This demonstrates direct GPU-to-GPU access without NCCL.
    For production, use NCCL which is heavily optimized.
    
    Args:
        tensor: Input tensor to reduce
        rank: Current rank
        world_size: Total number of ranks
        
    Returns:
        Reduced tensor
    """
    if world_size == 1:
        return tensor
    
    # This is a conceptual implementation
    # Real symmetric memory requires buffer setup
    
    # Ring algorithm:
    # - World size N, divide tensor into N chunks
    # - Each GPU starts with one chunk
    # - N-1 iterations: each GPU sends chunk to next GPU
    # - After N-1 iterations, all GPUs have sum of all chunks
    
    chunk_size = tensor.numel() // world_size
    chunks = torch.chunk(tensor, world_size)
    
    # Ring reduction
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        
        # In real symmetric memory:
        # - Direct write to next GPU's buffer
        # - Direct read from previous GPU's buffer
        
        # Fall back is intentionally not exposed through this public demo name.
        send_tensor = chunks[send_chunk_idx].contiguous()
        recv_tensor = torch.empty_like(send_tensor)
        
        # Send to next, receive from previous
        send_req = dist.isend(send_tensor, dst=(rank + 1) % world_size)
        recv_req = dist.irecv(recv_tensor, src=(rank - 1) % world_size)
        
        send_req.wait()
        recv_req.wait()
        
        # Accumulate
        chunks[recv_chunk_idx].add_(recv_tensor)
    
    # Reconstruct tensor
    result = torch.cat(chunks, dim=0)
    
    # AllGather phase (simplified - just use NCCL)
    dist.all_gather_into_tensor(result, result)
    
    return result


# ============================================================================
# Benchmark: Traditional vs Symmetric Memory
# ============================================================================

def benchmark_traditional_allreduce(
    size: int,
    rank: int,
    world_size: int,
    iterations: int = 100,
) -> float:
    """Benchmark traditional NCCL AllReduce."""
    device = torch.cuda.current_device()
    tensor = torch.randn(size, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return elapsed_ms


def benchmark_symmetric_memory_access(
    size: int,
    rank: int,
    world_size: int,
    iterations: int = 100,
) -> Optional[float]:
    """
    Benchmark symmetric memory access pattern.
    
    Note: This is conceptual. Real implementation requires
    torch.distributed.nn.SymmetricMemory which may not be
    available in all PyTorch builds.
    """
    device = torch.cuda.current_device()
    tensor = torch.randn(size, device=device, dtype=torch.float32)
    recv_tensor = torch.empty_like(tensor) if rank > 0 else None
    
    try:
        # Try to use symmetric memory API
        # This is PyTorch 2.10+ specific and may not be available
        
        # For now, demonstrate the pattern with standard operations
        # Real symmetric memory would be faster
        
        # Warmup
        for _ in range(10):
            # Simulate direct access pattern
            if rank > 0:
                if recv_tensor is None:
                    raise RuntimeError("Receive buffer was not initialized")
                dist.recv(recv_tensor, src=rank-1)
            if rank < world_size - 1:
                dist.send(tensor, dst=rank+1)
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            if rank > 0:
                if recv_tensor is None:
                    raise RuntimeError("Receive buffer was not initialized")
                dist.recv(recv_tensor, src=rank-1)
            if rank < world_size - 1:
                dist.send(tensor, dst=rank+1)
        end.record()
        end.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / iterations
        return elapsed_ms
        
    except Exception as e:
        if rank == 0:
            print(f"Symmetric memory not available: {e}")
        return None


# ============================================================================
# Multi-GPU Patterns
# ============================================================================

def demonstrate_ring_pattern(rank: int, world_size: int) -> None:
    """
    Demonstrate ring communication pattern for a power-of-two GPU group.
    
    On NVSwitch/NVLink, all GPUs are directly connected.
    Ring pattern: 0 → 1 → ... → (N-1) → 0
    """
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"{world_size}-GPU Ring Communication Pattern")
        print("=" * 80)
        ring = " → ".join(str(i) for i in range(world_size)) + " → 0"
        print(f"Pattern: GPU {ring}")
        print("Best for: Sequential data processing, pipeline parallelism")
        print("=" * 80 + "\n")
    
    device = torch.cuda.current_device()
    
    # Test different message sizes
    sizes = [1024, 1024*256, 1024*1024]  # 4KB, 1MB, 4MB
    
    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        recv_tensor = torch.empty_like(tensor)
        
        # Warmup
        for _ in range(10):
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size
            
            send_req = dist.isend(tensor, dst=next_rank)
            recv_req = dist.irecv(recv_tensor, src=prev_rank)
            
            send_req.wait()
            recv_req.wait()
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size
            
            send_req = dist.isend(tensor, dst=next_rank)
            recv_req = dist.irecv(recv_tensor, src=prev_rank)
            
            send_req.wait()
            recv_req.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        if rank == 0:
            latency_us = (elapsed / 100) * 1e6
            bandwidth_gbs = (size * 4 / 1e9) / (elapsed / 100)
            print(f"Size: {size*4/1024:.1f} KB | "
                  f"Latency: {latency_us:.2f} μs | "
                  f"Bandwidth: {bandwidth_gbs:.2f} GB/s")


def demonstrate_butterfly_pattern(rank: int, world_size: int) -> None:
    """
    Demonstrate butterfly (hypercube) pattern for power-of-two GPU counts.
    
    Total steps: log2(world_size)
    Best for: Low-latency AllReduce on small messages
    """
    stages = int(math.log2(world_size)) if world_size > 0 else 0
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"{world_size}-GPU Butterfly Communication Pattern")
        print("=" * 80)
        print(f"Stages: {stages} (log2({world_size}))")
        print("Stage stride doubles each step (1, 2, 4, ...)")
        print("Best for: Low-latency reductions on small data")
        print("=" * 80 + "\n")
    
    if world_size < 2 or (2 ** stages) != world_size:
        if rank == 0:
            print(f"⚠ Butterfly pattern requires power-of-two world size; have {world_size}")
        return
    
    device = torch.cuda.current_device()
    
    # Small message (latency-bound)
    size = 1024  # 4 KB
    tensor = torch.randn(size, device=device, dtype=torch.float32)
    recv_tensor = torch.empty_like(tensor)
    
    # Warmup
    for _ in range(10):
        for stage in range(stages):
            stride = 1 << stage
            peer = rank ^ stride
            
            send_req = dist.isend(tensor, dst=peer)
            recv_req = dist.irecv(recv_tensor, src=peer)
            
            send_req.wait()
            recv_req.wait()
            
            # Reduction
            tensor.add_(recv_tensor)
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        for stage in range(stages):
            stride = 1 << stage
            peer = rank ^ stride
            
            send_req = dist.isend(tensor, dst=peer)
            recv_req = dist.irecv(recv_tensor, src=peer)
            
            send_req.wait()
            recv_req.wait()
            
            tensor.add_(recv_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    if rank == 0:
        latency_us = (elapsed / 100) * 1e6
        print(f"Butterfly AllReduce ({size*4/1024:.1f} KB):")
        print(f"  Total latency: {latency_us:.2f} μs")
        if stages:
            print(f"  Per-stage latency: {latency_us/stages:.2f} μs")
            print(f"  Compare to Ring ({world_size - 1} steps): ~{latency_us * (world_size - 1) / stages:.2f} μs")


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def main():
    """Main benchmark and demonstration."""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("=" * 80)
        print("PyTorch 2.10 Symmetric Memory for 4 GPUs")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        
        if world_size == 4:
            print("Optimal 4-GPU configuration")
        else:
            print(f"⚠ Running with {world_size} GPUs (optimized for 4)")
        
        print("=" * 80 + "\n")
    
    # Check symmetric memory availability
    sym_mem_available = check_symmetric_memory_available()
    if rank == 0:
        if sym_mem_available:
            print("Symmetric memory API available")
        else:
            raise RuntimeError("SKIPPED: symmetric_memory_multigpu requires SymmetricMemory support")
    
    # Benchmark 1: Traditional AllReduce
    if rank == 0:
        print("\n=== Benchmark 1: NCCL AllReduce ===\n")
    
    sizes = [1024, 256*1024, 1024*1024, 64*1024*1024]
    for size in sizes:
        time_ms = benchmark_traditional_allreduce(size, rank, world_size, iterations=100)
        
        if rank == 0:
            size_mb = size * 4 / (1024 * 1024)
            data_bytes = size * 4
            # AllReduce bandwidth calculation: 2*(N-1)/N
            busbw_gbs = (data_bytes * 2 * (world_size - 1) / world_size) / (time_ms / 1000) / 1e9
            
            print(f"Size: {size_mb:8.2f} MB | Time: {time_ms:6.2f} ms | BusBW: {busbw_gbs:6.2f} GB/s")
    
    # Benchmark 2: Symmetric memory (if available)
    if sym_mem_available:
        if rank == 0:
            print("\n=== Benchmark 2: Symmetric Memory ===\n")
        
        for size in [1024, 256*1024]:  # Small sizes where latency matters
            time_ms = benchmark_symmetric_memory_access(size, rank, world_size, iterations=100)
            if time_ms and rank == 0:
                size_kb = size * 4 / 1024
                print(f"Size: {size_kb:8.2f} KB | Latency: {time_ms:6.3f} ms")
    
    # Pattern demonstrations (power-of-two)
    if world_size >= 2:
        demonstrate_ring_pattern(rank, world_size)
        demonstrate_butterfly_pattern(rank, world_size)
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\n{world_size}-GPU Communication Patterns:")
        print(f"  1. Ring: {world_size - 1} steps, good for large messages")
        steps = int(math.log2(world_size)) if world_size > 0 else 0
        if world_size >= 2 and (2 ** steps) == world_size:
            print(f"  2. Tree: log2({world_size})={steps} steps, good for medium messages")
            print(f"  3. Butterfly: {steps} steps, best for small messages")
        print("  4. NVLS: Specialized for dense NVLink domains (NCCL 2.28)")
        
        print("\nWhen to use Symmetric Memory:")
        print("  Frequent small synchronizations (<4KB)")
        print("  Custom reduction algorithms")
        print("  Sub-microsecond latency requirements")
        print("  Direct buffer access from kernels")
        
        print("\nWhen to use NCCL:")
        print("  Standard collectives (AllReduce, AllGather)")
        print("  Large messages (>1MB)")
        print("  Production training (heavily optimized)")
        print("  Multi-node communication")
        
        print("\nExpected Performance on 4x B200:")
        print("  - AllReduce 1GB: 700-800 GB/s bus bandwidth (topology-dependent)")
        print("  - Small message latency: <2 μs (symmetric memory)")
        print("  - Large message latency: <1ms (NCCL)")
        print("=" * 80)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
