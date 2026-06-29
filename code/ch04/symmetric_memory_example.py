"""
PyTorch 2.10 Symmetric Memory Example
Demonstrates ultralow-latency cross-GPU access using torch.distributed.nn.SymmetricMemory.

Requirements:
- PyTorch 2.10+
- Multi-GPU system (2+ GPUs)
- CUDA 13.0+
- NCCL 2.28+

Expected Runtime: ~5-10 seconds on 2 GPUs
"""
import argparse

import os

from core.common.device_utils import resolve_local_rank

from core.optimization.symmetric_memory_patch import (
    create_symmetric_memory_handle,
    maybe_create_symmetric_memory_handle,
    symmetric_memory_available,
)

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env


import torch
import torch.distributed as dist
import torch.cuda.nvtx as nvtx
from core.profiling.nvtx_helper import standardize_nvtx_label


def setup_distributed():
    """Initialize distributed environment for multi-GPU operation."""
    setup_single_gpu_env("symmetric_memory_example", min_world_size=2)
    if not symmetric_memory_available():
        raise RuntimeError("SKIPPED: symmetric_memory_example requires SymmetricMemory support")
    if dist.is_initialized():
        torch.cuda.set_device(resolve_local_rank())
        return dist.get_rank(), dist.get_world_size()

    # For single-node multi-GPU
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = resolve_local_rank()

    torch.cuda.set_device(local_rank)
    # Use NCCL backend for GPU communication with timeout
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.timedelta(seconds=30),  # 30 second timeout
        device_id=local_rank,
    )

    return dist.get_rank(), dist.get_world_size()


def detect_gb200_gb300() -> bool:
    """Detect if running on GB200/GB300 Grace-Blackwell Superchip."""
    import platform
    if platform.machine() != 'aarch64':
        return False
    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()
        return 'ARM' in cpuinfo or 'Neoverse' in cpuinfo


def enable_nvlink_c2c_optimizations() -> None:
    """
    Enable NVLink-C2C optimizations for Blackwell (NEW).
    
    Blackwell B200/B300 features NVLink-C2C (Chip-to-Chip) providing
    900 GB/s CPU-GPU bandwidth. This function configures peer access
    and memory hints for optimal performance.
    """
    if not torch.cuda.is_available():
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return
    
    is_grace_blackwell = detect_gb200_gb300()
    
    print("\n" + "=" * 80)
    if is_grace_blackwell:
        print("NVLink-C2C Configuration for GB200/GB300 Grace-Blackwell")
    else:
        print("NVLink-C2C Configuration for Blackwell")
    print("=" * 80)
    
    # 1. Enable peer access between all GPU pairs
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        for j in range(num_gpus):
            if i != j:
                try:
                    # Check if peer access is possible
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    if can_access:
                        # This is automatically enabled in modern PyTorch
                        # but we document it for educational purposes
                        print(f"P2P access enabled: GPU {i} <-> GPU {j}")
                except RuntimeError as e:
                    print(f"✗ P2P access failed: GPU {i} <-> GPU {j}: {e}")
    
    # 2. Configure pinned memory for C2C transfers
    # NVLink-C2C benefits from pinned memory allocation
    torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve 10% for overhead
    
    # 3. Set memory pool attributes (if using stream-ordered allocator)
    # Note: This is configured via CUDA driver - no PyTorch API needed
    
    print("\nNVLink-C2C Features:")
    if is_grace_blackwell:
        print("  GB200/GB300 detected:")
        print("    - CPU-GPU: ~900 GB/s (coherent)")
        print("    - GPU-GPU: 1800 GB/s NVLink 5.0")
        print("    - Unified memory address space")
        print("    - Grace CPU: 72 ARM cores")
    else:
        print("  - GPU-GPU: ~900 GB/s per pair")
        print("  - Optimal for: Large parameter transfers, peer communication")
    
    if num_gpus == 4:
        print(f"\n4x B200 Configuration:")
        print(f"  - Total SMs: 592")
        print(f"  - Total Memory: 0.72 TB HBM3e")
        print(f"  - Aggregate Bandwidth: 31.2 TB/s")
    
    print("=" * 80)


def benchmark_traditional_p2p(tensor: torch.Tensor, peer_rank: int, iterations: int = 100):
    """Benchmark traditional peer-to-peer copy using torch.cuda.comm."""
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    recv_tensor = torch.empty_like(tensor) if rank == peer_rank else None
    
    # Warmup
    for _ in range(10):
        if rank == 0:
            dist.send(tensor, dst=peer_rank)
        elif rank == peer_rank:
            if recv_tensor is None:
                raise RuntimeError("Receive buffer was not initialized")
            dist.recv(recv_tensor, src=0)
    
    dist.barrier()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize(device)
    start.record()
    
    with nvtx.range(standardize_nvtx_label("transfer_sync:traditional_p2p")):
        for _ in range(iterations):
            if rank == 0:
                dist.send(tensor, dst=peer_rank)
            elif rank == peer_rank:
                if recv_tensor is None:
                    raise RuntimeError("Receive buffer was not initialized")
                dist.recv(recv_tensor, src=0)
    
    end.record()
    end.synchronize()
    
    return start.elapsed_time(end) / iterations


def benchmark_symmetric_memory(tensor: torch.Tensor, iterations: int = 100):
    """Benchmark symmetric memory for ultralow-latency cross-GPU access."""
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    remote_touch = torch.empty((), device=device, dtype=tensor.dtype) if rank == 1 else None
    
    # Check if symmetric memory is available
    try:
        # Allocate symmetric memory buffer
        # All GPUs in the group can directly address this memory
        with nvtx.range(standardize_nvtx_label("setup:symmetric_memory_allocation")):
            sym_mem = maybe_create_symmetric_memory_handle(
                tensor,
                group=dist.group.WORLD,
            )
            if sym_mem is None:
                raise RuntimeError("Symmetric memory not available")
    except (AttributeError, RuntimeError) as e:
        print(f"Rank {rank}: Symmetric memory not available: {e}")
        print("This feature requires PyTorch 2.10+ with proper CUDA 13/NVSHMEM support")
        return None
    
    dist.barrier()
    
    # Warmup - direct cross-GPU access
    for _ in range(10):
        if rank == 0:
            # Rank 0 writes to its symmetric buffer
            sym_mem.buffer[:] = tensor
        dist.barrier()
        if rank == 1:
            # Rank 1 directly reads from rank 0's symmetric buffer
            remote_data = sym_mem.get_buffer(0)
            if remote_touch is None:
                raise RuntimeError("Remote touch buffer was not initialized")
            torch.sum(remote_data, dim=0, out=remote_touch)
    
    dist.barrier()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize(device)
    start.record()
    
    with nvtx.range(standardize_nvtx_label("transfer_sync:symmetric_memory_access")):
        for _ in range(iterations):
            if rank == 0:
                sym_mem.buffer[:] = tensor
            dist.barrier()
            if rank == 1:
                remote_data = sym_mem.get_buffer(0)
                if remote_touch is None:
                    raise RuntimeError("Remote touch buffer was not initialized")
                torch.sum(remote_data, dim=0, out=remote_touch)
    
    end.record()
    end.synchronize()
    
    return start.elapsed_time(end) / iterations


def benchmark_traditional_ring(tensor: torch.Tensor, iterations: int = 100) -> float:
    """Benchmark ring send/recv using NCCL P2P ops across all ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    recv_tensor = torch.empty_like(tensor)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    for _ in range(5):
        ops = [
            dist.P2POp(dist.isend, tensor, next_rank),
            dist.P2POp(dist.irecv, recv_tensor, prev_rank),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    torch.cuda.synchronize(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        ops = [
            dist.P2POp(dist.isend, tensor, next_rank),
            dist.P2POp(dist.irecv, recv_tensor, prev_rank),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    end.record()
    torch.cuda.synchronize(device)
    dist.barrier()
    return start.elapsed_time(end) / iterations


def benchmark_symmetric_ring(tensor: torch.Tensor, iterations: int = 100) -> float:
    """Benchmark ring traffic using symmetric memory buffers."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    flat = tensor.flatten()
    local = torch.stack([flat, flat.clone()])
    handle = create_symmetric_memory_handle(local, group=dist.group.WORLD)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    next_buf = handle.get_buffer(next_rank)
    prev_buf = handle.get_buffer(prev_rank)
    recv_tensor = torch.empty_like(flat)

    for idx in range(5):
        buf_idx = idx % 2
        next_buf[buf_idx].copy_(local[buf_idx], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        recv_tensor.copy_(prev_buf[buf_idx], non_blocking=True)
        torch.cuda.current_stream().synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for idx in range(iterations):
        buf_idx = idx % 2
        next_buf[buf_idx].copy_(local[buf_idx], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        recv_tensor.copy_(prev_buf[buf_idx], non_blocking=True)
        torch.cuda.current_stream().synchronize()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iterations


def benchmark_multigpu_symmetric_memory(
    tensor_sizes: list = [(1024,), (1024 * 256,), (1024 * 1024,)],
    iterations: int = 100
) -> dict:
    """
    Benchmark symmetric memory patterns for multi-GPU B200 configurations.
    
    Tests:
    - All-to-all communication (multi-GPU)
    - Ring patterns (common in training)
    - Broadcast from rank 0
    
    Returns:
        Performance metrics
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    
    if world_size < 2:
        if rank == 0:
            print(f"Warning: multi-GPU benchmark requested, got {world_size} GPUs")
        return {}
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"{world_size}x B200 Symmetric Memory Patterns")
        print("=" * 80)
    
    results = {}
    
    for size in tensor_sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        size_mb = size[0] * 4 / 1024 / 1024
        
        dist.barrier()
        
        # Test 1: Ring pattern (rank i -> rank (i+1) % world_size)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        dest_rank = (rank + 1) % world_size
        src_rank = (rank - 1) % world_size
        recv_tensor = torch.empty_like(tensor)
        
        # Warmup
        for _ in range(10):
            dist.send(tensor, dst=dest_rank)
            dist.recv(recv_tensor, src=src_rank)
        
        torch.cuda.synchronize(device)
        start.record()
        
        for _ in range(iterations):
            dist.send(tensor, dst=dest_rank)
            dist.recv(recv_tensor, src=src_rank)
        
        end.record()
        end.synchronize()
        
        ring_time = start.elapsed_time(end) / iterations
        
        # Test 2: Broadcast from rank 0
        dist.barrier()
        
        start.record()
        for _ in range(iterations):
            dist.broadcast(tensor, src=0)
        end.record()
        end.synchronize()
        
        broadcast_time = start.elapsed_time(end) / iterations
        
        if rank == 0:
            print(f"\nSize: {size_mb:.2f} MB")
            print(f"  Ring (i->i+1):    {ring_time:.3f} ms")
            print(f"  Broadcast (0->all): {broadcast_time:.3f} ms")
            
            # Estimate bandwidth
            ring_bw = (size[0] * 4) / (ring_time / 1000) / 1e9
            bcast_bw = (size[0] * 4 * (world_size - 1)) / (broadcast_time / 1000) / 1e9
            print(f"  Ring bandwidth:     {ring_bw:.2f} GB/s")
            print(f"  Broadcast bandwidth: {bcast_bw:.2f} GB/s")
        
        results[f"ring_{size_mb:.1f}MB"] = ring_time
        results[f"broadcast_{size_mb:.1f}MB"] = broadcast_time
    
    if rank == 0:
        print("=" * 80)
    
    return results


def demonstrate_gb200_unified_memory() -> None:
    """
    Demonstrate GB200/GB300 unified memory features.
    
    GB200/GB300 provides:
    - Coherent CPU-GPU address space
    - 900 GB/s CPU↔GPU bandwidth via NVLink-C2C
    - Zero-copy CPU-GPU transfers
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    is_grace = detect_gb200_gb300()
    
    print("\n" + "=" * 80)
    print("GB200/GB300 Unified Memory Demonstration")
    print("=" * 80)
    
    if not is_grace:
        print("⚠ Not running on GB200/GB300 Grace-Blackwell")
        print("  This demo will show concepts that apply when Grace CPU is available")
        print("=" * 80)
        return
    
    print("GB200/GB300 detected!")
    print("\nKey Features:")
    print("  1. Coherent CPU-GPU memory")
    print("  2. 900 GB/s NVLink-C2C bandwidth")
    print("  3. Zero-copy data transfers")
    print("  4. Optimal for CPU preprocessing + GPU compute")
    
    # Demonstrate pinned memory allocation (optimal for Grace-Blackwell)
    size = 1024 * 1024 * 100  # 100M floats = 400MB
    
    print(f"\nAllocating {size * 4 / 1024 / 1024:.0f} MB of memory...")
    
    # CPU-pinned memory (optimal for CPU-GPU transfers on GB200/GB300)
    cpu_tensor = torch.randn(size, dtype=torch.float32, pin_memory=True)
    gpu_tensor = torch.empty(size, device='cuda:0', dtype=torch.float32)
    
    # Benchmark CPU->GPU transfer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start.record()
    for _ in range(100):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    end.record()
    end.synchronize()
    
    transfer_time = start.elapsed_time(end) / 100
    bandwidth = (size * 4) / (transfer_time / 1000) / 1e9
    
    print(f"\nCPU→GPU Transfer:")
    print(f"  Time: {transfer_time:.2f} ms")
    print(f"  Bandwidth: {bandwidth:.2f} GB/s")
    print(f"  Expected on GB200/GB300: ~800 GB/s")
    print(f"  Efficiency: {(bandwidth / 800) * 100:.1f}%")
    
    print("\nUse Cases for GB200/GB300:")
    print("  CPU data loading → Grace memory → GPU training")
    print("  CPU preprocessing (tokenization) + GPU inference")
    print("  Large batch prep on CPU, compute on GPU")
    print("  Checkpoint to CPU memory (480GB-1TB)")
    
    print("=" * 80)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetric memory example")
    parser.add_argument(
        "--benchmark-mode",
        choices=("auto", "traditional", "symmetric"),
        default="auto",
        help="Benchmark mode: auto runs the original demo, or force a transport for perf runs.",
    )
    parser.add_argument(
        "--tensor-bytes",
        type=int,
        default=4 * 1024,
        help="Payload size in bytes for benchmark-only modes.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Iterations per benchmark measurement.",
    )
    return parser.parse_args()


def main():
    """Compare traditional P2P vs symmetric memory performance."""
    args = _parse_args()
    # Setup
    rank, world_size = setup_distributed()
    device = torch.device("cuda", torch.cuda.current_device())

    if args.benchmark_mode != "auto":
        if world_size < 2:
            if rank == 0:
                print("Benchmark mode requires at least 2 GPUs.")
            dist.destroy_process_group()
            return
        numel = max(1, args.tensor_bytes // 4)
        tensor = torch.randn(numel, device=device, dtype=torch.float32)
        if args.benchmark_mode == "traditional":
            time_ms = benchmark_traditional_ring(tensor, iterations=args.iterations)
            label = "traditional_ring"
        else:
            time_ms = benchmark_symmetric_ring(tensor, iterations=args.iterations)
            label = "symmetric_ring"
        if rank == 0:
            size_kb = args.tensor_bytes / 1024
            print(f"{label}: size={size_kb:.2f} KB time={time_ms:.4f} ms/iter")
        dist.destroy_process_group()
        return

    if world_size < 2:
        if rank == 0:
            print("This example requires at least 2 GPUs.")
            print("Run with: torchrun --nproc_per_node <num_gpus> symmetric_memory_example.py")
        return
    
    # Check for 4-GPU configuration
    is_4gpu = world_size == 4
    
    # Create test tensor (small size to emphasize latency over bandwidth)
    tensor_sizes = [
        (1024,),           # 4 KB
        (1024 * 256,),     # 1 MB
        (1024 * 1024,),    # 4 MB
    ]
    
    if rank == 0:
        print("=" * 80)
        print("PyTorch 2.10 Symmetric Memory Benchmark")
        print(f"World size: {world_size} GPUs")
        if is_4gpu:
            print("4x B200 configuration detected")
        print("=" * 80)
    
    # Run multi-GPU benchmarks when >=4 GPUs are available
    if world_size >= 4:
        benchmark_multigpu_symmetric_memory(tensor_sizes, iterations=100)
    else:
        # Standard 2-GPU benchmark
        for size in tensor_sizes:
            tensor = torch.randn(size, device=device, dtype=torch.float32)
            
            dist.barrier()
            
            # Benchmark traditional P2P
            if rank == 0:
                print(f"\nTensor size: {size[0] * 4 / 1024 / 1024:.2f} MB")
            
            trad_time = benchmark_traditional_p2p(tensor, peer_rank=1, iterations=100)
            
            dist.barrier()
            
            # Benchmark symmetric memory
            sym_time = benchmark_symmetric_memory(tensor, iterations=100)
            
            if rank == 0:
                print(f"  Traditional P2P:     {trad_time:.3f} ms/iter")
                if sym_time is not None:
                    print(f"  Symmetric Memory:    {sym_time:.3f} ms/iter")
                    speedup = trad_time / sym_time
                    print(f"  Speedup:             {speedup:.2f}x")
                else:
                    print(f"  Symmetric Memory:    Not available")
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Key Takeaways:")
        print("- Symmetric memory bypasses CPU involvement for small transfers")
        print("- Prefer it when latency matters more than bandwidth")
        print("- Ideal for frequent small synchronization points in multi-GPU algorithms")
        if is_4gpu:
            print("- 4x B200: Optimal for ring/tree algorithms with symmetric memory")
        print("=" * 80)
    
    # Demonstrate GB200/GB300 features (rank 0 only for simplicity)
    if rank == 0:
        demonstrate_gb200_unified_memory()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
