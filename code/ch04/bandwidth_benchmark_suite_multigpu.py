"""
Comprehensive Bandwidth Benchmark Suite for Multi-GPU B200 Nodes
================================================================

Measures all communication patterns and generates detailed analysis:
1. P2P bandwidth matrix (all GPU pairs)
2. Collective operations (AllReduce, AllGather, ReduceScatter)
3. Latency vs bandwidth curves
4. NVLink topology visualization
5. Scaling efficiency analysis

Hardware:
- Multi-GPU Blackwell B200 nodes
- NVLink 5.0: 1800 GB/s bidirectional per GPU pair
- Total bandwidth scales with GPU count

Expected Results:
- P2P: 800-900 GB/s per GPU pair
- AllReduce 1GB: 700-800 GB/s bus bandwidth
- Small message latency: <2 μs

Requirements:
- PyTorch 2.10+
- >=2 GPUs (requires torchrun)

Usage:
    torchrun --nproc_per_node=<num_gpus> bandwidth_benchmark_suite_multigpu.py --full

    # Quick test
    torchrun --nproc_per_node=<num_gpus> bandwidth_benchmark_suite_multigpu.py --quick

    # Save results to file
    torchrun --nproc_per_node=<num_gpus> bandwidth_benchmark_suite_multigpu.py --output results.json
"""

import argparse
import json
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env
from core.benchmark.gpu_requirements import require_min_gpus, warn_optimal_gpu_count
from core.benchmark.utils import scalar_tensor_to_float
from core.common.device_utils import resolve_local_rank


def setup_distributed():
    """Initialize distributed environment."""
    require_min_gpus(2)
    setup_single_gpu_env("bandwidth_benchmark_suite_multigpu", min_world_size=2)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=local_rank)

    return dist.get_rank(), dist.get_world_size()


def _bandwidth_summary(bandwidth_matrix: Dict[Tuple[int, int], float]) -> Tuple[float, float, float]:
    """Return average, min, and max bandwidth without materializing values."""
    values = iter(bandwidth_matrix.values())
    try:
        first = next(values)
    except StopIteration:
        return 0.0, 0.0, 0.0

    total = first
    min_bw = first
    max_bw = first
    count = 1
    for bw in values:
        total += bw
        min_bw = min(min_bw, bw)
        max_bw = max(max_bw, bw)
        count += 1
    return total / count, min_bw, max_bw


# ============================================================================
# P2P Bandwidth Matrix
# ============================================================================

def benchmark_p2p_bandwidth(
    src_rank: int,
    dst_rank: int,
    size_mb: int = 256,
    iterations: int = 100,
) -> float:
    """
    Benchmark point-to-point bandwidth between two GPUs.
    
    Args:
        src_rank: Source GPU rank
        dst_rank: Destination GPU rank
        size_mb: Transfer size in MB
        iterations: Number of iterations
        
    Returns:
        Bandwidth in GB/s
    """
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    size = size_mb * 1024 * 1024 // 4  # Convert to float32 elements
    tensor = torch.empty(size, device=device, dtype=torch.float32)
    recv_tensor = torch.empty_like(tensor) if rank == dst_rank else None
    
    # Warmup
    for _ in range(10):
        if rank == src_rank:
            dist.send(tensor, dst=dst_rank)
        elif rank == dst_rank:
            dist.recv(recv_tensor, src=src_rank)
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    if rank == src_rank:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iterations):
            dist.send(tensor, dst=dst_rank)
        end.record()
        end.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / iterations
        bandwidth_gbs = (size_mb / 1024) / (elapsed_ms / 1000)
        
        return bandwidth_gbs
    elif rank == dst_rank:
        for _ in range(iterations):
            dist.recv(recv_tensor, src=src_rank)
        
        return 0.0  # Only src_rank returns valid bandwidth
    else:
        return 0.0


def measure_p2p_matrix(
    rank: int,
    world_size: int,
    quick: bool = False,
) -> Dict[Tuple[int, int], float]:
    """
    Measure P2P bandwidth for all GPU pairs.
    
    Returns:
        Dictionary mapping (src, dst) pairs to bandwidth in GB/s
    """
    if rank == 0:
        pair_count = world_size * (world_size - 1) // 2
        print("\n" + "=" * 80)
        print(f"P2P Bandwidth Matrix ({world_size}x{world_size} = {pair_count} unique pairs)")
        print("=" * 80)
        print("Testing bidirectional bandwidth for all GPU pairs...")
        print("Transfer size: 256 MB per direction\n")
    
    bandwidth_matrix = {}
    bw_tensor = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.float32)
    
    # Test all pairs (or first pair only in quick mode)
    for src in range(world_size):
        for dst in range(src + 1, world_size):
            # Test src -> dst
            if quick:
                bw = benchmark_p2p_bandwidth(src, dst, size_mb=64, iterations=3)
            else:
                bw = benchmark_p2p_bandwidth(src, dst, size_mb=256, iterations=50)
            if rank == src:
                bandwidth_matrix[(src, dst)] = bw
                bw_tensor.fill_(bw)
            
            # Broadcast result to all ranks
            dist.broadcast(bw_tensor, src=src)
            bw = scalar_tensor_to_float(bw_tensor[0])
            bandwidth_matrix[(src, dst)] = bw
            
            if rank == 0:
                print(f"  GPU {src} → GPU {dst}: {bw:6.2f} GB/s")
        if quick:
            break
    
    if rank == 0:
        # Calculate statistics
        avg_bw, min_bw, max_bw = _bandwidth_summary(bandwidth_matrix)
        
        print(f"\nStatistics:")
        print(f"  Average: {avg_bw:.2f} GB/s")
        print(f"  Min: {min_bw:.2f} GB/s")
        print(f"  Max: {max_bw:.2f} GB/s")
        print(f"  Target: 800-900 GB/s per pair (NVLink 5.0)")
        print("=" * 80)
    
    return bandwidth_matrix


# ============================================================================
# Collective Operations Benchmark
# ============================================================================

def benchmark_collective(
    op_type: str,
    size: int,
    rank: int,
    world_size: int,
    iterations: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark a collective operation.
    
    Args:
        op_type: "allreduce", "allgather", or "reducescatter"
        size: Tensor size in elements
        rank: Current rank
        world_size: Total ranks
        iterations: Number of iterations
        
    Returns:
        (latency_ms, bandwidth_gbs)
    """
    device = torch.cuda.current_device()
    tensor = torch.empty(size, device=device, dtype=torch.float32)
    
    # Warmup
    if op_type == "allreduce":
        for _ in range(10):
            dist.all_reduce(tensor)
    elif op_type == "allgather":
        allgather_output = [torch.empty_like(tensor) for _ in range(world_size)]
        for _ in range(10):
            dist.all_gather(allgather_output, tensor)
    elif op_type == "reducescatter":
        shard_size = (size + world_size - 1) // world_size
        padded_size = shard_size * world_size
        reducescatter_source = (
            tensor if padded_size == size else torch.empty(padded_size, device=device, dtype=torch.float32)
        )
        reducescatter_input = reducescatter_source.view(world_size, shard_size)
        reducescatter_output = torch.empty(shard_size, device=device, dtype=torch.float32)
        for _ in range(10):
            dist.reduce_scatter_tensor(reducescatter_output, reducescatter_input)
    else:
        raise ValueError(f"Unsupported collective op: {op_type}")
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    if op_type == "allreduce":
        for _ in range(iterations):
            dist.all_reduce(tensor)
    elif op_type == "allgather":
        for _ in range(iterations):
            dist.all_gather(allgather_output, tensor)
    else:
        for _ in range(iterations):
            dist.reduce_scatter_tensor(reducescatter_output, reducescatter_input)
    end.record()
    end.synchronize()
    
    latency_ms = start.elapsed_time(end) / iterations
    
    # Calculate bandwidth
    data_bytes = size * 4  # float32
    if op_type == "allreduce":
        # AllReduce: 2*(N-1)/N algorithm bandwidth
        bandwidth_gbs = (data_bytes * 2 * (world_size - 1) / world_size) / (latency_ms / 1000) / 1e9
    elif op_type == "allgather":
        # AllGather: N * data
        bandwidth_gbs = (data_bytes * world_size) / (latency_ms / 1000) / 1e9
    elif op_type == "reducescatter":
        # ReduceScatter: (N-1)/N algorithm bandwidth
        bandwidth_gbs = (data_bytes * (world_size - 1) / world_size) / (latency_ms / 1000) / 1e9
    
    return latency_ms, bandwidth_gbs


def measure_collectives(rank: int, world_size: int, quick: bool = False) -> Dict[str, Dict]:
    """
    Measure all collective operations across different sizes.
    
    Returns:
        Dictionary with results for each operation and size
    """
    if rank == 0:
        print("\n" + "=" * 80)
        print("Collective Operations Benchmark")
        print("=" * 80)
    
    operations = ["allreduce", "allgather", "reducescatter"]
    
    if quick:
        sizes = [1024, 1024 * 1024]  # 4KB, 4MB
    else:
        sizes = [
            1024,           # 4 KB (latency-bound)
            256*1024,       # 1 MB
            1024*1024,      # 4 MB
            16*1024*1024,   # 64 MB
            64*1024*1024,   # 256 MB
            256*1024*1024,  # 1 GB (bandwidth-bound)
        ]
    
    results = {}
    
    for op in operations:
        if rank == 0:
            print(f"\n{op.upper()}:")
            print("-" * 80)
        
        results[op] = {}
        
        for size in sizes:
            iters = 5 if quick else 100
            latency_ms, bandwidth_gbs = benchmark_collective(
                op, size, rank, world_size, iterations=iters
            )
            
            size_mb = size * 4 / (1024 * 1024)
            results[op][size_mb] = {
                "latency_ms": latency_ms,
                "bandwidth_gbs": bandwidth_gbs,
            }
            
            if rank == 0:
                print(f"  {size_mb:8.2f} MB | {latency_ms:7.2f} ms | {bandwidth_gbs:7.2f} GB/s")
    
    if rank == 0:
        print("=" * 80)
    
    return results


# ============================================================================
# Latency vs Bandwidth Curves
# ============================================================================

def measure_latency_bandwidth_curve(rank: int, world_size: int) -> Dict[int, Tuple[float, float]]:
    """
    Measure latency and bandwidth across wide range of message sizes.
    
    Returns:
        Dictionary mapping size (bytes) to (latency_us, bandwidth_gbs)
    """
    if rank == 0:
        print("\n" + "=" * 80)
        print("Latency vs Bandwidth Curve (AllReduce)")
        print("=" * 80)
        print("Measuring across 4B to 1GB range...\n")
    
    # Logarithmic sweep from 4 bytes to 1GB
    sizes_bytes = [4 * (2**i) for i in range(30)]  # 4B, 8B, 16B, ..., 1GB
    
    results = {}
    device = torch.cuda.current_device()
    
    for size_bytes in sizes_bytes:
        size_elements = size_bytes // 4
        if size_elements == 0:
            size_elements = 1
        
        tensor = torch.empty(size_elements, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = max(10, 10000 // size_elements)  # More iterations for small sizes
        
        start.record()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        end.record()
        end.synchronize()
        
        latency_ms = start.elapsed_time(end) / iterations
        latency_us = latency_ms * 1000
        
        # Calculate bandwidth
        data_bytes = size_elements * 4
        bandwidth_gbs = (data_bytes * 2 * (world_size - 1) / world_size) / (latency_ms / 1000) / 1e9
        
        results[size_bytes] = (latency_us, bandwidth_gbs)
        
        if rank == 0 and size_bytes in [4, 1024, 1024*1024, 256*1024*1024]:
            if size_bytes < 1024:
                size_str = f"{size_bytes}B"
            elif size_bytes < 1024*1024:
                size_str = f"{size_bytes//1024}KB"
            else:
                size_str = f"{size_bytes//(1024*1024)}MB"
            
            print(f"  {size_str:>10s} | Latency: {latency_us:8.2f} μs | BW: {bandwidth_gbs:7.2f} GB/s")
    
    if rank == 0:
        print("\nKey Observations:")
        print("  - Small messages (<1KB): Latency-bound, <2 μs")
        print("  - Medium messages (1KB-1MB): Transition region")
        print("  - Large messages (>1MB): Bandwidth-bound, ~700-800 GB/s")
        print("=" * 80)
    
    return results


# ============================================================================
# Topology Visualization
# ============================================================================

def visualize_topology(rank: int, world_size: int, bandwidth_matrix: Dict[Tuple[int, int], float]):
    """Print ASCII visualization of GPU topology and bandwidth."""
    if rank != 0:
        return
    
    print("\n" + "=" * 80)
    print(f"NVLink Topology Visualization ({world_size}x B200)")
    print("=" * 80)
    
    # Print matrix
    print("\nBandwidth Matrix (GB/s):")
    print("     ", end="")
    for i in range(world_size):
        print(f"GPU{i:1d}  ", end="")
    print()
    
    for i in range(world_size):
        print(f"GPU{i}: ", end="")
        for j in range(world_size):
            if i == j:
                print("  -   ", end="")
            elif (i, j) in bandwidth_matrix:
                bw = bandwidth_matrix[(i, j)]
                print(f"{bw:5.0f} ", end="")
            elif (j, i) in bandwidth_matrix:
                bw = bandwidth_matrix[(j, i)]
                print(f"{bw:5.0f} ", end="")
            else:
                print("  ?   ", end="")
        print()
    
    # Topology detection
    print("\nTopology Analysis:")
    avg_bw, _, _ = _bandwidth_summary(bandwidth_matrix)
    
    if avg_bw > 700:
        print("  NVSwitch all-to-all topology detected")
        print("  → All GPUs have direct high-bandwidth connections")
        print("  → Optimal for: AllReduce, AllGather with NVLS")
    elif avg_bw > 400:
        print("  ⚠ Partial NVLink topology")
        print("  → Some GPUs connected via multi-hop")
        print("  → Performance may vary by GPU pair")
    else:
        print("  ⚠ PCIe-based topology")
        print("  → Limited by PCIe bandwidth")
    
    print("=" * 80)


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def main():
    # Check GPU requirements early - this script needs >=2 GPUs
    if torch.cuda.is_available():
        warn_optimal_gpu_count(torch.cuda.device_count(), "bandwidth_benchmark_suite_multigpu.py")
    require_min_gpus(2, "bandwidth_benchmark_suite_multigpu.py")

    parser = argparse.ArgumentParser(description="Multi-GPU Bandwidth Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer sizes)")
    parser.add_argument("--full", action="store_true", help="Full benchmark suite")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()
    
    rank, world_size = setup_distributed()
    
    if world_size < 2 and rank == 0:
        print("⚠ Warning: This benchmark is designed for multi-GPU runs")
    
    if rank == 0:
        print("=" * 80)
        print("Multi-GPU B200 Comprehensive Bandwidth Benchmark Suite")
        print("=" * 80)
        print(f"GPUs: {world_size}")
        print(f"Mode: {'Quick' if args.quick else 'Full' if args.full else 'Standard'}")
        print("=" * 80)
    
    all_results = {}
    
    # Benchmark 1: P2P Matrix
    if args.full or world_size <= 8:
        bandwidth_matrix = measure_p2p_matrix(rank, world_size, quick=args.quick)
        all_results["p2p_matrix"] = {f"{k[0]}-{k[1]}": v for k, v in bandwidth_matrix.items()}
        
        # Visualize topology
        visualize_topology(rank, world_size, bandwidth_matrix)
    
    # Benchmark 2: Collective Operations
    collective_results = measure_collectives(rank, world_size, quick=args.quick)
    all_results["collectives"] = collective_results
    
    # Benchmark 3: Latency vs Bandwidth
    if args.full:
        latency_bw_curve = measure_latency_bandwidth_curve(rank, world_size)
        all_results["latency_bandwidth_curve"] = {
            str(k): {"latency_us": v[0], "bandwidth_gbs": v[1]} 
            for k, v in latency_bw_curve.items()
        }
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        
        # Get AllReduce 1GB result
        if 1024.0 in collective_results.get("allreduce", {}):
            allreduce_1gb = collective_results["allreduce"][1024.0]
            print(f"\nKey Metric - AllReduce 1GB:")
            print(f"  Latency: {allreduce_1gb['latency_ms']:.2f} ms")
            print(f"  Bus Bandwidth: {allreduce_1gb['bandwidth_gbs']:.2f} GB/s")
            print(f"  Target: 700-800 GB/s")
            
            if allreduce_1gb['bandwidth_gbs'] >= 700:
                print(f"  Target achieved!")
            else:
                efficiency = (allreduce_1gb['bandwidth_gbs'] / 750) * 100
                print(f"  ⚠ {efficiency:.1f}% of target")
        
        print("\nRecommendations:")
        print("  1. Ensure NCCL 2.28+ with multi-GPU optimizations")
        print("  2. Check NVLink topology with nvidia-smi topo -m")
        print("  3. Verify no CPU throttling or power limits")
        print("  4. Use NCCL_NVLS_ENABLE=1 for dense NVLink domains")
        print("=" * 80)
    
    # Save results
    if args.output and rank == 0:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
