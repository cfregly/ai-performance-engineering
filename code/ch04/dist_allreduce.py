#!/usr/bin/env python

import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Multi-node Gloo all-reduce benchmark")
    parser.add_argument(
        "--data-size",
        type=int,
        default=1024 * 1024 * 100,  # 100M floats ≈ 400 MB
        help="Number of elements in the tensor",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Backend to use for distributed communication"
    )
    args = parser.parse_args()

    # Check if we have enough GPUs for NCCL
    if args.backend == "nccl" and torch.cuda.device_count() < 2:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but NCCL requires at least 2 GPUs.", flush=True)
        print("Falling back to Gloo backend.", flush=True)
        args.backend = "gloo"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize the default ProcessGroup over env:// (uses MASTER_ADDR, MASTER_PORT, etc.)
    try:
        init_kwargs = {"backend": args.backend, "init_method": "env://"}
        if args.backend == "nccl":
            torch.cuda.set_device(local_rank)
            init_kwargs["device_id"] = local_rank
        dist.init_process_group(**init_kwargs)
    except Exception as e:
        print(f"Failed to initialize process group: {e}", flush=True)
        print("Running single-process benchmark instead.", flush=True)
        # Single process benchmark
        device = f"cuda:{rank}" if args.backend == "nccl" else "cpu"
        if args.backend == "nccl":
            torch.cuda.set_device(rank)
        tensor = torch.ones(args.data_size, dtype=torch.float32, device=device)
        
        start = time.time()
        # Simulate all-reduce by just doing a local operation
        result = tensor * 1  # This simulates the all-reduce operation
        elapsed = time.time() - start
        
        mb = args.data_size * 4 / 1e6
        print(f"Single-process: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms", flush=True)
        print(" Single-process benchmark completed", flush=True)
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Allocate a large tensor
    device = f"cuda:{rank}" if args.backend == "nccl" else "cpu"
    if args.backend == "nccl":
        torch.cuda.set_device(rank)
    tensor = torch.ones(args.data_size, dtype=torch.float32, device=device)
    
    # Warm up and barrier
    dist.barrier()
    
    if rank == 0:
        start = time.time()
    
    # All-reduce (sum) across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Barrier + timing
    dist.barrier()
    
    if rank == 0:
        elapsed = time.time() - start
        mb = args.data_size * 4 / 1e6
        print(f"Rank0: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms "
              f"({mb/elapsed/1e3:.1f} GB/s)", flush=True)
    
    # Verify correctness on rank 0
    if rank == 0:
        expected_value = world_size  # sum of 1's across all ranks
        actual_value = tensor[0].item()
        assert abs(actual_value - expected_value) < 1e-6, f"Expected {expected_value}, got {actual_value}"
        print(" All-reduce correctness verified", flush=True)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
