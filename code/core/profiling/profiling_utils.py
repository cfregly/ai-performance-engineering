"""profiling_utils.py - PyTorch profiling helpers."""

import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile
from core.profiling.nvtx_helper import standardize_nvtx_label
from core.profiling.torch_profiler_compat import prepare_torch_cuda_profiler


ProfileFn = Callable[[], Any]


def profile_with_chrome_trace(
    func: ProfileFn,
    trace_path: str = "./trace.json",
    activities: Optional[List[ProfilerActivity]] = None,
    with_stack: bool = False,
    with_modules: bool = True
) -> None:
    """
    Profile a function and export Chrome trace.
    
    Args:
        func: Function to profile
        trace_path: Output path for trace file
        activities: List of profiler activities (default: [CPU, CUDA])
        with_stack: Include Python stack traces
        with_modules: Include module hierarchy
    
    Example:
        def my_training_step():
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
        
        profile_with_chrome_trace(my_training_step, "training_trace.json")
    """
    if activities is None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    prepare_torch_cuda_profiler(
        cuda_profile_requested=(
            ProfilerActivity.CUDA in activities and torch.cuda.is_available()
        )
    )
    
    with profile(
        activities=activities,
        with_stack=with_stack,
        with_modules=with_modules,
        record_shapes=True,
    ) as prof:
        func()
    
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace exported to: {trace_path}")
    print(f"View in Chrome at: chrome://tracing")


def profile_memory(
    func: ProfileFn,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Profile memory usage of a function.
    
    Args:
        func: Function to profile
        device: CUDA device (if None, will use current device)
    
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != "cuda":
        print("Memory profiling only available for CUDA")
        return {}
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    mem_before = torch.cuda.memory_allocated(device)
    
    func()
    
    torch.cuda.synchronize(device)
    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)
    
    stats = {
        "allocated_mb": (mem_after - mem_before) / (1024**2),
        "peak_mb": mem_peak / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
    }
    
    print(f"\nMemory Usage:")
    print(f"  Allocated: {stats['allocated_mb']:.2f} MB")
    print(f"  Peak:      {stats['peak_mb']:.2f} MB")
    print(f"  Reserved:  {stats['reserved_mb']:.2f} MB")
    
    return stats


def profile_with_nvtx(
    func: ProfileFn,
    name: str = "profile_range"
) -> None:
    """
    Profile function with NVTX markers for Nsight Systems.
    
    Args:
        func: Function to profile
        name: Name for the NVTX range
    
    Example:
        profile_with_nvtx(lambda: model(input), "forward_pass")
        # Then run: nsys profile -t cuda,nvtx python script.py
    """
    torch.cuda.nvtx.range_push(standardize_nvtx_label(name))
    try:
        func()
    finally:
        torch.cuda.nvtx.range_pop()


class ProfilerContext:
    """
    Context manager for comprehensive profiling.
    
    Example:
        with ProfilerContext("training", trace_dir="./traces"):
            for batch in dataloader:
                train_step(batch)
    """
    
    def __init__(
        self,
        name: str = "profile",
        trace_dir: str = "./traces",
        enable_cuda: bool = True,
        enable_memory: bool = True,
        enable_nvtx: bool = True
    ):
        self.name = name
        self.trace_dir = trace_dir
        self.enable_cuda = enable_cuda
        self.enable_memory = enable_memory
        self.enable_nvtx = enable_nvtx
        
        os.makedirs(trace_dir, exist_ok=True)
        
        activities = [ProfilerActivity.CPU]
        cuda_profile_requested = enable_cuda and torch.cuda.is_available()
        if cuda_profile_requested:
            activities.append(ProfilerActivity.CUDA)
        prepare_torch_cuda_profiler(
            cuda_profile_requested=cuda_profile_requested
        )
        
        self.profiler = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=enable_memory,
            with_stack=True,
            with_modules=True,
        )
    
    def __enter__(self) -> "ProfilerContext":
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(standardize_nvtx_label(self.name))
        self.profiler.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()
        
        # Export traces
        trace_path = os.path.join(self.trace_dir, f"{self.name}_trace.json")
        self.profiler.export_chrome_trace(trace_path)
        print(f"\nProfile exported to: {trace_path}")
        
        # Print summary
        print("\nTop 10 operations by CUDA time:")
        print(self.profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        ))


def compare_with_profiler(
    baseline: ProfileFn,
    optimized: ProfileFn,
    trace_dir: str = "./traces"
) -> None:
    """
    Profile and compare baseline vs optimized implementations.
    
    Args:
        baseline: Baseline function
        optimized: Optimized function
        trace_dir: Directory to save traces
    """
    os.makedirs(trace_dir, exist_ok=True)
    
    print("\nProfiling baseline...")
    baseline_trace = os.path.join(trace_dir, "baseline_trace.json")
    profile_with_chrome_trace(baseline, baseline_trace)
    
    print("\nProfiling optimized...")
    optimized_trace = os.path.join(trace_dir, "optimized_trace.json")
    profile_with_chrome_trace(optimized, optimized_trace)
    
    print(f"\nCompare traces in Chrome:")
    print(f"  1. Open chrome://tracing")
    print(f"  2. Load {baseline_trace}")
    print(f"  3. Load {optimized_trace}")
