import os
from contextlib import nullcontext

from core.utils.architecture_runtime import (
    get_arch_config,
    get_architecture,
    get_architecture_info,
)
import torch
import torch.nn as nn
from core.common.device_utils import get_preferred_device

_ARCH_CFG = get_arch_config()

def demonstrate_memory_profiling():
    """Demonstrate PyTorch memory profiling capabilities."""
    device_obj, cuda_err = get_preferred_device()
    device = device_obj.type
    cuda_ok = device == 'cuda' and cuda_err is None

    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); running demo on CPU.")

    # Clear any existing allocations
    if cuda_ok:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("=== Memory Profiling Demo ===")
    
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)  # reduced sizes for fast profiling
            self.linear2 = nn.Linear(1024, 2048)
            self.linear3 = nn.Linear(2048, 512)
            
        def forward(self, x):
            x = torch.relu_(self.linear1(x))
            x = torch.relu_(self.linear2(x))
            return self.linear3(x)
    
    model = SimpleModel().to(device)
    
    # Create input data
    batch_size = 16  # smaller batch keeps demo responsive during profiling
    input_data = torch.randn(batch_size, 512, device=device)
    target = torch.randint(0, 512, (batch_size,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    if cuda_ok:
        # Memory snapshot before training
        print("\nInitial memory state:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        # Take memory snapshot
        torch.cuda.memory._record_memory_history(max_entries=200000)
    
    # Training loop with memory tracking
    for epoch in range(2):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if cuda_ok:
            print(f"\nEpoch {epoch + 1}:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
            print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    if cuda_ok:
        # Memory snapshot
        torch.cuda.memory._dump_snapshot("memory_snapshot.json")
        snapshot = torch.cuda.memory.memory_snapshot()
        print(f"\nMemory snapshot contains {len(snapshot)} records")
        
        # Load memory_snapshot.json in https://pytorch.org/memory_viz for interactive analysis
        
        # Stop recording
        torch.cuda.memory._record_memory_history(enabled=None)
        
        # Detailed memory stats
        memory_stats = torch.cuda.memory_stats()
        print(f"\nDetailed Memory Statistics:")
        print(f"Peak allocated bytes: {memory_stats.get('allocated_bytes.all.peak', 0) / 1e6:.1f} MB")
        print(f"Peak reserved bytes: {memory_stats.get('reserved_bytes.all.peak', 0) / 1e6:.1f} MB")
        print(f"Number of allocations: {memory_stats.get('num_alloc_retries', 0)}")
        print(f"Number of OOM: {memory_stats.get('num_ooms', 0)}")

def demonstrate_memory_optimization():
    """Show memory optimization techniques."""
    print("\n=== Memory Optimization Techniques ===")

    device_obj, cuda_err = get_preferred_device()
    device = device_obj.type
    cuda_ok = device == 'cuda' and cuda_err is None

    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); running on CPU.")

    # 1. Gradient Checkpointing
    class CheckpointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(512, 512) for _ in range(5)
            ])

        def forward(self, x):
            for layer in self.layers:
                # Use checkpoint to trade compute for memory
                x = torch.utils.checkpoint.checkpoint(torch.relu, layer(x), use_reentrant=False)
            return x

    print("1. Gradient Checkpointing:")
    model = CheckpointModel().to(device)
    x = torch.randn(16, 512, device=device, requires_grad=True)

    if cuda_ok:
        torch.cuda.reset_peak_memory_stats()

    y = model(x)
    loss = y.sum()
    loss.backward()

    if cuda_ok:
        print(f"Peak memory with checkpointing: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # 2. Memory-efficient attention (scaled_dot_product_attention)
    print("\n2. Memory-efficient attention:")

    def efficient_attention_demo() -> None:
        batch_size, seq_len, embed_dim = 8, 256, 256

        query = torch.randn(batch_size, seq_len, embed_dim, device=device)
        key = torch.randn(batch_size, seq_len, embed_dim, device=device)
        value = torch.randn(batch_size, seq_len, embed_dim, device=device)

        if cuda_ok:
            torch.cuda.reset_peak_memory_stats()

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            _ = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
            if cuda_ok:
                print(f"Memory-efficient attention peak: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        else:
            print("scaled_dot_product_attention not available in this PyTorch version")

    efficient_attention_demo()

    # 3. Mixed precision
    print("\n3. Mixed precision training:")

    def mixed_precision_demo() -> None:
        model = nn.Linear(512, 512).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler(enabled=cuda_ok)

        x = torch.randn(16, 512, device=device)
        target = torch.randn(16, 512, device=device)

        if cuda_ok:
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()
        autocast_ctx = torch.autocast("cuda") if cuda_ok else nullcontext()

        with autocast_ctx:
            output = model(x)
            loss = nn.functional.mse_loss(output, target)

        if cuda_ok:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"Mixed precision peak memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        else:
            loss.backward()
            optimizer.step()

    mixed_precision_demo()

def demonstrate_pytorch_29_memory_features():
    """
    Demonstrate PyTorch 2.10 memory features (NEW).
    
    PyTorch 2.10 adds:
    - Improved memory snapshot v2 API
    - Better profiler integration
    - Blackwell-specific metrics
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping PyTorch 2.10 memory demos")
        return
    
    print("\n=== PyTorch 2.10 Memory Features ===")
    
    device = 'cuda'
    
    # 1. Memory snapshot v2 with enhanced metadata
    print("\n1. Enhanced Memory Snapshot (v2 API):")
    
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 1024)
    ).to(device)
    
    x = torch.randn(32, 1024, device=device)
    
    # Enable memory history with context
    torch.cuda.memory._record_memory_history(
        enabled=True,
        context="pytorch_29_demo",  # NEW in 2.9: context tagging
        stacks="python",  # Record Python stack traces
        max_entries=10000
    )
    
    # Run forward pass
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Get snapshot with enhanced metadata
    snapshot = torch.cuda.memory._snapshot()
    
    print(f"   Snapshot entries: {len(snapshot)}")
    print(f"   Memory events tracked with Python stacks")
    print(f"   Export with: torch.cuda.memory._dump_snapshot('snapshot.pkl')")
    
    torch.cuda.memory._record_memory_history(False)
    
    # 2. Memory-efficient attention backend selection (PyTorch 2.10)
    print("\n2. Memory-Efficient Attention Backend (FlashAttention-3 for Blackwell):")
    
    # Enable specific backends
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)  # FlashAttention-3
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        
        print("    FlashAttention-3 enabled (Blackwell-optimized)")
        print("    Memory-efficient backend enabled")
        print("    Math backend disabled (slow fallback)")
        
        # Check which backend is selected
        if hasattr(torch.backends.cuda, "preferred_sdp_backend"):
            backend = torch.backends.cuda.preferred_sdp_backend
            print(f"   Preferred backend: {backend}")
    
    # 3. Architecture-specific memory metrics
    print("\n3. Architecture-Specific Memory Metrics:")
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    arch = _ARCH_CFG.arch
    
    if arch == "blackwell":
        print(f"   Detected: Blackwell B200/B300 (CC {compute_capability})")
        print(f"   HBM3e Total: {device_props.total_memory / 1e9:.1f} GB")
        print("   Memory bandwidth: ~7.8 TB/s")
        print(f"   L2 Cache: {device_props.l2_cache_size / 1024 / 1024:.1f} MB")
        
        # Check HBM3e utilization
        allocated = torch.cuda.memory_allocated() / device_props.total_memory
        print(f"   HBM3e utilization: {allocated * 100:.1f}%")
    elif arch == "grace_blackwell":
        print(f"   Detected: Grace-Blackwell GB10 (CC {compute_capability})")
        print(f"   Total GPU memory: {device_props.total_memory / 1e9:.1f} GB")
        print(f"   L2 Cache: {device_props.l2_cache_size / 1024 / 1024:.1f} MB")
        print("   Bandwidth metrics: consult Nsight Compute on GB10 for peak values")
    else:
        print(f"   Non-Blackwell GPU detected (CC {compute_capability})")
    
    # 4. Advanced profiler integration (PyTorch 2.10)
    print("\n4. PyTorch Profiler with Blackwell Features:")
    
    from torch.profiler import profile, ProfilerActivity
    
    model_profiler = nn.Sequential(
        nn.Linear(512, 1024),
        nn.GELU(),
        nn.Linear(1024, 512)
    ).to(device)
    
    x_profiler = torch.randn(16, 512, device=device)
    
    # Check if experimental config is available (PyTorch 2.10+)
    experimental_config = None
    use_experimental = False
    if hasattr(torch._C, '_profiler') and hasattr(torch._C._profiler, '_ExperimentalConfig'):
        experimental_config = torch._C._profiler._ExperimentalConfig(
            verbose=True,
            enable_cuda_sync_events=True,  # Blackwell-specific sync tracking
            adjust_timestamps=True,
        )
        use_experimental = True
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        experimental_config=experimental_config if use_experimental else None,
    ) as prof:
        output = model_profiler(x_profiler)
        loss = output.sum()
        loss.backward()
    
    print(f"   {'' if use_experimental else ''} Experimental Blackwell features: {use_experimental}")
    print(f"   Profiler captured {len(prof.key_averages())} events")
    
    # Print top memory consumers
    print("\n   Top memory-consuming operations:")
    for event in prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=3).split('\n')[:5]:
        print(f"   {event}")
    
    print("\n=== End PyTorch 2.10 Memory Features ===")


if __name__ == "__main__":
    demonstrate_memory_profiling()
    demonstrate_memory_optimization()
    demonstrate_pytorch_29_memory_features()  # NEW in PyTorch 2.10

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if _ARCH_CFG.arch in {"blackwell", "grace_blackwell"} and triton_cfg is not None:
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
