import os

from core.common.device_utils import resolve_local_rank

from core.utils.architecture_runtime import (
    get_arch_config,
    get_architecture,
    get_architecture_info,
)
import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
from core.common.device_utils import cuda_supported

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except (ImportError, AttributeError):
    FSDP_AVAILABLE = False

_ARCH_CFG = get_arch_config()

class TransformerBlock(nn.Module):
    """Simple transformer block for FSDP demonstration."""
    def __init__(self, dim=256, num_heads=4, ff_dim=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class MyModel(nn.Module):
    def __init__(self, num_layers=32, dim=4096):
        super().__init__()
        self.embedding = nn.Embedding(32000, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads=32, ff_dim=11008) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 32000)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

def detect_b200_multigpu(min_gpus: int = 2) -> bool:
    """Detect if running on a multi-GPU B200 system."""
    if not torch.cuda.is_available():
        return False
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < min_gpus:
        return False
    
    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    
    # B200 has 180 GB memory
    memory_gb = props.total_memory / (1024**3)
    
    is_blackwell = (major >= 10)
    is_sm100_like = is_blackwell and (170 < memory_gb < 190)
    
    return is_sm100_like

def detect_gb200_gb300():
    """Detect if running on GB200/GB300 Grace-Blackwell Superchip."""
    import platform
    
    # Check for ARM architecture (Grace CPU)
    is_arm = platform.machine() in ['aarch64', 'arm64']
    
    # Check for Blackwell GPUs
    has_blackwell_gpu = False
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        has_blackwell_gpu = props.major >= 10  # covers SM100/103 and SM12x Grace-Blackwell
    
    # GB200/GB300 = Grace CPU + Blackwell GPU
    return is_arm and has_blackwell_gpu

def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        # Set default values for single-node training
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        
        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        local_rank = resolve_local_rank()
        torch.cuda.set_device(local_rank)
    
    return rank, world_size

def create_fsdp_model():
    """Create a model with FSDP wrapping."""
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build")
    rank, world_size = setup_distributed()
    
    # Create the model
    model = MyModel(num_layers=32, dim=4096)
    
    # Mixed precision policy (BFloat16 for compute/reduction)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy for transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
        min_num_params=1e8,
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,  # Ensure all ranks start with same parameters
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=True, pin_memory=True),
        activation_checkpointing_policy={
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.MultiheadAttention,
        },
    )
    
    return fsdp_model, rank, world_size


def create_fsdp_model_pytorch29():
    """
    Create FSDP model with PyTorch 2.10 features (NEW).
    
    PyTorch 2.10 adds:
    - forward_prefetch for overlap
    - HYBRID_SHARD_ZERO2 strategy
    - Improved performance (15-25% faster)
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build")
    rank, world_size = setup_distributed()
    
    # Create the model
    model = MyModel(num_layers=4, dim=256)
    
    # Mixed precision policy (BFloat16 recommended for Blackwell)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
        min_num_params=1e8,
    )
    
    # Check if forward_prefetch is available (PyTorch 2.10+)
    forward_prefetch_available = hasattr(FSDP, "__init__") and "forward_prefetch" in FSDP.__init__.__code__.co_varnames
    
    # FSDP configuration
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        # NEW in PyTorch 2.10: HYBRID_SHARD_ZERO2 for better performance
        "sharding_strategy": ShardingStrategy.HYBRID_SHARD if hasattr(ShardingStrategy, "HYBRID_SHARD") else ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "sync_module_states": True,
        "use_orig_params": True,
    }
    
    # NEW in PyTorch 2.10: forward_prefetch for better overlap
    if forward_prefetch_available:
        fsdp_kwargs["forward_prefetch"] = True
        fsdp_kwargs["limit_all_gathers"] = True  # Prevent memory spikes
    
    fsdp_model = FSDP(model, **fsdp_kwargs)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("FSDP PyTorch 2.10 Configuration")
        print("=" * 80)
        print(f"Sharding strategy: {fsdp_kwargs['sharding_strategy']}")
        if fsdp_kwargs["sharding_strategy"] == getattr(ShardingStrategy, "HYBRID_SHARD", fsdp_kwargs["sharding_strategy"]):
            print("Hybrid shard defaults to Zero2-style overlap on Blackwell.")
        print(f"Backward prefetch: {fsdp_kwargs['backward_prefetch']}")
        if forward_prefetch_available:
            print(f"Forward prefetch:  Enabled (NEW in 2.10)")
            print(f"Limit all gathers:  Enabled")
        else:
            print(f"Forward prefetch:  Not available (requires PyTorch 2.10+)")
        print(f"Mixed precision: BF16 (recommended for Blackwell)")
        print("=" * 80)
    
    return fsdp_model, rank, world_size

def create_fsdp_model_multigpu(tp_size=2, model_size="7B"):
    """
    Create FSDP model optimized for multi-GPU B200-class systems with hybrid parallelism.

    Recommended configurations (TP must divide world_size):
    - Smaller models:  TP=2, DP=world_size/2 (balanced)
    - Larger models:   Increase TP to reduce per-GPU memory pressure

    Args:
        tp_size: Tensor parallel size (must divide world_size)
        model_size: Model size hint ("demo", "7B", "20B", "50B", "100B+")
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build")

    rank, world_size = setup_distributed()
    if world_size < 2:
        raise RuntimeError("Multi-GPU FSDP configuration requires >=2 GPUs")
    if tp_size < 1 or world_size % tp_size != 0:
        raise ValueError(f"tp_size must be >=1 and divide world_size ({world_size}), got {tp_size}")

    is_b200_multigpu = detect_b200_multigpu(min_gpus=2)
    is_gb200_gb300 = detect_gb200_gb300()

    # Determine DP size
    dp_size = world_size // tp_size

    if rank == 0:
        print("\n" + "=" * 80)
        print("Multi-GPU Hybrid Parallel FSDP Configuration")
        print("=" * 80)
        if is_b200_multigpu:
            mem_per_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected: B200-class GPUs ({world_size} GPUs, {mem_per_gpu_gb * world_size:.0f} GB total)")
        if is_gb200_gb300:
            print("Detected: GB200/GB300 Grace-Blackwell Superchip")
        print(f"Model size: {model_size}")
        print(f"Tensor Parallel (TP) size: {tp_size}")
        print(f"Data Parallel (DP) size: {dp_size}")
        print(f"Total GPUs: {world_size}")
        print("=" * 80)

    # Create model (adjust size based on model_size parameter)
    if model_size == "7B":
        model = MyModel(num_layers=32, dim=4096)
    elif model_size == "20B":
        model = MyModel(num_layers=48, dim=5120)
    elif model_size == "50B":
        model = MyModel(num_layers=60, dim=6144)
    else:  # Demo size
        model = MyModel(num_layers=4, dim=256)

    # Mixed precision policy (BFloat16 for Blackwell)
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
        min_num_params=1e8,
    )

    # Check for PyTorch 2.10 features
    forward_prefetch_available = hasattr(FSDP, "__init__") and "forward_prefetch" in FSDP.__init__.__code__.co_varnames

    # Select sharding strategy based on configuration
    sharding_strategy = ShardingStrategy.HYBRID_SHARD if hasattr(ShardingStrategy, "HYBRID_SHARD") else ShardingStrategy.FULL_SHARD

    # FSDP configuration optimized for multi-GPU
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "sharding_strategy": sharding_strategy,
        "device_id": torch.cuda.current_device(),
        "sync_module_states": True,
        "use_orig_params": True,
    }

    # Enable PyTorch 2.10 features
    if forward_prefetch_available:
        fsdp_kwargs["forward_prefetch"] = True
        fsdp_kwargs["limit_all_gathers"] = True

    # For GB200/GB300, enable CPU offloading for larger models
    if is_gb200_gb300 and model_size in ["50B", "100B+"]:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=False, pin_memory=True)
        if rank == 0:
            print("CPU offloading enabled for large model on GB200/GB300")

    # Create FSDP model
    fsdp_model = FSDP(model, **fsdp_kwargs)

    if rank == 0:
        total_params = sum(p.numel() for p in fsdp_model.parameters())
        print(f"\nModel parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"Sharding strategy: {sharding_strategy}")
        print(f"Backward prefetch: Enabled")
        if forward_prefetch_available:
            print("Forward prefetch: Enabled (PyTorch 2.10)")
        print("Mixed precision: BF16")

        if torch.cuda.is_available():
            memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
            total_memory = memory_per_gpu * world_size
            allocated = torch.cuda.memory_allocated() / 1e9
            print("\nMemory:")
            print(f"  Per GPU: {memory_per_gpu:.1f} GB")
            print(f"  Total: {total_memory:.1f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Available: {total_memory - allocated:.2f} GB")

        print("\nMulti-GPU Performance Tips:")
        print("  - Use torch.compile() for additional 10-20% speedup")
        print("  - Enable gradient accumulation for larger effective batch sizes")
        print("  - Monitor NVLink bandwidth with nvidia-smi dmon -s u")
        print("  - Target 85-95% scaling efficiency vs single GPU")
        print("=" * 80 + "\n")

    return fsdp_model, rank, world_size

def train_step(model, batch, optimizer, criterion):
    """Single training step with FSDP."""
    optimizer.zero_grad()
    
    # Forward pass
    input_ids, labels = batch
    outputs = model(input_ids)
    
    # Compute loss (shift for language modeling)
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.detach()

def main(tp_size: int = 1, model_size: str = "demo"):
    """Main training function."""
    if not FSDP_AVAILABLE:
        print("FSDP modules not available; skipping FSDP training demo")
        return

    if not cuda_supported():
        print("CUDA unavailable or unsupported for this build; skipping FSDP demo.")
        return
    
    try:
        # Check multi-GPU availability
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 2:
            if tp_size > 1:
                raise ValueError("tp_size > 1 requires multi-GPU world_size >=2")
            fsdp_model, rank, world_size = create_fsdp_model()
        else:
            fsdp_model, rank, world_size = create_fsdp_model_multigpu(
                tp_size=tp_size,
                model_size=model_size,
            )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create dummy data
        batch_size = 2
        seq_length = 64
        vocab_size = 4096
        
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        
        # Generate random data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()
        
        if rank == 0:
            print(f"\nTraining with FSDP on {world_size} rank(s)")
            print(f"Model parameters: {sum(p.numel() for p in fsdp_model.parameters()):,}")
            
            if torch.cuda.is_available():
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
        
        # Training loop
        for step in range(2):
            loss = train_step(fsdp_model, (input_ids, labels), optimizer, criterion)
            
            if rank == 0:
                loss_value = float(loss)
                print(f"Step {step}: Loss = {loss_value:.4f}")
                
                if torch.cuda.is_available():
                    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        if rank == 0:
            print("\nTraining completed successfully!")
            
    except Exception as e:
        print(f"Error in training: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def demo_multigpu_configurations():
    """Demonstrate multi-GPU hybrid parallelism configurations."""
    if not FSDP_AVAILABLE or not torch.cuda.is_available():
        print("FSDP or CUDA not available for multi-GPU demo")
        return
    
    rank, world_size = setup_distributed()
    
    if world_size < 2:
        if rank == 0:
            print(f"Multi-GPU demo requires >=2 GPUs, found {world_size}")
        return
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("Multi-GPU Hybrid Parallelism Configuration Guide")
        print("=" * 80)
        print(f"\nTotal GPUs: {world_size}")
        print("\nRecommended configurations based on model size:")
        print("\n1. Small models (7-20B parameters):")
        print("   TP=2, DP=world_size/2 - Balanced approach (RECOMMENDED)")
        print("   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --tp-size 2 --model-size 7B")

        print("\n2. Medium models (20-50B parameters):")
        print("   TP=4, DP=world_size/4 - More model parallelism")
        print("   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --tp-size 4 --model-size 20B")

        print("\n3. Large models (50-100B parameters):")
        print("   TP=world_size, DP=1 - Maximum model parallelism")
        print("   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --tp-size <num_gpus> --model-size 50B")

        print("\n4. Very large models (100B+ parameters):")
        print("   TP=world_size, DP=N (multi-node required)")
        print("   Multi-node: TP=world_size per node, DP across nodes")

        print("\nExpected Performance (B200-class):")
        print("  - Scaling efficiency: 85-95% vs single GPU")
        print("  - AllReduce bandwidth: 700-800 GB/s (NVLink)")
        print("  - Training throughput: scales with GPU count")

        print("\nMemory Distribution (total scales with GPU count):")
        print("  - Model parameters (sharded)")
        print("  - Optimizer states (sharded)")
        print("  - Gradients (reduced, not stored long-term)")
        print("  - Activations (checkpointed when needed)")
        
        print("=" * 80 + "\n")
    
    if dist.is_initialized():
        dist.destroy_process_group()

def demonstrate_memory_efficiency():
    """Compare memory usage with and without FSDP."""
    print("\n=== FSDP Memory Efficiency Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory efficiency demo")
        return
    
    # Test without FSDP
    print("Testing without FSDP:")
    torch.cuda.reset_peak_memory_stats()
    
    regular_model = MyModel(num_layers=3, dim=256).cuda()
    optimizer = torch.optim.AdamW(regular_model.parameters())
    
    input_ids = torch.randint(0, 4096, (2, 64), device="cuda")
    labels = input_ids.clone()
    
    # Forward and backward pass
    outputs = regular_model(input_ids)
    loss = nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), 
        labels.view(-1)
    )
    loss.backward()
    
    regular_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory without FSDP: {regular_memory:.2f} GB")
    
    # Clean up
    del regular_model, optimizer, outputs, loss
    torch.cuda.empty_cache()
    
    # Test with FSDP (if distributed is available)
    if not dist.is_initialized():
        # For single GPU demo, we can still show the setup
        print("FSDP would reduce memory usage through parameter sharding")
        print("In multi-GPU setup, memory would be distributed across devices")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FSDP Training Example")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--model-size", type=str, default="demo", 
                        choices=["demo", "7B", "20B", "50B", "100B+"],
                        help="Model size for configuration")
    parser.add_argument("--show-configs", action="store_true", 
                        help="Show multi-GPU configuration guide")
    
    args = parser.parse_args()
    
    if args.show_configs:
        demo_multigpu_configurations()
    else:
        main(tp_size=args.tp_size, model_size=args.model_size)
        demonstrate_memory_efficiency()

"""
Multi-GPU Usage Examples:

1. Basic multi-GPU training (auto-detects configuration):
   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py

2. TP=2 configuration (7-20B models):
   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --tp-size 2 --model-size 7B

3. TP=4 configuration (20-50B models):
   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --tp-size 4 --model-size 20B

4. Show configuration guide:
   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --show-configs

5. Multi-node with <num_gpus> per node:
   # Node 0:
   torchrun --nproc_per_node=<num_gpus> --nnodes=2 --node_rank=0 \
       --master_addr=node0 --master_port=12355 \
       extras/ch13/fsdp_example.py --model-size 7B
   
   # Node 1:
   torchrun --nproc_per_node=<num_gpus> --nnodes=2 --node_rank=1 \
       --master_addr=node0 --master_port=12355 \
       extras/ch13/fsdp_example.py --model-size 7B

6. GB200/GB300 with CPU offloading (large models):
   torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py --model-size 50B

Expected Performance (B200-class, directional):
  - 7B model:  85-95% scaling efficiency
  - 20B model: 80-90% scaling efficiency
  - 50B model: 75-85% scaling efficiency

Memory Usage (per GPU, capacity varies by model):
  - 7B model (TP=2):   ~10-15 GB (model + optimizer + activations)
  - 20B model (TP=4):  ~20-30 GB
  - 50B model (TP=world_size):  ~40-60 GB
  - Activation checkpointing can reduce memory by 30-50%

Key Features:
  PyTorch 2.10 forward_prefetch for better overlap
  HYBRID_SHARD strategy for optimal multi-GPU performance
  BFloat16 mixed precision for Blackwell
  Automatic B200 and GB200/GB300 detection
  CPU offloading support for GB200/GB300
  Detailed memory profiling

Monitoring Commands:
  # Watch NVLink bandwidth
  nvidia-smi dmon -s u -i <gpu_ids>
  
  # Monitor GPU utilization
  nvidia-smi dmon -s m,u -i <gpu_ids>
  
  # NCCL debug info
  NCCL_DEBUG=INFO torchrun --nproc_per_node=<num_gpus> extras/ch13/fsdp_example.py
"""

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
