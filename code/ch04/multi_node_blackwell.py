"""Multi-Node Distributed Training for Blackwell GPUs.

Chapter 4: Multi-GPU and Multi-Node Training

Demonstrates multi-node distributed training optimizations for Blackwell 
B200/B300 GPUs with NVLink-C2C and NCCL. Includes tensor parallelism, 
FSDP, and gradient compression for multi-node scaling.

NOTE: This file uses F.scaled_dot_product_attention (SDPA), which is covered
in depth in Chapter 9 (arithmetic intensity, FlashAttention). Here we use it
as a building block for the attention layer in our training pipeline.
See ch09/baseline_sdpa_attention.py and ch09/optimized_sdpa_attention.py for
detailed SDPA analysis and optimization techniques.
"""

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import os

from core.common.device_utils import resolve_local_rank

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple
import time
# ============================================================================
# Environment Setup for Blackwell Multi-Node
# ============================================================================

def detect_gb200_gb300() -> bool:
    """Detect if running on GB200/GB300 Grace-Blackwell Superchip."""
    try:
        import platform
        if platform.machine() == 'aarch64':
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'ARM' in cpuinfo or 'Neoverse' in cpuinfo:
                    return True
    except FileNotFoundError:
        pass  # /proc/cpuinfo doesn't exist (not Linux)
    return False

def setup_multigpu_optimized(
    backend: str = "nccl",
    tp_size: int = 2,
    dp_size: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    Initialize distributed environment optimized for multi-GPU B200 systems.

    Args:
        backend: Communication backend (default: nccl)
        tp_size: Tensor parallel size (must divide world_size)
        dp_size: Data parallel size (defaults to world_size / tp_size)

    Returns:
        (rank, local_rank, world_size, local_world_size)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Multi-GPU setup requires WORLD_SIZE >= 2.")
    if dp_size is None:
        if world_size % tp_size != 0:
            raise ValueError(
                f"tp_size must divide world_size ({world_size}), got {tp_size}"
            )
        dp_size = world_size // tp_size
    if tp_size * dp_size != world_size:
        raise ValueError(
            f"tp_size * dp_size must equal world_size ({world_size}), got {tp_size} * {dp_size}"
        )

    # Use optimized NCCL config
    try:
        from ch04.nccl_blackwell_config import (
            configure_nccl_for_gb200_gb300,
            configure_nccl_for_multigpu,
        )
        is_grace = detect_gb200_gb300()
        if is_grace:
            configure_nccl_for_gb200_gb300(verbose=False)
        else:
            configure_nccl_for_multigpu(num_gpus=world_size, verbose=False)
    except ImportError:
        pass

    # Fall back to manual config
    return setup_blackwell_distributed(backend=backend)

def setup_blackwell_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int, int]:
    """Initialize distributed environment optimized for Blackwell.
    
    Returns:
        (rank, local_rank, world_size, local_world_size)
    """
    # Get environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = resolve_local_rank()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = torch.cuda.device_count()
    
    # Initialize process group
    if not dist.is_initialized():
        setup_single_gpu_env("multi_node_blackwell", min_world_size=2)
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                device_id=local_rank,
            )
        else:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Configure NCCL for Blackwell
    if backend == "nccl":
        # Detect multi-GPU configuration
        is_large_node = local_world_size >= 8
        
        # Blackwell-specific NCCL optimizations
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")  # Adjust as needed
        os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
        os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")  # GPU Direct RDMA
        os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # NVLink for intra-node
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")  # NVLink-C2C
        os.environ.setdefault("NCCL_PROTO", "Simple")  # Protocol
        
        # Optimize for larger nodes
        if is_large_node:
            os.environ.setdefault("NCCL_ALGO", "Tree,Ring,NVLS")
            os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "16")
            os.environ.setdefault("NCCL_BUFFSIZE", str(64 * 1024 * 1024))
            os.environ.setdefault("NCCL_MIN_NCHANNELS", "8")
            os.environ.setdefault("NCCL_MAX_NCHANNELS", "32")
        elif local_world_size >= 4:
            os.environ.setdefault("NCCL_ALGO", "Tree,Ring")
            os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "12")
            os.environ.setdefault("NCCL_BUFFSIZE", str(32 * 1024 * 1024))
        else:
            os.environ.setdefault("NCCL_ALGO", "Tree,Ring")
            os.environ.setdefault("NCCL_NCHANNELS_PER_NET_PEER", "8")
            os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
        
        if rank == 0:
            print("=" * 70)
            print(f"NCCL Configuration for Blackwell ({local_world_size} GPU node)")
            print("=" * 70)
            print(f" P2P Level: {os.environ.get('NCCL_P2P_LEVEL', 'default')}")
            print(f" NVLS (NVLink-C2C): {os.environ.get('NCCL_NVLS_ENABLE', '0')}")
            print(f" Algorithms: {os.environ.get('NCCL_ALGO', 'default')}")
            print(f" Channels per peer: {os.environ.get('NCCL_NCHANNELS_PER_NET_PEER', 'default')}")
            print(f" Buffer size: {int(os.environ.get('NCCL_BUFFSIZE', '2097152')) / 1024 / 1024:.1f} MB")
            if detect_gb200_gb300():
                print(" GB200/GB300 Grace-Blackwell detected")
                print(" → CPU-GPU: 900 GB/s NVLink-C2C")
            print("=" * 70)
        
        if rank == 0:
            print("\n" + "=" * 70)
            print("Distributed Training Setup")
            print("=" * 70)
            print(f" World size: {world_size}")
            print(f" Ranks per node: {local_world_size}")
            print(f" Number of nodes: {world_size // local_world_size}")
            print(f" Backend: {backend}")
            if local_world_size >= 2:
                print("\n Recommended Parallelism (single-node):")
                print(" - Choose TP to fit model memory")
                print(" - Use remaining GPUs for DP replicas")
                print(" - Ensure TP divides local_world_size")
            print("=" * 70 + "\n")
    
    return rank, local_rank, world_size, local_world_size

# ============================================================================
# Multi-Node Model with Hybrid Parallelism
# ============================================================================ 

class MultiNodeTransformerBlock(nn.Module):
    """Transformer block with hybrid parallelism for multi-node training."""
    
    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        
        # Multi-head attention (simplified)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.1 if self.training else 0.0,
        )
        x = self.o_proj(attn_out)
        x = self.dropout(x)
        x = x + residual
        
        # Feed-forward block
        residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + residual
        return x 


class MultiNodeTransformer(nn.Module):
    """Transformer model for multi-node training."""
    
    def __init__(
        self,
        vocab_size: 
            
            int = 50257,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.register_buffer(
            "_position_ids",
            torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            MultiNodeTransformerBlock(d_model, num_heads, d_ff, dropout)
    for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, T = input_ids.shape
        if T > self._position_ids.size(1):
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self._position_ids.size(1)}"
            )
        
        # Embeddings
        x = self.embedding(input_ids)
        pos = self._position_ids[:, :T]
        x = x + self.pos_embedding(pos)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

def create_multigpu_device_mesh(tp_size: int = 2, dp_size: Optional[int] = None):
    """
    Create optimal 2D device mesh for multi-GPU B200 GPUs.

    Args:
        tp_size: Tensor parallel size (must divide world_size)
        dp_size: Data parallel size (defaults to world_size / tp_size)
    
    Returns:
        2D device mesh with "dp" and "tp" dimensions
    """
    world_size = dist.get_world_size()
    if dp_size is None:
        if world_size % tp_size != 0:
            raise ValueError(
                f"tp_size must divide world_size ({world_size}), got {tp_size}"
            )
        dp_size = world_size // tp_size
    if tp_size * dp_size != world_size:
        raise ValueError(
            f"tp_size * dp_size must equal world_size ({world_size}), got {tp_size} * {dp_size}"
        )
    
    # Create 2D mesh: [data_parallel, tensor_parallel]
    # TP uses intra-node NVLink (low latency, high bandwidth)
    # DP uses inter-node or remaining GPUs (optimized by NCCL)
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    rank = dist.get_rank()
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-GPU Device Mesh Configuration")
        print("=" * 70)
        print(f" Mesh shape: ({dp_size}, {tp_size}) - (DP, TP)")
        print(f" Total GPUs: {dp_size * tp_size}")
        print(f"\nCommunication Patterns:")
        print(f" Tensor Parallel: {tp_size} GPUs via NVLink 5.0")
        print(f" → Within-group: {1800 * (tp_size - 1) // 2:.0f} GB/s aggregate")
        print(f" Data Parallel: {dp_size} groups")
        print(f" → Cross-group: NCCL AllReduce")
        print("\nMemory Distribution:")
        mem_per_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" Per TP group: {mem_per_gpu_gb * tp_size:.0f} GB (model sharded {tp_size}x)")
        print(f" Total capacity: {mem_per_gpu_gb * world_size:.0f} GB HBM3e")
        print("=" * 70)
    return device_mesh

def apply_tensor_parallelism(
    model: nn.Module,
    device_mesh,
) -> nn.Module:
    """
    Apply tensor parallelism to model using PyTorch 2.10 API.
    Optimized for intra-node parallelism via NVLink-C2C.
    """
    # Parallelize attention projections
    for layer in model.layers:
        # Column-wise parallel for Q, K, V projections
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "q_proj": ColwiseParallel(),
                "k_proj": ColwiseParallel(),
                "v_proj": ColwiseParallel(),
            },
        )
        # Row-wise parallel for output projection
        parallelize_module(
            layer,
            device_mesh["tp"],
            {"o_proj": RowwiseParallel()},
        )
        # Column-wise for FF1, Row-wise for FF2
        parallelize_module(
            layer,
            device_mesh["tp"],
            {
                "ff1": ColwiseParallel(),
                "ff2": RowwiseParallel(),
            },
        )
    return model

def apply_fsdp(
    model: nn.Module,
    device_mesh,
    mixed_precision: bool = True,
) -> FSDP:
    """
    Apply FSDP (Fully Sharded Data Parallel)
    for data parallelism.
    Optimized for inter-node communication.
    """
    # Mixed precision policy
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        device_mesh=device_mesh["dp"],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        use_orig_params=True,  # For optimizer compatibility
    )
    return model

# ============================================================================
# Multi-Node Training Loop
def train_multi_node(
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    gradient_accumulation_steps: int = 4,
    use_compile: bool = True,
) -> dict:
    """
    Multi-node distributed training with optimizations.
    
    Optimizations:
    - Gradient accumulation
    - Mixed precision (BF16)
    - torch.compile
    - Communication overlap
    - Gradient compression (future)
    
    Returns:
        Training statistics
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()
    
    # Compile model for Blackwell
    if use_compile:
        model = torch.compile(
            model,
            mode="max-autotune",
            backend="inductor",
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_data),
    )
    
    # Training stats
    stats = {
        'losses': [],
        'throughputs': [],
        'step_times': [],
    }
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Starting Multi-Node Training")
        print("=" * 70)
        print(f" Nodes: {world_size // torch.cuda.device_count()}")
        print(f" GPUs per node: {torch.cuda.device_count()}")
        print(f" Total GPUs: {world_size}")
        print(f" Mixed precision: BF16")
        print(f" torch.compile: {use_compile}")
        print(f" Gradient accumulation: {gradient_accumulation_steps}")
        print("=" * 70 + "\n")
    
    model.train()
    global_step = 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=f"cuda:{device}")
    epoch_loss_sum = torch.zeros((), dtype=torch.float64, device=f"cuda:{device}")
    
    for epoch in range(num_epochs):
        epoch_loss_sum.zero_()
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_data):
            step_start = time.perf_counter()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            # Forward pass
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            # Backward pass with gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            step_time = time.perf_counter() - step_start
            # Log statistics
            if rank == 0 and step % 10 == 0:
                loss_value_buffer[0].copy_(loss.detach())
                loss_value = float(loss_value_buffer.detach().cpu()[0]) * gradient_accumulation_steps
                tokens_per_step = input_ids.numel() * world_size
                throughput = tokens_per_step / step_time
                stats['losses'].append(loss_value)
                stats['throughputs'].append(throughput)
                stats['step_times'].append(step_time)
                print(f"Epoch {epoch} | Step {step:4d} | "
                      f"Loss: {loss_value:.4f} | "
                      f"Throughput: {throughput/1e6:.2f}M tokens/s | "
                      f"Time: {step_time*1000:.1f}ms")
            
            if rank == 0:
                epoch_loss_sum.add_(loss.detach())
            epoch_time = time.perf_counter() - epoch_start
        if rank == 0:
            loss_value_buffer[0].copy_(epoch_loss_sum)
            epoch_loss = float(loss_value_buffer.detach().cpu()[0])
            avg_loss = epoch_loss / len(train_data)
            print(f"\nEpoch {epoch} complete:")
            print(f" Average loss: {avg_loss:.4f}")
            print(f" Epoch time: {epoch_time:.2f}s")
            print(f" Throughput: {len(train_data) * input_ids.numel() * world_size / epoch_time / 1e6:.2f}M tokens/s")
    
    return stats 

# ============================================================================
# Bandwidth Benchmark
# ============================================================================ 

def benchmark_multigpu_bandwidth(
    sizes: List[int] | None = None,
    num_iters: int = 100,
) -> Dict[str, float]:
    """Run a suite of collectives tuned for a multi-GPU topology."""
    if sizes is None:
        sizes = [1 << 20, 10 << 20, 100 << 20, 500 << 20]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    results: Dict[str, float] = {}

    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-GPU Bandwidth Benchmark")
        print("=" * 70)

    def _bench(collective: Callable[[], None]) -> float:
        for _ in range(10):
            collective()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(num_iters):
            collective()
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    # All-reduce
    if rank == 0:
        print("\n[1/4] All-Reduce (FP32)")
    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        bytes_transferred = tensor.numel() * tensor.element_size()
        elapsed = _bench(lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM))
        algo_factor = 2.0 * (world_size - 1) / world_size
        bandwidth_gbs = (bytes_transferred * algo_factor * num_iters) / elapsed / 1e9
        results[f"allreduce_{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        if rank == 0:
            print(
                f" Size: {bytes_transferred/1e6:8.1f} MB | "
                f"Bandwidth: {bandwidth_gbs:6.2f} GB/s | "
                f"Time: {elapsed*1000/num_iters:6.2f} ms"
            )

    # P2P NVLink measurement
    if rank in (0, 1):
        if rank == 0:
            print("\n[2/4] P2P (GPU0 ↔ GPU1)")
        for size in sizes[:3]:
            tensor = torch.randn(size, device=device, dtype=torch.float32)
            bytes_transferred = tensor.numel() * tensor.element_size()
            elapsed = _bench(
                lambda: dist.send(tensor, dst=1) if rank == 0 else dist.recv(tensor, src=0)
            )
            bandwidth_gbs = (bytes_transferred * num_iters) / elapsed / 1e9
            if rank == 0:
                print(
                    f" Size: {bytes_transferred/1e6:8.1f} MB | "
                    f"Bandwidth: {bandwidth_gbs:6.2f} GB/s"
                )
    dist.barrier()

    # All-gather
    if rank == 0:
        print("\n[3/4] All-Gather")
    for size in sizes[1:3]:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        out = [torch.empty_like(tensor) for _ in range(world_size)]
        bytes_transferred = tensor.numel() * tensor.element_size()
        elapsed = _bench(lambda: dist.all_gather(out, tensor))
        bandwidth_gbs = (
            bytes_transferred * (world_size - 1) * num_iters / elapsed / 1e9
        )
        results[f"allgather_{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        if rank == 0:
            print(
                f" Size: {bytes_transferred/1e6:8.1f} MB | "
                f"Bandwidth: {bandwidth_gbs:6.2f} GB/s"
            )

    # Reduce-scatter
    if rank == 0:
        print("\n[4/4] Reduce-Scatter")
    for size in sizes[1:3]:
        tensor = torch.randn(size * world_size, device=device, dtype=torch.float32)
        output = torch.empty(size, device=device, dtype=torch.float32)
        reducescatter_input = tensor.view(world_size, size)
        bytes_transferred = tensor.numel() * tensor.element_size()
        elapsed = _bench(lambda: dist.reduce_scatter_tensor(output, reducescatter_input))
        bandwidth_gbs = (bytes_transferred * num_iters) / elapsed / 1e9
        results[f"reducescatter_{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        if rank == 0:
            print(
                f" Size: {bytes_transferred/1e6:8.1f} MB | "
                f"Bandwidth: {bandwidth_gbs:6.2f} GB/s"
            )

    if rank == 0:
        allreduce_total = 0.0
        allreduce_count = 0
        for key, value in results.items():
            if key.startswith("allreduce"):
                allreduce_total += value
                allreduce_count += 1
        if allreduce_count:
            avg_allreduce = allreduce_total / allreduce_count
            efficiency = (avg_allreduce / 750.0) * 100  # 750 GB/s target
            print("\n" + "=" * 70)
            print(f"Average All-Reduce Bandwidth: {avg_allreduce:.2f} GB/s")
            print(f"Efficiency vs 750 GB/s target: {efficiency:.1f}%")
            print("=" * 70 + "\n")
    return results


def benchmark_multi_node_bandwidth(
    sizes: List[int] | None = None,
    num_iters: int = 100,
) -> Dict[str, float]:
    """Benchmark collectives across multiple nodes."""
    if sizes is None:
        sizes = [1 << 20, 10 << 20, 100 << 20]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = torch.cuda.device_count()

    if local_world_size == world_size and local_world_size >= 2:
        return benchmark_multigpu_bandwidth(sizes, num_iters)

    device = torch.device("cuda", torch.cuda.current_device())
    results: Dict[str, float] = {}

    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Bandwidth Benchmark")
        print("=" * 70)

    def _bench(collective: Callable[[], None]) -> float:
        for _ in range(5):
            collective()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(num_iters):
            collective()
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    for size in sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float32)
        bytes_transferred = tensor.numel() * tensor.element_size()
        elapsed = _bench(lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM))
        algo_factor = 2.0 * (world_size - 1) / world_size
        bandwidth_gbs = (bytes_transferred * algo_factor * num_iters) / elapsed / 1e9
        results[f"allreduce_{bytes_transferred/1e6:.1f}MB"] = bandwidth_gbs
        if rank == 0:
            print(
                f" Size: {bytes_transferred/1e6:8.1f} MB | "
                f"Bandwidth: {bandwidth_gbs:6.2f} GB/s | "
                f"Time: {elapsed*1000/num_iters:6.2f} ms"
            )

    if rank == 0:
        print("=" * 70 + "\n")
    return results


class DummyDataset(torch.utils.data.Dataset):
    """Lightweight synthetic dataset for demonstration."""

    def __init__(self, size: int = 1000, seq_len: int = 512) -> None:
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        del idx
        return {
            "input_ids": torch.randint(0, 50257, (self.seq_len,)),
            "labels": torch.randint(0, 50257, (self.seq_len,)),
        }


def main() -> None:
    """Main entry point for the multi-node benchmarking demo."""
    (
        rank,
        local_rank,
        world_size,
        local_world_size,
    ) = setup_blackwell_distributed()

    num_nodes = max(1, world_size // max(1, local_world_size))
    if num_nodes <= 1:
        if rank == 0:
            print(
                "\nWARNING: multi_node_blackwell.py requires at least 2 nodes.\n"
                "Launch with torchrun --nnodes>=2 to run the full benchmark.\n"
            )
        dist.destroy_process_group()
        return

    if rank == 0:
        print("\n[1/3] Running bandwidth benchmark...")
    bandwidth_results = benchmark_multi_node_bandwidth()

    if rank == 0:
        print("\n[2/3] Setting up hybrid parallel configuration...")
    if local_world_size == world_size and local_world_size >= 2:
        device_mesh = create_multigpu_device_mesh(tp_size=2)
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(num_nodes, local_world_size),
            mesh_dim_names=("dp", "tp"),
        )

    model = MultiNodeTransformer(
        vocab_size=50257,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ).cuda()
    model = apply_tensor_parallelism(model, device_mesh)
    model = apply_fsdp(model, device_mesh, mixed_precision=True)

    if rank == 0:
        print("Hybrid parallelization complete")

    if rank == 0:
        print("\n[3/3] Starting training loop...")
    dataset = DummyDataset(size=100, seq_len=512)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
    )
    stats = train_multi_node(
        model,
        dataloader,
        num_epochs=2,
        gradient_accumulation_steps=4,
        use_compile=True,
    )

    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-Node Training Complete!")
        print("=" * 70)
        if stats["throughputs"]:
            avg_throughput = sum(stats["throughputs"]) / len(stats["throughputs"])
            print(f"Average throughput: {avg_throughput/1e6:.2f}M tokens/s")
        print("Bandwidth benchmark samples:")
        for key, value in bandwidth_results.items():
            print(f"  {key}: {value:.2f} GB/s")
        print("=" * 70 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
