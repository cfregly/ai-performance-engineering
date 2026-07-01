"""
Training with torch.compile - Demonstrating 1.3-1.5x Speedup
============================================================

This shows how to achieve 1.3-1.5x training speedup with torch.compile.
Key factors for high speedup:

1. Model size: 500M-2B parameters (large enough to amortize compilation)
2. Batch size: Large batches fully utilize GPU
3. Sequence length: 1024-2048 tokens
4. Proper warmup: at least 10 iterations
5. Full graph mode: fullgraph=True for maximum optimization

Training benchmark (end-to-end):
- Forward pass
- Backward pass  
- Optimizer step

Hardware: NVIDIA B200 (SM 10.0, 178 GB HBM3e)
"""

import os
import sys


import torch
from core.utils.compile_utils import enable_tf32
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    n_layers: int = 32
    d_model: int = 1536
    n_heads: int = 24
    d_ff: int = 6144
    vocab_size: int = 50304
    seq_len: int = 1024
    

def configure_for_training():
    """Configure PyTorch for peak training performance"""
    print("Configuring for Blackwell B200 training...")
    
    # TF32 for training
    enable_tf32()
    
    # Training optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Inductor settings for training
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.triton.cudagraph_trees = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = True
    
    print("Configuration complete\n")


class TransformerBlock(nn.Module):
    """Transformer block with modern optimizations"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Attention
        self.ln1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # FFN
        self.ln2 = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        
    def forward(self, x):
        # Attention with residual
        residual = x
        x = self.ln1(x)
        
        batch, seq_len, d_model = x.shape
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        
        qkv = self.qkv(x).reshape(batch, seq_len, 3, n_heads, head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        
        # Flash Attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, d_model)
        x = self.out_proj(attn_out) + residual
        
        # FFN with residual
        residual = x
        x = self.ln2(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = x + residual
        
        return x


class LargeLanguageModel(nn.Module):
    """Large language model for training"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_training_batch(config: ModelConfig, batch_size: int):
    """Create a training batch"""
    input_ids = torch.randint(
        0, config.vocab_size,
        (batch_size, config.seq_len),
        device='cuda'
    )
    labels = input_ids.clone()
    
    return input_ids, labels


def training_step(model, input_ids, labels, optimizer):
    """Single training step"""
    # Forward
    logits = model(input_ids)
    
    # Loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )
    
    # Backward
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return loss.detach()


def benchmark_training(model, input_ids, labels, optimizer, name, num_warmup=10, num_iters=100):
    """Benchmark training performance"""
    print(f"\nBenchmarking: {name}")
    print(f"  Batch shape: {input_ids.shape}")
    print(f"  Warmup: {num_warmup} iterations")
    print(f"  Benchmark: {num_iters} iterations")
    
    # Warmup
    print(f"  Warming up...", end='', flush=True)
    for i in range(num_warmup):
        if i % 20 == 0:
            print('.', end='', flush=True)
        _ = training_step(model, input_ids, labels, optimizer)
    torch.cuda.synchronize()
    print(" done")
    
    # Benchmark
    print(f"  Running benchmark...", end='', flush=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(input_ids.device)
    start.record(current_stream)
    for i in range(num_iters):
        if i % 20 == 0:
            print('.', end='', flush=True)
        _ = training_step(model, input_ids, labels, optimizer)
    end.record(current_stream)
    end.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    print(" done")
    
    avg_time_ms = (elapsed / num_iters) * 1000
    throughput_samples = num_iters / elapsed
    tokens_per_sec = (input_ids.numel() * num_iters) / elapsed
    
    print(f"  Average time: {avg_time_ms:.2f} ms/step")
    print(f"  Throughput: {throughput_samples:.2f} samples/sec")
    print(f"  Token throughput: {tokens_per_sec:.1f} tokens/sec")
    
    return avg_time_ms, throughput_samples


def estimate_memory(config: ModelConfig, batch_size: int):
    """Estimate memory usage"""
    # Parameters
    params = config.n_layers * (
        4 * config.d_model * config.d_model +  # Attention
        2 * config.d_model * config.d_ff       # FFN
    ) + config.vocab_size * config.d_model
    
    # Activations (forward + backward)
    activations = batch_size * config.seq_len * config.d_model * 2
    
    # Gradients
    gradients = params
    
    # Optimizer states (Adam: 2x params)
    optimizer_states = params * 2
    
    total_gb = (params + activations + gradients + optimizer_states) * 4 / 1e9
    
    return {
        'params_gb': params * 4 / 1e9,
        'activations_gb': activations * 4 / 1e9,
        'gradients_gb': gradients * 4 / 1e9,
        'optimizer_gb': optimizer_states * 4 / 1e9,
        'total_gb': total_gb
    }


def main():
    """Run training benchmark"""
    configure_for_training()
    
    # Check available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Available GPU memory: {total_memory:.1f} GB\n")
    
    print("=" * 80)
    print("TRAINING BENCHMARK - TARGETING 1.3-1.5x SPEEDUP")
    print("=" * 80)
    
    # Test different configurations
    test_configs = [
        {'name': 'Medium (400M)', 'layers': 24, 'd_model': 1280, 'batch': 4},
        {'name': 'Large (700M)', 'layers': 32, 'd_model': 1536, 'batch': 4},
        {'name': 'XL (1.2B)', 'layers': 48, 'd_model': 1536, 'batch': 2},
    ]
    
    selected = None
    
    for test in test_configs:
        config = ModelConfig(
            n_layers=test['layers'],
            d_model=test['d_model'],
            n_heads=test['d_model'] // 64,  # 64 dim per head
            d_ff=test['d_model'] * 4
        )
        
        mem = estimate_memory(config, test['batch'])
        
        print(f"\nTesting: {test['name']}")
        print(f"  Layers: {test['layers']}, Hidden: {test['d_model']}")
        print(f"  Batch size: {test['batch']}")
        print(f"  Estimated memory: {mem['total_gb']:.1f} GB")
        
        if mem['total_gb'] < total_memory * 0.75:  # 75% utilization
            selected = (test, config)
            print(f"  Status: FITS ({mem['total_gb']/total_memory*100:.0f}% memory)")
        else:
            print(f"  Status: TOO LARGE")
            break
    
    if selected is None:
        print("\nNo configuration fits in memory!")
        return
    
    test, config = selected
    batch_size = test['batch']
    
    print(f"\n" + "=" * 80)
    print(f"SELECTED: {test['name']}")
    print("=" * 80)
    print(f"Layers: {config.n_layers}")
    print(f"Hidden size: {config.d_model}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print()
    
    # Create model
    print("Creating model...")
    model = LargeLanguageModel(config).cuda()
    total_params = model.count_parameters()
    print(f"Model parameters: {total_params / 1e9:.2f}B")
    
    # Create batch
    input_ids, labels = create_training_batch(config, batch_size)
    
    # Benchmark 1: Eager mode
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Eager Mode (baseline)")
    print("=" * 80)
    
    model.train()
    optimizer_eager = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    
    eager_time, eager_throughput = benchmark_training(
        model, input_ids, labels, optimizer_eager,
        "Eager Mode",
        num_warmup=10, num_iters=50
    )
    
    # Benchmark 2: Compiled mode
    print("\n" + "=" * 80)
    print("BENCHMARK 2: torch.compile (optimized)")
    print("=" * 80)
    
    # Reset model and optimizer
    model = LargeLanguageModel(config).cuda()
    model.train()
    
    # Compile model
    model_compiled = torch.compile(
        model,
        mode='max-autotune',
        fullgraph=True,
        dynamic=False,
        backend='inductor'
    )
    
    optimizer_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=1e-4, fused=True)
    
    compiled_time, compiled_throughput = benchmark_training(
        model_compiled, input_ids, labels, optimizer_compiled,
        "Compiled Mode",
        num_warmup=10, num_iters=50
    )
    
    # Results
    speedup = eager_time / compiled_time
    throughput_gain = compiled_throughput / eager_throughput
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Configuration: {test['name']} ({total_params / 1e9:.2f}B parameters)")
    print(f"Batch size: {batch_size}, Sequence length: {config.seq_len}")
    print()
    print(f"Eager Mode:         {eager_time:.2f} ms/step ({eager_throughput:.2f} samples/s)")
    print(f"Compiled Mode:      {compiled_time:.2f} ms/step ({compiled_throughput:.2f} samples/s)")
    print(f"Speedup:            {speedup:.2f}x")
    print(f"Throughput gain:    {throughput_gain:.2f}x")
    print()
    
    # Assessment
    if speedup >= 1.5:
        print("OUTSTANDING! Achieved 1.5x+ speedup target!")
        print("This demonstrates torch.compile's value for large-scale training")
    elif speedup >= 1.3:
        print("EXCELLENT! Achieved 1.3x+ speedup target!")
        print("This is realistic performance for training workloads")
    elif speedup >= 1.2:
        print("GOOD! Above 1.2x speedup")
        print("Training workloads typically see 1.2-1.5x due to backward pass complexity")
    else:
        print("Below target. Training speedup factors:")
        print("  - Backward pass less optimizable than forward")
        print("  - Optimizer step overhead")
        print("  - Memory bandwidth constraints")
    
    print("\n" + "=" * 80)
    print("KEY LEARNINGS FOR BOOK")
    print("=" * 80)
    print("1. Training speedup (1.2-1.5x) is lower than inference (1.3-2.0x)")
    print("2. Backward pass is harder to optimize than forward pass")
    print("3. Larger models (500M-2B) show better speedup than small models")
    print("4. Large batch sizes maximize GPU utilization")
    print("5. Use fused optimizers (fused=True) for AdamW")
    print("6. Warmup is still important: run ≥10 iterations")
    print("7. fullgraph=True gives best results when possible")
    print("8. TF32 is essential for Blackwell training performance")
    print("9. Combine with mixed precision (BF16) for further speedup")
    print("10. End-to-end training speedup includes forward + backward + optimizer")
    print("=" * 80)
    
    return speedup


if __name__ == "__main__":
    speedup = main()
    
    SUCCESS_THRESHOLD = float(os.getenv("TRAINING_SPEEDUP_THRESHOLD", "1.05"))
    sys.exit(0 if speedup >= SUCCESS_THRESHOLD else 1)
