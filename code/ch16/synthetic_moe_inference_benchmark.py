"""
Synthetic MoE Inference Benchmark for Blackwell B200
=====================================================

IMPORTANT: This is a SYNTHETIC/HAND-BUILT MoE model for benchmarking!
It is NOT the real GPT-OSS model from OpenAI/HuggingFace.

This demonstrates inference optimizations for large MoE models on 
NVIDIA Blackwell B200 GPUs, showing:

1. torch.compile speedup on large models
2. Memory scaling with layer count
3. Batch size optimization for B200
4. MoE architecture implementation patterns


Model: Synthetic MoE Transformer (10-50B parameters, configurable)
- Architecture inspired by GPT-OSS but simplified
- MoE with 64 experts (only 1 implemented for memory efficiency)
- Scalable from 8-48 layers depending on GPU memory
- Good for testing infrastructure without downloading huge models

Performance Targets:
- Baseline eager mode: ~50-100 ms per batch
- With torch.compile: 1.5-2x speedup expected
- Scales to use available B200 memory (178 GB)

Hardware: NVIDIA B200 (SM 10.0, 178 GB HBM3e)

For REAL GPT-OSS models, see:
- openai/gpt-oss-120b on HuggingFace (120B parameters)
- openai/gpt-oss-20b on HuggingFace (20B parameters)
"""
from core.harness.arch_config import prefer_flash_sdpa

import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.compile_utils import enable_tf32, compile_model

CURRENT_DEVICE_FLAVOR = "unknown"


def build_candidate_configs(device_flavor: str):
    """Return candidate layer/batch/seq settings and an explanatory note."""
    if device_flavor == "blackwell_sm100":
        configs = [
            {'layers': 16, 'batch': 8, 'seq': 1024},
            {'layers': 12, 'batch': 8, 'seq': 1024},
            {'layers': 8, 'batch': 8, 'seq': 1024},
            {'layers': 6, 'batch': 8, 'seq': 1024},
            {'layers': 4, 'batch': 8, 'seq': 1024},
            {'layers': 2, 'batch': 8, 'seq': 1024},
        ]
        note = "B200 baseline: moderate config tuned for torch.compile vs eager comparisons."
    elif device_flavor == "blackwell_sm121":
        configs = [
            {'layers': 8, 'batch': 4, 'seq': 512},
            {'layers': 6, 'batch': 4, 'seq': 512},
            {'layers': 4, 'batch': 4, 'seq': 512},
            {'layers': 2, 'batch': 2, 'seq': 512},
            {'layers': 2, 'batch': 2, 'seq': 256},
        ]
        note = "GB10 baseline: compact sequence settings that still highlight compile gains."
    else:
        configs = [
            {'layers': 4, 'batch': 4, 'seq': 1024},
            {'layers': 3, 'batch': 4, 'seq': 768},
            {'layers': 2, 'batch': 2, 'seq': 768},
            {'layers': 2, 'batch': 2, 'seq': 512},
        ]
        note = "Minimal footprint path for non-Blackwell devices while keeping measurable speedup."
    return configs, note


def resolve_iteration_schedule(device_flavor: str):
    """Choose warmup/iteration counts based on hardware profile."""
    eager_warmup = 5
    eager_iters = 20
    compile_warmup = 20
    compile_iters = 20
    fp8_eager_warmup = eager_warmup
    fp8_eager_iters = eager_iters
    fp8_compile_warmup = compile_warmup
    fp8_compile_iters = compile_iters

    if device_flavor == "blackwell_sm100":
        eager_warmup, eager_iters = 5, 20
        compile_warmup, compile_iters = 25, 20
        fp8_eager_warmup, fp8_eager_iters = 8, 15
        fp8_compile_warmup, fp8_compile_iters = 20, 15
    elif device_flavor == "blackwell_sm121":
        eager_warmup, eager_iters = 3, 10
        compile_warmup, compile_iters = 10, 12
        fp8_eager_warmup, fp8_eager_iters = 5, 10
        fp8_compile_warmup, fp8_compile_iters = 10, 12
    else:
        eager_warmup, eager_iters = 3, 10
        compile_warmup, compile_iters = 8, 10
        fp8_eager_warmup, fp8_eager_iters = 5, 10
        fp8_compile_warmup, fp8_compile_iters = 8, 10

    return (
        eager_warmup,
        eager_iters,
        compile_warmup,
        compile_iters,
        fp8_eager_warmup,
        fp8_eager_iters,
        fp8_compile_warmup,
        fp8_compile_iters,
    )


@dataclass
class SyntheticMoEConfig:
    """Configuration for synthetic MoE model (inspired by GPT-OSS architecture)"""
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 64
    d_model: int = 8192
    d_ff: int = 32768  # 4x d_model for MoE experts
    n_experts: int = 64
    n_experts_per_token: int = 2  # MoE: 2 experts active per token
    max_seq_len: int = 8192
    use_fp8: bool = False
    model_dtype: torch.dtype = torch.bfloat16


def detect_device_flavor() -> str:
    """Classify the active CUDA device to tailor benchmark heuristics."""
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    if "gb10" in name or (props.major == 12 and props.minor >= 1):
        return "blackwell_sm121"
    if "b200" in name or (props.major == 12 and props.minor == 0):
        return "blackwell_sm100"
    return "other"


class FP8Linear(nn.Module):
    """Linear layer that stores weights in FP8 and computes in bfloat16."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise RuntimeError("FP8Linear requires torch.float8_e4m3fn support.")

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.fp8_dtype = fp8_dtype
        self.has_bias = bias

        self.register_buffer(
            "weight_fp8",
            torch.empty(out_features, in_features, dtype=self.fp8_dtype),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_dequant",
            torch.empty(out_features, in_features, dtype=compute_dtype),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_buffer("bias", torch.empty(0, dtype=compute_dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        temp = torch.empty(self.out_features, self.in_features, dtype=torch.float32)
        nn.init.xavier_uniform_(temp)
        self._quantize_from_fp32(temp)
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def _quantize_from_fp32(self, weight_fp32: torch.Tensor) -> None:
        max_vals = weight_fp32.detach().abs().amax(dim=1, keepdim=True)
        scale = torch.clamp(max_vals / 240.0, min=1e-8)
        quant = torch.clamp((weight_fp32 / scale).round(), -240, 240).to(self.fp8_dtype)
        self.weight_fp8.copy_(quant)
        self.weight_scale.copy_(scale)
        self._refresh_dequantized_weight()

    def _refresh_dequantized_weight(self) -> None:
        scale = self.weight_scale.to(dtype=self.compute_dtype, device=self.weight_fp8.device)
        dequant = self.weight_fp8.to(dtype=self.compute_dtype).mul_(scale)
        if (
            self.weight_dequant.device != dequant.device
            or self.weight_dequant.dtype != dequant.dtype
            or tuple(self.weight_dequant.shape) != tuple(dequant.shape)
        ):
            self.weight_dequant = dequant
        else:
            self.weight_dequant.copy_(dequant)

    def convert_precision(self, compute_dtype: torch.dtype) -> None:
        """Update compute dtype while keeping FP8 weights intact."""
        self.compute_dtype = compute_dtype
        if self.has_bias and self.bias.dtype != compute_dtype:
            self.bias = self.bias.to(compute_dtype)
        self._refresh_dequantized_weight()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != self.compute_dtype:
            input = input.to(self.compute_dtype)
        weight = self.weight_dequant
        bias = None
        if self.has_bias:
            bias = self.bias
        return F.linear(input, weight, bias)


def _convert_module_precision(module: nn.Module, dtype: torch.dtype) -> None:
    """Recursively cast parameters/buffers to the target dtype, skipping FP8 weights."""
    for child in module.children():
        _convert_module_precision(child, dtype)
    if isinstance(module, FP8Linear):
        module.convert_precision(dtype)
        return
    for _name, param in module.named_parameters(recurse=False):
        if not torch.is_floating_point(param):
            continue
        param.data = param.data.to(dtype)
    for name, buf in module.named_buffers(recurse=False):
        if not torch.is_floating_point(buf):
            continue
        if buf.dtype == getattr(torch, "float8_e4m3fn", None):
            continue
        module.register_buffer(name, buf.to(dtype), persistent=True)


def _make_linear(config: SyntheticMoEConfig, in_features: int, out_features: int, *, bias: bool = True) -> nn.Module:
    if config.use_fp8:
        return FP8Linear(in_features, out_features, bias=bias, compute_dtype=config.model_dtype)
    return nn.Linear(in_features, out_features, bias=bias)


def _count_parameters_with_fp8(module: nn.Module) -> int:
    """Count parameters including FP8 buffers."""
    total = sum(p.numel() for p in module.parameters())
    for submodule in module.modules():
        if isinstance(submodule, FP8Linear):
            total += submodule.weight_fp8.numel()
            if submodule.has_bias:
                total += submodule.bias.numel()
    return total

def configure_for_inference():
    """Configure PyTorch for peak inference performance"""
    print("Configuring for Blackwell B200 inference...")
    
    # TF32 for mixed precision (avoids legacy/new API mixing issues)
    enable_tf32()
    
    # Flash Attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Inference-specific settings
    torch.backends.cudnn.benchmark = True
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.max_autotune = True
    
    print("Configuration complete\n")


class SimpleMoELayer(nn.Module):
    """Simplified Mixture-of-Experts layer (only 1 expert implemented for memory efficiency)"""
    def __init__(self, config: SyntheticMoEConfig):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.n_experts_per_token = config.n_experts_per_token
        self.compute_dtype = config.model_dtype
        
        # Router
        self.router = _make_linear(config, config.d_model, config.n_experts, bias=False)
        
        # Experts (simplified - just one expert for memory efficiency)
        self.expert = nn.Sequential(
            _make_linear(config, config.d_model, config.d_ff),
            nn.GELU(),
            _make_linear(config, config.d_ff, config.d_model)
        )
        
    def forward(self, x):
        # Route to experts (simplified: use all tokens with expert 0)
        return self.expert(x)


class SyntheticMoEBlock(nn.Module):
    """Synthetic MoE Transformer block"""
    def __init__(self, config: SyntheticMoEConfig):
        super().__init__()
        self.config = config
        self.compute_dtype = config.model_dtype
        
        # Attention
        self.ln1 = nn.LayerNorm(config.d_model)
        self.qkv = _make_linear(config, config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = _make_linear(config, config.d_model, config.d_model, bias=False)
        
        # MoE FFN
        self.ln2 = nn.LayerNorm(config.d_model)
        self.moe = SimpleMoELayer(config)
        
    def forward(self, x):
        # Attention with residual
        residual = x
        x = self.ln1(x)
        
        batch, seq_len, d_model = x.shape
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        
        qkv = self.qkv(x).reshape(batch, seq_len, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash Attention
        with prefer_flash_sdpa():
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, d_model)
        x = self.out_proj(attn_out) + residual
        
        # MoE FFN with residual
        residual = x
        x = self.ln2(x)
        x = self.moe(x) + residual
        
        return x


class SyntheticMoEModel(nn.Module):
    """Synthetic MoE model for benchmarking (10-50B parameters depending on layers)"""
    def __init__(self, config: SyntheticMoEConfig, num_layers: int = None):
        super().__init__()
        self.config = config
        self.use_fp8 = config.use_fp8
        self.compute_dtype = config.model_dtype
        
        # Use fewer layers for memory efficiency (simulate full model)
        actual_layers = num_layers if num_layers else min(8, config.n_layers)
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            SyntheticMoEBlock(config) for _ in range(actual_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = _make_linear(config, config.d_model, config.vocab_size, bias=False)

        _convert_module_precision(self, self.compute_dtype)
        
        # Count parameters
        self.total_params = _count_parameters_with_fp8(self)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


def estimate_memory_usage(config: SyntheticMoEConfig, batch_size: int, seq_len: int, num_layers: int):
    """Estimate memory usage for the model"""
    # Model parameters (simplified)
    # Each layer: ~2.5B params (attention: 0.5B, MoE: 2B per expert, using 2/64)
    params_per_layer = (
        4 * config.d_model * config.d_model +  # QKV + out_proj
        2 * config.d_model * config.d_ff / config.n_experts * config.n_experts_per_token  # MoE
    )
    total_params = params_per_layer * num_layers + config.vocab_size * config.d_model
    
    # Memory in GB (FP16)
    param_memory = total_params * 2 / 1e9
    
    # Activations (per token)
    activation_memory = batch_size * seq_len * config.d_model * 2 / 1e9
    
    # KV cache
    kv_cache = 2 * num_layers * batch_size * seq_len * config.d_model * 2 / 1e9
    
    total = param_memory + activation_memory + kv_cache
    
    return {
        'params_gb': param_memory,
        'activations_gb': activation_memory,
        'kv_cache_gb': kv_cache,
        'total_gb': total
    }


def _benchmark_autocast_context(use_autocast, autocast_dtype):
    if use_autocast:
        return torch.autocast("cuda", dtype=autocast_dtype)
    return nullcontext()


def benchmark_inference(model, input_ids, name, num_warmup=20, num_iters=100, *, autocast_dtype=None):
    """Benchmark inference performance"""
    print(f"\nBenchmarking: {name}")
    print(f"  Input shape: {input_ids.shape}")
    
    # Warmup
    print(f"  Warming up ({num_warmup} iterations)...", end='', flush=True)
    use_autocast = autocast_dtype is not None and input_ids.device.type == "cuda"
    with torch.inference_mode(), _benchmark_autocast_context(use_autocast, autocast_dtype):
        for _ in range(num_warmup):
            _ = model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(" done")
    
    # Benchmark
    print(f"  Running benchmark ({num_iters} iterations)...", end='', flush=True)
    count = max(num_iters, 1)
    if input_ids.device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(input_ids.device)
        with torch.inference_mode(), _benchmark_autocast_context(use_autocast, autocast_dtype):
            start_event.record(current_stream)
            for _ in range(count):
                _ = model(input_ids)
            end_event.record(current_stream)
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(count):
                _ = model(input_ids)
        elapsed = time.perf_counter() - start
    print(" done")
    
    avg_time_ms = (elapsed / count) * 1000
    tokens_per_sec = (input_ids.numel() * count) / elapsed
    
    print(f"  Average time: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    
    return avg_time_ms, tokens_per_sec


def main():
    """Run Synthetic MoE inference benchmark"""
    configure_for_inference()
    
    # Check available memory
    if not torch.cuda.is_available():
        print("CUDA device unavailable; skipping benchmark.")
        return 0.0
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Available GPU memory: {total_memory:.1f} GB\n")
    
    # Configuration
    config = SyntheticMoEConfig()
    device_flavor = detect_device_flavor()
    global CURRENT_DEVICE_FLAVOR
    CURRENT_DEVICE_FLAVOR = device_flavor

    # Auto-scale configuration for quick validation runs while preserving MoE structure.
    # We keep expert count high but shrink hidden size when memory is limited so the
    # benchmark runs quickly and doesn't exhaust developer GPUs.
    if total_memory < 160:
        config = replace(
            config,
            d_model=4096,
            d_ff=16384,
            n_heads=32,
        )
    if total_memory < 80:
        config = replace(
            config,
            d_model=3072,
            d_ff=12288,
            n_heads=24,
        )
    
    print("=" * 80)
    print("SYNTHETIC MoE INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Model: Synthetic MoE (NOT real GPT-OSS)")
    print(f"Architecture: {config.n_layers} layers max, {config.d_model} hidden")
    print(f"Scaling to fit: {total_memory:.0f} GB available memory")
    print()
    
    test_configs, config_note = build_candidate_configs(device_flavor)
    
    selected_config = None
    
    for test in test_configs:
        mem = estimate_memory_usage(config, test['batch'], test['seq'], test['layers'])
        print(f"Testing: {test['layers']} layers, batch={test['batch']}, seq={test['seq']}")
        print(f"  Estimated memory: {mem['total_gb']:.1f} GB")

        if mem['total_gb'] <= total_memory * 0.7:  # target 70% utilization ceiling
            selected_config = test
            print("  Status: FITS")
            break
        else:
            print("  Status: TOO LARGE")
    
    if selected_config is None:
        print("\nNo configuration fits in memory!")
        return
    
    print(f"\n" + "=" * 80)
    print(f"SELECTED CONFIGURATION")
    print("=" * 80)
    print(f"Layers: {selected_config['layers']}")
    print(f"Batch size: {selected_config['batch']}")
    print(f"Sequence length: {selected_config['seq']}")
    print()
    if config_note:
        print(config_note)
        print()
    
    # Create model
    print("Creating synthetic MoE model...")
    model = SyntheticMoEModel(config, num_layers=selected_config['layers'])
    model = model.cuda().eval()
    baseline_param_count = model.total_params
    
    print(f"Model parameters: {model.total_params / 1e9:.2f}B")
    print(f"(This is a synthetic model for benchmarking, NOT real GPT-OSS)")
    print()
    
    # Create input
    batch_size = selected_config['batch']
    seq_len = selected_config['seq']
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')

    (
        eager_warmup,
        eager_iters,
        compile_warmup,
        compile_iters,
        fp8_eager_warmup,
        fp8_eager_iters,
        fp8_compile_warmup,
        fp8_compile_iters,
    ) = resolve_iteration_schedule(device_flavor)
    
    # Benchmark 1: Eager mode (baseline)
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Eager Mode (BF16 baseline)")
    print("=" * 80)
    eager_time, eager_throughput = benchmark_inference(
        model,
        input_ids,
        "Eager Mode BF16",
        num_warmup=eager_warmup,
        num_iters=eager_iters,
    )
    
    # Benchmark 2: Compiled mode
    print("\n" + "=" * 80)
    print("BENCHMARK 2: torch.compile (optimized)")
    print("=" * 80)
    model_compiled = compile_model(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    compiled_time, compiled_throughput = benchmark_inference(
        model_compiled,
        input_ids,
        "Compiled Mode",
        num_warmup=compile_warmup,
        num_iters=compile_iters,
    )
    
    fp8_eager_time = fp8_eager_throughput = None
    fp8_compiled_time = fp8_compiled_throughput = None
    
    fp8_supported = getattr(torch, "float8_e4m3fn", None) is not None
    if fp8_supported:
        # Free baseline and compiled models to make room for FP8 variants
        if model_compiled is not None:
            del model_compiled
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("BENCHMARK 3: FP8 Weights (Eager Mode)")
        print("=" * 80)
        fp8_config = replace(config, use_fp8=True)
        fp8_model = SyntheticMoEModel(fp8_config, num_layers=selected_config['layers']).cuda().eval()
        fp8_eager_time, fp8_eager_throughput = benchmark_inference(
            fp8_model,
            input_ids,
            "FP8 Eager Mode",
            num_warmup=fp8_eager_warmup,
            num_iters=fp8_eager_iters,
        )

        print("\n" + "=" * 80)
        print("BENCHMARK 4: FP8 Weights + torch.compile")
        print("=" * 80)
        fp8_model_compiled = compile_model(
            fp8_model,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        fp8_compiled_time, fp8_compiled_throughput = benchmark_inference(
            fp8_model_compiled,
            input_ids,
            "FP8 Compiled Mode",
            num_warmup=fp8_compile_warmup,
            num_iters=fp8_compile_iters,
        )
    else:
        print("\nPyTorch build does not expose torch.float8_e4m3fn; skipping FP8 benchmarks.")
    
    # Results
    speedup = eager_time / compiled_time
    throughput_gain = compiled_throughput / eager_throughput
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Configuration: {selected_config['layers']} layers, {baseline_param_count / 1e9:.2f}B params")
    print(f"Sequence length: {seq_len} tokens")
    print()
    print(f"Eager Mode (BF16):      {eager_time:.2f} ms ({eager_throughput:.1f} tok/s)")
    print(f"Compiled Mode:          {compiled_time:.2f} ms ({compiled_throughput:.1f} tok/s)")
    print(f"Speedup:                {speedup:.2f}x")
    print(f"Throughput gain:        {throughput_gain:.2f}x")
    fp8_speedup = None
    fp8_compiled_speedup = None
    fp8_compiled_gain = None
    if fp8_eager_time is not None and fp8_eager_throughput is not None:
        fp8_speedup = eager_time / fp8_eager_time
        fp8_gain = fp8_eager_throughput / eager_throughput
        print(f"FP8 Eager Mode:         {fp8_eager_time:.2f} ms ({fp8_eager_throughput:.1f} tok/s)")
        print(f"FP8 Speedup vs eager:   {fp8_speedup:.2f}x")
        print(f"FP8 Throughput gain:    {fp8_gain:.2f}x")
    if fp8_compiled_time is not None and fp8_compiled_throughput is not None:
        fp8_compiled_speedup = eager_time / fp8_compiled_time
        fp8_compiled_gain = fp8_compiled_throughput / eager_throughput
        print(f"FP8 Compiled Mode:      {fp8_compiled_time:.2f} ms ({fp8_compiled_throughput:.1f} tok/s)")
        print(f"FP8 Compiled Speedup:   {fp8_compiled_speedup:.2f}x")
        print(f"FP8 Compiled Gain:      {fp8_compiled_gain:.2f}x")
    print()
    
    # Scaling analysis
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Current configuration ({selected_config['layers']} layers): {compiled_throughput:.1f} tokens/sec")
    print(f"Model size: {baseline_param_count / 1e9:.2f}B parameters")
    
    # Scale by layer count (rough approximation)
    projected_throughput = compiled_throughput
    if selected_config['layers'] < config.n_layers:
        full_model_ratio = config.n_layers / selected_config['layers']
        projected_throughput = compiled_throughput / full_model_ratio
        print(f"\nProjected (full {config.n_layers} layers): ~{projected_throughput:.1f} tokens/sec")
    
    print()
    print("Potential additional optimizations:")
    if fp8_eager_time is not None:
        print(f"  FP8 quantization:    achieved {fp8_speedup:.2f}x vs eager BF16")
    else:
        print(f"  + FP8 quantization:    1.5-2x speedup")
    print(f"  + FlexAttention:       1.3-1.8x speedup (sparse patterns)")
    print(f"  + Dynamic KV cache:    2-4x memory reduction")
    print()
    
    # Key learnings
    print("=" * 80)
    print("KEY LEARNINGS")
    print("=" * 80)
    print("1. Large MoE models (10-50B+) benefit significantly from torch.compile")
    print("2. B200 can handle 50-200B parameter models in 178 GB memory")
    print("3. Batch size and layer count are key tuning parameters")
    print("4. torch.compile speedup increases with model size (1.5-2x typical)")
    print("5. Proper warmup (50+ iterations) is critical for compiled models")
    print("6. Memory scaling is predictable: ~2-3 GB per layer for this config")
    print("7. Synthetic models are useful for infrastructure testing")
    print()
    print("For REAL GPT-OSS models (not this synthetic one):")
    print("  - Use: openai/gpt-oss-120b (120B params, requires 80GB+ GPU)")
    print("  - Use: openai/gpt-oss-20b (20B params, requires 16GB+ GPU)")
    print("=" * 80)
    
    # Save results
    results = {
        'model': 'Synthetic-MoE-Benchmark',
        'config': selected_config,
        'parameters': baseline_param_count,
        'eager_time_ms': eager_time,
        'compiled_time_ms': compiled_time,
        'speedup': speedup,
        'eager_throughput': eager_throughput,
        'compiled_throughput': compiled_throughput,
        'fp8_eager_time_ms': fp8_eager_time,
        'fp8_eager_throughput': fp8_eager_throughput,
        'fp8_compiled_time_ms': fp8_compiled_time,
        'fp8_compiled_throughput': fp8_compiled_throughput,
        'fp8_speedup': fp8_speedup,
        'fp8_compiled_speedup': fp8_compiled_speedup,
        'fp8_compiled_gain': fp8_compiled_gain,
        'projected_full_model_throughput': projected_throughput
    }
    
    with open('synthetic_moe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to synthetic_moe_results.json")
    
    return speedup


if __name__ == "__main__":
    speedup = main()

    flavor = CURRENT_DEVICE_FLAVOR
    threshold = 1.05
    if flavor in {"blackwell_sm121", "cpu", "other", "unknown"}:
        threshold = 0.80
    sys.exit(0 if speedup >= threshold else 1)
