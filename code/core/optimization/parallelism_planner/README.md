# Parallelism Strategy Advisor

A comprehensive tool for analyzing model architectures and recommending optimal distributed training/inference parallelism strategies based on actual hardware topology.

## Features

### Core Features

- **Real-time Hardware Topology Detection**
  - NVLink mesh and NVSwitch detection
  - P2P accessibility matrix
  - NUMA node mapping
  - Interconnect bandwidth estimation
  - Grace-Blackwell specific features (NVLink-C2C)

- **Multi-Node Cluster Support**
  - Cross-node parallelism strategies
  - InfiniBand/RoCE/Ethernet network detection
  - GPUDirect RDMA support
  - Cluster presets (DGX H100, GB200 NVL72)

- **Model Architecture Analysis**
  - HuggingFace model support (auto-fetch config)
  - Built-in presets for popular models (Llama, Mixtral, DeepSeek, etc.)
  - Custom architecture specifications
  - Memory estimation with precision options

- **Smart Parallelism Recommendations**
  - Data Parallel (DP)
  - Tensor Parallel (TP)
  - Pipeline Parallel (PP)
  - Context Parallel (CP) - for long sequences
  - Expert Parallel (EP) - for MoE models

### Advanced Features

- **ZeRO/FSDP/HSDP Sharding Recommendations**
  - ZeRO Stage 1/2/3 analysis
  - PyTorch FSDP configurations
  - Hybrid Sharded Data Parallel (HSDP)
  - Memory/communication tradeoff analysis

- **Profiling-Based Calibration**
  - Load historical benchmark data
  - Calibrate throughput/memory estimates
  - Architecture-specific adjustments

- **Cost/Throughput Pareto Analysis**
  - Multi-objective optimization
  - Pareto-optimal configuration identification
  - GPU pricing models (A100, H100, B200)
  - Cost-per-million-tokens analysis

- **Framework Launch Commands**
  - PyTorch torchrun (single & multi-node)
  - DeepSpeed with config generation
  - HuggingFace Accelerate
  - NVIDIA Megatron-LM
  - SLURM job script generation

### NEW: Validation & Testing

- **Configuration Validation**
  - Check parallelism strategy for correctness
  - Hardware compatibility verification
  - Memory fit validation
  - Framework config validation (DeepSpeed, FSDP)

- **Dry-Run Testing**
  - Quick sanity check before long training runs
  - GPU memory allocation verification
  - Forward/backward pass testing
  - Catch OOM errors early

### NEW: Advanced Optimizations

- **Compound Techniques**
  - Mixed precision strategies (FP8, BF16, FP16, TF32)
  - Gradient checkpointing (full, selective, block-wise)
  - Communication-computation overlap
  - Memory-efficient optimizers (8-bit Adam, Adafactor, Lion, CAME)
  - Pipeline scheduling (1F1B, Interleaved, Zero-bubble)
  - Kernel fusion recommendations (Flash Attention, fused LayerNorm, etc.)

- **Hardware-Aware Optimizations**
  - Automatic FP8 recommendations for Hopper/Blackwell
  - NVLink-aware communication strategies
  - NCCL tuning recommendations

### NEW: Workload-Specific Performance Profiles

- **Pre-training** - Maximum throughput, large batch optimization
- **Fine-tuning** - Quality-focused with LoRA/PEFT support
- **RLHF** - Memory optimization for actor/critic/reference models
- **Inference** - Batch, streaming, and real-time modes
- **Long Context** - Ring attention, context parallelism
- **MoE Training** - Expert parallelism optimization
- **Cost-Optimized** - Budget-friendly configurations

## Quick Start

### CLI Usage (Typer via `cli.aisp`)

```bash
# Get parallelism recommendations
python -m cli.aisp distributed plan --model-size 70 --nodes 1 --gpus 8

# Cluster topology discovery
python -m cli.aisp distributed topology

# NCCL tuning guidance
python -m cli.aisp distributed nccl --nodes 2 --gpus 8 --diagnose

# ZeRO/FSDP sizing
python -m cli.aisp distributed zero --model-size 70 --gpus 8

# Auto-generate launch commands
python -m cli.aisp distributed launch-plan --model-params 70 --nodes 2 --gpus 4 --tp 2 --pp 2 --dp 1 --script train.py
```

The current Typer CLI exposes the distributed planning surfaces above. The more
specialized validation, advanced-optimization, and workload-profile flows are
available through the Python API in this package.

### Python API

```python
from core.optimization.parallelism_planner import ParallelismAdvisor

advisor = ParallelismAdvisor()

# Quick recommendation
result = advisor.recommend("meta-llama/Llama-3.1-70B")
print(result.summary())
# Output: Llama-3.1-70B: TP=4×PP=1×DP=2 (Score: 85/100)

# Get the best strategy
best = result.best_strategy
print(f"Use TP={best.tp}, PP={best.pp}, DP={best.dp}")

# Generate full report
report = advisor.generate_report("mixtral-8x7b", is_training=True)
print(report)
```

### Validation API

```python
from core.optimization.parallelism_planner import validate_full_configuration

strategy = {"data_parallel": 8, "tensor_parallel": 1, "pipeline_parallel": 1}
hardware = {"num_gpus": 8, "gpu_memory_gb": 80, "has_nvlink": True}
model = {"num_layers": 80, "hidden_size": 8192}

result = validate_full_configuration(strategy, hardware, model)
print(f"Valid: {result['valid']}, Passed: {result['summary']['passed']}")
```

### Advanced Optimizations API

```python
from core.optimization.parallelism_planner import get_advanced_optimization_report

model_config = {
    "parameters_billions": 70,
    "num_layers": 80,
    "hidden_size": 8192,
    "max_sequence_length": 4096,
    "batch_size": 1,
}

hardware_config = {
    "gpu_arch": "hopper",
    "gpu_memory_gb": 80,
    "num_gpus": 8,
    "has_nvlink": True,
}

report = get_advanced_optimization_report(model_config, hardware_config, "throughput")
print(f"Memory Savings: {report['compound_optimization']['total_memory_savings_gb']:.1f} GB")
print(f"Throughput Boost: ~{report['compound_optimization']['total_throughput_boost_pct']:.0f}%")
```

### Performance Profiles API

```python
from core.optimization.parallelism_planner import get_performance_profile, list_available_profiles

# List profiles
profiles = list_available_profiles()
for p in profiles:
    print(f"{p['name']}: {p['description']}")

# Get specific profile
hardware_config = {
    "gpu_arch": "hopper",
    "gpu_memory_gb": 80,
    "num_gpus": 8,
    "has_nvlink": True,
}

profile = get_performance_profile(
    model_params_b=70,
    workload="pretraining",
    hardware_config=hardware_config,
    seq_length=4096,
)
print(f"Parallelism: TP={profile['parallelism']['tensor_parallel']}")
print(f"Optimizer: {profile['memory']['optimizer']}")
```

### Sharding Recommendations

```python
from core.optimization.parallelism_planner import ModelAnalyzer, ShardingOptimizer

analyzer = ModelAnalyzer()
model = analyzer.analyze("llama-3.1-70b")

sharding = ShardingOptimizer()
recommendations = sharding.recommend(
    model=model,
    dp_size=4,
    gpu_memory_gb=80,  # H100 80GB
    batch_size=4,
    seq_length=4096,
)

print(sharding.format_recommendations(recommendations))
```

### Launch Command Generation

```python
from core.optimization.parallelism_planner import LaunchCommandGenerator, LaunchConfig, ShardingStrategy

config = LaunchConfig(
    num_nodes=2,
    gpus_per_node=4,
    tp_size=2,
    pp_size=2,
    dp_size=2,
    sharding=ShardingStrategy.ZERO_3,
    micro_batch_size=1,
    gradient_accumulation_steps=8,
    master_addr="node0.cluster",
)

gen = LaunchCommandGenerator()
print(gen.format_launch_guide(config, "ch13/train.py"))

# Or get raw commands/configs
commands = gen.generate_all(config, "ch13/train.py")
deepspeed_config = commands["deepspeed"]["config"]
```

## CLI Commands Summary

| Command | Description |
|---------|-------------|
| `recommend` | Get parallelism recommendations (TP/PP/DP/CP/EP) |
| `sharding` | Get ZeRO/FSDP/HSDP sharding recommendations |
| `launch` | Generate framework launch commands |
| `pareto` | Cost/throughput Pareto analysis |
| `calibrate` | Calibrate from benchmark data |
| `presets` | List available model presets |
| `topology` | Detect or display hardware topology |
| `analyze` | Analyze model architecture |
| `estimate` | Estimate training time and cost |
| `compare` | Compare parallelism for multiple models |
| `slurm` | Generate SLURM job script |
| `validate` | Validate parallelism configuration |
| `dry-run` | Quick dry-run test before training |
| `optimize` | Get advanced optimization recommendations |
| `profile` | Get workload-specific performance profiles |

## Available Model Presets

| Preset | Parameters | Type | Notes |
|--------|-----------|------|-------|
| `llama-7b` | 7B | Dense | LLaMA 1/2 |
| `llama-13b` | 13B | Dense | LLaMA 1/2 |
| `llama-70b` | 70B | Dense | GQA attention |
| `llama-3.1-8b` | 8B | Dense | 128K context |
| `llama-3.1-70b` | 70B | Dense | 128K context |
| `llama-3.1-405b` | 405B | Dense | 128K context |
| `mixtral-8x7b` | 47B | MoE | 8 experts, top-2 |
| `mixtral-8x22b` | 141B | MoE | 8 experts, top-2 |
| `deepseek-v2` | 120B | MoE | 160 experts, top-6 |
| `deepseek-v3` | 671B | MoE | 256 experts, top-8 |
| `deepseek-r1` | 701B | MoE | 256 experts, top-8 |
| `qwen-72b` | 72B | Dense | |
| `gpt-4-estimated` | ~1.8T | MoE | Estimated architecture |

## Architecture

```
core/optimization/parallelism_planner/
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point
├── cli.py                   # Comprehensive CLI
├── advisor.py               # Main ParallelismAdvisor class
├── topology_detector.py     # Hardware topology detection
├── model_analyzer.py        # Model architecture analysis
├── strategy_optimizer.py    # Strategy recommendation engine
├── cluster_config.py        # Multi-node cluster support
├── sharding_strategies.py   # ZeRO/FSDP/HSDP analysis
├── calibration.py           # Profiling-based calibration
├── pareto_analysis.py       # Cost/throughput Pareto frontier
├── launch_commands.py       # Framework launch command generation
├── validation.py            # Configuration validation & dry-run
├── advanced_optimizations.py # Compound techniques & optimizations
├── performance_profiles.py  # Workload-specific profiles
├── extras.py                # Training estimation, SLURM, comparison
└── README.md               # This file
```

## Parallelism Strategy Guide

### Tensor Parallel (TP)
- **Best for**: Within NVSwitch/NVLink domains (high bandwidth)
- **Bandwidth**: Requires >200 GB/s interconnect
- **Splits**: Attention heads, FFN columns
- **Optimal sizes**: 2, 4, 8 (power of 2)

### Pipeline Parallel (PP)
- **Best for**: Spanning slower interconnects
- **Bandwidth**: Works with ~50 GB/s
- **Splits**: Model layers across stages
- **Note**: Introduces pipeline bubble overhead

### Data Parallel (DP)
- **Best for**: Scaling throughput
- **Bandwidth**: Flexible, works everywhere
- **Splits**: Batch across replicas
- **Note**: Communication only during gradient sync

### Context Parallel (CP)
- **Best for**: Long sequences (>32K tokens)
- **Bandwidth**: Needs ring communication bandwidth
- **Splits**: Sequence dimension
- **Note**: Uses ring attention pattern

### Expert Parallel (EP)
- **Best for**: MoE models
- **Bandwidth**: Requires all-to-all communication
- **Splits**: Experts across GPUs
- **Note**: Should divide total experts evenly

### Sharding Strategies (ZeRO/FSDP)

| Strategy | Memory Savings | Communication | Best For |
|----------|---------------|---------------|----------|
| ZeRO-1 | ~33% | Low | Large batches |
| ZeRO-2 | ~50% | Medium | Moderate memory |
| ZeRO-3/FSDP | ~N× (N=DP size) | High | Huge models |
| HSDP | ~8× within node | Medium | Multi-node |

## Advanced Optimization Techniques

### Precision Optimization
| Mode | Memory Savings | Throughput | Accuracy Impact |
|------|---------------|------------|-----------------|
| FP8 (Hopper+) | 50%+ | 80%+ boost | Minimal |
| BF16 | 50% | 40% boost | None |
| FP16 | 50% | 35% boost | Minimal (with loss scaling) |

### Gradient Checkpointing
| Strategy | Memory Savings | Compute Overhead |
|----------|---------------|------------------|
| Full | ~95% | ~33% |
| Block-wise | ~50-80% | ~15-25% |
| Selective | ~30-50% | ~10-15% |

### Memory-Efficient Optimizers
| Optimizer | State Memory | Convergence | Notes |
|-----------|-------------|-------------|-------|
| AdamW | 8 bytes/param | Excellent | Baseline |
| 8-bit AdamW | 2 bytes/param | Very Good | 75% reduction |
| Adafactor | 0.5 bytes/param | Good | 94% reduction |
| Lion | 4 bytes/param | Good | Needs LR tuning |

## Example Output

```
======================================================================
ADVANCED OPTIMIZATION REPORT
======================================================================

Model: llama-3.1-70b (69.5B params)
Optimization Goal: THROUGHPUT

──────────────────────────────────────────────────────────────────────
COMPOUND OPTIMIZATION: Optimal Throughput Configuration
──────────────────────────────────────────────────────────────────────
Complexity: MEDIUM
Memory Savings: 590.8 GB
Throughput Boost: ~95%

Techniques Applied:
  • Mixed Precision (mixed_fp8)
  • Memory-Efficient Optimizer (adafactor)
  • Fused Kernels + Flash Attention

Implementation Steps:
  1. Enable mixed_fp8 precision
  2. Use adafactor: saves 521.3GB optimizer states
  3. Enable flash_attn_3: memory-efficient attention
======================================================================
```

## Integration

This tool integrates with:
- `labs/moe_parallelism/plan.py` - Compatible data structures
- `core/harness/hardware_capabilities.py` - Hardware detection
- `ch04/training_multigpu_pipeline.py` - Training examples
