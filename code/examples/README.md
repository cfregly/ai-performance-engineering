# 📚 Examples

This directory contains example scripts demonstrating the usage of the AI Performance Engineering tools.

## Contents

| File | Description |
|------|-------------|
| `optimize_examples.py` | Auto-optimizer usage examples |
| `profiling_examples.py` | GPU profiling suite examples |
| `zymtrace_gpu_smoke.py` | Timed CUDA workload for validating Zymtrace GPU capture |
| `ch19_dynamic_precision_zymtrace_probe.py` | Timed Chapter 19 decode workload for Zymtrace GPU capture |
| `mcp_client_example.py` | MCP client lifecycle and tool-call examples |
| `optimize_config.yaml` | Sample configuration file |

## Running Examples

### Auto-Optimizer Examples

```bash
# Show all examples (without running actual optimization)
python examples/optimize_examples.py

# Run actual optimization (requires LLM API key)
python examples/optimize_examples.py --run-optimization
```

### Profiling Examples

```bash
# List available examples
python examples/profiling_examples.py

# Run a specific example
python examples/profiling_examples.py --example 1  # UnifiedProfiler
python examples/profiling_examples.py --example 4  # Flame Graph
python examples/profiling_examples.py --example 6  # torch.compile

# Run all examples
python examples/profiling_examples.py --all
```

### Zymtrace GPU Smoke

```bash
core/scripts/profiling/profile.sh examples/zymtrace_gpu_smoke.py --tool zymtrace -- --seconds 30
core/scripts/profiling/profile.sh examples/ch19_dynamic_precision_zymtrace_probe.py --tool zymtrace -- --seconds 30 --mode dynamic
```

### MCP Client Example

```bash
# Run the end-to-end MCP client examples
python examples/mcp_client_example.py
```

## Configuration

Copy `optimize_config.yaml` to your project root to customize optimizer behavior:

```bash
cp examples/optimize_config.yaml ./optimize_config.yaml
# Edit as needed
```

The optimizer will automatically find and use `optimize_config.yaml` if present.

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ (for torch.compile and torch.profiler)
- CUDA-capable GPU
- API key for LLM provider (ANTHROPIC_API_KEY or OPENAI_API_KEY)

## Quick Start

```python
# Optimize a file
from core.optimization.auto import AutoOptimizer

optimizer = AutoOptimizer()
result = optimizer.optimize_file("examples/optimize_examples.py", output_path="/tmp/optimize_examples_optimized.py")
print(f"Speedup: {result.speedup:.2f}x")
```

```python
# Profile GPU code
from core.profiling import UnifiedProfiler

profiler = UnifiedProfiler()
with profiler.profile("my_model") as session:
    output = model(input)
    
print(f"Time: {session.total_time_ms:.2f}ms")
print(f"Memory: {session.peak_memory_mb:.1f}MB")
```

## Output Files

Examples generate output files in `/tmp/`:

- `/tmp/flame.html` - Interactive flame graph
- `/tmp/timeline.html` - CPU/GPU timeline
- `/tmp/memory_profile.json` - Memory usage data
- `/tmp/compile_report.html` - torch.compile analysis
