# 📚 Examples

This directory contains example scripts demonstrating the usage of the AI Performance Engineering tools.

## Contents

| File | Description |
|------|-------------|
| `optimize_examples.py` | Evidence-first campaign initialization example |
| `profiling_examples.py` | GPU profiling suite examples |
| `zymtrace_gpu_smoke.py` | Timed CUDA workload for validating Zymtrace GPU capture |
| `ch19_dynamic_precision_zymtrace_probe.py` | Timed Chapter 19 decode workload for Zymtrace GPU capture |
| `mcp_client_example.py` | MCP client lifecycle and tool-call examples |
| `optimize_config.yaml` | Sample frozen workload contract |

## Running Examples

### Optimization campaign example

```bash
python examples/optimize_examples.py \
  --workspace /tmp/latency-campaign \
  --objective "Reduce representative latency" \
  --metric latency_ms \
  --initial-control-commit "$(git rev-parse HEAD)" \
  --primary-case representative \
  --frozen-case boundary \
  --workload-spec examples/optimize_config.yaml \
  --environment-spec /path/to/environment.json
```

This creates a fail-closed experiment template. Collect measurements with the trusted benchmark harness before recording an experiment.

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

## Campaign workload contract

Copy `optimize_config.yaml` and replace every placeholder before initializing a campaign:

```bash
cp examples/optimize_config.yaml /tmp/workload.yaml
```

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ (for torch.compile and torch.profiler)
- CUDA-capable GPU

## Quick Start

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
