# 📊 GPU Profiling Suite

**Comprehensive GPU profiling toolkit** with flame graphs, memory analysis, CPU/GPU timelines, HTA integration, and torch.compile diagnostics.

## Features

- **UnifiedProfiler**: Complete profiling with torch.profiler integration
- **MemoryProfiler**: Track GPU memory usage over time
- **FlameGraphGenerator**: Interactive flame graph visualizations
- **TimelineGenerator**: CPU/GPU parallel timeline views
- **HTAAnalyzer**: Meta's Holistic Trace Analysis integration
- **TorchCompileAnalyzer**: Analyze torch.compile behavior and graph breaks

## Quick Start

```bash
# Generate flame graph from trace
python -m core.profiling flame trace.json -o flame.html

# Generate CPU/GPU timeline
python -m core.profiling timeline trace.json -o timeline.html

# Run HTA analysis
python -m core.profiling hta trace.json -o report.html
```

## CLI Reference

```
usage: python -m core.profiling [-h] {profile,memory,flame,timeline,hta,compile} ...

Commands:
  profile    Profile GPU code and generate trace
  memory     Profile memory usage
  flame      Generate flame graph from trace
  timeline   Generate CPU/GPU timeline visualization
  hta        Run Holistic Trace Analysis
  compile    Analyze torch.compile behavior
```

### Generate Flame Graph

```bash
python -m core.profiling flame trace.json --output flame.html
python -m core.profiling flame trace.json --output flame.json --json
```

### Generate Timeline

```bash
python -m core.profiling timeline trace.json --output timeline.html
```

### HTA Analysis

```bash
python -m core.profiling hta trace.json --output hta_report.html
python -m core.profiling hta trace.json --output hta_report.json --json
```

## Python API

### UnifiedProfiler

Complete profiling with automatic trace export:

```python
from core.profiling import UnifiedProfiler

profiler = UnifiedProfiler(
    output_dir="./profiles",
    enable_trace=True,
    enable_memory=True,
    enable_flame_graph=True,
)

# Profile a code block
with profiler.profile("my_benchmark") as session:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Access results
print(f"Total time: {session.total_time_ms:.2f}ms")
print(f"GPU time: {session.cuda_time_ms:.2f}ms")
print(f"Peak memory: {session.peak_memory_mb:.1f}MB")
print(f"Bottlenecks: {session.bottlenecks}")
print(f"Recommendations: {session.recommendations}")

# Top kernels
for kernel in session.kernels[:10]:
    print(f"  {kernel.name}: {kernel.cuda_time_us:.0f}μs")
```

### Profile a Function

```python
session = profiler.profile_function(
    model.forward,
    input_tensor,
    name="forward_pass",
    warmup=5,
    iterations=20,
)
```

### Compare Sessions

```python
baseline_session = profiler.profile_function(baseline_model, input)
optimized_session = profiler.profile_function(optimized_model, input)

comparison = profiler.compare_sessions(baseline_session, optimized_session)
print(f"Speedup: {comparison['speedup']:.2f}x")
print(f"Memory saved: {comparison['memory_diff']['peak_mb']:.1f}MB")
```

### MemoryProfiler

Track memory usage over time:

```python
from core.profiling import MemoryProfiler

mem_profiler = MemoryProfiler(
    sample_interval_ms=1.0,
    record_history=True,
)

with mem_profiler.track("training_step"):
    output = model(input)
    loss.backward()
    optimizer.step()

# Get timeline
timeline = mem_profiler.get_timeline()
for point in timeline:
    print(f"t={point['time_ms']:.1f}ms: {point['allocated_mb']:.1f}MB")

# Peak analysis
peak = mem_profiler.get_peak_analysis()
print(f"Peak: {peak['peak_allocated_mb']:.1f}MB at t={peak['peak_time_ms']:.1f}ms")

# Export
mem_profiler.export("memory_profile.json")
```

### FlameGraphGenerator

Create interactive flame graphs:

```python
from core.profiling import FlameGraphGenerator

generator = FlameGraphGenerator(
    min_duration_us=10.0,
    max_depth=50,
    group_small_kernels=True,
)

# From torch.profiler
with torch.profiler.profile() as prof:
    model(input)

flame_data = generator.from_profiler(prof)
generator.export(flame_data, "flame.html", format="html")

# From Chrome trace
flame_data = generator.from_chrome_trace("trace.json")
generator.export(flame_data, "flame.json", format="json")
```

### TimelineGenerator

Visualize CPU/GPU parallelism:

```python
from core.profiling import TimelineGenerator

generator = TimelineGenerator(
    min_duration_us=1.0,
    include_python_events=True,
)

# From profiler
with torch.profiler.profile() as prof:
    model(input)

timeline = generator.from_profiler(prof)

# Statistics
print(f"Total: {timeline.total_time_us/1000:.2f}ms")
print(f"CPU active: {timeline.cpu_active_time_us/1000:.2f}ms")
print(f"GPU active: {timeline.gpu_active_time_us/1000:.2f}ms")
print(f"Overlap: {timeline.overlap_time_us/1000:.2f}ms")

# Export
generator.export_chrome_trace(timeline, "timeline_trace.json")
generator.generate_html_viewer(timeline, "timeline.html")
```

### HTAAnalyzer

Run Holistic Trace Analysis:

```python
from core.profiling import HTAAnalyzer

analyzer = HTAAnalyzer(output_dir="artifacts/runs/analysis/hta")

# Analyze existing trace
report = analyzer.analyze_trace("trace.json")

print(f"GPU idle: {report.gpu_idle_time_pct:.1f}%")
print(f"Compute: {report.compute_time_pct:.1f}%")
print(f"Communication: {report.communication_time_pct:.1f}%")

# Top kernels
for kernel in report.top_kernels[:5]:
    print(f"  {kernel['name']}: {kernel['time_us']:.0f}μs")

# Recommendations
for rec in report.recommendations:
    print(f"💡 {rec}")

# Export report
analyzer.export_report(report, "hta_report.html", format="html")
```

### TorchCompileAnalyzer

Analyze torch.compile behavior:

```python
from core.profiling.torch_compile import TorchCompileAnalyzer

analyzer = TorchCompileAnalyzer(
    backend="inductor",
    mode="default",
)

# Analyze model
report = analyzer.analyze(
    model,
    sample_input,
    warmup=3,
    iterations=10,
)

print(f"Speedup: {report.speedup:.2f}x")
print(f"Compile time: {report.compile_time_ms/1000:.1f}s")
print(f"Graph breaks: {report.total_graph_breaks}")
print(f"Fusion ratio: {report.fusion_ratio:.1f}x")

# Graph breaks
for break_info in report.graph_breaks:
    print(f"⚠️ {break_info.reason}")
    print(f"   💡 {break_info.suggestion}")

# Compare modes
results = analyzer.compare_modes(
    model,
    sample_input,
    modes=["default", "reduce-overhead", "max-autotune"]
)

for mode, report in results.items():
    print(f"{mode}: {report.speedup:.2f}x speedup")

# Export
analyzer.export_report(report, "compile_report.html", format="html")
```

## Output Formats

### ProfileSession

```python
@dataclass
class ProfileSession:
    total_time_ms: float       # Total execution time
    cuda_time_ms: float        # GPU execution time
    cpu_time_ms: float         # CPU execution time
    peak_memory_mb: float      # Peak GPU memory
    allocated_memory_mb: float # Current allocated memory
    reserved_memory_mb: float  # Reserved memory pool
    kernels: List[KernelInfo]  # Kernel timing breakdown
    bottlenecks: List[str]     # Identified bottlenecks
    recommendations: List[str] # Optimization suggestions
    trace_path: Path           # Chrome trace file
    flame_graph_data: Dict     # Flame graph JSON
```

### HTAReport

```python
@dataclass
class HTAReport:
    gpu_idle_time_pct: float      # GPU idle percentage
    compute_time_pct: float       # Compute percentage
    communication_time_pct: float # Communication overhead
    top_kernels: List[Dict]       # Top kernels by time
    recommendations: List[str]    # HTA recommendations
    bottlenecks: List[str]        # Identified bottlenecks
```

### CompileReport

```python
@dataclass
class CompileReport:
    speedup: float             # Compiled vs eager speedup
    compile_time_ms: float     # Compilation time
    graph_breaks: List[CompileBreak]  # Graph break info
    fusion_ratio: float        # Op fusion ratio
    ops_before_fusion: int     # Ops before fusion
    ops_after_fusion: int      # Ops after fusion
    recommendations: List[str] # Optimization suggestions
```

## Examples

See [`examples/profiling_examples.py`](../../examples/profiling_examples.py) for more usage examples.

## Architecture

```
core/profiling/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── profiler.py          # UnifiedProfiler, ProfileSession
├── memory.py            # MemoryProfiler, MemorySnapshot
├── flame_graph.py       # FlameGraphGenerator
├── timeline.py          # TimelineGenerator, TimelineData
├── hta_integration.py   # HTAAnalyzer, HTAReport
├── torch_compile.py     # TorchCompileAnalyzer, CompileReport
└── README.md            # This file
```

## Requirements

- PyTorch >= 2.0 (for torch.profiler and torch.compile)
- Optional: `hta` package for full HTA support (`pip install HolisticTraceAnalysis`)
