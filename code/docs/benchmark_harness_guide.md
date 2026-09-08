# The Benchmark Harness: Power and Architecture

## Overview

The `BenchmarkHarness` is a production-grade benchmarking framework that provides **automatic discovery, execution, profiling, and comparison** of 264+ baseline/optimized example pairs across 20 chapters. It's designed to handle real-world GPU workloads reliably, with comprehensive timeout protection, profiling integration, and reproducibility guarantees.

---

## Core Architecture

### Simple Protocol, Powerful Execution

The harness uses a **minimal protocol** that benchmarks must implement:

```python
class Benchmark(Protocol):
    def setup(self) -> None:          # Initialize models, data, etc.
    def benchmark_fn(self) -> None:    # The code to measure
    def teardown(self) -> None:        # Cleanup
    def get_config(self) -> Optional[BenchmarkConfig]:  # Optional overrides
    def validate_result(self) -> Optional[str]:  # Optional validation
```

**The power**: Benchmarks focus on their logic. The harness handles:
- ✅ Accurate timing (CUDA Events, Triton do_bench, PyTorch Timer)
- ✅ Statistical analysis (mean, median, std, percentiles)
- ✅ Memory tracking
- ✅ Profiling integration (nsys, ncu, PyTorch profiler)
- ✅ Timeout protection
- ✅ Error handling and recovery
- ✅ Reproducibility (seeds, deterministic mode)
- ✅ GPU state management

---

## Key Features

### 1. **Automatic Benchmark Discovery**

The harness automatically discovers benchmarks using a simple naming convention:

```
ch*/baseline_*.py  →  ch*/optimized_*.py
```

**Example**: `ch20/baseline_moe.py` pairs with `ch20/optimized_moe.py`

**Discovery logic**:
- Scans all chapter directories (`ch01` through `ch20`)
- Finds `baseline_*.py` files
- Matches with `optimized_{name}*.py` files
- Extracts example name (e.g., `ch20/baseline_moe.py` → `moe`)
- Returns tuples: `(baseline_path, [optimized_paths], example_name)`

**Result**: Run **264 benchmarks** across **20 chapters** with a single command:
```bash
python -m cli.aisp bench run
```

Use `--accept-regressions` when you intentionally want to refresh expectation files on improvements only, or `--update-expectations` to force-write observed metrics even if they are slower (full refresh). The harness will surface **SKIPPED** for known hardware gaps like DSMEM-only cluster kernels on GB10, NVFP4-only trainers on non-FP4 parts, or TF32 being disabled via the new helper APIs—those no longer count as failures.

---

### 2. **Multiple Execution Modes**

#### Subprocess Mode (Default - Production-Grade)

**Why subprocess?**
- ✅ **True isolation**: Each benchmark runs in a separate process
- ✅ **Reliable timeouts**: Can kill hung processes (CUDA kernels can't be interrupted from Python)
- ✅ **Clean GPU state**: No contamination between benchmarks
- ✅ **Crash protection**: One benchmark crash doesn't kill the entire suite

**How it works**:
1. Serializes benchmark class and config to JSON
2. Spawns isolated Python subprocess via `python -m core.harness.isolated_runner`
3. Subprocess imports module, instantiates benchmark, runs harness
4. Returns results via JSON (Pydantic models)
5. Parent process handles timeouts and cleanup

**Timeout handling**: Uses `subprocess.communicate(timeout=...)` with process group killing (`os.killpg`) for reliable termination.

#### Threading Mode (Fallback)

**When used**:
- Module file path cannot be determined
- Subprocess isolation unavailable
- Quick prototyping

**Limitations**:
- ⚠️ Cannot force-stop hung CUDA kernels
- ⚠️ Less isolation (shared memory space)
- ⚠️ Timeout enforcement less reliable

#### Torchrun launch (multi-GPU)
- Enable with `--launch-via torchrun` plus `--nproc-per-node`, optional `--nnodes/--rdzv-backend/--rdzv-endpoint`.
- Harness records `world_size`/`launch_via` in results and only parses rank0-style output (duplicate rank logs are deduped).
- Forward env vars via `--torchrun-env CUDA_VISIBLE_DEVICES=0,1` and per-target overrides with `--target-extra-arg target="--flag value"`.
- Multi-GPU targets (`multi_gpu_required=True`) return `SKIPPED` instead of hanging when launched with a single process.
- Example: `PYTHONPATH=. python -m cli.aisp bench run --targets labs/train_distributed:ddp_multigpu --launch-via torchrun --nproc-per-node 2 --target-extra-arg 'labs/train_distributed:ddp_multigpu=--compile'`.

---

### 3. **⚠️ CRITICAL: Warmup Requirements**

**Warmup iterations are MANDATORY** to ensure accurate benchmark measurements. Low warmup causes JIT/compile overhead to be INCLUDED in timing results, leading to incorrect speedup calculations.

#### Why Warmup Matters

| What Happens Without Warmup | Impact |
|----------------------------|--------|
| `torch.compile` JIT compilation | First 1-3 calls trigger compilation (seconds of overhead) |
| Triton kernel compilation | First call compiles the kernel |
| cuDNN autotuning | First call selects optimal algorithm |
| CUDA driver initialization | First CUDA call has extra overhead |
| Memory allocator warmup | First allocations are slower |

#### Minimum Requirements

| Feature Used | Minimum Warmup | Recommended |
|--------------|----------------|-------------|
| Basic CUDA | 5 | 5-10 |
| torch.compile | 10 | 10-15 |
| Triton kernels | 10 | 10-15 |
| CUDA Graphs | 10 | 10-15 |

#### Automatic Enforcement

The harness **automatically enforces minimum warmup**:

```python
# This triggers a warning and auto-corrects:
config = BenchmarkConfig(warmup=2)  # → warmup=5, warning issued

# This is correct:
config = BenchmarkConfig(warmup=10)  # → warmup=10, no warning
```

#### Validation Tools

```bash
# Check all benchmarks
make audit-warmup

# Stricter check (recommended thresholds)
make audit-warmup-strict

# Pre-commit hook runs automatically on commits
```

See `CONTRIBUTING.md` for detailed warmup guidelines.

---

### 4. **Comprehensive Timeout Protection**

The harness provides **per-stage timeouts** to prevent hangs:

| Stage | Default Timeout | Purpose |
|-------|----------------|---------|
| **Setup** | 30s | Model loading, CUDA extension compilation, torch.compile() |
| **Warmup** | 15s | GPU warmup, cuDNN autotuning |
| **Measurement** | 15s | Actual benchmark iterations |
| **Profiling** | 180s | nsys/ncu profiling (can be slow) |

**Timeout multiplier**: Scale all timeouts with `timeout_multiplier=2.0` for slower systems.

**Timeout behavior**:
- Creates structured timeout results (not just failures)
- Logs detailed diagnostics (which stage timed out, elapsed time, suggestions)
- Automatically cleans up GPU resources
- Continues with remaining benchmarks

**Example timeout result**:
```json
{
  "timeout_stage": "measurement",
  "timeout_duration_seconds": 45.2,
  "timeout_limit_seconds": 15,
  "errors": ["TIMEOUT: Benchmark measurement stage exceeded timeout..."],
  "watchdog": {
    "setup": {"status": "completed", "duration": 2.1},
    "warmup": {"status": "completed", "duration": 1.5},
    "measurement": {"status": "timeout", "duration": 45.2}
  }
}
```

---

### 4. **Integrated Profiling**

The harness seamlessly integrates **three profiling tools**:

#### Nsight Systems (nsys)
- **Purpose**: Timeline profiling (CPU/GPU activity, memory transfers, kernel launches)
- **Output**: `.nsys-rep` files viewable in `nsys-ui`
- **When enabled**: `enable_nsys=True` (default: True)
- **Timeout**: 120s per benchmark

#### Nsight Compute (ncu)
- **Purpose**: Kernel-level metrics (SM utilization, memory throughput, occupancy)
- **Output**: `.ncu-rep` files viewable in `ncu-ui`
- **When enabled**: `enable_ncu=True` (default: True)
- **Timeout**: 180s per benchmark

#### PyTorch Profiler
- **Purpose**: Python-level profiling (operator-level timing, memory usage)
- **Output**: Chrome trace JSON (viewable in `chrome://tracing`)
- **When enabled**: `enable_profiling=True` (default: True)

**Automatic NVTX**: When profiling is enabled, NVTX markers are automatically added for better trace visualization.

**Profiling orchestration**:
- Runs profiling **alongside** timing (not separate runs)
- Captures metrics automatically (SM throughput, memory bandwidth, etc.)
- Stores artifacts in structured output directory
- Extracts key metrics into `ProfilerMetrics` Pydantic models

**Result**: Every benchmark run produces:
- Timing statistics (mean, median, p99, etc.)
- Memory statistics (peak, allocated, reserved)
- Profiler artifacts (`.nsys-rep`, `.ncu-rep`, `.json`)
- Profiler metrics (SM utilization, bandwidth, occupancy)

---

### 5. **Reproducibility Guarantees**

The harness ensures **bitwise reproducibility** when configured:

#### Seed Management
```python
config = BenchmarkConfig(
    seed=42,                    # Set all random seeds
    deterministic=True          # Enable deterministic algorithms
)
```

**What gets seeded**:
- Python `random` module
- NumPy random state
- PyTorch random state
- CUDA random state (all GPUs)

**Deterministic mode**:
- `torch.use_deterministic_algorithms(True)` - Uses slower but reproducible algorithms
- `torch.backends` -> `cudnn.deterministic = True` - Disables cuDNN autotuning
- **Trade-off**: Falls back to slower kernels (often 5–20% hit) and ops without deterministic support raise at runtime

#### Run Manifest

Every benchmark run captures a **complete manifest**:

```python
harness.benchmark_with_manifest(benchmark, run_id="my_run")
# Returns BenchmarkRun with:
# - Manifest: Hardware, software, git state, environment
# - Result: Timing, memory, profiling data
# - Metadata: Run ID, timestamp, configuration
```

**Manifest captures**:
- Hardware: GPU model, compute capability, memory, clocks
- Software: PyTorch version, CUDA version, driver version, Triton version
- Git state: Commit hash, branch, dirty flag
- Environment: Python version, OS, environment variables
- Seeds: All random seed values used
- Configuration: All benchmark config parameters

**Use case**: Debug "why did performance change?" by comparing manifests across runs.

---

### 6. **Memory Tracking**

Automatic GPU memory tracking:

```python
config = BenchmarkConfig(enable_memory_tracking=True)
```

**Tracks**:
- **Peak memory**: Maximum memory allocated during benchmark
- **Allocated memory**: Current memory allocated
- **Reserved memory**: Memory reserved by PyTorch allocator

**Memory context manager**:
```python
with self._memory_tracking(config) as mem_result:
    # Run benchmark
    times_ms = self._benchmark_without_profiling(fn, config)
# mem_result now contains MemoryStats
```

**Result**: Every benchmark reports memory usage, enabling:
- Memory optimization analysis
- OOM debugging
- Memory efficiency comparisons (baseline vs optimized)

---

### 7. **Statistical Analysis**

The harness computes **comprehensive statistics**:

**Timing statistics**:
- Mean, median, standard deviation
- Min, max
- Percentiles: p25, p50 (median), p75, p90, p95, p99
- Custom percentiles via `config.percentiles`

**Inference timing** (for LLM workloads):
- **TTFT** (Time To First Token): Mean, p50, p90, p95, p99
- **TPOT** (Time Per Output Token): Mean, p50, p90, p95, p99
- Request and token counts

**Raw data preservation**: All raw timing measurements stored in `raw_times_ms` for custom analysis.

**Example result**:
```json
{
  "timing": {
    "mean_ms": 12.5,
    "median_ms": 12.3,
    "std_ms": 0.8,
    "p99_ms": 14.2,
    "iterations": 100,
    "raw_times_ms": [12.1, 12.3, 12.5, ...]
  }
}
```

---

### 8. **Multiple Benchmarking Modes**

The harness supports **three timing modes**:

#### CUSTOM Mode (Default)
- Uses **CUDA Events** for GPU timing (most accurate)
- Uses `time.perf_counter()` for CPU timing
- **Advantages**: Minimal overhead, accurate GPU timing
- **Best for**: Most GPU benchmarks

#### TRITON Mode
- Uses `triton.testing.do_bench()` internally
- **Advantages**: Optimized for Triton kernels
- **Best for**: Custom Triton kernel benchmarks

#### PYTORCH Mode
- Uses `torch.utils.benchmark.Timer`
- **Advantages**: Automatic iteration count based on `min_run_time_ms`
- **Best for**: PyTorch operator benchmarks

**Mode selection**:
```python
harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM)
```

---

### 9. **Automatic Comparison**

Built-in comparison function:

```python
from core.harness.benchmark_harness import compare_benchmarks

result = compare_benchmarks(
    baseline=baseline_benchmark,
    optimized=optimized_benchmark,
    name="MoE Optimization",
    regression_threshold_pct=5.0
)
```

**Returns**:
- Speedup ratio (baseline / optimized)
- Regression detection (if optimized is >5% slower)
- Full statistics for both benchmarks
- Complete `BenchmarkResult` objects for further analysis

Benchmarks whose verification output contains tensors with different numeric
scales can opt into exact-keyed tolerances:

```python
def get_output_tolerances(self) -> dict[str, tuple[float, float]]:
    return {
        "prediction": (0.4, 1.0),
        "parameter.weight": (1e-3, 7.5e-4),
        "parameter.bias": (1e-3, 5e-5),
    }
```

The map must cover every key returned by `get_verify_output()`. Baseline and
optimized maps must be identical, and every `rtol` and `atol` must be finite
and nonnegative. The existing `get_output_tolerance()` tuple remains required
as global compatibility metadata. When the per-output hook is present, numeric
comparison uses the keyed map and records it under
`verification.output_tolerances`.

**Use case**: Automatic performance regression detection in CI/CD.

---

### 10. **GPU State Management**

The harness ensures **clean GPU state** between benchmarks:

**Automatic cleanup**:
- Resets CUDA state between benchmarks (prevents cascading failures)
- Clears GPU cache (torch CUDA module `empty_cache()`)
- Resets peak memory stats
- Handles cold-start resets via `reset_gpu_state()` for cache/memory cleanup
- Orchestrators may invoke `core/scripts/reset_gpu.py` for deeper resets; override the script path with `AISP_GPU_RESET_SCRIPT`

**Cold start mode**:
```python
config = BenchmarkConfig(enable_cleanup=True)  # Force cleanup
# Or use cold_start flag in CLI
```

**Result**: Each benchmark starts with a clean GPU state, ensuring fair comparisons.

---

### 11. **Error Handling and Recovery**

**Comprehensive error handling**:

1. **Timeout errors**: Structured timeout results with diagnostics
2. **Execution errors**: Captured in `errors` list, benchmark continues
3. **Validation errors**: Custom validation via `validate_result()`
4. **Profiling errors**: Graceful fallback if profilers unavailable
5. **Import errors**: Fallback to threading mode if subprocess unavailable

**Error reporting**:
- Errors stored in `BenchmarkResult.errors` list
- Detailed error messages with context
- Stack traces for debugging
- Partial results preserved (e.g., memory stats even if timing fails)

**Recovery**: One benchmark failure doesn't stop the suite. All 264 benchmarks can run even if some fail.

---

### 12. **Configuration Flexibility**

**Single source of truth**: `BenchmarkDefaults` provides all defaults.

**Configuration hierarchy**:
1. **BenchmarkDefaults** (hardcoded defaults)
2. **BenchmarkConfig** (instance-level overrides)
3. **Benchmark.get_config()** (benchmark-specific overrides)
4. **CLI flags** (runtime overrides)

**Example**:
```python
# Global defaults (core/benchmark/defaults.py)
iterations = 100
warmup = 10

# Instance config
config = BenchmarkConfig(iterations=50)  # Override

# Benchmark-specific
class MyBenchmark(BaseBenchmark):
    def get_config(self):
        return BenchmarkConfig(iterations=25)  # Further override
```

**Result**: Flexible configuration without code changes.

---

## The Power: End-to-End Automation

### Running All 264 Benchmarks

**Single command**:
```bash
python -m cli.aisp bench run
```

**What happens**:
1. ✅ Discovers all `baseline_*.py` / `optimized_*.py` pairs
2. ✅ Loads each benchmark module dynamically
3. ✅ Runs baseline → optimized comparison
4. ✅ Collects timing, memory, profiling data
5. ✅ Computes speedups and statistics
6. ✅ Generates JSON + Markdown reports
7. ✅ Handles failures gracefully (continues with remaining benchmarks)
8. ✅ Produces comprehensive summary

**Output**:
- `benchmark_test_results` (JSON, machine-readable)
- `benchmark_test_results.md` (human-readable)
- Profiling artifacts in `artifacts/runs/<run_id>/profiles/`
- Per-chapter summaries with speedup statistics

---

### Example Output

```markdown
## Overall Summary
- **Chapters tested:** 20/20
- **Total benchmarks:** 264
- **Successful:** 258
- **Failed:** 6
- **Average speedup:** 3.2x
- **Best speedup:** 89x (ch20: end-to-end optimization)
- **Worst speedup:** 0.95x (ch14: torch.compile on memory-bound model)

## Per-Chapter Summary
| Chapter | Status | Benchmarks | Successful | Avg Speedup |
|---------|--------|------------|------------|-------------|
| ch01     | PASS   | 21         | 21         | 2.5x        |
| ch13    | PASS   | 23         | 23         | 4.1x        |
| ch19    | PASS   | 15         | 15         | 6.8x        |
...
```

---

## Advanced Features

### 1. **Inference Timing Support**

For LLM inference benchmarks, the harness captures **TTFT** and **TPOT**:

```python
def benchmark_fn(self):
    # Return dict with inference timing
    return {
        "ttft_times_ms": [50.2, 48.1, 52.3],  # One per request
        "tpot_times_ms": [12.1, 11.9, 12.5, ...]  # One per token
    }
```

**Result**: Automatic inference timing statistics (mean, p50, p90, p99 for both TTFT and TPOT).

### 2. **Hardware Limitation Detection**

The harness detects and skips benchmarks with known hardware limitations:

- Triton SM 12.1 support issues
- Device-side assert cascades
- Architecture-specific features (e.g., TMA on non-Blackwell GPUs)

**Result**: Benchmarks gracefully skip with clear messages instead of failing.

### 3. **Workload Scaling**

`BaseBenchmark` provides automatic workload scaling based on GPU memory:

```python
size = self._scale_workload_by_memory(base_size=4096)
# >=16GB: 4096 (100%)
# >=8GB:  2048 (50%)
# >=4GB:  1024 (25%)
# <4GB:   409 (10%)
```

**Result**: Benchmarks adapt to available hardware automatically.

### 4. **NVTX Range Management**

Automatic NVTX markers for profiling:

```python
with self._nvtx_range("my_operation"):
    # Code automatically marked in nsys traces
    result = model(input)
```

**Result**: Better trace visualization in Nsight Systems.

---

## Real-World Usage Examples

### Example 1: Run Single Chapter

```bash
python -m cli.aisp bench run --targets ch13
```

**Result**: Runs all 23 benchmarks in Chapter 13 (PyTorch Profiling), compares baseline vs optimized, generates report.

### Example 2: Reproducible Run

```bash
python -m cli.aisp bench run --reproducible
```

**Result**: All seeds set to 42, deterministic algorithms enabled; expect matching outputs at the cost of slower kernels and possible op errors if deterministic paths are missing.

### Example 3: Extended Timeouts for Slow Systems

```bash
python -m cli.aisp bench run --timeout-multiplier 2.0
```

**Result**: All timeouts doubled (30s → 60s setup, 15s → 30s measurement, etc.).

### Example 4: Cold Start Measurements

```bash
python -m cli.aisp bench run --cold-start
```

**Result**: GPU state reset between benchmarks, additional cleanup, cold start performance measurements.

### Example 5: Profiling Disabled (Faster Runs)

```bash
python -m cli.aisp bench run --no-profile
```

**Result**: Timing-only runs (no nsys/ncu/PyTorch profiler), faster execution, still collects timing and memory stats.

---

## Integration Points

### 1. **Discovery System**

`discover_benchmarks()` automatically finds benchmark pairs:
- Scans chapter directories
- Matches baseline/optimized files
- Returns structured tuples

### 2. **Comparison Template**

`core/utils/chapter_compare_template.py` provides:
- `load_benchmark()`: Dynamic module loading
- `compare_baseline_optimized()`: Standard comparison workflow
- Integration with harness

### 3. **Manifest System**

`core/benchmark/run_manifest.py` captures:
- Complete environment state
- Hardware/software versions
- Git state
- Configuration

### 4. **Artifact Management**

`core/benchmark/artifact_manager.py` organizes:
- Profiling outputs
- Logs
- Reports
- Timestamped run directories

---

## Why This Harness is Powerful

### 1. **Zero-Configuration Benchmarking**

Write a benchmark class, the harness handles everything else:
- ✅ Accurate timing
- ✅ Statistical analysis
- ✅ Profiling integration
- ✅ Error handling
- ✅ GPU state management

### 2. **Production-Grade Reliability**

- ✅ Subprocess isolation prevents cascading failures
- ✅ Comprehensive timeout protection
- ✅ Graceful error recovery
- ✅ Clean GPU state between benchmarks

### 3. **Comprehensive Data Collection**

Every benchmark produces:
- Timing statistics (mean, median, p99, etc.)
- Memory statistics (peak, allocated)
- Profiler artifacts (nsys, ncu, PyTorch traces)
- Profiler metrics (SM utilization, bandwidth, occupancy)
- Complete environment manifest

### 4. **Automatic Comparison**

- ✅ Discovers baseline/optimized pairs automatically
- ✅ Computes speedups automatically
- ✅ Detects regressions automatically
- ✅ Generates reports automatically

### 5. **Scalability**

- ✅ Runs 264 benchmarks with one command
- ✅ Handles failures gracefully (continues with remaining)
- ✅ Suite-level timeouts (4 hours default)
- ✅ Parallel execution support (via subprocess)

### 6. **Reproducibility**

- ✅ Seed management (all RNGs)
- ✅ Deterministic mode support
- ✅ Complete environment capture
- ✅ Git state tracking

### 7. **Developer Experience**

- ✅ Simple protocol (just implement 3 methods)
- ✅ Rich error messages with diagnostics
- ✅ Structured output (JSON + Markdown)
- ✅ Integration with existing tools (nsys-ui, ncu-ui, Chrome tracing)

## Profiling Presets (Minimal vs Deep-Dive)

- **Minimal (default)** keeps profiler overhead low for baseline/optimized A/B runs. Harness emits:
  - Nsight Systems: `nsys profile --force-overwrite=true -o <out> -t cuda,nvtx,osrt --sample=cpu --backtrace=none [--nvtx-include <range> ...] python <wrapper>`.
  - Nsight Compute: `ncu --set speed-of-light --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.avg --replay-mode kernel [--pm-sampling-interval <cycles>] --target-processes all [--nvtx-include <range> ...] -o <out> python <wrapper>`.
  - PyTorch profiler (CUDA-only, sparse schedule): 
    ```python
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False,
        with_stack=False, with_flops=False,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    ) as prof:
        for _ in range(min(config.iterations, 10)):
            fn(); prof.step()
    ```
- NVTX filters are auto-derived from the benchmark’s NVTX ranges (prefill/decode/router/moe_ep, etc.) so the traces stay focused; override with `BenchmarkConfig(nsys_nvtx_include=["prefill","decode"])`.
- Switch to a fuller capture with `profile_type="deep_dive"` (CLI: `--profile deep_dive`) and optionally `ncu_metric_set="deep_dive"`—the metric set auto-switches when profile_type is deep_dive. Keep the same preset for both baseline_ and optimized_ when comparing. Default safer settings are `--ncu-metric-set minimal --ncu-replay-mode kernel`; use `--profile minimal|roofline` and `--ncu-metric-set auto|minimal|deep_dive|roofline` to tune mix vs. overhead.

---

## Summary

The `BenchmarkHarness` is a **production-grade benchmarking framework** that:

1. **Automatically discovers** 264 benchmarks across 20 chapters
2. **Executes reliably** with subprocess isolation and timeout protection
3. **Profiles comprehensively** with nsys, ncu, and PyTorch profiler integration
4. **Analyzes statistically** with mean, median, percentiles, and custom metrics
5. **Compares automatically** baseline vs optimized implementations
6. **Reports comprehensively** with JSON and Markdown outputs
7. **Handles errors gracefully** with recovery and partial results
8. **Ensures reproducibility** with seed management and manifest capture

**The result**: A single command runs the entire benchmark suite, producing comprehensive performance analysis, profiling data, and comparison reports - enabling systematic performance engineering at scale.
