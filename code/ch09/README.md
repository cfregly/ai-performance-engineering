# Chapter 9 - Arithmetic Intensity & Kernel Fusion

## Summary
Explores how to move workloads along the roofline: raise arithmetic intensity with tiling, fuse memory-bound kernels, and deploy CUTLASS/Triton/inline-PTX paths built for Blackwell tensor cores.

## Problem
Chapter 9 is where roofline reasoning has to cash out in actual kernels. The useful question is not "is this compute-bound or memory-bound?" but "which arithmetic-intensity and fusion changes create a measurable gain once the same harness measures both sides?"

## Baseline Path
- lower-intensity or less-fused kernels
- more time spent moving data than doing useful math
- easier to inspect, but often too far from the hardware roofline

## Optimized Path
- CUTLASS/Triton/custom-kernel paths with better tiling and reuse
- fused or higher-intensity schedules that reduce redundant memory work
- the same benchmark contract as the rest of the repo, so the gains are not script-local illusions

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `cutlass_gemm` | `0.178 ms` | `0.045 ms` | `3.95x` | better GEMM schedule and kernel implementation |
| `memory_bound` | `3.491 ms` | `0.205 ms` | `17.05x` | less wasted memory movement on a bandwidth-limited workload |
| `sdpa_attention` | `0.762 ms` | `0.446 ms` | `1.71x` | attention path with improved compute/memory balance |

The right chapter-level read is not that every CUTLASS or Triton change is dramatic. It is that arithmetic-intensity gains and fusion gains show up very differently depending on whether the workload is math-limited or traffic-limited.

## Profiler Evidence
Use deep-dive runs when you want to see whether the improvement came from better tensor-core utilization, less memory traffic, or simply fewer kernels:

```bash
python -m cli.aisp bench run --targets ch09:cutlass_gemm --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch09:memory_bound --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch09:sdpa_attention --profile deep_dive --single-gpu
```

Those three targets give you a balanced view of the chapter:
- `cutlass_gemm`: math-path scheduling
- `memory_bound`: bandwidth-path improvement
- `sdpa_attention`: mixed compute/memory behavior in a more realistic primitive

## Repro Commands
```bash
python -m ch09.compare
python -m cli.aisp bench list-targets --chapter ch09
python -m cli.aisp bench run --targets ch09 --profile minimal
python -m cli.aisp bench run --targets ch09:cutlass_gemm --profile deep_dive --single-gpu
```

## Learning Goals
- Separate compute-bound vs memory-bound behaviors and adjust kernels accordingly.
- Design micro-tiling schedules that balance register pressure with data reuse.
- Leverage CUTLASS and Triton for rapid iteration while keeping custom CUDA fallbacks.
- Fuse reduction-heavy kernels (e.g., norm + activation) to eliminate redundant memory trips.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py` | Reference kernels that isolate compute vs bandwidth ceilings and demonstrate tuning strategies. |
| `baseline_micro_tiling_matmul.cu`, `baseline_micro_tiling_matmul.py`, `optimized_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.py` | Naive and 16x16 shared-memory tiled matmuls with per-thread scalar accumulation. |
| `baseline_cublaslt_gemm.cu`, `baseline_cublaslt_gemm.py`, `optimized_cublaslt_gemm.cu`, `optimized_cublaslt_gemm.py`, `tcgen05_pipelined.cu` | cuBLASLt-driven matmuls and tcgen05 pipeline kernels showcasing tcgen05 lowering and occupancy tuning. |
| `baseline_cublaslt_gemm_fp4.cu`, `baseline_cublaslt_gemm_fp4.py`, `optimized_cublaslt_gemm_fp4.cu`, `optimized_cublaslt_gemm_fp4.py` | FP4 comparison path: naive block-scaled FP4 baseline vs native cuBLASLt NVFP4 when the driver/toolchain exposes the required heuristic. |
| `baseline_fused_l2norm.cu`, `baseline_fused_l2norm.py`, `optimized_fused_l2norm.cu`, `optimized_fused_l2norm.py`, `fusedL2Norm/` | Fusion examples that merge L2 norm + scaling while staying numerically stable. |
| `baseline_triton.py`, `optimized_triton.py` | Triton counterparts for quick prototyping and verifying compiler-generated PTX on Blackwell. |
| `baseline_tcgen05_tma_pipeline.py`, `optimized_tcgen05_tma_pipeline.py`, `two_stage_pipeline.cu` | Producer/consumer pipelines emphasizing staged TMA loads and inline PTX hooks. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness hooks plus regression thresholds for every example. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch09.compare
python -m cli.aisp bench list-targets --chapter ch09
python -m cli.aisp bench run --targets ch09 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m cli.aisp bench run --targets ch09:compute_bound ch09:memory_bound --profile minimal` reports much higher arithmetic intensity for the compute-bound path than the memory-bound path, matching the roofline plots.
- `python -m cli.aisp bench run --targets ch09:cublaslt_gemm --profile minimal` confirms the optimized path improving throughput over the baseline on the same device.
- `python -m ch09.compare --examples fused_l2norm` confirms numerically identical outputs before and after fusion.

## Notes
- `inline_ptx_example.cu` demonstrates how to wrap tcgen05 intrinsics safely with architecture guards.
- `requirements.txt` includes the shared `requirements_latest.txt` pins for PyTorch 2.9.1 and its compatible Triton release.
- `optimized_cublaslt_gemm_fp4` is intentionally capability-gated: if cuBLASLt cannot provide the native block-scaled NVFP4 heuristic, the benchmark reports a clean skip instead of silently falling back to a different FP4 mode.
