# Lab - CUTLASS Profiler Kernel Selector

## Summary
Automates CUTLASS profiler sweeps for transformer-style GEMMs, records Triton or custom kernel results, and compares everything so you can prove custom kernels beat the fastest stock CUTLASS option.
This is a workflow-oriented matrix/playbook lab, not currently a clean benchmark-pair lab.

## Learning Goals
- Generate per-shape CUTLASS profiler logs and store the best kernel metadata.
- Optionally benchmark Triton or custom paths on the same shapes.
- Compare providers (CUTLASS, Triton, DeepEP, custom) with a uniform JSON schema.
- Adjust shapes quickly by editing a single definition file.

## Directory Layout
| Path | Description |
| --- | --- |
| `run_cutlass_profiler_sweep.py` | Invokes `cutlass_profiler` for every shape in `shapes.py` and stores JSON summaries. |
| `run_triton_matmul.py` | Optional Triton matmul runner for parity checks. |
| `compare_against_baselines.py` | Reads CUTLASS + competitor JSON files and emits TFLOP/s + speedup tables. |
| `shapes.py` | Central list of GEMM shapes (prefill, decode, KV proj, etc.). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering/code
python labs/cutlass_profiler_kernel_selector/run_cutlass_profiler_sweep.py --output-dir artifacts/cutlass_profiler
python labs/cutlass_profiler_kernel_selector/run_triton_matmul.py --output-dir artifacts/cutlass_profiler
python labs/cutlass_profiler_kernel_selector/compare_against_baselines.py --include-default-triton
```
- `code/setup.sh` clones CUTLASS into `code/third_party/cutlass` but does not build `cutlass_profiler`. Build the `cutlass_profiler` CMake target in `code/third_party/cutlass/build_profiler`, where the sweep looks by default, or point `CUTLASS_PROFILER_BIN` or `--profiler-bin` at an existing binary.
- Add extra providers by writing JSON files matching the documented schema (see `compare_against_baselines.py`).

## Validation Checklist
- Profiler runs emit `artifacts/cutlass_profiler/cutlass_profiler_results.json` with per-shape winners. Rerun the sweep when upgrading CUDA or GPUs.
- Triton baselines land in `artifacts/cutlass_profiler/triton_matmul_results.json` and should stay within a few percent of CUTLASS for supported shapes.
- `compare_against_baselines.py` exits non-zero when the baseline or a provider file is missing, or when any file lacks a positive TFLOP/s record for a shape in `shapes.py` (or the `--shapes` subset) or lists that shape as failed. CI therefore catches stale or partial outputs.

## Notes
- Shapes can be overridden via CLI flags (e.g., `--shapes decode_mlp_m4096_n4096_k8192`).
- Provider JSON files may include metadata (kernel names, launch params) for additional debugging.
