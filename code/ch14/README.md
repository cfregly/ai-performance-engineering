# Chapter 14 - Compiler & Triton Optimization

## Summary
Highlights compiler-driven acceleration: `torch.compile` workflows, Triton kernels, CUTLASS/TMA experimentation, and quantization-aware communication, all validated through the shared harness. The repo chapter focuses on CUDA/Triton/Inductor paths; the broader XLA backend discussion from the manuscript is not represented as runnable chapter code here. The explicit `cublas_vs_cutlass` pair remains an informational comparison surface rather than a chapter-native speed-claim benchmark.

## Problem
Chapter 14 is where compiler claims have to turn into measured wins. The useful question is not "can `torch.compile` or Triton work?" but "which compiler-driven optimizations still deliver real latency and memory reductions on current Blackwell-class hardware?"

## Baseline Path
- eager or minimally fused PyTorch execution
- generic Triton/CUTLASS paths without persistent or regional specialization
- easier to reason about, but heavy on launch overhead, graph breaks, and redundant staging

## Optimized Path
- `torch.compile` and regional compilation where the graph is stable enough to pay back compile cost
- Triton persistent kernels and TMA-fed schedules where memory movement dominates
- the same harness contract as every other benchmarked chapter, so the speedups are comparable instead of script-local

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `model_compile_reduced_precision` | `29.873 ms` | `7.978 ms` | `3.74x` | eager reduced precision vs reduced precision + `torch.compile` |
| `regional_triton` | `1.944 ms` | `0.863 ms` | `2.25x` | regional compilation and Triton fusion |
| `triton_persistent` | `0.830 ms` | `0.086 ms` | `9.68x` | persistent Triton kernel scheduling |

These are chapter-level proof points, not vendor peak numbers. The chapter is most useful when you separate "compiler removes Python/graph overhead" from "kernel schedule removes memory-movement overhead."

## Profiler Evidence
Use the same benchmark targets with deep-dive profiling when you want launch-count and kernel-attribution evidence instead of only the wall-clock delta:

```bash
python -m cli.aisp bench run --targets ch14:model_compile_reduced_precision --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch14:regional_triton --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch14:triton_persistent --profile deep_dive --single-gpu
```

The expected story is different per workload:
- `model_compile_reduced_precision`: fewer graph breaks and lower framework overhead, with both paths using the same reduced-precision dtype and the optimized path adding `torch.compile`
- `regional_triton`: fewer unfused launches and better steady-state scheduling
- `triton_persistent`: materially longer-lived kernels with less relaunch churn

## Repro Commands
```bash
python -m ch14.compare --profile none
python -m cli.aisp bench list-targets --chapter ch14
python -m cli.aisp bench run --targets ch14 --profile minimal
python -m cli.aisp bench run --targets ch14:triton_persistent --profile deep_dive --single-gpu
```

## Learning Goals
- Adopt `torch.compile` modes for large models while tracking compile-time and steady-state gains.
- Author Triton kernels (including TMA schedules) that rival custom CUDA.
- Profile FlexAttention and regional compilation strategies end-to-end.
- Blend quantization with NCCL and pipeline overlap without regressions.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_model_compile_reduced_precision.py`, `optimized_model_compile_reduced_precision.py`, `model_eager_common.py`, `torch_compile_large_model.py`, `torch_compiler_examples.py`, `training_large_model_1_5x.py` | Model-scale examples showcasing the eager-vs-compiled reduced-precision pair, shared transformer scaffolding, compile modes, guard rails, and large-model sanity tests. |
| `baseline_cublas_vs_cutlass.py`, `optimized_cublas_vs_cutlass.py`, `triton_examples.py`, `triton_tma_blackwell.py`, `triton_fp8_advanced.py`, `triton_nvshmem_example.py` | Explicit cuBLAS-vs-CUTLASS comparison pair plus advanced TMA/NVSHMEM Triton kernels. The comparison pair remains informational rather than a canonical chapter speed claim. |
| `baseline_attention_eager_sdpa.py`, `optimized_attention_eager_sdpa.py`, `baseline_flex_attention_sparse.py`, `optimized_flex_attention_sparse.py`, `flex_attention_sparse_demo.py` | Eager-vs-SDPA attention plus FlexAttention sparse workloads that validate custom score mods, masks, sparsity, and compile speedups. |
| `baseline_nccl_quantization.py`, `optimized_nccl_quantization.py`, `deepseek_innovation_l2_bypass.py` | Quantization-aware communication and the DeepSeek-inspired L2 bypass experiment. |
| `baseline_regional_triton.py`, `optimized_regional_triton.py`, `inspect_compiled_code.py`, `benchmark_tma_configs.py` | Regional compilation and TMA parameter sweeps for auto-tuning generated kernels. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `train.py`, `transformer.py` | Harness entry plus model definitions and dependency pins. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch14.compare
python -m cli.aisp bench list-targets --chapter ch14
python -m cli.aisp bench run --targets ch14 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m cli.aisp bench run --targets ch14:model_compile_reduced_precision --profile minimal` produces compile-time summaries followed by steady-state throughput gains vs an eager baseline running the same reduced-precision model.
- `python -m ch14.triton_tma_blackwell --validate` compares Triton and CUDA outputs to double-check TMA scheduling logic.
- `python -m ch14.compare --examples attention_eager_sdpa` shows the fused SDPA path reducing kernel launch count without changing accuracy.

## Notes
- `inspect_compiled_code.py` dumps Triton/PTX/Graph captures for any target; edit the helper to introspect new workloads.
- `requirements.txt` includes the shared `requirements_latest.txt` pins for stable PyTorch and Triton wheels aligned with the CUDA 13 toolchain.
- For repo-native supporting examples that fill the training hot-path gaps without changing this chapter's primary compile narrative, see `labs/training_hotpath`.
- `cublas_vs_cutlass` is a supplementary comparison pair. Chapter-native performance claims stay anchored on `model_compile_reduced_precision`, `regional_triton`, and `triton_persistent`.
