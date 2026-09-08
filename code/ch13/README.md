# Chapter 13 - PyTorch Profiling & Memory Tuning

## Summary
Focuses on PyTorch-centric optimizations: compiled autograd, memory profiling, FSDP/context/expert parallelism, and FP8/quantization workflows backed by the same harness infrastructure. The chapter README is fairness-refreshed so canonical pairs stay separate from informational variants such as `torchao_quantization_compiled`, `kv_cache_naive_flash_blockwise`, and the torchao FP8 recipe demos (`precisionfp8`, `precisionfp8_rowwise`, `precisionfp8_rowwise_gw_hp`). The manuscript walkthrough uses a fuller Hugging Face/MoE profiling example, while the repo chapter keeps the runnable surface lighter and more harness-native.

## Problem
Chapter 13 is where high-level PyTorch optimizations have to prove they are doing more than rearranging framework overhead. The useful question is not "can PyTorch do this optimization?" but "which profiling, compilation, precision, and memory changes actually improve the workload under the shared harness?"

## Baseline Path
- eager or less-optimized PyTorch execution
- higher-overhead cache, precision, and dataloader paths
- easier to debug, but often too expensive once memory and framework overhead dominate

## Optimized Path
- compiled, quantized, or allocator-aware PyTorch paths where they produce a real measured benefit
- lower-overhead cache and attention paths
- still benchmarked through the same harness contract, so the numbers stay comparable to the lower-level chapters

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `kv_cache_naive` | `1664.672 ms / 1194.135 MB` | `1739.080 ms / 370.394 MB` | `68.98% less memory` | token-by-token paged allocation preserves the batch contract while cutting KV-cache footprint |
| `memory_profiling` | allocator fragmentation baseline | gradient-checkpointed path | memory-goal benchmark | checkpointing is judged by lower peak allocation and cleaner allocator behavior, not by raw speedup |
| `autograd_standard` | `1.644 ms` | `0.204 ms` | `8.04x` | compiled/optimized autograd path |

This chapter is one of the easiest places to fool yourself with framework overhead. That is why the benchmark contract and side-by-side baseline/optimized structure matter here more than almost anywhere else.

The prior `precisionfp8_te` number compared eager FP16 with CUDA-graph-replayed FP8, so it is retired. The pair now runs both sides eagerly to isolate Transformer Engine FP8. A fresh direct-B200 sweep with Transformer Engine 2.18 measured the full training update after five setup and ten warmup updates:

| Matched batch | Eager FP16 median | Eager TE FP8 median | FP16 / FP8 | Disposition |
| ---: | ---: | ---: | ---: | --- |
| 256 | `0.4786 ms` | `0.6662 ms` | `0.7183x` | no measured speedup |
| 1,024 | `0.5483 ms` | `0.6644 ms` | `0.8252x` | no measured speedup |
| 4,096 | `1.2303 ms` | `0.9616 ms` | `1.2795x` | candidate speedup for this workload |

The 24 observations cover two seeds, two repeats, both arms, and all three batches. Every observation compared the actual prediction and all 67,121,152 post-step parameters and passed the frozen output policy. Batch 256 remains the default, so the batch-4,096 result establishes a workload-specific crossover rather than a general default-workload speedup. Whole-call cost is retained separately from the CUDA-event update timing.

The TE 2.18 pair now verifies its captured prediction and every post-step parameter with a calibrated per-output policy. Previously, all outputs inherited the global `(rtol=0.5, atol=5.0)` threshold. B200 calibration of the unchanged batch-256, hidden-4096 training step selected these stricter budgets across seed 44 and fixed holdouts 45, 1044, and 1045:

| Captured outputs | Calibrated `(rtol, atol)` | Largest observed absolute error |
| --- | --- | --- |
| Prediction | `(0.4, 1.0)` | `0.5949707` |
| Both weight tensors | `(0.001, 0.00075)` | `0.000534058` |
| Both bias tensors | `(0.001, 0.00005)` | `0.000041008` |

The frozen map passed all holdouts and independently rejected zeroed and localized corrupted copies of all five outputs. These bounds apply to this TE 2.18 workload and establish numerical verification only; they do not establish a speedup.

## Matched Batch Controls
`precisionfp8_te` keeps batch size 256 as its default workload. Omitting a target override preserves that default for both the eager FP16 baseline and eager Transformer Engine FP8 candidate:

```bash
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu
```

To test optional larger matched controls, pass one pair-wide override through the harness. This sends the requested batch to both arms and updates their workload metadata consistently:

```bash
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu --target-extra-arg 'ch13:precisionfp8_te=--batch-size 1024'
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu --target-extra-arg 'ch13:precisionfp8_te=--batch-size 4096'
```

Compare results only when both arms report the same requested batch size and workload signature. Fresh B200 checks pass at 256, 1,024, and 4,096 with the frozen policy. The first two batches are slower under FP8; the observed 1.2795x result applies only to the matched batch-4,096 workload and does not change the default.

## Profiler Evidence
Use deep-dive runs when you want to see whether the gain came from framework overhead reduction, memory behavior, or the lower-precision path itself:

```bash
python -m cli.aisp bench run --targets ch13:kv_cache_naive --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch13:autograd_standard --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu
```

Those targets cover three different PyTorch optimization stories:
- `kv_cache_naive`: cache-path and memory behavior, with memory reduction treated as the primary win
- `autograd_standard`: framework/compile overhead
- `precisionfp8_te`: lower-precision execution with real library support

Matched batch-4,096 Nsys captures pass for both `precisionfp8_te` arms. The FP8 trace shows shorter main GEMMs alongside quantization and scale-update work. Each trace contains one profiled update after setup and one profiler warmup, so trace durations are diagnostic; the repeated sweep above remains the timing authority.

The torchao FP8 recipe demos (`precisionfp8`, `precisionfp8_rowwise`, `precisionfp8_rowwise_gw_hp`) remain useful implementation references, but they are treated as informational examples rather than canonical speed-claim surfaces.

## Repro Commands
```bash
python -m ch13.compare
python -m cli.aisp bench list-targets --chapter ch13
python -m cli.aisp bench run --targets ch13 --profile minimal
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu --target-extra-arg 'ch13:precisionfp8_te=--batch-size 1024'
python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile deep_dive --single-gpu --target-extra-arg 'ch13:precisionfp8_te=--batch-size 4096'
```

## Learning Goals
- Profile PyTorch training loops end-to-end, capturing goodput, memory, and kernel traces.
- Apply `torch.compile`, regional compilation, and custom allocators to reduce overhead.
- Tune DataLoader, KV-cache, and optimizer states to eliminate fragmentation.
- Exercise FP8/quantized training recipes with Transformer Engine integration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_training_standard.py`, `optimized_training_standard.py`, `train.py`, `train_deepseek_v3.py`, `train_deepseek_coder.py` | Reference training loops showcasing eager vs compiled paths and DeepSeek-inspired configs. |
| `baseline_dataloader_default.py`, `optimized_dataloader_default.py`, `baseline_memory_profiling.py`, `optimized_memory_profiling.py`, `memory_profiling.py` | DataLoader/memory studies that explain how to read allocator stats and fix leaks. |
| `baseline_attention_standard.py`, `optimized_attention_standard.py`, `baseline_long_context_attention.py`, `optimized_long_context_attention.py`, `baseline_arithmetic_intensity.py`, `optimized_arithmetic_intensity.py`, `baseline_matmul_pytorch.py`, `optimized_matmul_pytorch.py` | Attention and matmul microbenchmarks tuned purely within PyTorch, including long-context Flash SDP. |
| `baseline_context_parallel_multigpu.py`, `optimized_context_parallel_multigpu.py`, `context_parallel_benchmark_common.py` | Context-parallel attention benchmarks comparing all-gather vs ring-style streaming across ranks. |
| `baseline_sequence_parallel_multigpu.py`, `optimized_sequence_parallel_multigpu.py`, `sequence_parallel_benchmark_common.py` | Sequence-parallel TP+SP hybrid benchmark contrasting per-layer full-sequence all-gather against keeping activations sequence-sharded between tensor-parallel layers. |
| `baseline_expert_parallel_multigpu.py`, `optimized_expert_parallel_multigpu.py`, `expert_parallel_common.py` | Expert-parallel all-to-all benchmarks contrasting per-iteration list allocations vs pre-allocated all_to_all_single. |
| `context_parallelism.py`, `fsdp_example.py` | Context and FSDP sharding demos for scaling beyond a single GPU. (Tools; not benchmark targets.) |
| `baseline_precisionfp8*.py`, `optimized_precisionfp8*.py`, `baseline_precisionmixed.py`, `optimized_precisionmixed.py`, `compiled_autograd.py` | Precision-management suites covering Transformer Engine, torchao FP8 recipe demos, and compiled autograd recipes. |
| `baseline_quantization.py`, `optimized_quantization.py`, `baseline_kv_cache_naive.py`, `optimized_kv_cache_naive.py`, `optimized_kv_cache_naive_pool.py` | Quantization and KV-cache pipelines for inference/training memory savings, including the quantization-only canonical pair and a token-by-token decode with naive concat cache versus paged cache allocation. |
| `compare.py`, `compare_perf.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `workload_config.py` | Harness entry, performance comparison helper, dependencies, and regression baselines. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch13.compare
python -m cli.aisp bench list-targets --chapter ch13
python -m cli.aisp bench run --targets ch13 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch13.compare --examples training_standard` shows optimized training runs producing higher goodput with identical metrics.
- `python -m cli.aisp bench run --targets ch13:precisionfp8_te --profile minimal` confirms Transformer Engine calibration plus NVFP8 execution with max error tolerances enforced.
- Matched B200 `precisionfp8_te` checks at batches 256, 1,024, and 4,096 pass the full-output policy; only batch 4,096 shows a measured candidate speedup (`1.2795x`) for this exact workload.
- `python -m ch13.memory_profiling --dump` and the optimized variant demonstrate allocator fragmentation dropping after applying the recommended knobs, with memory reduction treated as the primary benchmark outcome.

## Notes
- `custom_allocator.py` contains a standalone torch allocator shim that can be re-used in other chapters when debugging fragmentation.
- `compiled_autograd.py` doubles as a tutorial on partial graph capture; the README here references it directly.
- `precisionfp8_te` defaults to batch 256. Larger `--batch-size` values are explicit pair-wide workload overrides; the B200 crossover appeared at batch 4,096 and does not change the default.
- `torchao_quantization_compiled`, `kv_cache_naive_flash_blockwise`, `precisionfp8`, `precisionfp8_rowwise`, and `precisionfp8_rowwise_gw_hp` remain informational variants.
- `kv_cache_naive` and `memory_profiling` are memory-goal benchmarks; they are expected to reduce memory pressure even when the timed path is not faster.
