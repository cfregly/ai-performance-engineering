# Lab - Real-World Model Optimizations

## Summary
Applies the course-wide optimization patterns to representative models (Llama 3.1 8B, DeepSeek-R1 MoE, GPT-4-style) so you can practice end-to-end tuning on Blackwell and Grace-Blackwell hardware.

## Problem
Microbenchmarks are useful, but they can hide whether the repo's optimizations still matter on a real model path. This lab is the end-to-end check.

## Baseline Path
- Llama uses eager preferred SDPA with FP32 residuals and one stable RMSNorm implementation
- the baseline and optimized Llama arms have identical weights, inputs, attention, and normalization
- materialized manual attention remains a separate numerical diagnostic, not the performance baseline

## Optimized Path
- the Llama pair changes only max-autotune `torch.compile`
- topology-aware and memory-aware configuration choices
- the same benchmark harness contract as the lower-level labs

## Current Llama Evidence Boundary
The Wave 40 private B200 design probe used identical 32-layer,
7,784,890,368-parameter models at batch one and 2,048 tokens. All
8,388,608 outputs matched bitwise for eight distinct inputs. Its
CUDA-event medians were `30.481 ms` eager and `28.233 ms` compiled
(`1.080x`). This supports the compile-only source design; it does
not qualify the subsequently changed factories or replace a normal
harness and profiler run.

The retained materialized-attention comparisons exceeded the
unchanged `rtol=0.02`, `atol=0.02` gate. That path remains available
as `attention_mode="manual"` for diagnosis and is not used as the
performance baseline.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile deep_dive --single-gpu
```

That path keeps the same evidence model as the rest of the repo: baseline/optimized timing, validation, and profiler artifacts in one run tree.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/real_world_models
python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile minimal --single-gpu
```

## Learning Goals
- Exercise attention, MoE, and memory optimizations on realistic architectures instead of toy kernels.
- Use the benchmark harness to collect reproducible throughput/latency metrics across models.
- Track expert balance, routing entropy, and KV-cache pressure while iterating on serving choices.
- Compare eager and compiled execution without changing the Llama math or workload.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_llama_3_1_8b.py`, `optimized_llama_3_1_8b.py` | Compile-only Llama pair with shared preferred SDPA, FP32 residuals, stable RMSNorm, and complete-output verification. |
| `llama_3_1_8b_optimization.py` | Shared 32-layer model plus an explicit materialized-attention diagnostic mode. |
| `deepseek_r1_moe_optimization.py` | 64-expert top-6 routing demo with balance/Gini/entropy metrics and auxiliary loss. |
| `gpt4_architecture_optimization.py` | GPT-4-style MoE + context-parallel sketch with FP8 support and memory estimation. |
| `__init__.py` | Exports harness targets for the CLI. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/real_world_models
python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile minimal --single-gpu
```
- Override per-model flags via `--target-extra-arg labs/real_world_models:<target>="--flag value"` when using the harness.

## Validation Checklist
- `llama_3_1_8b` preserves batch one, 2,048 tokens, 4,096 hidden size, and 32 layers; it compares every output value at `rtol=0.02`, `atol=0.02`.
- The normal source factories and profiler must pass on the target GPU before publishing a performance result from the compile-only pair.
- `deepseek_r1_moe_optimization.py` reports balanced experts (Gini < 0.2) and stable router entropy across batches.
- `gpt4_architecture_optimization.py` runs the context-parallel path without OOM on appropriately sized clusters; memory estimates match the printed budget.
- Harness runs emit comparable baseline/optimized timings for every target without manual wiring.

## Notes
- These are architecture-shaped random-weight benchmarks, not checkpoint-backed production models; validate production settings in the actual serving stack.
- Hardware expectations: B200/GB200 for best results; GPT-4-scale examples assume 24+ GPUs with NVLink/NVL fabrics.
- Metrics (balance loss, entropy, KV cache) are emitted alongside throughput so you can gate deployments with more than raw speed.
