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

## Historical Delta (portable B200, earlier multi-head stack)
Wave 46 measured the earlier decoder stack, whose K and V
projections were full 4,096 x 4,096 matrices. It ran the actual
factories with identical 32-layer, 7,784,890,368-parameter models
at batch one and 2,048 tokens. The current stack uses 8 key/value
heads and has 6,979,584,000 decoder parameters, so these numbers
do not describe it. It needs a fresh B200 measurement. Four
ABBA blocks covered eight fresh inputs and eight observations per
arm. Every complete 8,388,608-element output matched bitwise.
CUDA-event medians were `30.3838 ms` eager and `28.1446 ms` compiled
(`1.07956x`); sample standard deviations were `0.3793 ms` and
`0.4166 ms`, with block ratios from `1.0639x` to `1.0970x`.

Separate Nsight Systems captures observed three steady-state
requests per arm: 1,731 eager kernels versus 1,443 compiled kernels
and exactly three CUDA graph launches. Complete final outputs
remained bitwise equal. These portable observations support the
compilation mechanism for this workload, not a cross-machine or
locked-clock performance guarantee. See the repository's
[repair checkpoint](../../../docs/reviews/2026-09-06-codebase-repair-checkpoint.md)
for the retained attempt history and evidence boundaries.

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
| `llama_3_1_8b_optimization.py` | Shared 32-layer Llama 3.1 8B decoder stack with 8 key/value heads (no token embedding, final norm, LM head, or RoPE) plus an explicit materialized-attention diagnostic mode. |
| `deepseek_r1_moe_optimization.py` | 64-expert top-6 routing demo with balance/Gini/entropy metrics and auxiliary loss. |
| `gpt4_architecture_optimization.py` | GPT-4-scale memory estimate that can assume stored MoE experts and FP8 weights and KV cache, plus a four-layer dense BF16 timing proxy. The proxy does not run MoE, FP8, or context parallelism. |
| `__init__.py` | Exports harness targets for the CLI. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering/code
python -m cli.aisp bench list-targets --chapter labs/real_world_models
python -m cli.aisp bench run --targets labs/real_world_models:llama_3_1_8b --profile minimal --single-gpu
```
- These targets define no `apply_target_overrides` hook, so the harness rejects `--target-extra-arg` for them.

## Validation Checklist
- `llama_3_1_8b` preserves batch one, 2,048 tokens, 4,096 hidden size, 32 layers, and 8 key/value heads. It compares every output value at `rtol=0.02`, `atol=0.02`.
- The normal source factories and profiler must pass on the target GPU before publishing a performance result from the compile-only pair.
- `deepseek_r1_moe_optimization.py` reports balanced experts (Gini < 0.2) and stable router entropy across batches.
- `gpt4_architecture_optimization.py` times only its dense BF16 proxy on one GPU and reports `use_moe`, `use_fp8`, and `use_context_parallel` as `0.0`. Its GPU count is a memory-only estimate at 192 GB per B200, not a tested multi-GPU run.
- Harness runs emit comparable baseline/optimized timings for every target without manual wiring.

## Notes
- These are architecture-shaped random-weight benchmarks, not checkpoint-backed production models; validate production settings in the actual serving stack.
- Hardware expectations: B200/GB200 for best results. The GPT-4 proxy needs one CUDA GPU, and no example here runs a GPT-4-scale model across GPUs.
- Metrics (balance loss, entropy, KV cache) are emitted alongside throughput so you can gate deployments with more than raw speed.
