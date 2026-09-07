# Lab - NanoChat Fullstack

## Summary
Provides a harness-comparable NanoChat inference pair for a fixed prefill-plus-decode workload. The baseline launches the eager model for both phases; the optimized path replays the complete request in one CUDA graph.

## Problem
Full-stack LLM comparisons can look successful while checking degenerate outputs or timing different execution scopes. This lab holds the model, inputs, attention settings, KV-cache layout, and decode work constant, then verifies the final decode logits from both paths.

## Baseline Path
- Runs a batch of 4 with a 512-token prefill followed by 64 one-token decode steps.
- Uses the eager NanoChat GPT model for prefill and decode.
- Captures the final decode logits as the verification output.

## Optimized Path
- Runs the same model configuration, inputs, attention path, KV cache, and decode loop as the baseline.
- Captures the fixed 512-token prefill and all 64 ordered decode steps in one CUDA graph, preserving the baseline kernels and supplied decode tokens.
- Warms and captures during setup; reports those costs separately from steady-state replay. Inputs remain mutable in place, and each replay overwrites the complete request's KV-cache state.

## Historical Delta
No current performance delta is published. Earlier README timings and the retained expectation files were produced by older benchmark source. The current pair reinitializes projections that the training initializer leaves at zero, rejects all-zero or non-finite logits, and captures the complete request for replay. Prior numbers therefore do not establish the performance of this workload.

Publish a new delta only from the current source after output verification, repeated interleaved baseline/optimized measurements on the target hardware, noise reporting, and profiler evidence.

## Profiler Evidence
```bash
cd code
python -m cli.aisp bench run -t labs/nanochat_fullstack:nanochat_inference --profile deep_dive --single-gpu
```

Use this path to compare eager kernel launches with whole-request CUDA graph replay. A successful correctness run alone does not establish a performance win. Keep the `speedrun.sh` workflow separate from this harness pair; they exercise different scopes.

## Repro Commands
```bash
cd code
python -m cli.aisp bench list-targets --chapter labs/nanochat_fullstack
python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference
python -m cli.aisp bench run -t labs/nanochat_fullstack:nanochat_inference --profile minimal --single-gpu
```

## Learning Goals
- Keep a full-stack LLM workload in the benchmark suite rather than reducing the comparison to one kernel.
- Reduce host launch overhead while preserving the complete prefill and decode work.
- Require meaningful final logits before interpreting timing or profiler results.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nanochat_inference.py`, `optimized_nanochat_inference.py` | Harness pair for eager prefill/decode versus whole-request CUDA graph replay. |
| `benchmark_incremental_optimizations.py` | Incremental benchmarking helper inside the NanoChat tree. |
| `speedrun.sh`, `run1000.sh`, `README_FAST.md` | Broader NanoChat quick-start and end-to-end project entrypoints. |
| `nanochat/`, `scripts/`, `tasks/`, `tests/` | Core NanoChat project tree and operational helpers. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd code
python -m cli.aisp bench list-targets --chapter labs/nanochat_fullstack
python -m cli.aisp bench run -t labs/nanochat_fullstack:nanochat_inference --profile minimal --single-gpu
```
- The benchmark requires CUDA and uses one visible GPU with `--single-gpu`.
- Benchmark validity defaults to `strict`; use `portable` only when its documented compatibility tradeoffs are acceptable.
- Treat the retained expectation files as historical until current-source measurements pass the verification and evidence gates above.

## Validation Checklist
- `python -m cli.aisp bench verify -t labs/nanochat_fullstack:nanochat_inference` must compare finite, nonzero final decode logits before timing is interpreted.
- Do not require or claim that the optimized path wins until a current-source, equivalent-workload run supplies repeated measurements and profiler attribution.

## Project Context
NanoChat is intentionally bigger than a single benchmark pair. This lab entry provides one narrow, auditable inference comparison inside that tree; it does not replace the broader NanoChat project documentation.

- Use [README_FAST.md](README_FAST.md) for the faster end-to-end project walkthrough.
- Use [speedrun.sh](speedrun.sh) when you want the broader "train and talk to a small model" experience.
- Use [rustbpe/README.md](rustbpe/README.md) for the tokenizer-specific component work.

## Notes
- This README covers the harness pair. Use `README_FAST.md` and the project scripts for the broader training and serving walkthrough.
- The decode microbenchmarks live separately in `labs/decode_optimization`; this lab is the broader inference-stack companion.
