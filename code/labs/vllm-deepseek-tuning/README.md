# Lab - vLLM DeepSeek Tuning Harness

## Summary
A matrix-driven tuning harness for DeepSeek + vLLM scenarios: scenario sweeps, plots, reports, and startup-failure capture. It is a comparison matrix, not a single honest baseline/optimized benchmark pair yet.

## Problem
Large serving-stack comparisons rarely reduce to one "optimized" switch. TP/EP choices, MTP on/off, model family, and concurrency sweeps all matter, so this lab keeps the experiment matrix explicit instead of pretending there is one universal optimized path.

## What This Lab Is
- a matrix harness around `vllm serve` + `vllm bench serve`
- scenario/config/report tooling
- useful for comparative serving experiments and artifact generation

It is not currently a clean benchmark-pair lab because the important comparisons are multiple named variants, not one generic baseline/optimized route.

## Current Artifact State
The checked-in artifact set under `results/` is currently startup-failure/report oriented, not a canonical baseline/optimized pair history. That is still valuable, but it should be described honestly.

If we want benchmark-pair docs here later, we should first produce canonical successful runs for one or two concrete comparisons instead of extrapolating from the matrix harness.

## What Proper Benchmark Pairs Would Look Like
If we productize this into benchmark targets, the clean shape is to create explicit comparison pairs such as:

- `baseline_vllm_deepseek_tp2.py` vs `optimized_vllm_deepseek_ep2.py`
- `baseline_vllm_deepseek_mtp0.py` vs `optimized_vllm_deepseek_mtp1.py`

Each pair would need fixed prompts, fixed ISL/OSL/concurrency, stable serve lifecycle handling, and a clear validation/report artifact contract. That is better than inventing one generic `optimized_vllm_deepseek.py` wrapper that hides what actually changed.

## Learning Goals
- Keep DeepSeek + vLLM comparison work reproducible and auditable.
- Separate matrix-harness experimentation from benchmark-pair performance claims.
- Provide a cleaner path to future canonical vLLM serving benchmark pairs.

## Directory Layout
| Path | Description |
| --- | --- |
| `configs/benchmark_matrix.yaml`, `configs/smoke_tiny.yaml` | Scenario matrices and smoke-test configs. |
| `scripts/run_matrix.py`, `scripts/plot_results.py`, `scripts/report_results.py` | Serve/bench orchestration plus plotting/report generation. |
| `results/`, `plots/`, `reports/` | Structured outputs from the matrix harness. |
| `Makefile`, `scripts/vllm_docker.sh`, `scripts/teardown.sh` | Operational entrypoints for running and cleaning up the matrix. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd labs/vllm-deepseek-tuning
make smoke
make full
make artifacts
```
- This lab is Makefile/script driven today, not harness-target driven.
- The PyTorch 2.14.1 compatibility smoke must use the repository's host serving stack: PyTorch `2.14.1+cu130`, source-built vLLM `0.33.0.dev0+cu130.torch2141`, and FlashInfer `0.7.2`.
- `scripts/vllm_docker.sh` defaults to the moving `cu130-nightly` image. It is useful for exploratory isolation, but it does not prove compatibility with the repository's exact source-built stack unless the image is overridden by a digest-pinned equivalent.

## Validation Checklist
- `make smoke` should prove the orchestration path is alive before a full matrix run.
- Inspect the emitted JSON `returncode` values and the final `bench_success`/`bench_failed` counts. `run_matrix.py` continues after a request-benchmark failure and can exit zero after retaining the failure.
- `make full` should emit raw logs plus structured `results/*.json` records.
- `make artifacts` should regenerate plots and markdown/csv reports from collected results.
- If this lab is promoted into benchmark-pair targets later, create explicit named comparison pairs instead of a fake one-size-fits-all optimized wrapper.

## Notes
- This README stays intentionally honest: useful matrix harness, not yet a canonical baseline/optimized benchmark lab.
