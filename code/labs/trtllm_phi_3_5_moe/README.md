# Lab - TRT-LLM Phi-3.5 MoE

## Summary
Benchmarks a TensorRT-LLM style Phi-3.5 MoE serving path against a slower reference path so the repo has a measured inference-stack example, not just kernel-level microbenches.

## Problem
End-to-end serving optimizations are easy to misread because setup, engine build, and runtime execution all blur together. This lab keeps a reference path and an optimized TRT-LLM path in the same harness contract so the serving win is measurable.

## Baseline Path
- slower reference serving/inference path
- stable end-to-end anchor for the optimized TRT-LLM route
- useful for showing the cost of not using the optimized engine stack

## Optimized Path
- TensorRT-LLM-oriented optimized serving path
- same workload and verification contract
- tuned to show the practical inference-stack win, not just a kernel-local result
- verifies a deterministic generated-token prefix rather than the more fragile full-logits or full-suffix path

## Measured Delta
Representative validated result from `artifacts/runs/20260303_trtllm_phi35moe_minimal_expectations_mixedprov_clean17/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `trtllm_phi_3_5_moe` | `9767.635 ms` | `1065.477 ms` | `9.17x` |

That is a substantial end-to-end win, and it only became worth documenting after the earlier failure and verification-cleanup passes were sorted out. This is exactly why the repo keeps the validation history instead of hiding the false starts.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/trtllm_phi_3_5_moe
python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile minimal
```

## Learning Goals
- Keep a real serving-stack optimization in the repo's benchmark story.
- Measure TRT-LLM value under the same harness discipline as the kernel labs.
- Make it clear when an optimized inference stack is really worth the complexity.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_trtllm_phi_3_5_moe.py`, `optimized_trtllm_phi_3_5_moe.py` | Baseline and TensorRT-LLM benchmark entrypoints. |
| `trtllm_common.py` | Shared helpers and workload setup for the pair. |
| `expectations_{hardware_key}.json` | Regression thresholds for the lab. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/trtllm_phi_3_5_moe
python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe --profile minimal
```
- Targets follow the `labs/trtllm_phi_3_5_moe:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/trtllm_phi_3_5_moe:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe --profile minimal` should keep the optimized path verification-clean on the shared generated-token prefix and materially ahead.

## Notes
- This lab is one of the best repo examples for "serving-stack optimization" as opposed to pure kernel tuning.
- `trtllm_phi_3_5_moe` accepts `--model-path`, `--engine-path`, `--prompt-len`, `--max-new-tokens`, `--batch-size` and `--vocab-slice`, for example `--target-extra-arg labs/trtllm_phi_3_5_moe:trtllm_phi_3_5_moe="--model-path /path/to/model"`. Any other argument fails the target.
