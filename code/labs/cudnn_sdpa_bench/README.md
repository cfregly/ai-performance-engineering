# Lab - cuDNN SDPA Bench

## Summary
Microbenchmarks cuDNN fused scaled-dot-product attention against Flash and math backends with explicit CLI backend selection.

## Problem
Attention backend choices are often treated as an implementation detail. This lab exists to keep that choice explicit and benchmarked so you can tell whether cuDNN, Flash, or the math path is actually the right answer for this exact shape family.

## Baseline Path
- pins the unfused math SDPA backend by default
- stable reference for correctness and shape coverage
- useful when fused paths are unavailable or unstable

## Optimized Path
- pins the fused Flash SDPA backend by default
- same shapes and validation contract
- tuned to answer "does backend choice alone move the result?" rather than mixing in unrelated changes

## Historical Delta (backend attribution unverified)
Earlier representative result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flash_sdp` | `0.345 ms` | `0.282 ms` | `1.22x` |

That run timed the default pair. The baseline used `auto`, which lets PyTorch choose among cuDNN, Flash, and efficient attention, and the run did not record which kernel it dispatched. The optimized side pinned Flash. The table therefore does not establish a cuDNN-versus-Flash delta or any other backend-specific delta. The b200 and 4x_gb200 expectation entries came from the same auto-versus-Flash default and were retired. A new entry needs a strict run in which both variants pin explicit, different backends. The default pair now pins math against Flash. A pinned backend fails if it is unavailable. Fresh GPU verification and profiler evidence are required before attributing a speedup to backend selection.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--baseline-backend cudnn --optimized-backend flash"
```

The first run profiles the default math-versus-Flash pair. The second profiles cuDNN against Flash. Pin a different explicit backend on each side when you profile. The point is to attribute the gain to backend behavior, not to mixed runtime heuristics.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--baseline-backend cudnn --optimized-backend flash"
```

## Learning Goals
- Compare cuDNN fused SDPA to Flash and math backends on identical shapes.
- Capture Nsight traces per backend to inspect kernel fusion and launch counts.
- Keep regression thresholds per architecture in `expectations_{hardware_key}.json`.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py` | Shared attention microbenchmark. The baseline pins math and the optimized variant pins Flash. Override each side with `--baseline-backend` or `--optimized-backend` (`auto`, `cudnn`, `flash`, or `math`) passed with `--target-extra-arg`. |
| `expectations_{hardware_key}.json` | Current golden timings for the active hardware key. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench --profile minimal
```
- Targets follow the `labs/cudnn_sdpa_bench:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/cudnn_sdpa_bench:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal` times the default pair, math against Flash.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--baseline-backend cudnn --optimized-backend flash"` times pinned cuDNN against pinned Flash.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--baseline-backend math --optimized-backend cudnn"` times the math backend against pinned cuDNN.

## Notes
- Backend selection comes only from `--baseline-backend` and `--optimized-backend`. Environment variables are ignored. Any other override argument, including a bare `--backend`, makes the lab raise an error. The harness then fails the target instead of running the default pair.
- `auto` lets PyTorch choose the kernel. An expectation entry records the override arguments in `provenance.target_override_argv`, but not the backend each side ran. Do not write expectations from an `auto` run or from a run that pins the same backend on both sides.
- Profiling outputs are stored under `artifacts/runs/<run_id>/profiles/bench/labs_cudnn_sdpa_bench/` with harness artifacts in `artifacts/runs/<run_id>/`.
