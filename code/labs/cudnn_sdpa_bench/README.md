# Lab - cuDNN SDPA Bench

## Summary
Microbenchmarks cuDNN fused scaled-dot-product attention against Flash and math backends with explicit CLI backend selection.

## Problem
Attention backend choices are often treated as an implementation detail. This lab exists to keep that choice explicit and benchmarked so you can tell whether cuDNN, Flash, or the math path is actually the right answer for this exact shape family.

## Baseline Path
- attention path with conservative backend selection
- stable reference for correctness and shape coverage
- useful when fused paths are unavailable or unstable

## Optimized Path
- fused SDPA backend path
- same shapes and validation contract
- tuned to answer "does backend choice alone move the result?" rather than mixing in unrelated changes

## Historical Delta (backend attribution unverified)
Earlier representative result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `flash_sdp` | `0.345 ms` | `0.282 ms` | `1.22x` |

The earlier implementation allowed a requested cuDNN run to fall back to Flash. This table therefore does not establish a cuDNN-versus-Flash delta. Explicit backend requests are now pinned and fail if unavailable; only `auto` permits fallback. Fresh GPU verification and profiler evidence are required before attributing a speedup to backend selection.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile deep_dive --single-gpu --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"
```

Keep the backend fixed per run when you profile. The point is to attribute the gain to backend behavior, not to mixed runtime heuristics.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"
```

## Learning Goals
- Compare cuDNN fused SDPA to Flash and math backends on identical shapes.
- Capture Nsight traces per backend to inspect kernel fusion and launch counts.
- Keep regression thresholds per architecture in `expectations_{hardware_key}.json`.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py` | Shared attention microbenchmarks; backend chosen via `--backend {auto,cudnn,flash,math}` passed with `--target-extra-arg`. |
| `expectations_{hardware_key}.json` | Current golden timings for the active hardware key. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench --profile minimal
```
- Targets follow the `labs/cudnn_sdpa_bench:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/cudnn_sdpa_bench:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"` captures cuDNN with Nsight traces.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"` compares the Flash path against cuDNN.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend math"` sanity-checks the math backend where fused kernels are unsupported.

## Notes
- Backend selection is CLI-only; environment variables are intentionally ignored.
- Profiling outputs are stored under `artifacts/runs/<run_id>/profiles/bench/labs_cudnn_sdpa_bench/` with harness artifacts in `artifacts/runs/<run_id>/`.
