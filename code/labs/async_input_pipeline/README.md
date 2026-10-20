# Lab - Async Input Pipeline

## Summary
Compares a blocking input path to an overlapped asynchronous staging path so you can see whether host/device feeding is the bottleneck on this workload.

## Problem
Host-side staging can make a GPU workload look slower even when the kernel is fine. This lab keeps the input path measurable instead of letting pipeline overhead hide inside model-level numbers.

## Baseline Path
- synchronous input preparation and transfer
- simple end-to-end reference
- intentionally leaves overlap on the table

## Optimized Path
- asynchronous input staging
- overlaps host/device preparation with compute
- same benchmark contract, but less visible pipeline stall

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `async_input_pipeline` | `135.105 ms` | `88.690 ms` | `1.52x` |

Earlier exploratory runs showed larger numbers, but the strict rerun is the one worth publishing. This lab is about making overlap measurable under the harness contract, not about keeping the biggest scalar.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile deep_dive --single-gpu
```

Nsight is useful here because the overlap story should show up directly in the timeline: less host-visible stall, not just a smaller latency number.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/async_input_pipeline
python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile minimal
```

## Learning Goals
- Make input staging cost visible under the same harness contract as the rest of the repo.
- Show when async overlap matters and when it does not.
- Keep host-side data movement from masquerading as a kernel optimization problem.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_async_input_pipeline.py`, `optimized_async_input_pipeline.py` | Benchmark pair for blocking vs asynchronous staging. |
| `expectations_{hardware_key}.json` | Regression thresholds for the async pipeline pair. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/async_input_pipeline
python -m cli.aisp bench run --targets labs/async_input_pipeline --profile minimal
```
- Targets follow the `labs/async_input_pipeline:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/async_input_pipeline:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/async_input_pipeline:async_input_pipeline --profile minimal` should show lower end-to-end latency for the overlapped path on this host.
- Deep-dive runs should show the optimized path reducing host-visible stall rather than changing the math workload.

## Notes
- This is an end-to-end pipeline lab, so the value is in the timeline and total latency, not just kernel-local timing.
- Keep this lab as the primary copy-stream overlap example. `labs/training_hotpath` reuses that concept by reference and only adds the missing reduction and padding-aware cases.
