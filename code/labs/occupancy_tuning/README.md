# Lab - Triton Occupancy & Schedule Sweep

## Summary
Sweeps Triton matmul schedules for ProtonNet-style workloads on Blackwell, comparing the baseline schedule against optimized block/warp dimensions and reporting how each choice affects occupancy and FLOP/s.

## Problem
Occupancy work is easy to oversell. This lab exists to measure schedule choices directly and show whether better resident work actually lands a throughput win on the same matmul workload.

## Baseline Path
- one baseline Triton schedule
- stable correctness and a clean occupancy reference
- not tuned for this GPU/shape family

## Optimized Path
- curated block/warp schedule variants
- measured through the same harness contract
- designed to answer "which schedule is actually best here?" instead of assuming bigger blocks win

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `proton_matmul` baseline | `0.251 ms` | `0.197 ms` (`bm64_bn256_bk32`) | `1.28x` |
| `proton_matmul` baseline | `0.251 ms` | `0.206 ms` (`bm128_bn256_bk64`) | `1.22x` |

The lab is valuable because it keeps the schedule sweep honest. The win is real, but it is a schedule-selection win, not magic.

Keep `proton_matmul_bm64_bn64_bk32_nw2` as a supplementary local-contract schedule benchmark. It also remains a supplementary comparison schedule benchmark for low-warp occupancy and Proton-vs-Nsight agreement checks, while the canonical speed claims stay on `proton_matmul`, `proton_matmul_bm64_bn256_bk32`, `proton_matmul_bm128_bn128_bk32_nw8`, and `proton_matmul_bm128_bn256_bk64`.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/occupancy_tuning:proton_matmul --profile deep_dive --single-gpu
python -m labs.occupancy_tuning.sweep_schedules --csv artifacts/occupancy_tuning.csv
```

Use the deep-dive harness run for Nsight evidence and the sweep script when you want to explore candidate schedules before promoting one into the benchmark pair.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/occupancy_tuning
python -m cli.aisp bench run --targets labs/occupancy_tuning:proton_matmul --profile minimal
python -m labs.occupancy_tuning.sweep_schedules --csv artifacts/occupancy_tuning.csv
```

## Learning Goals
- Measure how Triton block sizes map to achieved occupancy on SM100/121.
- Autogenerate schedule sweeps and record best-performing parameter sets.
- Compare baseline schedules to curated optimized variants packaged with the lab.
- Integrate selected schedules into harness targets for regression tracking.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_proton_matmul.py`, `optimized_proton_matmul_bm128_bn128_bk32_nw8.py`, `optimized_proton_matmul_bm64_bn64_bk32_nw2.py`, `optimized_proton_matmul_bm64_bn256_bk32.py`, `optimized_proton_matmul_bm128_bn256_bk64.py` | Baseline and optimized Triton schedules covering multiple block/warp configurations. |
| `triton_matmul.py`, `triton_matmul_schedules.py` | Core Triton kernel and schedule definitions used by the harness. |
| `sweep_schedules.py` | Utility for enumerating candidate schedules and logging throughput/occupancy to `artifacts/`. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/occupancy_tuning
python -m cli.aisp bench run --targets labs/occupancy_tuning --profile minimal
```
- Targets follow the `labs/occupancy_tuning:<workload>` naming convention listed by `list-targets`.
- Each harness target binds one named schedule. Use `python -m labs.occupancy_tuning.sweep_schedules --schedule <name> --csv <path>` to explore schedule knobs before adding a new target.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/occupancy_tuning --profile minimal` executes every schedule defined in the lab.
- `python -m labs.occupancy_tuning.sweep_schedules --csv artifacts/occupancy_tuning.csv` enumerates schedules and records every result; it exits nonzero if any schedule fails.
- `python -m cli.aisp bench run --targets labs/occupancy_tuning:proton_matmul --profile minimal` runs the harness's baseline-versus-optimized output verification on CUDA.

## Notes
- Add new schedules to `triton_matmul_schedules.py` and regenerate the harness targets by rerunning the sweep script.
- `proton_matmul_bm64_bn64_bk32_nw2` is a supplementary local-contract schedule benchmark and supplementary comparison schedule benchmark; use the larger winning schedules when you want the lab's canonical speed claims.
- `expectations_{hardware_key}.json` records FLOP/s per schedule so improvements show up in CI.
