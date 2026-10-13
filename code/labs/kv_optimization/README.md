# Lab - KV Cache Optimization

## Summary
Compares a standard FP16 KV cache path against a compressed KV-cache implementation so longer context lengths fit without treating memory reduction as a free lunch.

## Problem
KV cache growth is one of the fastest ways to turn a good inference path into an unusable one. This lab exists to measure how much memory the cache optimization actually gives back, and what latency tradeoff you pay to get it.

## Baseline Path
- standard FP16 KV cache
- simple, high-fidelity, and expensive in HBM
- useful as the correctness and memory reference

## Optimized Path
- compressed KV cache with lower memory footprint
- benchmarked through the same harness path, so the speed/memory tradeoff is explicit
- designed to answer "does the memory saving justify the latency change?" instead of assuming quantization is always a win

## Measured Delta
Current validated expectation-backed B200 result from `labs/kv_optimization/expectations_b200.json`:

| Target | Baseline | Optimized | Measured delta | Memory change |
| --- | ---: | ---: | ---: | ---: |
| `kv_standard` | `3782.365 ms` | `1777.506 ms` | `2.13x` | `49.77%` lower memory |

That run recorded:
- baseline memory: `32916.315 MB`
- optimized memory: `16534.378 MB`

This lab is useful because it makes the speed/memory tradeoff explicit instead of treating KV compression as a free optimization.

## Profiler Evidence
The first thing to trust here is the benchmark pair and its recorded memory delta. When you want deeper attribution, run the same target through the harness with profiling enabled:

```bash
python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu
```

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/kv_optimization
python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile minimal
python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu
```

## Learning Goals
- Measure the latency and memory tradeoff of KV-cache compression instead of optimizing for one metric in isolation.
- Use the shared harness to keep the baseline and optimized cache paths directly comparable.
- Validate that memory savings survive the same contract checks as every other repo benchmark.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_kv_standard.py` | Reference FP16 KV-cache path. |
| `optimized_kv_standard.py` | Compressed KV-cache path used for the optimized benchmark. |
| `expectations_{hardware_key}.json` | Stored speedup and memory-savings baselines for the lab. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/kv_optimization
python -m cli.aisp bench run --targets labs/kv_optimization --profile minimal
```
- Targets follow the `labs/kv_optimization:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/kv_optimization:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile minimal` reports both latency and memory deltas for the pair.
- The optimized path should reduce memory materially without violating the benchmark contract or correctness checks.
- `python -m cli.aisp bench run --targets labs/kv_optimization:kv_standard --profile deep_dive --single-gpu` produces profiler artifacts for the same measured path.

## Notes
- This README is now generator-owned; update the source of truth in `core/scripts/refresh_readmes.py`, not the rendered file.
- The current public numbers come from the stored expectation baseline because there is no newer canonical tier-1 history artifact for this lab yet.
