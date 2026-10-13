# Lab - Blackwell Matmul Suite

## Summary
Ports the four-part Blackwell matmul deep dive into the harness: start with a naive CUDA kernel, then layer pipeline loads, real TMA, and cluster DSMEM broadcasts until you surpass the baseline roofline.

## Problem
This lab exists to answer a very specific Blackwell question: which part of the matmul stack is actually buying the win on this machine? Pipeline staging, TMA, and cluster/DSMEM support do not always move together, so the lab keeps them as separate benchmark targets instead of hiding everything behind one "optimized" label.

## Baseline Path
- naive CUDA matmul kernel
- no TMA or DSMEM cluster help
- useful roofline reference, but not a realistic Blackwell schedule

## Optimized Path
- pipelined staging path
- TMA-enabled path for lower copy/staging overhead
- cluster/DSMEM variants when the hardware and shape make them worthwhile

## Measured Delta
Representative validated results from `artifacts/runs/20260301_1032__bench__profile_none_targets_labs26_recheck/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `blackwell_matmul_pipeline` | `29.254 ms` | `5.045 ms` | `5.80x` | pipeline staging only |
| `blackwell_matmul_tma` | `29.303 ms` | `4.373 ms` | `6.70x` | TMA staging path |
| `blackwell_matmul_cluster` | `29.259 ms` | `16.307 ms` | `1.79x` | cluster/DSMEM path |

The useful reading is that the current local winner is the TMA path, not the cluster path. The cluster target is still valuable because it keeps the DSMEM route benchmarked and verified, but this repo does not pretend it is the latency leader on every shape.

## Profiler Evidence
Use deep-dive runs when you want Nsight evidence for each schedule family:

```bash
python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_pipeline --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_tma --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile deep_dive --single-gpu
```

Keep the targets separate when you analyze them. The point of this lab is to attribute the gain to the schedule family, not to blur TMA and cluster behavior together.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/blackwell_matmul
python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_tma --profile minimal
python labs/blackwell_matmul/run_blackwell_matmul.py --variant tma --size 4096
```

## Learning Goals
- Reproduce the reference matmul trajectory (baseline -> pipelined -> TMA -> cluster).
- Compare PyTorch harness timings against the CUDA extensions while reusing the same shapes.
- Validate kernels on SM100/103 targets and gracefully skip DSMEM-only paths on SM121.
- Capture dual roofline metadata (SM vs TMEM) for every variant.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_blackwell_matmul.py`, `optimized_blackwell_matmul_pipeline.py`, `optimized_blackwell_matmul_tma.py`, `optimized_blackwell_matmul_cluster.py` | Python entrypoints for each stage of the matmul tutorial. |
| `blackwell_benchmarks.py`, `run_blackwell_matmul.py` | Harness adapters and standalone runner for quick sweeps and metadata capture. |
| `grace_blackwell_extension.py`, `grace_blackwell_kernels.cu` | PyTorch extension and CUDA kernels implementing the baseline and optimized kernels. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/blackwell_matmul
python -m cli.aisp bench run --targets labs/blackwell_matmul --profile minimal
```
- Targets follow the `labs/blackwell_matmul:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/blackwell_matmul:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile minimal` delivers higher TFLOP/s than the baseline and emits artifacts under `artifacts/labs_blackwell_matmul*`.
- `python labs/blackwell_matmul/run_blackwell_matmul.py --variant pipeline --size 4096 --roofline-meta artifacts/labs_blackwell_matmul/matmul_meta.csv` saves roofline metadata alongside timings.
- DSM-aware variants error out early on GPUs that lack cluster DSMEM support, preventing misleading results.

## Notes
- `run_blackwell_matmul.py` accepts `--variant baseline|pipeline|tma|cluster` plus `--size` to mirror the blog walkthrough.
- TMA kernels require CUDA 13.0+ and SM100/103 hardware; on GB10 they log a warning and skip execution.
- For a schedule-design companion to this implementation-heavy matmul lab, see `labs/software_pipelining`, which makes ring-buffer reuse and dependency legality explicit in a smaller benchmark and analyzer surface.
