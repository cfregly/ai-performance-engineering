# Chapter 3 - System Tuning

## Summary
Captures the host-level changes-NUMA pinning, governor tweaks, container settings, and Kubernetes manifests-that keep GPU workloads fed before kernel-level optimization begins.

## Problem
Chapter 3 is where "the GPU is slow" often turns out to be a host problem. The chapter matters when CPU affinity, container defaults, or orchestration choices are quietly capping the work that later CUDA kernels can ever see.

## Baseline Path
- NUMA-unaware or scheduler-default execution
- untuned container and Kubernetes settings
- host configuration that leaves throughput on the floor before kernels even matter

## Optimized Path
- NUMA pinning and topology-aware process placement
- container and cluster settings that stop starving the GPU
- host-level tuning that is measurable through the same shared harness

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `pinned_prefetch_mlp` | `4.456 ms` | `1.225 ms` | `3.64x` | pinned-memory prefetch removes blocking host staging |
| `gemm` | `0.548 ms` | `0.189 ms` | `2.90x` | comparison GEMM shows how host/runtime launch overhead caps achievable FLOP/s |
| `double_buffered_batch_provisioning` | `1.734 ms` | `1.076 ms` | `1.61x` | double-buffered provisioning overlaps batch copies with compute |

The magnitude is smaller than the headline CUDA chapters, but the lesson is important: host tuning changes are often prerequisite wins, not optional polish.
`gemm` is intentionally a comparison workload for host/runtime overhead, while `rack_prep` is the more chapter-native staged-copy example for locality-aware host preparation. Structured metrics mark `gemm` with `story.comparison_pair=1` and `story.chapter_native_exemplar=0`, and structured story metadata marks it as a supplementary comparison pair with chapter-native targets like `pageable_copy` and `rack_prep`, so downstream reports can keep that distinction explicit.

## Profiler Evidence
For this chapter, pair the harness runtime with host/GPU traces so the bottleneck story stays grounded:

```bash
python -m cli.aisp bench run --targets ch03:pinned_prefetch_mlp --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch03:gemm --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch03:double_buffered_batch_provisioning --profile deep_dive --single-gpu
```

Expected evidence:
- `pinned_prefetch_mlp`: pinned-memory prefetch removes blocking host staging
- `gemm`: lower host overhead around the same kernel without changing the math
- `double_buffered_batch_provisioning`: double buffering reduces batch-provisioning stalls

## Repro Commands
```bash
python -m ch03.compare
python -m cli.aisp bench list-targets --chapter ch03
python -m cli.aisp bench run --targets ch03 --profile minimal
python -m ch03.power_tuning_tool --power-limits 300,350 --iterations 5 --warmup 1
```

## Learning Goals

For packet paths, TCP windows, path MTU and retained packet captures, use the
[network diagnosis tool and guide](network_diagnosis.md). It collects evidence
and runs explicitly selected probes through `aisp tools network-diagnose`.

- Diagnose CPU and memory affinity issues that throttle GPU pipelines.
- Harden Docker and Kubernetes environments for sustained GPU throughput on shared clusters.
- Automate repeatable system tuning via shell scripts so lab machines stay consistent.
- Use comparison workloads like GEMM and rack-prep to quantify host/runtime overhead, locality, and launch latency.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_pageable_copy.py`, `optimized_pageable_copy.py`, `bind_numa_affinity.py`, `numa_topology_script.sh` | Host-transfer and NUMA-adjacent helpers: the benchmark pair covers pageable-vs-pinned async copies, while the scripts handle CPU/GPU socket placement and topology inspection. |
| `baseline_rack_prep.py`, `optimized_rack_prep.py`, `grace_blackwell_topology.py` | Topology-aware staging comparison pair: baseline uses blocking pageable staging, while optimized adds affinity planning plus pinned double-buffered copy/compute overlap. |
| `baseline_pinned_prefetch_mlp.py`, `optimized_pinned_prefetch_mlp.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh` | Pinned-memory prefetch comparison pair plus host/container setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility. |
| `baseline_double_buffered_batch_provisioning.py`, `optimized_double_buffered_batch_provisioning.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` | Batch-provisioning overlap pair plus Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets. |
| `cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh` | Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness. |
| `baseline_gemm.py`, `optimized_gemm.py`, `train.py` | Comparison GEMM + training loops that expose host/runtime launch overhead in measurable FLOP/s without claiming a NUMA-specific kernel optimization. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json` | Harness entry, Python deps, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch03.compare
python -m cli.aisp bench list-targets --chapter ch03
python -m cli.aisp bench run --targets ch03 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch03.mig_mps_tool --device 0` reports the active MIG/MPS state before you change host-level scheduling policy.
- `python -m ch03.power_tuning_tool --power-limits 300,350 --iterations 5 --warmup 1` produces a short perf-per-watt sweep with the harness clock-lock path.
- `python -m ch03.compare` keeps the chapter baseline/optimized tuning pairs runnable through the shared harness.

## Notes
- `cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.
- Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.
