# Chapter 2 - GPU Hardware Architecture

## Summary
Provides architecture awareness tooling for Blackwell-era systems-query SM and memory specs, validate NVLink throughput, and experiment with CPU-GPU coherency so optimizations stay grounded in measured hardware limits.

## Problem
Chapter 2 is the "know the machine first" chapter. The point is not to collect pretty hardware facts; it is to tie optimization decisions to measured fabric, memory, and coherency behavior on the actual target system.

## Baseline Path
- generic transfer paths that do not exploit topology or coherency
- untuned cuBLAS defaults
- hardware assumptions based on specs instead of measured bandwidth/latency

## Optimized Path
- topology-aware transfer and coherency choices
- tuned cuBLAS invocation parameters
- system bring-up driven by measured bandwidth ceilings rather than marketing numbers

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `memory_transfer` | `18.901 ms` | `3.637 ms` | `5.20x` | optimized transfer path fits the actual link behavior |
| `cublas` | `0.590 ms` | `0.114 ms` | `5.17x` | tuned cuBLAS settings match the hardware better |

`grace_coherent_memory` has no current measured delta. The same run reported `22468.299 ms` to `970.890 ms` (`23.14x`) for it, but that run predates the Grace host check and timed an older optimized path that staged copies through pinned memory. Both variants now do the same work each iteration: read the host buffer, compute `2x + 1`, and leave the result in host memory. The baseline stages that data through pageable H2D and D2H copies. The optimized variant has the GPU read and write a pinned host buffer in place over NVLink-C2C, with no copies. The pair needs a fresh GB200 or GB300 run before it can claim a speedup.

`grace_coherent_memory` is only valid on Grace-Blackwell coherent-memory hosts. On PCIe-only Blackwell or other non-Grace systems, the target fails fast with `SKIPPED:` instead of emitting fallback transfer numbers under the wrong benchmark name.

This chapter is the hardware sanity anchor for later claims: if these numbers drift, everything that depends on them deserves scrutiny.

## Profiler Evidence
These are mostly hardware-path benchmarks, so the main evidence is topology, transfer, and kernel traces rather than high-level model metrics:

```bash
python -m cli.aisp bench run --targets ch02:cublas --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch02:memory_transfer --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch02:grace_coherent_memory --profile deep_dive --single-gpu
```

The expected story is:
- `cublas`: better math-mode and launch configuration behavior
- `memory_transfer`: less time lost to the wrong host/device path
- `grace_coherent_memory`: on GB200/GB300 the baseline trace should show a pageable H2D copy, two elementwise kernels, and a pageable D2H copy per iteration, while the optimized trace shows one elementwise kernel that reads and writes host memory and no copies. On non-Grace hosts the benchmark should fail fast with `SKIPPED:`

## Repro Commands
```bash
python -m ch02.compare
python -m ch02.hardware_info
python -m cli.aisp bench list-targets --chapter ch02
python -m cli.aisp bench run --targets ch02 --profile minimal
```

## Learning Goals
- Query and log GPU, CPU, and fabric capabilities before running performance studies.
- Measure NVLink, PCIe, and memory-bandwidth ceilings using purpose-built microbenchmarks.
- Validate Grace-Blackwell coherency paths to know when zero-copy buffers help or hurt.
- Contrast baseline vs optimized cuBLAS invocations to highlight architecture-specific tuning levers.

## Directory Layout
| Path | Description |
| --- | --- |
| `hardware_info.py`, `cpu_gpu_topology_aware.py` | System scanners that record GPU capabilities, NUMA layout, NVLink/NVSwitch connectivity, and affinity hints. |
| `nvlink_c2c_bandwidth_benchmark.py`, `baseline_memory_transfer.py`, `optimized_memory_transfer.py`, `memory_transfer_pcie_demo.cu`, `memory_transfer_nvlink_demo.cu`, `memory_transfer_zero_copy_demo.cu`, `baseline_memory_transfer_multigpu.cu`, `optimized_memory_transfer_multigpu.cu` | Peer-to-peer and zero-copy experiments for quantifying NVLink, PCIe, and coherent memory performance. |
| `cpu_gpu_grace_blackwell_coherency.cu`, `cpu_gpu_grace_blackwell_coherency_sm121` | Grace-Blackwell cache-coherent samples that compare explicit transfers vs shared mappings. |
| `baseline_cublas.py`, `optimized_cublas.py` | cuBLAS GEMM benchmark pair that toggles TF32, tensor op math, and stream affinity to highlight architecture knobs. |
| `compare.py`, `Makefile`, `expectations_{hardware_key}.json` | Harness driver, CUDA build rules, and expectation file for automated pass/fail checks. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch02.compare
python -m cli.aisp bench list-targets --chapter ch02
python -m cli.aisp bench run --targets ch02 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python -m ch02.hardware_info` records the correct device name, SM count, and HBM size for every GPU in the system.
- `python -m ch02.nvlink_c2c_bandwidth_benchmark` reports the host↔device and bidirectional bandwidth table for the active topology.
- Running the coherency sample on GB200/GB300 shows zero-copy benefiting sub-MB transfers while large transfers favor explicit H2D copies; on non-Grace hosts the benchmark should stop immediately with a `SKIPPED:` capability diagnostic.

## Notes
- Grace-only coherency tests require GB200/GB300 nodes; on PCIe-only or discrete-GPU Blackwell hosts the Python benchmark pair fails fast with an explicit unsupported-capability diagnostic.
- `Makefile` builds both CUDA and CPU tools so results can be compared without leaving the chapter.
