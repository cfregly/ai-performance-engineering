# Lab - CUDA MoE Decode Toolkit

## Summary
Implements mixture-of-experts decode helpers directly in CUDA: decode kernels, KV-transfer overlap/graph variants, router policies, and validation math so you can iterate on Blackwell-friendly pipelines.

## Problem
MoE serving paths usually fail for one of four reasons: decode kernels are too launch-heavy, KV movement is too slow, backend selection is naïve, or routers do too much scalar work. This lab keeps those costs separated so you can see which one actually improved.

## Baseline Path
- eager CUDA helpers for decode, routing, and KV transfer
- good correctness references
- too much overhead in the hot path for steady-state serving

## Optimized Path
- staged decode kernels
- overlapped and graph-assisted KV transfer
- batched direct KV destination writes for serving layouts that colocate compute and KV placement
- backend and router kernels tuned for Blackwell-friendly execution

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `decode_attention` | `0.259 ms` | `0.207 ms` | `1.25x` |
| `kv_transfer` | `1.224 ms` | `1.085 ms` | `1.13x` |
| `kv_transfer_direct` | baseline transfer path | batched direct KV writes | destination-placement + launch-removal variant |
| `kv_transfer_graphs` | `1.224 ms` | `0.315 ms` | `3.88x` |
| `kv_transfer_direct_graphs` | baseline transfer path | direct KV writes + graph replay | destination + launch-removal variant |
| `moe_backend_selection` | `1.747 ms` | `0.308 ms` | `5.67x` |
| `router` | `67.265 ms` | `8.674 ms` | `7.75x` |

That spread is the point of the lab. Not every MoE subsystem gets the same win: overlap-only KV transfer is a modest directional step, direct destination placement removes the copy when the serving layout allows it, batched direct placement removes per-chunk launches, and graphed direct placement removes most of the remaining scheduling overhead. The router/backend work is still where the biggest local payoff is showing up.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/moe_cuda:decode_attention --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/moe_cuda:kv_transfer_direct_graphs --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/moe_cuda:router_vectorized --profile deep_dive --single-gpu
```

Those three targets cover the highest-value slices: decode kernel efficiency, KV movement/orchestration, and router kernel behavior.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_cuda
python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal
python -m cli.aisp bench verify -t labs/moe_cuda:decode_attention
```

## Learning Goals
- Benchmark decode kernels that stage tokens through shared memory and cp.async pipelines.
- Optimize KV-transfer strategies (manual, CUDA Graphs) across NVLink fabrics.
- Prototype routers that understand MoE grouping, locality, and vectorized loads.
- Validate CUDA kernels against Python math models before integrating into serving stacks.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_decode_attention.py`, `optimized_decode_attention.py` | Attention microbenchmarks that validate correctness while optimizing kernel schedules. |
| `baseline_decode_kernel.py`, `optimized_decode_kernel.py`, `decode_kernels.py`, `kernels/` | CUDA kernels and wrappers for the decode core. |
| `baseline_kv_transfer.py`, `baseline_kv_transfer_direct.py`, `baseline_kv_transfer_direct_graphs.py`, `optimized_kv_transfer.py`, `optimized_kv_transfer_direct.py`, `optimized_kv_transfer_graphs.py`, `optimized_kv_transfer_direct_graphs.py` | KV-transfer samples comparing eager copies, overlap, direct destination placement, and CUDA Graph orchestration. |
| `baseline_router.py`, `optimized_router.py`, `optimized_router_vectorized.py` | MoE router logic fit for device execution. |
| `expectations_{hardware_key}.json`, `__init__.py` | Metadata and module exports needed by the harness. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_cuda
python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal
```
- Targets follow the `labs/moe_cuda:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_cuda:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal` runs every baseline/optimized pair and captures NVTX traces.
- `python -m cli.aisp bench verify -t labs/moe_cuda:decode_attention` compares the CUDA path to the math reference and fails loudly if drift is detected.
- KV transfer graphs print latency breakdowns showing overlap improvements relative to the baseline script.

## Notes
- `kernels/` houses the raw CUDA sources split by component; edit schedules there before rebuilding via the harness.
- `optimized_kv_transfer_graphs.py` and `optimized_kv_transfer_direct_graphs.py` emit CUDA Graph captures under `artifacts/` for reproducibility.
