# Lab - Parameterized CUDA Graph Launch

## Summary
A narrow PyTorch-first benchmark pair for parameterized CUDA Graph replay: fixed shapes, fixed device buffers, and fixed graph topology, with request-specific host bindings updated on the executable graph before replay.

## Problem
Normal CUDA Graph capture wants stable addresses. Real serving loops often keep tensor shapes static while rotating through different request slots, scalar values, and output buffers. If the graph is recaptured every time those bindings change, capture and instantiation overhead can erase the benefit.

## Baseline Path
- eager-style warmup followed by per-request graph recapture
- fresh capture and instantiation for each request slot
- same math as the optimized path, but full recapture tax stays in the hot path

## Optimized Path
- one `torch.cuda.CUDAGraph(keep_graph=True)` capture on stable device buffers
- executable memcpy-node parameter updates for the input slot, scalar slot, and output slot
- graph replay without changing shapes, weights, or graph topology

## Measured Delta
Representative local strict result from `artifacts/runs/20260320_parameterized_cuda_graphs_local/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `parameterized_graph_launch` | `0.938 ms` | `0.096 ms` | `9.74x` |

This run was verification-clean and used harness clock locking, but the host was also flagged as virtualized. Treat it as solid local evidence for the parameterized replay technique, not as a publish-grade bare-metal number.

## Profiler Evidence
Use the same target with a profiler-backed run when you want Nsight artifacts instead of only the wall-clock delta:

```bash
python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile deep_dive --single-gpu
```

The thing to look for is unchanged compute and kernels with the recapture overhead removed from the optimized path.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/parameterized_cuda_graphs
python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile none --single-gpu --gpu-sm-clock-mhz 1500
```

## Learning Goals
- Show a clean benchmark pair for graph recapture versus executable-graph parameter mutation.
- Keep the claim narrow: stable-shape PyTorch work with changing request bindings, not generic dynamic-shape graphing.
- Make the measured artifact prove that replay stays correct while per-request graph construction disappears from the hot path.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_parameterized_graph_launch.py`, `optimized_parameterized_graph_launch.py` | Benchmark wrappers for per-request recapture versus parameterized executable-graph replay. |
| `parameterized_cuda_graphs_common.py` | Shared PyTorch block, request-slot buffers, graph capture, memcpy-node mutation, and verification helpers. |
| `__init__.py` | Package export for the shared benchmark classes and config. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/parameterized_cuda_graphs
python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs --profile minimal
```
- Targets follow the `labs/parameterized_cuda_graphs:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/parameterized_cuda_graphs:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench list-targets --chapter labs/parameterized_cuda_graphs` should discover `parameterized_graph_launch`.
- `python -m cli.aisp bench run --targets labs/parameterized_cuda_graphs:parameterized_graph_launch --profile none --single-gpu --gpu-sm-clock-mhz 1500` should keep the optimized path verification-clean and faster than the recapture baseline.
- The optimized path should report executable memcpy-node parameter updates while avoiding per-iteration graph recapture.

## Notes
- This lab overlaps conceptually with `labs/persistent_decode` and `labs/decode_optimization`, but it is intentionally the narrower companion focused on executable-graph parameter mutation in PyTorch.
- The current measured delta is backed by a local virtualized B200 run; replace the artifact when a canonical bare-metal run is collected.
- Optional local shape sweeps use `AISP_PARAMETERIZED_CUDA_GRAPHS_*` environment variables without changing the core fixed-shape benchmark contract.
