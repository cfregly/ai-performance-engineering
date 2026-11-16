# Chapter 12 - CUDA Graphs & Dynamic Workloads

## Summary

Covers CUDA Graph replay, conditional execution, dynamic parallelism, and work
queues that can reduce repeated host submission work. The chapter primarily
implements single-GPU graph and work-queue examples. The native communication
lab adds rank-local NCCL graph capture across compute and communication streams.

Start with [CUDA Graphs: define, instantiate, and replay](cuda_graph_lifecycle.md)
for operation nodes and dependencies, CPU host functions, copies and memset,
graph allocation/free, external synchronization, child graphs, and their actual
repository coverage. The guide distinguishes template creation from executable
instantiation and replay, explains flag and update choices, and separates a DOT
dependency drawing from a measured timeline. Repeated launches of one
`cudaGraphExec_t` serialize with that same executable. A launch stream still
permits the graph's independent internal branches to run concurrently.

## Problem
Chapter 12 is where launch overhead and dynamic work management have to justify themselves with measured wins. The useful question is not "can we capture a graph?" but "which graph or dynamic-work techniques actually reduce the real runtime once correctness and workload shape stay fixed?"

## Baseline Path
- eager launches and more CPU-visible scheduling work
- less reuse of graph capture or GPU-resident work management
- easy to inspect, but often too expensive for irregular steady-state workloads

## Optimized Path
- CUDA Graph replay where the steady-state workload is stable enough
- fused or GPU-resident queueing/dispatch where it actually removes launch overhead
- measured through the shared harness instead of hand-timed one-off scripts

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `cuda_graphs` | `529.532 ms` | `125.874 ms` | `4.21x` | graph replay instead of repeated eager launch overhead |
| `kernel_fusion` | `1.776 ms` | `0.654 ms` | `2.72x` | fewer launches through fused graph-friendly execution |
| `work_queue` | `2.100 ms` | `0.442 ms` | `4.75x` | GPU-resident work queue path |

This chapter is useful because it separates "graphs help" from "graphs help on a workload that is actually stable enough to benefit." The work-queue target also keeps the chapter from being only about graph replay.

## Profiler Evidence
Use deep-dive runs when you want hard evidence for launch reduction and work scheduling changes:

```bash
python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch12:kernel_fusion --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch12:work_queue --profile deep_dive --single-gpu
```

Those targets answer slightly different questions:
- `cuda_graphs`: replay payoff
- `kernel_fusion`: launch-count reduction
- `work_queue`: GPU-side dynamic dispatch effectiveness

## Repro Commands
```bash
python -m ch12.compare
python -m cli.aisp bench list-targets --chapter ch12
python -m cli.aisp bench run --targets ch12 --profile minimal
python -m cli.aisp bench run --targets ch12:cuda_graphs --profile deep_dive --single-gpu
```

## Learning Goals
- Capture steady-state workloads into CUDA Graphs and study the delta vs eager launches.
- Map operation dependencies and separate definition, instantiation, and replay costs.
- Explain executable reuse, independent instances, parameter updates, and resource lifetimes.
- Use conditional nodes and graph memory pools for adaptive pipelines.
- Experiment with device-side launches (dynamic parallelism) to reduce CPU involvement.
- Implement GPU-resident work queues and uneven partition schedulers.

## Graph Replay with Distributed Communication

The [common halo-exchange lab](../labs/multi_gpu_communication/README.md) connects
capture to a repeated compute/communication schedule. Keep buffer addresses,
halo parity, and communication order consistent across replay. Capture a complete
buffer cycle or select the graph for the current parity. Make convergence checks
and rank control flow agree so one rank cannot leave its peers waiting.
The implemented solver uses fixed iteration counts. Each rank captures its own
two-step A → B → A graph and a separate odd tail on its local GPU. This is not a
single global graph spanning MPI processes, and the pair/tail executables are
not a parent/child-graph hierarchy.
[NCCL graph capture](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)
requires matching capture and replay participation. The
[group guide](../ch04/nccl_groups.md) explains why groups do not replace stream
dependencies.

Chapter 4 already has a
[captured CUDA symmetric-memory ring](../ch04/symmetric_memory_example.py), with
separate publication and consumption barriers. That ring uses the CUDA
symmetric-memory backend. It is distinct from capturing NCCL halo exchange.
Graph setup and stable-buffer management add cost, so report capture separately
from replay and keep eager launches when too few iterations amortize that work.
Check CPU API activity and GPU intervals. Replay targets launch, driver, and
scheduling overhead around short kernels rather than making unchanged kernel
math inherently faster. CPU-side norm checks or MPI reductions still need
completion and can remove the intended savings. Follow the
[graph exercises](../labs/multi_gpu_communication/TASKS.md#5-capture-a-stable-schedule)
to inspect capture dependencies, parity, flags, and evidence limits.

## Directory Layout
| Path | Description |
| --- | --- |
| [`cuda_graph_lifecycle.md`](cuda_graph_lifecycle.md) | Graph operation families, template/executable lifecycle, capture fork/join, concurrency, flags, updates, and links to implemented examples. |
| `baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu` | Graph capture demos that evolve from simple replay to conditional and DSM-aware execution, including a supplementary host-dispatch-versus-device-branching conditional graph demo. |
| `baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py` | Launch- and bandwidth-focused studies illustrating how graphs reduce CPU overhead. |
| `baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu`, `dynamic_parallelism_sm121/` | Device-side launch samples showing when dynamic parallelism helps or hurts. |
| `baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh` | GPU work queues for irregular batch sizes, including NVTX instrumentation. |
| `baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu` | Uneven workload partitioners that rebalance CTA assignments at runtime. |
| `baseline_kernel_fusion.py`, `optimized_kernel_fusion.py`, `kernel_fusion_cuda_demo.cu` | Kernel fusion exercises within graph capture so you can remove CPU synchronization entirely. (`kernel_fusion_cuda_demo.cu` is a standalone tool; not a benchmark target.) |
| `compare.py`, `cuda_extensions/`, `expectations_{hardware_key}.json` | Harness entry, extension stubs, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch12.compare
python -m cli.aisp bench list-targets --chapter ch12
python -m cli.aisp bench run --targets ch12 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python optimized_cuda_graphs.py --iterations 100` should report lower wall-clock time than the baseline while matching outputs.
- Device-side dynamic parallelism samples emit warnings on unsupported hardware, ensuring you only trust data from GPUs with the feature enabled.
- `python optimized_work_queue.py --trace` exposes balanced dequeue times across CTAs when compared to the baseline's stragglers.

## Notes
- `cuda_graphs_workload.cuh` holds the standalone graph example's stage parameters and scalar output reference.
- `helper_*.cu` files contain host/device glue for the dynamic-parallelism case studies-copy them when bootstrapping new experiments.
- `graph_conditional_runtime` is the canonical conditional-node benchmark. Keep `cuda_graphs_conditional` as a supplementary demo rather than a canonical speedup target.
- For a smaller supporting lab that keeps Chapter 12's CUDA Graph story primary while adding fused reduction and padding-aware projection examples, see `labs/training_hotpath`.
