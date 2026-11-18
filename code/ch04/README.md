# Chapter 4 - Distributed Communication & Multi-GPU Distribution

## Summary
Demonstrates how to scale training and inference across multiple Blackwell GPUs with NVLink/NVSwitch fabric awareness, NCCL tuning, NVSHMEM collectives, and symmetric memory patterns. The current repo chapter emphasizes NCCL/NVSHMEM-style communication overlap; the broader Magnum IO + NIXL connector story from the manuscript is only partially represented here.

## Problem
Chapter 4 is where multi-GPU claims have to survive contact with real communication cost. The useful question is not "can this scale?" but "which overlap, fusion, and topology choices actually move the latency or throughput needle under the shared harness?"

## Baseline Path
- naive or minimally overlapped communication
- CPU-visible coordination where device-driven orchestration would be cheaper
- topology-agnostic execution that pays the full cost of bad placement

## Optimized Path
- explicit overlap between compute and communication
- fusion and pre-staging to reduce collective overhead
- topology-aware or NVSHMEM/symmetric-memory variants where the fabric is the bottleneck

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta | What changed |
| --- | ---: | ---: | ---: | --- |
| `gradient_fusion` | `3.791 ms` | `0.055 ms` | `68.83x` | fused gradient reduction path |
| `dataparallel` | `7.601 ms` | `0.968 ms` | `7.86x` | direct GPU execution over DataParallel overhead |
| `grace_blackwell_locality` | `5.082 ms` | `0.317 ms` | `16.01x` | locality-aware placement |
| `bandwidth_benchmark_suite` | `4.704 ms` | `2.685 ms` | `1.75x` | cleaner communication path |

The chapter has both huge "remove obvious framework overhead" wins and smaller communication-optimization wins. Those should not be read as one uniform scaling story.

## Profiler Evidence
Use deep-dive harness runs when you want actual overlap evidence rather than only before/after runtime:

```bash
python -m cli.aisp bench run --targets ch04:gradient_fusion --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch04:dataparallel --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch04:grace_blackwell_locality --profile deep_dive --single-gpu
```

The expected profiler story differs by target:
- `gradient_fusion`: fewer communication phases and less launch fragmentation
- `dataparallel`: elimination of framework-side fan-out/fan-in overhead
- `grace_blackwell_locality`: better data placement and lower locality penalties

## Repro Commands
```bash
python -m ch04.compare
python -m cli.aisp bench list-targets --chapter ch04
python -m cli.aisp bench run --targets ch04 --profile minimal
python -m cli.aisp bench run --targets ch04:disaggregated_multigpu --profile none
python -m cli.aisp bench run --targets ch04:gradient_fusion --profile deep_dive --single-gpu
python -m cli.aisp tools ch04-nixl-tier-handoff -- --mode probe --json
```

## Learning Goals

Use the [collective diagnosis tool](collective_diagnosis.md) to distinguish late
rank arrival, contention and lost overlap. Pair it with the
[fabric diagnostics](../cluster/fabric/diagnostics.md) and
[transport isolation tool](../cluster/fabric/transport.md) when the collective
microbenchmark itself is slow.

- Benchmark data-parallel and tensor-parallel training loops with and without overlap.
- Quantify NVLink bandwidth and topology effects when mixing local and disaggregated GPUs.
- Experiment with NVSHMEM pipelines to reduce host involvement in GPU synchronization.
- Adopt symmetric memory pools to simplify KV-cache replication and optimizer state sharding.

## NCCL Groups and Neighbor Exchange

Use the [NCCL group guide](nccl_groups.md) and
[common halo-exchange lab](../labs/multi_gpu_communication/README.md) to connect
native `ncclGroupStart`/`ncclGroupEnd` with PyTorch's
`batch_isend_irecv`. Grouping lets coupled sends and receives progress together
and can reduce launch overhead. It adds responsibility for matching peers,
counts, dtypes, and operation order. Post the complete exchange before waiting.
Keep dependent computation outside the group, and use stream completion before
consuming or reusing its buffers. `ncclGroupEnd` is not proof of GPU completion.
See the official [group](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
and [P2P](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html) semantics.

This builds on existing grouped paths. The
[pipeline helper](pipeline_parallel_common.py) batches opposing neighbor
transfers, and the [symmetric-memory comparison](symmetric_memory_example.py)
uses batched NCCL P2P for its traditional ring. Grouping alone does not establish
compute/communication overlap. Use independent work, explicit stream dependencies,
and a GPU trace to check whether communication is hidden. A serialized exchange
is sufficient when no useful independent work remains.

Use the [NVSHMEM device lab](../labs/nvshmem_device/README.md) to continue from
host-enqueued exchange to scalar GPU puts, cooperative block transfers, and
fused device loops. Its five modes check complete outputs. Ready signals,
consumed acknowledgments, and a checked one-CTA launch make the synchronization
cost and reuse rules explicit.

## NCCL Symmetric-Memory AllReduce

The native symmetric-memory benchmark keeps one out-of-place float32 sum
AllReduce fixed while changing only buffer allocation and registration:

| Arm | Send and receive buffer path | Purpose |
| --- | --- | --- |
| `cuda_malloc` | `cudaMalloc` | Control used by the harness baseline. |
| `nccl_mem_alloc` | `ncclMemAlloc` | Isolates NCCL-compatible allocation. |
| `comm_register` | `ncclMemAlloc` plus `ncclCommRegister` | Isolates ordinary communicator registration. |
| `window_register` | `ncclMemAlloc` plus collective `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)` | Candidate used by the harness optimized path. |

Native results record how many ranks returned non-null send and receive
registration handles. Unregistered arms report zero. Registered arms must report
every rank for both buffers.

The build and runtime both require NCCL 2.27.3 or newer. The runtime version
must exactly match the build headers. The build receipt records the selected
`libnccl.so.2` path and SHA256. Before MPI starts, the runner checks `ldd`
against that exact file. Every MPI rank resolves each NCCL API used by the study
with `dlsym`, checks its canonical library path with `dladdr`, and reports the
runtime version. User-supplied `LD_PRELOAD` must be empty when the runner starts.
An unprofiled rank rejects a preload added by launcher arguments. The per-rank
profile wrapper also rejects one before it starts Nsight. Nsight may then inject
its own tooling library into the target. The runner marks only that target as
profiled, and every NCCL API still has to resolve from the receipted library.
The runner also requires one distinct GPU per MPI rank, a single shared-memory node, and
directed CUDA peer access between every selected GPU pair. It does not fall
back when any gate is unavailable. Every rank checks every output element.
Setup, warmup, output checking, artifact writing, and cleanup stay outside the
per-collective samples. Each timed sample is the maximum CUDA-event elapsed
time across ranks from an event pair around exactly one `ncclAllReduce`.

Build the native binary, then run a small two-GPU correctness smoke:

```bash
make -C code/ch04 nccl_symmetric_memory_native
CUDA_VISIBLE_DEVICES=0,1 python code/ch04/run_nccl_symmetric_memory.py \
  --all-arms --ranks 2 --elements 1048576 --iterations 2 --warmup 1
```

If a newer NCCL is installed outside `/usr`, point the build at that prefix:

```bash
make -C code/ch04 NCCL_HOME=/path/to/nccl nccl_symmetric_memory_native
```

The runner prepends the receipted NCCL directories only in the MPI child's
`LD_LIBRARY_PATH`. It does not change the parent shell or the host's default
linker configuration. It reserves the result path before creating process
artifacts, so a duplicate invocation cannot replace the active owner's result.

A passing smoke proves only that the selected host ran all four arms and passed the
complete-output checks. The direct runner defaults to 16,777,216 elements, 20
timed samples, five warmups, and all visible GPUs. It retains raw `MPI_MAX`
samples, summary statistics, build provenance, process logs, and complete
float32 output files under `code/artifacts/nccl_symmetric_memory/` unless
`--json-out` selects another location.

### Capture an AllReduce on every MPI rank

From the repository root on the allocated node, build with NVTX and run a fresh
diagnostic for each arm and profiler you want to inspect. Use the same NCCL
prefix as the timing build. The commands below capture the window arm on two GPUs.

```bash
make -B -C code/ch04 NVTX_ENABLED=1 nccl_symmetric_memory_native
CUDA_VISIBLE_DEVICES=0,1 python code/ch04/run_nccl_symmetric_memory.py \
  --arm window_register --ranks 2 --elements 1048576 --iterations 2 --warmup 1 --profile nsys
CUDA_VISIBLE_DEVICES=0,1 python code/ch04/run_nccl_symmetric_memory.py \
  --arm window_register --ranks 2 --elements 1048576 --iterations 2 --warmup 1 --profile ncu
```

The runner uses the executable on `PATH`. Nsys traces CUDA and NVTX throughout
the process. NCU records GPU duration over each steady AllReduce NVTX range.
All ranks participate in TCP lockstep during range replay. The runner checks
the installed NCU help before MPI starts and retains that help and the version
when the coordination controls are missing. Nsight's own target injection is
allowed only after the launcher and rank wrapper have rejected a user-supplied
preload. The native binding check still requires every NCCL API used by the
study to resolve from the receipted library. Inspect the rank logs for the
loaded-library diagnostic. NVIDIA documents the NCU controls in
the [Nsight Compute MPI profiling guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/#multi-process-support).

Each profile summary identifies the command, GPU mapping, full-output check,
and one nonempty native report with its SHA256 and logs per rank. Open the
reports in Nsight to inspect which kernels actually ran. Collect performance
samples in a separate invocation with profiling omitted.

Run the comparable control and candidate through the harness from `code/`:

```bash
python -m cli.aisp bench run --targets ch04:nccl_symmetric_memory --profile none \
  --suite-timeout 0 --execution-timeout-seconds 0
```

This command uses the repository's owned uncapped MPI process path. It does not
by itself provide publication-grade performance evidence. A performance claim
still needs locked clocks, application-clock provenance, repeated interleaving,
and profiler or counter evidence for the mechanism. See the official NCCL
[buffer registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)
and [communicator API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
documentation for the window lifetime and collective registration contract.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dataparallel.py`, `optimized_dataparallel.py` | Single-GPU DataParallel anti-pattern vs direct GPU execution. |
| `baseline_dataparallel_multigpu.py`, `optimized_dataparallel_multigpu.py` | Multi-GPU DataParallel vs manual gradient reduction with pre-staged shards. |
| `baseline_no_overlap.py`, `optimized_no_overlap.py` | Strict distributed DDP overlap benchmarks; they now require `torchrun` plus `>=2` GPUs and fail fast with `SKIPPED:` when launched on unsupported hosts. |
| `baseline_pcie_staging.py`, `optimized_pcie_staging.py`, `baseline_nvlink_topology_aware.py`, `optimized_nvlink_topology_aware.py`, `baseline_nvlink_multigpu.py`, `optimized_nvlink_multigpu.py`, `baseline_nvlink_topology_aware_multigpu.py`, `optimized_nvlink_topology_aware_multigpu.py` | PCIe host-staging comparison pair plus NVLink exercises for validating peer bandwidth and topology effects (single- and multi-GPU). |
| `baseline_nixl_tier_handoff.py`, `optimized_nixl_tier_handoff.py`, `nixl_tier_handoff_tool.py` | Chapter-local NIXL-style memory-tier handoff analogue built on the same selected-block transport workload used by the communication-stack lab, with both benchmark and probe entrypoints. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py`, `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py` | Continuous-batching comparisons for request pooling and scheduling. |
| `baseline_disaggregated.py`, `optimized_disaggregated.py`, `baseline_disaggregated_multigpu.py`, `optimized_disaggregated_multigpu.py` | Shared-model versus separate prefill/decode model-storage paths. The multi-GPU pair runs both complete phases on every rank and reduces both outputs over the common WORLD group. |
| `baseline_gradient_compression_fp16.py`, `optimized_gradient_compression_fp16.py`, `baseline_gradient_compression_int8.py`, `optimized_gradient_compression_int8.py`, `baseline_gradient_compression_fp16_multigpu.py`, `optimized_gradient_compression_fp16_multigpu.py`, `baseline_gradient_compression_int8_multigpu.py`, `optimized_gradient_compression_int8_multigpu.py` | Gradient compression all-reduce benchmarks comparing small-bucket vs full-buffer compression (single GPU and multi-GPU FP16/INT8 paths). |
| `baseline_gradient_compression_fp16_comm_only.py`, `optimized_gradient_compression_fp16_comm_only.py`, `baseline_gradient_compression_int8_comm_only.py`, `optimized_gradient_compression_int8_comm_only.py`, `baseline_gradient_compression_fp16_comm_only_multigpu.py`, `optimized_gradient_compression_fp16_comm_only_multigpu.py`, `baseline_gradient_compression_int8_comm_only_multigpu.py`, `optimized_gradient_compression_int8_comm_only_multigpu.py` | Communication-only gradient compression benchmarks with pre-quantized buffers (single GPU and multi-GPU FP16/INT8 paths). |
| `baseline_pipeline_parallel.py`, `optimized_pipeline_parallel_1f1b.py`, `baseline_tensor_parallel.py`, `optimized_tensor_parallel_async.py`, `baseline_torchcomms.py`, `optimized_torchcomms.py`, `baseline_pipeline_parallel_multigpu.py`, `optimized_pipeline_parallel_multigpu_1f1b.py`, `baseline_tensor_parallel_multigpu.py`, `optimized_tensor_parallel_multigpu.py`, `baseline_tensor_parallel_allgather_multigpu.py`, `optimized_tensor_parallel_allgather_multigpu.py`, `baseline_torchcomms_multigpu.py`, `optimized_torchcomms_multigpu.py` | Pipeline/tensor-parallel and torchcomms overlap studies (single- and multi-GPU). |
| `baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py`, `baseline_nvshmem_training_example_multigpu.py`, `optimized_nvshmem_training_example_multigpu.py` | NVSHMEM/symmetric-memory samples that require real multi-GPU launch plus symmetric-memory support; unsupported hosts fail fast with `SKIPPED:`. |
| `ddp_worker.py`, `nvshmem_worker.py`, `disaggregated_multigpu_worker.py`, `disaggregated_multigpu_result.py` | Dedicated `torchrun` workers and fresh full-rank result transport used by the harness. Harness-discovered baseline/optimized modules stay import-only. |
| `baseline_symmetric_memory_multigpu.py`, `optimized_symmetric_memory_multigpu.py` | Named ring pair: NCCL P2P baseline versus an explicit PyTorch `CUDA` symmetric-memory backend that replays one captured graph containing each peer publication and both fences. The optimized receive is the published local symmetric slot itself, with its final parity slot cloned only after timing for verification. Warmup and graph capture are outside steady-state timing and capture failure is fatal. The public ring helper retains its `NVSHMEM` default and NCCL fences; backend mismatch fails instead of falling back. |
| `baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py`, `baseline_symmetric_memory_perf_multigpu.py`, `optimized_symmetric_memory_perf_multigpu.py` | Symmetric memory utilities and perf probes for KV cache and optimizer shards. |
| `baseline_nccl_symmetric_memory.py`, `optimized_nccl_symmetric_memory.py`, `nccl_symmetric_memory_benchmark.py`, `run_nccl_symmetric_memory.py`, `nccl_symmetric_memory_native.cu` | Native NCCL 2.27.3 control versus symmetric-window AllReduce pair, with two intermediate attribution arms in the direct runner. |
| `compare.py`, `requirements.txt`, `expectations_{hardware_key}.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py` | Harness driver plus standalone NCCL/NVLink sweepers for topology bring-up. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch04.compare
python -m cli.aisp bench list-targets --chapter ch04
python -m cli.aisp bench run --targets ch04 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- `python compare.py --examples dataparallel_multigpu` shows the optimized pair overlapping compute and communication with lower latency.
- `python -m cli.aisp bench run --targets ch04:disaggregated_multigpu --profile none` launches two local worker ranks, reports CUDA-event iteration timing from rank 0, and verifies both ranks' complete prefill-plus-decode outputs.
- `python -m cli.aisp bench run --targets ch04:nixl_tier_handoff --profile minimal --single-gpu` verifies the Chapter 4 NIXL-style tier handoff analogue on the same selected-block workload.
- `python -m cli.aisp bench run --targets ch04:nccl_symmetric_memory --profile none --suite-timeout 0 --execution-timeout-seconds 0` compares the ordinary allocation control with the symmetric-window candidate and retains complete native outputs.
- `python -m cli.aisp bench run --targets ch04:bandwidth_benchmark_suite_multigpu --profile minimal` surfaces >=250 GB/s links on connected GPU pairs and highlights any slow hops on a host with >=2 visible GPUs.
- NVSHMEM samples now run only on real multi-GPU symmetric-memory hosts; unsupported environments fail fast with `SKIPPED:` instead of publishing fallback timings.

## Notes
- `disaggregated_multigpu` teaches separate phase model/storage paths and collective handling. Both phases run on every rank in one WORLD group; it does not model dedicated prefill/decode GPU pools or remote KV placement.
- `symmetric_memory_*` helpers hold user-space allocators for pooling KV-cache lines across GPUs without NVSwitch penalties.
- Use `nccl_blackwell_config.py` to seed NCCL env vars (min NRings, IB mapping) before launching multi-node tests.
