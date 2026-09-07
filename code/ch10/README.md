# Chapter 10 - Tensor Core Pipelines & Cluster Features

## Summary
Applies tensor-core friendly scheduling on Blackwell: warp specialization, TMA-powered pipelines, persistent kernels, and thread-block clusters with DSMEM and NVLink-C2C awareness.

## Problem
Chapter 10 is where the repo stops talking about tensor-core scheduling in the abstract and starts proving which pipeline and cluster choices actually matter on Blackwell.

## Baseline Path
- scalar-heavy or launch-heavy kernels that leave tensor cores underfed
- non-persistent pipelines that pay setup cost every iteration
- cluster-disabled variants that show the cost of ignoring DSMEM / multicast hardware

## Optimized Path
- warp-specialized and persistent kernels that keep producer/consumer work separated
- TMA-fed pipelines that reduce staging overhead
- cluster-enabled kernels that exploit DSMEM and multicast when the hardware supports it

## Measured Delta
Representative validated results from `artifacts/runs/20260303_163946__bench__profile_minimal_targets_20/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `cluster_group_single_cta` | `2.203 ms` | `0.031 ms` | `71.42x` |
| `batch` | `10.061 ms` | `0.185 ms` | `54.44x` |

These are good chapter-level "does the optimization concept work?" numbers, not universal hardware ceilings.
`book-after/ch10.md` is centered on intra-kernel pipelines, warp specialization, persistent kernels, and cluster workflows, so the canonical Chapter 10 surface stays anchored on targets such as `double_buffered_pipeline`, `pipeline_3stage`, `warp_specialized_pipeline`, and `cluster_group`.

## Profiler Evidence
Use the harness target directly when you want reproducible Nsight evidence instead of ad hoc scripts:

```bash
python -m cli.aisp bench run --targets ch10:cluster_group_single_cta --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets ch10:batch --profile deep_dive --single-gpu
```

The deep-dive path gives you a concrete before/after pairing for launch count, kernel duration, and memory/cluster behavior.

## Repro Commands
```bash
python -m ch10.compare --profile none
python -m cli.aisp bench list-targets --chapter ch10
python -m cli.aisp bench run --targets ch10 --profile minimal
```

## Learning Goals
- Use warp specialization and cp.async/TMA to keep tensor cores saturated.
- Prototype persistent matmuls that amortize launch overhead across iterations.
- Exercise thread-block clusters with and without DSMEM to understand hardware limits.
- Combine PyTorch, Triton, and CUDA kernels while keeping expectations synchronized.
- Study how [FA4 decode ping-pong and backward TMEM de-aliasing](../labs/flashattention4/colfax_optimization_diaries.md) change MMA issue order and synchronization scope without changing the attention math.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_attention.py`, `optimized_attention.py`, `baseline_flash_attention.py`, `optimized_flash_attention.py`, `analyze_scaling.py` | Attention workloads that span eager, fused, and `torch.compile` paths for modern decoder models. |
| `baseline_batch.py`, `optimized_batch.py`, `baseline_matmul_tcgen05_vs_cublas.py`, `optimized_matmul_tcgen05_vs_cublas.py` | Batch scheduling benchmarks plus a custom-tcgen05-versus-cuBLAS comparison target kept for manual tensor-core reference. |
| `baseline_tcgen05_warp_specialization.py`, `optimized_tcgen05_warp_specialization.py`, `tcgen05_warp_specialized.cu` | Warp-specialized tcgen05 GEMM with dedicated producer/consumer warps. |
| `baseline_tcgen05_warp_specialization_cutlass.py`, `optimized_tcgen05_warp_specialization_cutlass.py`, `tcgen05_warp_specialized_cutlass.cu`, `tcgen05_warpgroup_specialized.cu` | CUTLASS warp-specialized mainloop comparison (1-SM warp-specialized vs 2-SM warpgroup tile). |
| `warpgroup_specialization_demo.py`, `tcgen05_warpgroup_specialized.cu` | Demo of the CUTLASS warpgroup array mainloop using a 2-SM tile. |
| `baseline_double_buffered_pipeline.{py,cu}`, `optimized_double_buffered_pipeline.{py,cu}`, `baseline_tma_2d_pipeline.py`, `optimized_tma_2d_pipeline.py` | Async pipeline samples mixing cp.async, TMA, and manual double buffering. |
| `baseline_cluster_group*.{py,cu}`, `optimized_cluster_group*.{py,cu}`, `cluster_group_common.cuh`, `cluster_group_utils.py` | Clustered kernel suite covering DSMEM-enabled and DSMEM-free thread-block clusters. |
| `baseline_cluster_multicast.py`, `optimized_cluster_multicast.py`, `tma_multicast_baseline.cu`, `tma_multicast_cluster.cu` | Cluster multicast GEMM example (baseline vs cluster multicast) wrapped as CUDA-binary harness benchmarks. |
| `baseline_cooperative_persistent.{py,cu}`, `optimized_cooperative_persistent.{py,cu}`, `baseline_persistent_matmul_tma.py`, `optimized_persistent_matmul_tma.py` | Persistent kernels that keep the iteration loop on-device, contrasting synchronous staging with a two-stage shared-memory pipeline. |
| `baseline_warp_spec_pingpong.{py,cu}`, `optimized_warp_spec_pingpong.{py,cu}`, `baseline_flash_attn_tma_micro_pipeline.{py,cu}`, `optimized_flash_attn_tma_micro_pipeline.{py,cu}`, `baseline_warp_specialized_pipeline*.{py,cu}`, `optimized_warp_specialized_pipeline*.{py,cu}` | Micro-pipeline and warp specialization studies, including explicit producer/compute/consumer warp roles and ping-pong staging. |
| `baseline_warp_specialized_cluster_pipeline.{py,cu}`, `optimized_warp_specialized_cluster_pipeline.{py,cu}` | Thread-block-cluster warp specialization example: a synchronous DSMEM baseline versus a leader-CTA pipeline that stages tiles once per cluster. |
| `compare.py`, `workload_config.py`, `demo_both_examples.sh`, `profile.sh`, `requirements_cufile.txt` | Harness entry, workload dials, demo runner, Nsight automation, and optional cuFile deps. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m ch10.compare
python -m cli.aisp bench list-targets --chapter ch10
python -m cli.aisp bench run --targets ch10 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Expectation baselines live next to each chapter in `expectations_{hardware_key}.json`; refresh with `--update-expectations` after validating new hardware. In portable mode, add `--allow-portable-expectations-update` to write expectation files explicitly.

## Validation Checklist
- Cluster-enabled kernels fail fast on hardware without DSMEM support, while DSMEM-free variants still execute-use this to confirm cluster capability flags.
- `python -m cli.aisp bench run --targets ch10:flash_attention --profile minimal` produces fewer kernel launches and higher achieved FLOP/s than the baseline script.
- `python -m ch10.analyze_scaling` summarizes the chapter's scaling behavior without relying on path surgery.
- `python -m ch10.cufile_gds_example` runs the CUDA memory pipeline and GDS demo, highlighting launch amortization and IO overlap.

## Notes
- `cufile_gds_example.py` demonstrates integrating GPUDirect Storage into tensor-core pipelines for IO-heavy training loops.
- `requirements_cufile.txt` holds the optional `cufile` wheel; install it only on hosts with GPUDirect Storage enabled.
- The CUTLASS-style warp-specialization pair provides a reference implementation aligned with `sm100_mma_array_warpspecialized` for performance comparison.
