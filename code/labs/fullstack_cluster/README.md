# Lab - Full-Stack Blackwell Cluster

## Summary
Replays the entire performance-engineering arc as scenarios: from system prep to streaming inference, plus the original cluster GEMM CUDA kernels wired into the harness.

## Problem
This lab is where the repo stops being a pile of isolated kernels and starts behaving like a system story. The important question is whether the end-to-end scenario kernels still show the same directional wins once you put them back into a larger flow.

## Baseline Path
- scenario kernels without Blackwell-specific cluster tuning
- useful for a stable reference, but not a good steady-state throughput path
- keeps the extension and harness wiring honest

## Optimized Path
- optimized cluster GEMM variants
- tcgen05 route where available
- same harness contract and validation as the rest of the repo
- canonical hybrid-EP comparisons now keep the same default routing mode, with topology-aware routing available only through an explicit override

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `cluster_gemm` | `29.213 ms` | `4.583 ms` | `6.37x` |
| `cluster_gemm_tcgen05` | `0.240 ms` | `0.230 ms` | `1.04x` |

The useful split here is that `cluster_gemm` demonstrates the big end-to-end kernel win, while `cluster_gemm_tcgen05` is the fine-grained tcgen05 follow-up where the remaining headroom is much smaller.

Treat `cluster_gemm_tcgen05` as a supplementary comparison benchmark with a local contract. It is still useful for tcgen05 profiling and regression tracking, but the lab's canonical speed claim stays on `cluster_gemm`.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm_tcgen05 --profile deep_dive --single-gpu
```

The tcgen05 path is worth profiling separately. Its win is much smaller than the coarse cluster GEMM delta, so Nsight evidence matters more than headline speedup alone.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/fullstack_cluster
python -m cli.aisp bench run --targets labs/fullstack_cluster:moe_hybrid_ep --profile minimal
python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm --profile minimal
python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048
```

## Learning Goals
- Run scenario benchmarks that stitch together chapters into end-to-end workflows.
- Compare a baseline vs topology-aware DeepSeek-style hybrid expert-parallel optimizer step.
- Inspect cluster GEMM kernels (baseline and DSMEM/TMA optimized) via the CUDA extension.
- Track GPU requirements, expected shapes, and automation scripts in one place.
- Collect artifact bundles that summarize every phase of the scenario.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_moe_hybrid_ep.py`, `optimized_moe_hybrid_ep.py`, `baseline_moe_hybrid_ep_multigpu.py`, `optimized_moe_hybrid_ep_multigpu.py`, `moe_hybrid_ep_common.py` | DeepSeek-style hybrid EP optimizer-step benchmarks with explicit dispatch/combine phases, load-balance metrics, and intra-node fallback reporting. |
| `baseline_cluster_gemm.py`, `optimized_cluster_gemm.py`, `baseline_cluster_gemm_tcgen05.py`, `optimized_cluster_gemm_tcgen05.py` | Python entrypoints for the cluster GEMM kernels with tcgen05 fallbacks. |
| `capstone_extension.py`, `capstone_extension_tcgen05.py`, `capstone_kernels.cu`, `capstone_kernels_tcgen05.cu`, `capstone_benchmarks.py` | PyTorch extension, CUDA kernels, and harness hooks for the GEMM showcase. |
| `run_lab_fullstack_cluster.py`, `gpu_requirements.py`, `expectations_{hardware_key}.json` | Standalone runner, hardware requirement helper, and expectation file. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/fullstack_cluster
python -m cli.aisp bench run --targets labs/fullstack_cluster --profile minimal
```
- Targets follow the `labs/fullstack_cluster:<workload>` naming convention listed by `list-targets`.
- `--target-extra-arg labs/fullstack_cluster:<workload>="--flag value"` passes arguments to one target. The harness fails the target unless the benchmark defines `apply_target_overrides` or launches through torchrun.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/fullstack_cluster:moe_hybrid_ep --profile minimal` records a full optimizer step with routing/dispatch/combine/backward/grad-sync metrics.
- `python -m cli.aisp bench run --targets labs/fullstack_cluster --profile minimal` records per-phase metrics for the entire scenario suite.
- `python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048` builds the extension on first run and prints baseline vs optimized TFLOP/s.
- KF-specific kernels skip gracefully on hardware lacking tcgen05 or DSMEM, ensuring CI signal stays meaningful.

## Notes
- `gpu_requirements.py` reports the minimum GPU count, memory, and features for each scenario; consult it before scheduling runs.
- `capstone_extension.py` caches builds under `~/.cache/torch_extensions`; run `python cleanup.py --include-extensions` when switching CUDA versions.
- Single-GPU `moe_hybrid_ep` benchmark runs measure `HybridEPTrainer.run_step()` in-process so the timing reflects the optimizer step instead of single-rank launcher overhead; multi-rank runs still use `torchrun`.
- Canonical hybrid-EP comparisons now keep the same default routing mode; use `--route-mode topology_aware` when you want that alternate behavior to be visible instead of relying on a silent default.
- `cluster_gemm_tcgen05` is a supplementary comparison benchmark with a local contract; use `cluster_gemm` when you want the lab's canonical cluster-GEMM speed claim.
