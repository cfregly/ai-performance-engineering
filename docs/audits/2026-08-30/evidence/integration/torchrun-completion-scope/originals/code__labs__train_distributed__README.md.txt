# Lab - Distributed Training Playbook

## Summary
Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness.

## Problem
Distributed training has too many "optimized" labels that mean different things. This lab is here to keep DDP compression, pipeline schedules, and symmetric-memory training as separate benchmarked choices so you can see what actually helps on the current stack.

## Baseline Path
- conservative DDP, pipeline, and symmetric-memory paths
- useful for correctness and topology sanity
- enough communication overhead to make overlap/compression visible

## Optimized Path
- overlap-aware pipeline schedules
- compression-aware DDP variants
- symmetric-memory and sharding strategies run through the same harness

## Measured Delta
Representative strict result from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `ddp_compression` | `1135.768 ms` | `408.656 ms` (`powersgd`) | `2.78x` |
| `pipeline_1f1b` | `159.060 ms` | `105.125 ms` | `1.51x` |
| `pipeline_dualpipe` | `154.106 ms` | `105.111 ms` | `1.47x` |
| `symmem_training` | `177.269 ms` | `167.167 ms` | `1.06x` |

The useful point is that the lab shows more than one kind of "distributed optimization." Compression and pipeline scheduling move the needle more than the current symmetric-memory path on this local setup.

Treat single-GPU `fsdp2` on `b200` as a supplementary comparison surface with a local comparison contract. The real FSDP2 speed gate stays on the multi-GPU `2x_b200` contract where sharding and overlap can actually change the communication story.

## Profiler Evidence
```bash
python -m cli.aisp bench run --targets labs/train_distributed:ddp_compression --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/train_distributed:pipeline_1f1b --profile deep_dive --single-gpu
```

For the multi-GPU variants, keep using `torchrun` through the lab utilities. The single-GPU harness targets are the evidence-first entrypoint, not a replacement for real cluster validation.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --targets labs/train_distributed:ddp_compression --profile minimal
python -m cli.aisp bench run --targets labs/train_distributed:pipeline_1f1b --profile minimal
```

## Learning Goals
- Benchmark standard DDP vs optimized overlap-aware variants.
- Exercise FSDP and ZeRO strategies with shared helper utilities.
- Validate symmetric-memory training modes that pool NVLink bandwidth.
- Reuse launcher utilities (torchrun) with consistent configuration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `baseline_ddp_multigpu.py`, `optimized_ddp_multigpu.py`, `baseline_ddp_flash_multigpu.py`, `optimized_ddp_flash_multigpu.py`, `baseline_ddp_compression_multigpu_int8.py`, `optimized_ddp_compression_multigpu_int8.py`, `baseline_ddp_compression_multigpu_powersgd.py`, `optimized_ddp_compression_multigpu_powersgd.py`, `ddp.py` | DDP workloads including flash-attention and compression variants (single + multi GPU). |
| `baseline_fsdp.py`, `optimized_fsdp.py`, `baseline_fsdp_multigpu.py`, `optimized_fsdp_multigpu.py`, `baseline_fsdp2.py`, `optimized_fsdp2.py`, `baseline_fsdp2_multigpu.py`, `optimized_fsdp2_multigpu.py`, `train_fsdp.py`, `train_fsdp2.py` | FSDP/FSDP2 scripts that demonstrate shard-by-shard memory savings. |
| `baseline_pipeline_1f1b.py`, `optimized_pipeline_1f1b.py`, `baseline_pipeline_gpipe.py`, `optimized_pipeline_gpipe.py`, `baseline_pipeline_dualpipe.py`, `optimized_pipeline_dualpipe.py`, `baseline_pipeline_dualpipev.py`, `optimized_pipeline_dualpipev.py`, `baseline_pipeline_1f1b_multigpu.py`, `optimized_pipeline_1f1b_multigpu.py`, `baseline_pipeline_gpipe_multigpu.py`, `optimized_pipeline_gpipe_multigpu.py`, `baseline_pipeline_1f1b_to_gpipe_multigpu.py`, `optimized_pipeline_1f1b_to_gpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipe_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipe_multigpu.py`, `baseline_pipeline_gpipe_to_dualpipev_multigpu.py`, `optimized_pipeline_gpipe_to_dualpipev_multigpu.py`, `baseline_pipeline_dualpipe_multigpu.py`, `optimized_pipeline_dualpipe_multigpu.py`, `baseline_pipeline_dualpipev_multigpu.py`, `optimized_pipeline_dualpipev_multigpu.py`, `pipeline_*.py` | Pipeline parallelism schedules (single GPU simulations + multi-GPU execution). |
| `baseline_symmem_training.py`, `optimized_symmem_training.py`, `baseline_symmem_training_multigpu.py`, `optimized_symmem_training_multigpu.py` | Symmetric-memory strategies for optimizer state replication. |
| `baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `baseline_zero1_multigpu.py`, `baseline_zero2_multigpu.py`, `baseline_zero3_multigpu.py`, `optimized_zero1_multigpu.py`, `optimized_zero2_multigpu.py`, `optimized_zero3_multigpu.py`, `zero1.py`, `zero2.py`, `zero3.py` | ZeRO implementations (1/2/3) plus helpers for parameter partitioning. |
| `training_utils/`, `utils.py`, `__init__.py` | Shared launch utilities, argument parsing, and harness exports. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --targets labs/train_distributed --profile minimal
```
- Targets follow the `labs/train_distributed:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/train_distributed:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.
- `python labs/train_distributed/train_fsdp.py --validate` confirms numerical parity between FSDP shards and the baseline DDP path.
- `python labs/train_distributed/optimized_zero3_multigpu.py --summary` shows reduced peak memory vs the baseline script.

## Notes
- Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.
- `utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.
- FSDP/FSDP2 benchmarks default to `labs/train_distributed/data/tinystories_packed_seq128.jsonl` plus `labs/train_distributed/data/tinyllama_config.json`, with `AISP_TINYSTORIES_LAYERS=4` to keep the model small. Override with `AISP_TINYSTORIES_PACKED_PATH`, `AISP_TINYSTORIES_LOCAL_PATH`, `AISP_TINYSTORIES_CONFIG_PATH`, or `AISP_TINYSTORIES_LAYERS`.
- Scale up by increasing `AISP_TINYSTORIES_LAYERS` or swapping to a larger config and pairing it with a packed dataset that matches the new sequence length.
- Set `AISP_FSDP_DISABLE_FP8=1` to keep the minimal BF16 path; unset it when you want to exercise the FP8 conversion on larger workloads.
- On single-GPU `b200`, `fsdp2` remains runnable for regression tracking and profiler capture, but the benchmark is judged as a local comparison contract rather than a canonical speed claim.
