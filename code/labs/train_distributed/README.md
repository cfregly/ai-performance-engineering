# Lab - Distributed Training Playbook

## Summary
Collects distributed-training recipes: DDP, FSDP, ZeRO-1/2/3, symmetric memory and flash-attention-aware all-reduce handling. Direct training scripts remain available. Plain DDP and ZeRO-2 have dedicated child-result contracts; wrappers without a contract remain unsupported.

## Generic wrapper verification unavailable
The shared `training_utils/torchrun_harness.py` wrapper formerly verified an unrelated parent-side Linear model before launching the real child. That surrogate has been removed. Factories without a dedicated child-result contract remain discoverable, but their harness execution and verification stop explicitly before launch. Plain DDP publishes complete post-training logits and loss from the actual final batch and a changed input, with separately executed forwards through the same unwrapped trained weights. That checks the forward wrapper; it is not independent proof of the optimizer updates. ZeRO-2 has a separate result adapter. A failed launch-spec getter is propagated rather than replaced with a fallback script. Direct training entrypoints are unchanged; executing them alone is not correctness or performance acceptance. The separate ZeRO training tests do not supply a verification protocol for other wrappers.

## Training runtime prerequisites
The Hugging Face examples need `datasets` and `accelerate` in the Python
environment that launches the workers. The optimized FlashAttention training
paths use the **FlashAttention-2** distribution and its public `flash_attn`
APIs. A FlashAttention-4 namespace import alone does not satisfy that
requirement. Keep these environments separate when their packages overlap;
installing a training dependency must not replace another workload's PyTorch
or attention runtime.

Check the selected training interpreter before launching:

```bash
python -c 'import torch, datasets, accelerate; from flash_attn import flash_attn_func, flash_attn_varlen_func; print(torch.__version__, torch.version.cuda)'
```

The September 2026 direct B200 pass exercised all 61 discovered training
variants (27 one-rank and 34 two-rank runs) with Torch 2.9.1+cu130 and the
matching FlashAttention 2.8.3 CUDA 13/Torch 2.9/CXX11 ABI wheel. All scripts
exited successfully; this does not make their generic wrappers qualified
benchmarks. See the [dated validation checkpoint](../../../docs/reviews/2026-09-06-codebase-repair-checkpoint.md)
for the source identity, retained failures and execution limits. Direct
`torchrun` commands work without Slurm.

## DDP accumulation
In the optimized DDP and FlashAttention DDP entrypoints, `--steps` caps consumed
microbatches, up to the available data. `--grad-accum` groups them into optimizer
updates. A final partial group uses its actual size and still updates the model;
for example, `--steps 3 --grad-accum 2` performs two optimizer updates. Both
forward and backward stay inside the same gradient synchronization context.

The plain optimized multi-GPU DDP path also accepts the explicit
`--overlap-optimizer` experiment. It updates fused AdamW parameter buckets on a
dedicated CUDA stream after each DDP all-reduce. The synchronous optimizer stays
the default. The opt-in requires exactly two ranks, `--grad-accum 1`, and no
`--compile`; requesting it outside that scope fails before model construction.
The paired baseline remains synchronous. A three-update exact gate matched all
parameters and AdamW state. The retained two-B200, full-epoch, two-seed A/B/B/A
screen matched complete inputs, logits, and losses, while training-loop speedup
was `1.023626x` geometric mean. Whole-process ratios were mixed, so this is a
scoped loop result rather than an end-to-end speedup claim.

The DDP result transport retains all four full output/reference mappings.
After full reference checks, byte-identical references share serialized tensor
storage; distinct values, including signed zeros, keep their own storage. The
1 GiB per-rank file limit remains enforced. This avoids redundant full-logit
files without reducing the batch, output coverage, or accuracy requirements.

For this target, the harness maps `--iterations 3` to child `--steps 3`; the
fixed `--grad-accum 1` therefore performs three optimizer updates per rank. Run
the public integration check with:

```bash
python -m cli.aisp bench run --targets labs/train_distributed:ddp_multigpu \
  --launch-via torchrun --nproc-per-node 2 --iterations 3 --warmup 0 \
  --profile none --validity-profile portable \
  --target-extra-arg 'labs/train_distributed:ddp_multigpu=--overlap-optimizer'
```

## FSDP training semantics

In the FSDP and FSDP2 entrypoints, `--steps` counts optimizer updates. Each
update consumes `--grad-accum` full per-rank microbatches, continuing into
another data epoch when necessary. For example, `--steps 3 --grad-accum 2`
consumes six microbatches and performs three updates. Empty per-rank loaders fail
explicitly.

Packed and synthetic labels already contain the next token at each input
position. The shared training helper computes cross-entropy against these
targets directly, avoiding a second causal shift inside the model. Direct runs
seed model initialization and data order with 42; harness-owned seeds are
preserved. Synthetic data uses a separate generator so data creation does not
change model initialization.

Model weights use BF16 while rotary-position frequency buffers stay in FP32.
Device placement preserves those buffer dtypes, and FSDP1's mixed-precision
policy also keeps buffers in FP32. Rounding inverse frequencies to BF16 before
the rotary calculation introduces position-dependent errors that grow with
sequence length.

The FSDP and FSDP2 entrypoints report synchronized milliseconds per optimizer
update on rank 0, using the slowest rank's complete training interval. This
includes data loading, transfers, forward, backward, optimizer updates, and
training logging; it excludes model startup and teardown. The optimized paths
retain their FlashAttention, resharding, FP8, and fused AdamW behavior where
supported.

For reproducible FlashAttention 2 training checks, explicitly set
`FLASH_ATTENTION_DETERMINISTIC=1` for both compared runs. A fixed model/data seed
alone does not make the default FlashAttention backward pass bitwise repeatable.
Keep this setting consistent across compared runs and record it with timings.
Matching an independent model with the same attention backend does not establish
bitwise equality with eager attention or an application-level accuracy budget.
These direct-run diagnostics do not enable the generic wrapper's child-result
verification for either FSDP family; the wrappers remain fail-closed until they
publish actual trained outputs and an independent reference.

## Problem
Distributed training has too many "optimized" labels that mean different things. This lab is here to keep DDP compression, pipeline schedules, and symmetric-memory training as separate benchmarked choices so you can see what actually helps on the current stack.

## ZeRO comparison validity
The ZeRO-2 named pair uses a shared seven-linear-layer GELU model, FP32 model parameters, BF16 CUDA autocast, identical rank-specific inputs, accumulation, AdamW settings, clipping, one warmup update and the same measured training loop. The optional communication payload is BF16 in both variants. The optimized path shards optimizer state and uses reduce-scatter/all-gather to restore complete gradients; it does not overlap optimizer updates or keep gradients sharded through the optimizer step.

The original single baseline used ReLU and executed two training runs, while its optimized peer used GELU and one run. Other precision, clipping and timing differences also invalidated that comparison. Historical ZeRO timings are not accepted evidence for the repaired pair. The generic torchrun wrapper's small signature model has been withdrawn because it did not verify child-process training. Acceptance requires the separate full training tests, including actual two-GPU NCCL/BF16 checks, before any new speed or memory claim.

Inner training throughput counts batch rows as samples, not hidden-vector elements as tokens. Harness process-wall timing includes startup and warmup and must remain labeled separately.

## Baseline Path
- conservative DDP, pipeline, and symmetric-memory paths
- useful for correctness and topology sanity
- enough communication overhead to make overlap/compression visible

## Optimized Path
- overlap-aware pipeline schedules
- compression-aware DDP variants
- direct symmetric-memory and sharding scripts; generic harness qualification is unavailable

## Historical Delta (not requalified by this audit)
Stored historical results from `artifacts/runs/20260302_full_strict_chapter_lab_singlegpu_v2/`; this audit does not requalify their original verification or timing contracts:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `ddp_compression` | `1135.768 ms` | `408.656 ms` (`powersgd`) | `2.78x` |
| `pipeline_1f1b` | `159.060 ms` | `105.125 ms` | `1.51x` |
| `pipeline_dualpipe` | `154.106 ms` | `105.111 ms` | `1.47x` |
| `symmem_training` | `177.269 ms` | `167.167 ms` | `1.06x` |

These records remain available for lineage. Fresh source, workload, actual-output and timing evidence is required before repeating their performance claims; generic wrapper metadata is not child-training evidence.

FSDP2 wrapper execution is also unavailable until actual child training is verified. Future FSDP2 qualification must distinguish single-GPU behavior from real multi-GPU sharding; old labels do not certify either.

## Profiler workflow status
```bash
python -m cli.aisp profile torch --help
python -m cli.aisp bench run --help
```

These commands inspect options only. Generic training wrappers cannot currently produce a qualified harness profile or comparison. Profile a direct script only on an allocated, authorized target, and retain actual training verification separately.

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m pytest -q tests/test_audit_wave1_zero2_parity.py tests/test_audit_wave1_torchrun_verification.py
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

## Discovery and local validation
Run from code/. These commands inspect metadata and exercise local contracts; they do not launch a qualified training benchmark.
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --help
python -m pytest -q tests/test_audit_wave1_zero2_parity.py tests/test_audit_wave1_torchrun_verification.py
```
- Generic wrapper setup, verification and launch-spec requests fail explicitly; no speed or memory claim is accepted from them.
- Canonical or publish-grade GPU results require bare metal. A virtualized current-host rerun requires explicit user approval, locked clocks, recorded provenance and a non-canonical label.

## Validation Checklist
- From `code/`, run `python -m pytest -q tests/test_audit_wave1_zero2_parity.py tests/test_audit_wave1_zero2_single.py tests/test_audit_wave1_zero2.py` for the local source/CPU checks; CUDA skips remain unverified.
- Actual two-GPU training checks are in `tests/test_audit_wave1_zero2_parity_cuda.py`. Run only with an allocated compatible CUDA/NCCL target; local CPU passes do not qualify sharding, streams, memory savings or performance.
- Use `python -m cli.aisp bench list-targets --chapter labs/train_distributed` to inspect registered workload names before selecting a run. A successful launch alone does not verify child training.

## Notes
- Inspect `python -m cli.aisp bench run --help` and `training_utils/torchrun_harness.py` for the supported launcher configuration; use the allocated topology and preserve launcher arguments with results.
- FSDP and FSDP2 wrappers select `labs/train_distributed/data/tinystories_packed_seq1024.jsonl` and `labs/train_distributed/data/tinyllama_config.json`. FSDP selects 22 layers for single-GPU and 12 for multi-GPU; FSDP2 selects eight layers, per-rank microbatch size two, and accumulation two. Override direct-run inputs with `AISP_TINYSTORIES_PACKED_PATH`, `AISP_TINYSTORIES_LOCAL_PATH`, `AISP_TINYSTORIES_CONFIG_PATH`, or `AISP_TINYSTORIES_LAYERS`.
- Scale up by increasing `AISP_TINYSTORIES_LAYERS` or swapping to a larger config and pairing it with a packed dataset that matches the new sequence length.
- Set `AISP_FSDP_DISABLE_FP8=1` to keep the minimal BF16 path; unset it when you want to exercise the FP8 conversion on larger workloads.
- The generic `fsdp2` wrapper retains metadata but rejects harness execution. Direct script execution is not a substitute for a child-result contract or multi-GPU correctness evidence.
