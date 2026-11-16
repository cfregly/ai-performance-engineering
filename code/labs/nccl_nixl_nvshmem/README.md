# Lab - NCCL, NIXL, and NVSHMEM

## Summary
Explores communication-stack tradeoffs without pretending a `1x B200` host can exercise every path in the GTC deck. The lab exposes one honest benchmark pair for the decision we can measure locally, then pairs it with a stack probe runner that tells you when NCCL collectives, NIXL, or NVSHMEM are actually available.

## Problem
NCCL, NVSHMEM, and NIXL solve different communication problems. The practical mistake is to treat them as interchangeable before checking runtime and topology constraints. This lab keeps that straight by benchmarking the host-staged versus packed async tier-handoff decision we can run locally while making the unavailable paths explicit.

## Baseline Path
- CPU-staged, block-by-block handoff
- per-block D2H and H2D copies with explicit synchronizations
- models the control path that appears when movement and metadata bounce through the host

## Optimized Path
- packed async handoff over the same selected blocks
- coalesces noncontiguous blocks into one packed transfer plan
- uses a dedicated copy stream and one bulk roundtrip instead of per-block synchronization

## Stack Mapping
| Deck theme | What this lab does |
| --- | --- |
| NCCL symmetric memory and copy-engine collectives | Probes whether NCCL collectives are runnable on this host and records the repo-local `all_reduce_perf` binary path. True collective benchmarking still requires `>=2` GPUs. |
| NVSHMEM GPU-initiated one-sided communication | Probes PyTorch symmetric-memory availability plus NVSHMEM launcher/runtime presence. On the current host, the probe finds the versioned launcher install but still stays blocked instead of fabricating a one-sided result. |
| NIXL async, noncontiguous movement across tiers | The `tier_handoff` benchmark pair is the local analogue: scattered GPU blocks, CPU tier roundtrip, and the difference between host-staged control flow and packed async transport. |

## Measured Delta
Representative portable/shared-host result from `artifacts/review/labs/nccl_nixl_nvshmem_portable_shared_v3/20260320_041146__bench__profile_none_targets_labs_nccl_nixl_nvshmem_tier_handoff/`:

| Target | Baseline | Optimized | Measured delta |
| --- | ---: | ---: | ---: |
| `tier_handoff` | `17.07 ms` | `1.12 ms` | `15.28x` |

This is intentionally non-canonical: the host is virtualized and the successful dogfood run used `--validity-profile portable --allow-foreign-gpu-processes`. The lab keeps that provenance visible instead of pretending it is publish-grade NCCL or NVSHMEM evidence.

## Related Work
This lab sits between the low-level communication primitives in `ch04` and the higher-level workflow labs:

- `ch04` covers raw NCCL, symmetric-memory, and NVSHMEM building blocks.
- `labs/train_distributed` and `labs/fullstack_cluster` use communication as part of broader training or cluster stories.
- `labs/cache_aware_disagg_inference` and `labs/dynamic_router` use communication to support serving and routing policies.

The unique focus here is the communication-stack decision itself: what you can measure locally, what is blocked by runtime or topology, and how to tell those apart honestly.

For a common multi-GPU application, continue with the
[halo-exchange lab](../multi_gpu_communication/README.md). It connects neighbor
transfers, native NCCL groups, independent compute, and buffer reuse. The
[NCCL group guide](../../ch04/nccl_groups.md) explains matching and completion
rules. Grouping can reduce submission overhead, while overlap adds stream and
event bookkeeping. This lab's tier-handoff pair remains the selected-block
memory-tier comparison described above.

## Local Reality
| Probe field | Local outcome |
| --- | --- |
| GPU | `1x NVIDIA B200` |
| `torch.distributed` NCCL | available |
| `all_reduce_perf` | available at `tools/nccl-tests/build/all_reduce_perf` |
| `nixl` import | available |
| `nvshmem` import | missing |
| NVSHMEM launcher | available at `/usr/bin/nvshmem_13/nvshmrun` |
| PyTorch symmetric memory | available |
| Multi-GPU NCCL / NVSHMEM validation | blocked by topology (`>=2` GPUs required) |

## Learning Goals
- Separate host-staged control flow from packed async transport planning on a deterministic workload.
- Connect the deck's NIXL memory-tier story to a runnable single-GPU analogue.
- Make NCCL, NVSHMEM, and symmetric-memory availability visible before attempting the wrong experiment.
- Keep local dogfood results honest when the host cannot provide a clean multi-GPU transport path.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_tier_handoff.py`, `optimized_tier_handoff.py` | Auto-discoverable benchmark pair for the local host-staged vs packed async handoff comparison. |
| `comm_stack_common.py` | Shared workload, runtime probe, and benchmark implementation for the pair. |
| `run_lab_nccl_nixl_nvshmem.py` | Standalone probe, direct compare, and sweep entrypoint. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem
python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode probe --json
python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode compare --warmup 3 --iterations 8 --selected-blocks 96 --block-kib 64 --inner-iterations 4
python -m cli.aisp bench run --targets labs/nccl_nixl_nvshmem:tier_handoff --profile none --validity-profile portable --allow-foreign-gpu-processes --target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff='--selected-blocks 96 --block-kib 64 --inner-iterations 4'
```
- Targets follow the `labs/nccl_nixl_nvshmem:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff="--flag value"` to sweep block counts or block sizes through the harness.
- On clean bare-metal single-GPU hosts, the harness target should run in strict mode.
- On the current shared virtualized host, the successful dogfood run required `--validity-profile portable --allow-foreign-gpu-processes`.

## Validation Checklist
- `python -m py_compile labs/nccl_nixl_nvshmem/*.py`
- `python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode probe --json`
- `python labs/nccl_nixl_nvshmem/run_lab_nccl_nixl_nvshmem.py --mode compare --warmup 3 --iterations 8 --selected-blocks 96 --block-kib 64 --inner-iterations 4 --json`
- `python -m cli.aisp bench list-targets --chapter labs/nccl_nixl_nvshmem`
- `python -m cli.aisp bench run --targets labs/nccl_nixl_nvshmem:tier_handoff --profile none --validity-profile portable --allow-foreign-gpu-processes --target-extra-arg labs/nccl_nixl_nvshmem:tier_handoff='--selected-blocks 96 --block-kib 64 --inner-iterations 4' --artifacts-dir artifacts/review/labs/nccl_nixl_nvshmem_portable_shared_v3`

## Notes
- The strict harness path was attempted first and failed for a real reason: a separate isolated benchmark runner already owned GPU memory on the current host. The lab keeps that provenance visible instead of rewriting history.
- No expectation file is checked in yet. The current verified data point is portable and shared-host only, so it is useful for dogfood and documentation, not for a canonical baseline refresh.
- If a future pass adds `>=2` GPUs plus NIXL or NVSHMEM runtime support, keep the probe behavior intact and add new benchmark targets only when those paths are actually runnable and verifiable.
