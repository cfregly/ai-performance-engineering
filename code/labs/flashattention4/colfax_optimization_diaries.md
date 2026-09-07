# Colfax Optimization Diaries: FlashAttention-4 TMEM Scheduling

This guide turns two Colfax Research optimization diaries into comparable `baseline_*` / `optimized_*` lab targets. The pairs hold the attention workload fixed and isolate a Blackwell TMEM scheduling change. They complement the existing FlashAttention-4 lab; they do not replace its eager-versus-compiled FlexAttention comparison.

## Source material

- [S/P Ping-Pong for FlashAttention-4 Decode](https://research.colfax-intl.com/optimization-diaries-s-p-ping-pong-for-flashattention-4-decode/)
- [Improving FlashAttention-4 Backward for Head Dimension 64](https://research.colfax-intl.com/optimization-diaries-improving-flashattention-4-backward-for-head-dimension-64/)

The performance figures below are results reported by Colfax for its B200 environment. They are study targets, not measurements from this repository or the current host.

## Experiment contract

| Harness target | Baseline | Optimized | Invariant |
| --- | --- | --- | --- |
| `flashattention4_decode` | One TMEM S/P slot keeps `QK(i+1)` behind `softmax(S(i))` and `PV(i)` | Two TMEM slots alternate S and P so the next QK can issue during the current softmax | Same decode inputs, attention math, outputs, and KV-block work |
| `flashattention4_backward` | At head dimension 64, P aliases S and dS aliases dP, requiring compute-wide alias guards | Spare TMEM gives P and dS dedicated slots, enabling earlier S release and warp-local signaling | Same backward inputs, gradients, mode, dtype, and attention shape |

The corresponding modules follow the repository convention: `baseline_flashattention4_decode.py` with `optimized_flashattention4_decode.py`, and `baseline_flashattention4_backward.py` with `optimized_flashattention4_backward.py`. A valid comparison must fail clearly if the requested upstream FA4 path or Blackwell capability is unavailable. A different attention provider is not an equivalent fallback for either pair.

The default decode workload is non-causal BF16 with batch 32, one query token, 131,072 KV tokens, 16 query heads, 1 KV head, and head dimension 64, matching a representative cell in the upstream decode PR. The default backward workload is non-causal BF16 with batch 4, sequence length 16,384, 32 query and KV heads, head dimension 64, and deterministic backward enabled, matching the article's 64k-token deterministic dense row. These are representative performance workloads; the opt-in correctness tests use smaller tensors.

Both pairs measure steady-state CUDA Graph replay after JIT, warmup, and capture in setup. One decode replay contains one FA4 forward invocation. One backward replay contains FA4 backward preprocessing, mainloop, and postprocessing and returns dQ, dK, and dV; the common forward pass that prepares O and LSE runs once in setup and is outside the backward timing boundary. The baseline and optimized arms use the same lifecycle.

The two experiments intentionally pin separate upstream FA4 source identities and environments; follow this lab's README and experiment-specific requirement files rather than mixing them. Decode controls an upstream private ping-pong boolean during graph capture because the corresponding environment switch is read only once at import. Backward selects the source ablation explicitly: both `split_P_dS` and `warp_sync` are false for baseline and true for optimized.

## Decode: true S/P ping-pong

FA4 prefill assigns two 128-row Q tiles to a CTA and overlaps the low tile's QK work with softmax for the high tile. Single- and multi-token decode usually has only one padded Q tile, so that prefill schedule leaves TMEM columns 128-255 idle and serializes:

```text
QK(i) -> softmax(S(i)) -> PV(i) -> QK(i+1)
```

The optimized decode path uses TMEM columns 0-127 as slot 0 and 128-255 as slot 1. S and P alternate between them across KV blocks. The MMA warp can write `S(i+1)` into one slot while the softmax warps read `S(i)` and write `P(i)` in the other. `QK` therefore runs one block ahead of `PV`.

This is more than ordinary double buffering. Correct ping-pong requires all of the following:

- Alternate the TMEM destination for both S and P by KV-block parity.
- Track a barrier per slot. The slot is selected by the low bit of the global PV count, while the next bit tracks the phase when that slot is reused.
- Change the load order from `K0, V0, K1, V1, ...` to `K0, K1, V0, K2, V1, ...`, with the final V appended, so `QK(i+1)` is ready before `PV(i)`.
- Keep the single O accumulator safe while correction warps may rescale it. Slot acquisition must wait for both P production and any correction use to finish.

A host-side buffer toggle or alternate SMEM staging without these TMEM destinations, phases, and dependencies does not reproduce the optimization.

Colfax reports up to a 16% improvement for supported single- and multi-token decode shapes on B200. Because decode is memory-bound, the article reports achieved bandwidth and shows larger gains at longer KV lengths, where the steady-state overlap covers more iterations.

## Backward hdim-64: de-alias before relaxing synchronization

The head-dimension-64 backward kernel uses less tensor-core work per tile than the hdim-128 kernel, while its pointwise softmax and dS work remains. In the baseline TMEM layout, fp32 accumulators occupy columns 0-383 and columns 384-511 are unused. BF16 P overwrites S and BF16 dS overwrites dP.

Those aliases create a real cross-warp hazard. The eight compute warps span two warpgroups, and paired warps share 32-lane TMEM sectors. One warp can otherwise store P into a sector while its partner is still reading S. The baseline therefore needs alias guards: a P-over-S guard once per tile and a dS-over-dP guard in each stage of the two-stage dS loop. Along with signal barriers, the article counts five compute-wide barriers per mainloop iteration.

The optimized layout uses the spare 128 columns for packed BF16 tiles: P in columns 384-447 and dS in 448-511. Once P and dS no longer overwrite S and dP, the kernel can:

1. Remove the two kinds of cross-warp alias guard while retaining the required TMEM fences.
2. Release S as soon as each compute warp has copied it into registers, before softmax.
3. Split the old fused `S read and P written` signal into an early S-read pipeline and a separate P-written/P-consumed pipeline.
4. Issue the next tile's QK before the current tile's `P dO`, moving useful tensor-core work under softmax.
5. Replace the remaining compute-wide signal barriers with warp-local synchronization, allowing the eight compute warps to drift instead of waiting for the slowest warp.

Colfax reports 6-15% improvements across the highlighted B200 BF16 cases, including deterministic cases, and up to 903 TFLOP/s. The wider source sweep covers more modes and dtypes. These figures describe the source environment; this lab must establish its own correctness and performance evidence.

## What the profiler should test

Run each pair as a hypothesis test rather than assuming the upstream mechanism is active.

For decode, look for `QK(i+1)` overlapping the current softmax instead of following `PV(i)`, correct alternation and reuse of the two TMEM slots, and a gain that becomes clearer as KV length increases. Colfax demonstrates the overlap with in-kernel event tracing; a local deep-dive artifact should retain whatever equivalent timeline and counter evidence the installed profiler can capture.

For backward, look for the next QK moving under softmax, removal of the five compute-wide barrier rendezvous per tile, and compute warps progressing independently through softmax and dS. Confirm that the run used head dimension 64 and the de-aliased upstream path. Lower latency without those mechanism checks could come from a different dispatch or workload.

For both targets, compare equivalent work, preserve numerical verification, record the selected kernel/API identity, and reject unsupported or fallback execution. The source's trace and benchmark results explain what to seek; they do not qualify a local run.

## Chapter connections

- [Chapter 10](../../ch10/README.md): warp specialization, `tcgen05.mma`, TMEM allocation, TMA-fed operand pipelines, and choosing synchronization scope inside a tensor-core kernel.
- [Chapter 11](../../ch11/README.md): producer/consumer overlap, barrier phase tracking, and removing unnecessary ordering. This is intra-kernel concurrency rather than CUDA-stream concurrency.
- [Chapter 18](../../ch18/README.md): decode-specific attention shapes, KV-block iteration, GQA, and the difference between prefill and decode schedules.

The existing `flashattention4` target compares eager score materialization with a compiled provider-aware fused path. The separate `labs/flexattention` lab studies programmable masks, score modifications, and FlexAttention sweeps. Each Colfax pair uses one pinned upstream revision with its optimization disabled or enabled, isolating TMEM allocation and scheduling.

## Run and qualify

From `code/`:

```bash
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_decode --profile deep_dive --single-gpu
python -m cli.aisp bench run --targets labs/flashattention4:flashattention4_backward --profile deep_dive --single-gpu
```

Current-host CPU/static checks can verify discovery, imports, metadata, pair structure, and unsupported-hardware diagnostics. They cannot show that a Blackwell TMEM path executed or establish a speedup. Target-GPU qualification requires successful baseline and optimized execution on the intended Blackwell GPU, output or gradient parity, retained provenance, repeated comparable timings, and profiler evidence for the scheduling mechanism.
