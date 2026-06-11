# DRAFT — internal review pending

## Upstream candidate: CUTLASS 2x1SM atom B-split semantics + expect-tx trap docs

Filing package for Tier-1 item 3 of `code/docs/gb300-upstream-patches.md`
(B49/B53/B57 evidence, `code/docs/gb300-gemm-occupancy-rewrite.md` — Front U
design facts, B53 traps-banked list, V2 expect-tx deadlock fix).

**The claims being filed (all from banked evidence + one new host-only
print check, no new GPU measurement):**

1. `SM100_MMA_F16BF16_2x1SM_SS` splits B N/2-per-CTA across the pair —
   proven on GB300 by ncu smem footprints (64 KiB/CTA at (tile_n=256,
   stages=2); 72 KiB at (128,3) -> Block Limit Shared Mem 3 where full-N B
   would give 96 KiB -> limit 2) and by mbarrier tx-byte balance (B49).
   Tutorial 04's print annotations still claim full-N B.
2. `tma_partition`'s multicast slice is an OFFSET VIEW — expect-tx bytes
   are the full delivered slice with NO participant multiplier. Wrong
   formula = mbarrier never fires = deterministic deadlock; hit
   independently by two fronts (B53: 80 KiB armed vs 48 KiB delivered;
   B57/V2: the `kClusterN` multiplier -> 25-min hang, one-line fix,
   rel_err 0.0 after).

## Files

| file | what |
|---|---|
| `snippet.cu` | host-only print check (no GPU): actual `partition_shape_B` output vs tutorial 04's annotations, + the verified expect-tx formula and trap notes as comments |
| `PR_DRAFT.md` | CUTLASS docs/examples PR text |

## Verification record (2026-06-11, GB300 pod)

`snippet.cu` compiled (`nvcc -std=c++20 -arch=sm_103a`) and run against
BOTH `cutlass-main` (4.5.2) and the repo's `third_party/cutlass` (4.2.0)
headers — identical output:

```
mma_shape_B:  ((_128,_16),_1,_4)      <- tutorial 04 comment claims ((_256,_16),_1,_4)
sB_layout:    Sw<3,4,3> o ... ((_128,_16),_1,_4):((_64,_1),_0,_16)
per-CTA smem B bytes/stage: 16384  (full-N would be 32768)
leader expect-tx bytes/stage = 2 * (16384 + 16384) = 65536
```

16384 B-bytes/stage matches the banked B49 ncu footprints exactly. The
stale annotations were confirmed present in cutlass-main (4.5.2) at lines
~254 (`tCsB`), ~310-325 (`tBsB`), ~505-517 (`mma_shape_B`/`sB_layout`) of
`examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`. Pod artifacts:
`/tmp/frontP/{snippet_2x1sm.cu,snippet_2x1sm.stdout}`.

The deadlock narrative cites banked evidence only (B53 trap bank, B57/V2
WRONG-vs-RIGHT one-liner) — the wrong formula was NOT re-run.

## What the user must do to file

1. Fork NVIDIA/cutlass; suggested branch
   `docs/blackwell-tutorial04-2sm-b-split-expect-tx`.
2. Paste `PR_DRAFT.md` body; attach `snippet.cu`. The actual annotation
   edits to `04_mma_tma_2sm_sm100.cu` are mechanical (table in the draft);
   make them in the fork before opening the PR, or open as an issue first
   if maintainers prefer to fix annotations themselves.
3. Internal-only caveat to strip-check before posting: the draft already
   omits pod/host names, repo paths, and org names — verify once more
   after any edit.
4. Version note: the lab kernels were built against CUTLASS 4.2.0 headers
   (`third_party/cutlass`); the stale annotations persist on current main
   (4.5.2 checked 2026-06-11). The campaign doc previously said "CUTLASS
   4.3.0" — 4.2.0 is the primary-source-verified version, now corrected.
