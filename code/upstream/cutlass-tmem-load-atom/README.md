# DRAFT — internal review pending

## Upstream candidate: CUTLASS tutorial TMEM_LOAD atom width

Filing package for Tier-1 item 2 of `code/docs/gb300-upstream-patches.md`
(B55 evidence, `code/docs/gb300-capstone-f-decomposition.md`).

**The claim being filed:** all five Blackwell CuTe tutorials
(`examples/cute/tutorial/blackwell/01..05_*.cu`, verified present in both
CUTLASS 4.2.0 and current main) demonstrate the TMEM->register epilogue
with `SM100_TMEM_LOAD_32dp32b1x`. ptxas (CUDA 13.2, sm_103) lowers every
such load through a per-load LEPC + CALL.ABS.NOINC + WARPSYNC
convergence-helper call; at 256 loads/thread that dominated the lab kernel
(t2r 7.65-7.81 us of 23.8-24.2 us). One line to `32dp32b32x` = 1.49x kernel,
bit-identical (banked B55).

## Files

| file | what |
|---|---|
| `tmem_load_atom_repro.cu` | minimal standalone kernel (no torch/repo deps): TMEM alloc + t2r copy + store only |
| `sass_evidence.txt` | instruction counts + SASS excerpts from the verified pod build |
| `PR_DRAFT.md` | CUTLASS docs/examples PR text + optional NVIDIA bug-tracker text |

## Verification record (2026-06-11, GB300 pod)

Compiled with `nvcc -std=c++20 -arch=sm_103a` against BOTH
`cutlass-main` (4.5.2) and the repo's `third_party/cutlass` (4.2.0) headers
— no compat overlay needed; identical counts:

```
repro_1x  (32dp32b1x):  LDTM 256, CALL.ABS.NOINC 256, LEPC 256, WARPSYNC 259
repro_32x (32dp32b32x): LDTM   8, CALL.ABS.NOINC   8, LEPC   8, WARPSYNC  11
run on GPU 3 (flock lease, foreign-proc check clean):
  10.906 us/launch (1x)  vs  5.861 us/launch (32x), 2000 launches each
```

The per-load `LEPC ; CALL.ABS.NOINC ; WARPSYNC.ALL` wrap is visible in
`sass_evidence.txt`. Pod artifacts: `/tmp/frontP/{tmem_load_atom_repro.cu,`
`repro_1x.sass,repro_32x.sass,repro_tmem_run.stdout}`.

Numbers cited in the PR draft beyond this repro (7.65->0.42 us t2r, 1.49x
kernel, width sweep, ncu tensor-pipe 20.4->37.9%) are the banked B55
measurements — do not re-derive, cite as-is.

## What the user must do to file

1. Fork NVIDIA/cutlass; suggested branch
   `docs/blackwell-tutorial-tmem-load-atom-width`.
2. Decide the PR shape (the draft offers both): comment-only perf note in
   tutorials 01-05 (lowest-friction, no annotation churn) vs switching the
   atom (requires refreshing every affected layout-print annotation in the
   tutorials — more work, reviewers may prefer it).
3. Paste `PR_DRAFT.md` body; attach `tmem_load_atom_repro.cu` +
   `sass_evidence.txt`.
4. Optional: file the ptxas question at developer.nvidia.com bug tracker
   using the bottom section of `PR_DRAFT.md` (it stands alone).
5. Note for review: `examples/93_blackwell_low_latency_gqa/tgv_gqa.cuh`
   also uses a `32dp32b1x` atom — deliberately NOT touched in this draft
   (different access pattern, unmeasured); mention it in the PR only as a
   question to maintainers if desired.
