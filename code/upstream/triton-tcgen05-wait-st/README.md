# DRAFT — internal review pending

## Status package: Triton sm_103 `tcgen05.wait.st` abort (Tier-2 item 6)

Status package for Tier-2 item 6 of `code/docs/gb300-upstream-patches.md`
(toolchain section + B17/B18/B19/B22c-era evidence).

**There is NO ISSUE_DRAFT in this package, deliberately.** Per the
pre-filing check, the error signature is already filed upstream
(triton-lang/triton#8473 open, #8481 closed-dup; fixed on main by the
LLVM bump in #8299 — the genuine breakage was the Triton 3.5 release).
And on THIS stack (vanilla Triton 3.7.0, sm_103) the abort does not
reproduce at all: the campaign's aborts were caused by the repo's own
`ensure_triton_compat` patch rewriting the Triton target
`sm_103a -> sm_103`, which probe D proves is sufficient to produce the
exact abort. Filing upstream would be both a duplicate AND wrong.

See `STATUS.md` for the full upstream record, the probe matrix, the
mechanism, and the suggested in-repo fix.

## Files

| file | what |
|---|---|
| `STATUS.md` | the deliverable: upstream filing state + verified probe verdict + root cause |
| `repro.py` | 4-probe verification script (child-process isolation; probe D = mechanism proof) |

## Verification record (2026-06-11, GB300 pod, GPU 3, flock lease, foreign-proc check clean)

`repro.py` exit 0 on torch 2.12.0a0+5aff3928d8.nv26.05, Triton 3.7.0:

```
probe A: plain tl.dot matmul JIT (control)        -> clean
probe B: dead num_warps:tl.constexpr param        -> clean
probe C: torch.compile max-autotune attn+MLP      -> clean
probe D: probe A mis-targeted to sm_103 (no 'a')  -> ABORT-REPRODUCED
         stderr: LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st
```

Pod evidence: `/tmp/frontP2/tcgen05_probe.{cmd,stdout,stderr,exit}` and
the standalone mechanism probe `/tmp/frontP2/childD.{py,out,err}`
(rc=134).

Upstream references read this session (gh): triton-lang/triton #8473,
#8481, #7725, #8299, #8045.

## What the user must do

1. Nothing upstream. (Optional: a +1 comment on #8473 for the 3.5.1 ask
   only if we still care about pinned 3.5.0 anywhere — our 3.7 stack is
   unaffected when unpatched.)
2. The actionable follow-up is IN-REPO: fix
   `core/benchmark/triton_compat.py` to preserve the `a` suffix for all
   `sm_1xxa` arches (clamp only sm_121->sm_120), then re-test the llama
   max-autotune path with `ENABLE_TRITON_PATCH=0`/the fix and consider
   retiring the sm_103 max-autotune->default downgrade in
   `core/utils/compile_utils.get_optimal_compile_mode`.
3. Update the runbook's B19 "side effect not yet isolated" note to point
   at STATUS.md (mechanism now proven).
