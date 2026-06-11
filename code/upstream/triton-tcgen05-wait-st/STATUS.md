# DRAFT — internal review pending

## Status note: Triton sm_103 `tcgen05.wait.st` LLVM-selection abort

**Verdict (2026-06-11): DO NOT FILE.** The error signature is already
filed upstream and fixed on Triton main, and — the bigger finding — the
campaign's Triton-3.7 aborts do NOT reproduce on vanilla Triton 3.7.0:
they were self-inflicted by this repo's own Triton compat patch
mis-targeting `sm_103` instead of `sm_103a`. The actionable fix is
in-repo, not upstream.

Environment verified: GB300 sm_103, CUDA 13.2, torch 2.12.0a0 NGC 26.05,
Triton 3.7.0.

## Upstream filing state (checked via gh, 2026-06-11)

The exact error string
`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st` is
upstream-known as the **Triton 3.5 release breakage of sm_103**:

| ref | state | what |
|---|---|---|
| triton-lang/triton#8473 | OPEN (last activity 2025-10-27) | "Release 3.5 broke sm103 (GB300) support — a minor release desired"; names root cause + fix |
| triton-lang/triton#8481 | CLOSED (dup of #8473) | same error string, sglang on GB300 |
| triton-lang/triton#7725 | MERGED 2025-07-31 | the breaking change (inline PTX -> NVVM intrinsics) |
| triton-lang/triton#8299 | MERGED 2025-10-01 | the fix on main (LLVM bump); release 3.5 was cut between #7725 and #8299 |
| triton-lang/triton#8045 | CLOSED unmerged | proposed revert for a 3.5.1 that never shipped (per #8473 thread) |

No 2026-era upstream issue reports the abort on Triton 3.6/3.7 (searched
triton-lang/triton for `tcgen05` and `sm103`).

## What this session verified on the pod (repro.py, exit 0)

| probe | result |
|---|---|
| A: plain `tl.dot` matmul JIT (vanilla) | clean |
| B: same kernel + dead `num_warps: tl.constexpr` param (the B17/B18-suspected trigger) | clean |
| C: `torch.compile(mode="max-autotune")` on an attention+MLP block | clean |
| D: probe A with `sm_arch_from_capability` patched to return `sm_103` (no 'a') | **rc=134, `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`** |

Mechanism (probe D): Triton 3.7 plans tcgen05 codegen from the compute
capability (10.3), but the tcgen05 intrinsics are arch-conditional and
only selectable when the LLVM/ptxas target is `sm_103a`. Mis-target plain
`sm_103` and instruction selection dies with an uncatchable LLVM fatal.

## Root cause of the campaign's 3.7 aborts (resolves the B19 open question)

`code/core/benchmark/triton_compat.py:100-109` (`ensure_triton_compat`,
applied unconditionally at import, and imported by
`core/harness/arch_config.py` and `arch_config.py`) patches Triton's
`sm_arch_from_capability` to strip the trailing `a` from every arch
EXCEPT `sm_100a` — a GB10 (sm_121 -> sm_120) workaround that also rewrites
`sm_103a -> sm_103` on GB300. Probe D reproduces the abort by applying
exactly that transform to vanilla Triton; probes A/B/C show the unpatched
stack is clean. This is the "exact configure_optimizations side effect"
B19 left un-isolated, now mechanism-proven.

Consequences / hypotheses now testable in-repo (not re-run this session):

* The llama max-autotune abort (runbook, resolved via `_safe_compile_mode`
  sm_103 fallback) imports `core.harness.arch_config` -> the patch was
  active in every observed abort. Falsifiable: re-run llama max-autotune
  with `ENABLE_TRITON_PATCH=0` — expected clean.
* The B18 "dead `num_warps: tl.constexpr` param" co-trigger is likely a
  red herring: probe B (param present, patch absent) is clean.
* The central `get_optimal_compile_mode` max-autotune->default downgrade
  on sm_103 + Triton>=3.6 may be removable once the patch is fixed.

## Suggested in-repo fix (separate change, not this package)

In `_safe_sm_arch_from_capability` and `_canonicalize_triton_arch`:
preserve the `a` suffix for ALL `sm_1xxa` arches (the suffix is required
for arch-conditional ISA: tcgen05/TMA on sm_100a/sm_103a); clamp only
`sm_121 -> sm_120` (the original GB10 intent).

**SHIPPED 2026-06-11 (runbook B70):** both functions fixed (the patch
closure now delegates to `_canonicalize_triton_arch`; suffix preserved
verbatim for all arches, only `sm_121[a] -> sm_120` clamped), unit-tested
in `tests/test_triton_compat_arch.py` (pod-run, 15 passed, GB10 clamp
unchanged). Pod re-verified clean: occupancy_tuning raw `tl.dot` with
`arch_config` imported unguarded (harness 1.327x verify-pass) and llama
max-autotune. The `_safe_compile_mode` fallback and the central
`get_optimal_compile_mode` sm_103 downgrade are retired.

## Residual upstream angle (hypothesis, intentionally not filed)

Triton could raise a catchable Python error instead of an uncatchable
LLVM fatal when capability-driven tcgen05 codegen meets a non-`a` target
(probe D's failure mode). Low expected value: the mis-target is an
integrator error, and the same fatal-vs-exception complaint already
surfaces in the #8473 thread context. Revisit only if a vanilla trigger
ever appears.
