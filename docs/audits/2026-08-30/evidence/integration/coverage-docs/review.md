# Independent bounded coverage-documentation review

Scope: review only the factual coverage reconciliation in the generated code
README, its generator and the AGENTS inventory. No source, operational rule, or
old evidence was edited by this reviewer. This record precedes any follow-up
wording corrections and identifies its exact source snapshot below.

## Confirmed

- Both inventories have95rows in11categories. Blanket all-protected statements
  and unconditional OK/checkmark statuses have been removed.
- The requested missing-policy/scoped-check mapping is present. README and AGENTS
  mechanism/status cells agree, apart from harmless code parentheses/formatting.
- Unmapped rows explicitly remain unaudited rather than implying coverage.
- AGENTS first400lines match the pre-reconciliation snapshot exactly. Its explicit
  bare-metal operational restriction remains; the factual warning behavior is
  distinguished from policy. Existing safety/approval/runtime qualification rules
  were not weakened by the reviewed inventory changes.
- Both relative clock-lock receipt links resolve to
  `docs/audits/2026-08-30/evidence/validation/clock-lock-followup/receipt.json`,
  SHA256`e90d4b1e57815fa7e591594f9e1d604806761e2927a0c2dab8a5264d19d69463`.
  Both prior protection-receipt links also resolve, SHA256
  `809c67de9f024211e1dd64059dd99170d4eed515e9920b7449f85aef6dafef54`.
- Generator AST parsing passes. No GPU/runtime test or external incident-source
  revalidation was performed; historical incident facts are outside this review.

## Remaining corrections sent to root

1. **README virtualization wording conflicts with retained policy.** The generated
   note atREADME183–184/generator456–457 still says bare metal is merely preferred
   for final performance numbers. State that the checker warns, while virtualized
   current-host reruns require the existing explicit user approval, locked clocks,
   provenance and non-canonical label; canonical/publish-grade results require
   bare metal. This aligns factual prose with the preserved AGENTS241exception and
   AGENTS521restriction; it must not broaden authorization.
2. **Profiler cell overstates active disabling.** README196, generator469 and
   AGENTS531 say timing disables its profiler. The actual test observes inactive
   profiler state without an external profiler; no nested-profiler guard exists.
   Use “Harness timing does not enable its profiler; no nested-profiler rejection.”
3. **Existing jitter explanation needs a factual scope clarification.** AGENTS
   Advisory section around738–759 says jitter only catches both variants returning
   the same constant. `_run_jitter_check` evaluates one benchmark independently and
   has pre-perturbation unsupported exits returning(True,None). Clarify conditional
   per-benchmark sensitivity and that a successful jitter return does not alone
   establish complete verification. Do not relax full correctness requirements or
   present the existing subset-output example as full-output qualification.

The first two are narrow follow-ups to this reconciliation; the third is an
existing adjacent explanatory overclaim. No additional production guard is
requested. After these wording changes, this scoped inventory does not need to
claim all95policies are implemented in order to close the original test-quality
findings. Real CUDA gates and unsupported policies remain distinct limitations.

## Reviewed source snapshot

- `code/README.md`: `2c14642cf6188a098e24ff5e14814e4848ffa1c9f3c656774a6cdef79b0d7373`
- `code/AGENTS.md`: `e4cca5f7ac816f4a11410063b217d61b2e2146ab9d9cd80f8b6bc0b0d902d27e`
- `code/core/scripts/refresh_readmes.py`: `da6596dcf3b278fba6daeadaceea4b34013a0a0319a368ba6f582966d6011397`

## Final follow-up verification

Root applied all three requested wording corrections. Readback confirms generator/README profiler wording, explicit virtualization approval and canonical bare-metal limits, and AGENTS conditional independent jitter explanation. AGENTS first400lines remain unchanged. No further actionable issue found within this bounded factual-inventory scope. This is documentation/source review only, not runtime qualification.

- `code/README.md`: `2c14642cf6188a098e24ff5e14814e4848ffa1c9f3c656774a6cdef79b0d7373`

- `code/AGENTS.md`: `e4cca5f7ac816f4a11410063b217d61b2e2146ab9d9cd80f8b6bc0b0d902d27e`

- `code/core/scripts/refresh_readmes.py`: `da6596dcf3b278fba6daeadaceea4b34013a0a0319a368ba6f582966d6011397`
