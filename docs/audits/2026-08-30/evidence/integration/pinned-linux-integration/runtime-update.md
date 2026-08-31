# Remaining acceptance after Wave 2 source reconciliation

This adds to [the earlier runtime handoff](../runtime-handoff.md). Source and
metadata receipts retain their own identities. Final-source hosted CPU/static/
dashboard validation and a four-target CUDA compiler matrix now pass, as recorded
in the [Wave 2 complete-source receipt](../wave-2-complete-source/receipt.json).
Those workflows do not certify the target Linux installation, GPU correctness,
profiler capture, numerical acceptance, or performance.

1. **Finish pinned Linux installation provenance.** Final-source Benchmark
   Validation run `33391774956` completed on hosted Linux with **4,346 passed,
   461 capability skips, and zero failures**, and uploaded JUnit evidence. The
   workflow did not record every Python/wheel origin and hash or run the complete
   `pip check` acceptance described here. Reproduce the exact package/index cell
   on the supported target, preserving origins, hashes, resolved versions, and
   `pip check`. Current metadata shows 20 matching direct workflow pins, a
   56-package exact-source union, and a 49-package first-PyPI closure.
2. **Exercise the reviewed Linux/CUDA dependency graph.** All 90 current direct
   specifications resolve to 327 unique CPython 3.12 x86_64-manylinux packages
   under one reviewed GPUtil 1.4.0 source-distribution metadata exception.
   Execute the staged setup transactions on a supported Linux/CUDA host, verify
   origins/hashes and `pip check`, and run imports and native extension builds.
   Confirm that the six Darwin import holds — Triton in four entrypoints,
   cuda-python in one, and FlashInfer in one — load from the pinned target
   environment. Metadata resolution alone cannot close this gate.
3. **Run the current CUDA/NCCL correctness matrix after GPU custody returns.**
   The four ZeRO factories now pass source and CPU/Gloo child-result verification;
   they still require actual two-GPU CUDA/NCCL execution, completed
   reduce-scatter/all-gather observations, CUDA RNG checks, and cleanup evidence.
   The other 57 generic wrappers remain deliberately unsupported by the harness.
   Do not count their refusals or CPU/Gloo results as GPU coverage.
4. **Execute the finding-specific hardware gates from the ledger.** The current
   [Wave 2 matrix](../wave-2-complete-source/runtime-gates.md) contains 48 exact
   rows. These include
   CUDA compile/device-link and extension imports, chapter validation runners,
   full-output numerical checks and reviewed policies, stream/sanitizer checks,
   exact B200 Nsight Systems/Nsight Compute captures, and Grace/NUMA observations.
   Preserve model, workload, engine/build, device, acceptance cell, source, and
   outcome lineage. Do not attach historical performance numbers to the new
   source revision.
5. **Respect current custody and local-environment limits.** `HANDOFF.md` reserves
   both B200 GPUs for another task. This task has not probed them, changed clocks,
   registered a runner, or launched a job. The optional Docker Linux route is
   still blocked by the macOS Keychain helper (`-25293`); unlock the existing
   credential path before retrying and do not bypass it or disturb unrelated
   containers. Scheduled Tier-1 runs `33303612803` and `33379410190` were still
   queued at stale sources `b57e4c6a9` and `cf48c8481`; both were canceled before
   any step ran so they cannot later consume the returned runner or produce
   evidence against the wrong revision.
6. **Preserve the completed Wave 2 reconciliation.** The user-supplied 1,034-line
   attachment is captured with its SHA-256 and parsed into 141 rows. Every row has
   a source disposition at `3316e0efe`; 48 remain `awaiting_runtime`. Do not use
   the older public artifact frame or its “Not yet reviewed” footer to replace
   that authoritative capture.

The final local source checkpoint deliberately uses focused regression, related
seam, lint, static, and import-edge gates. Per the user's direction and the rule
now recorded in `code/AGENTS.md`, the full CPU suite was not repeatedly rerun
locally. The required final-source hosted workflow supplied the current broad CPU
result above. Earlier failing broad runs remain diagnostic evidence only.

The legacy HTTP microbenchmark/export routes remain absent and explicitly
unsupported. New HTTP checks cover the real ASGI surface and owned-child failure
handling; they do not restore retired routes. Wave 2 source reconciliation is
complete. The goal remains open until the retained runtime gates close with
current target evidence.
