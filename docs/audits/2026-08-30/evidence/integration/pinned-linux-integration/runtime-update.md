# Remaining acceptance after the targeted source checkpoint

This adds to [the earlier runtime handoff](../runtime-handoff.md). Source and
metadata receipts retain their own identities. No Linux installation, GitHub
workflow, CUDA build, GPU correctness, profiler capture, or performance result is
certified by local Darwin checks.

1. **Run the pinned Linux CPU workflow on an available isolated x86-64 host.**
   Install the exact workflow packages and indexes, record Python and wheel
   origins/hashes, resolved versions, and `pip check`, then execute the workflow's
   required tests with JUnit and explicit skip reasons. Current metadata shows 20
   matching direct workflow pins, a 56-package exact-source union, and a
   49-package first-PyPI closure. These are resolution results, not an installed
   Linux environment or a GitHub Actions run.
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
4. **Execute the finding-specific hardware gates from the ledger.** These include
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
   containers.
6. **Capture the completed second-wave report before final reconciliation.** The
   supplied artifact still reports the original 128 findings and a “Not yet
   reviewed” future-coverage footer. That footer is not wave 2, and missing input
   is not zero findings.

The final local source checkpoint deliberately uses focused regression, related
seam, lint, static, and import-edge gates. Per the user's direction and the rule
now recorded in `code/AGENTS.md`, the full CPU suite was not rerun after the final
tensor-parallel fixes. The preserved 4,181-pass/450-skip/1-failure run is a
diagnostic that led to those fixes, not post-fix integration evidence.

The legacy HTTP microbenchmark/export routes remain absent and explicitly
unsupported. New HTTP checks cover the real ASGI surface and owned-child failure
handling; they do not restore retired routes. The goal remains open until the
runtime gates above and the received second wave reconcile cleanly.
