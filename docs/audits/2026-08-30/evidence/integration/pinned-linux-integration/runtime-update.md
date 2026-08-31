# Remaining acceptance after Wave 2 source reconciliation

This adds to [the earlier runtime handoff](../runtime-handoff.md). Source and
metadata receipts retain their own identities. Final-source hosted CPU/static/
dashboard validation and a four-target CUDA compiler matrix now pass. The exact
row mapping is retained in the [non-GPU reconciliation](../hosted-non-gpu-runtime-closure/receipt.json)
and [CUDA 13 compile receipt](../hosted-cuda-compile-closure/receipt.json). Those
workflows do not certify the full target Linux installation, GPU correctness,
profiler capture, numerical acceptance, or performance.

1. **Retain the completed CPU provenance cell and finish target installation
   provenance.** Final-source Benchmark Validation run `33391774956` completed
   on hosted Linux with **4,346 passed, 461 explicit skips, and zero failures**.
   Its retained [hosted CPU closure receipt](../hosted-cpu-closure/receipt.json)
   confirms that the job explicitly installed `requests==2.34.2` and
   `tokenizers==0.22.2`, collected the complete `tests/` tree, and uploaded the
   reconciled JUnit artifact. This verifies LOCAL-025 and closes W1-006's Linux
   CPU full-directory CI sub-gate without a rerun. W1-006 remains pending for
   explicit supported-GPU execution of its CUDA protection cases. The 461 skips
   are heterogeneous and include 62 known missing-protection,
   protection-summary, or absent-route scope skips; none is converted to passing
   protection coverage.
   The later focused [Linux CPU provenance receipt](../linux-cpu-provenance/README.md)
   records a clean CPython 3.12 x86_64 installation of the reviewed 20-direct-pin,
   56-distribution lock at `cf801679b`: every selected origin and SHA-256 is
   retained, `pip check` passes, all 20 direct imports resolve inside the isolated
   environment, and Torch `2.9.1+cpu` completes a CPU tensor operation. That
   closes this bounded CPU provenance sub-cell. It intentionally excludes the
   separate CMake and Prometheus packages used by Benchmark Validation and does
   not certify the complete 90-specification/327-package target graph, CUDA
   imports, native extensions, or GPU execution. Reproduce those remaining cells
   on the supported target with origins, hashes, resolved versions, and
   `pip check` preserved.
   The same retained JUnit now verifies seven exact Wave 1 host/configuration
   findings: W1-007, W1-052, W1-055, W1-057, W1-067, W1-111, and W1-112.
   Their mechanisms are Make routing, failure propagation, architecture identity,
   or wrapper paths; no device result is intrinsic. The 76-row local matrix is
   therefore **7 verified and 69 pending**. All 48 Wave 2 runtime rows also have an
   exact final-source hosted CPU/source-regression subgate, but zero whole Wave 2
   rows close from CPU evidence.
2. **Exercise the reviewed Linux/CUDA dependency graph.** All 90 current direct
   specifications resolve to 327 unique CPython 3.12 x86_64-manylinux packages
   under one reviewed GPUtil 1.4.0 source-distribution metadata exception.
   Execute the staged setup transactions on a supported Linux/CUDA host, verify
   origins/hashes and `pip check`, and run imports and native extension builds.
   Confirm that the six Darwin import holds — Triton in four entrypoints,
   cuda-python in one, and FlashInfer in one — load from the pinned target
   environment. Metadata resolution alone cannot close this gate.
3. **Run the current CUDA/NCCL correctness matrix after GPU custody returns.**
   The four ZeRO factories now pass source, matched CPU/Gloo update parity, and
   fresh CPU/Gloo child-result verification under all four factory paths. They
   still require actual two-GPU CUDA/NCCL execution, completed
   reduce-scatter/all-gather observations, CUDA RNG checks, and cleanup evidence.
   The other 57 generic wrappers remain deliberately unsupported by the harness.
   Do not count their refusals or CPU/Gloo results as GPU coverage.
4. **Execute the finding-specific hardware gates from the ledger.** The current
   [Wave 2 matrix](../wave-2-complete-source/runtime-gates.md) contains 48 exact
   rows. Retained run `33391774950` already supplies bounded four-target compiler
   evidence for nine exact Wave 2 sources; `W2-078` has only a configured-target
   header result and still lacks `sm_90`/H100. It also supplies bounded compiler
   evidence for 17 pending Wave 1 rows. The remaining contracts include unbuilt
   consumers and extensions, chapter validation runners, full-output numerical
   checks and reviewed policies, stream/sanitizer checks, exact B200 Nsight
   Systems/Nsight Compute captures, and Grace/NUMA observations.
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
