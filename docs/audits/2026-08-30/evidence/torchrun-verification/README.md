# LOCAL-019: withdraw synthetic child-training verification

The generic `TorchrunScriptBenchmark` no longer verifies a parent-side Linear/meta computation as if it were the child training. Its execution and verification hooks now raise an explicit unsupported reason before launching anything, including when an old cached output is injected. All 61 factory callers remain discoverable; the unchanged `get_config` and process-count resolution methods retain their configuration behavior. Actual child-training verification is still unavailable. A GPU allocation alone cannot provide the missing output protocol or independent reference.

The harness now propagates errors from a declared `get_torchrun_spec` getter. A noncallable getter is an error; only a callable getter returning `None` selects the existing module fallback. CPU controls observe zero attempted subprocess launches after invalid or unsupported specifications. A real static-loopback `torchrun` process executes a real CPU child for the legitimate `None` fallback, and a separate real direct child-wrapper invocation also succeeds. These are launcher checks, not training correctness or performance measurements.

The owned changes are limited to `code/labs/train_distributed/training_utils/torchrun_harness.py`, the launch-spec selection region of `code/core/harness/benchmark_harness.py`, and the new `code/tests/test_audit_wave1_torchrun_verification.py`. Parent-owned training README/generator updates and removal of obsolete synthetic-forward hygiene assertions are coordinated separately. No original training script was changed by this slice.

## Evidence and attempts

- `before-source.json` records the two source hashes before this slice and all 61 factory callers. The full former wrapper and former harness method are preserved in the two `*.before.py.txt` files.
- `before-mechanism.json` records the initial actual CPU reproduction: different child arguments produce identical cached toy outputs before any child executes. `reproduce_archived_surrogate.py` replays the preserved old source, with an integrity check, and writes a separate `before-replay.json`. Both are explicitly CPU toy mechanisms, never GPU results.
- `before-tests.xml`: 27 failures and 1 pass against the original source. One failure was a 60-second real elastic `c10d` rendezvous timeout on this Mac; it is not evidence of a production defect. The other failures exposed unsupported synthetic verification and swallowed/invalid launch-spec handling.
- `after-tests.xml`: 30 passes, 1 skip, 1 failure. The remaining failure was a new test fixture expecting a returned error result after a deliberately raised spawn `OSError`; the actual harness correctly propagated it. The fixture was corrected to require propagation and inspect the already-built fallback command. Production error handling was not weakened.
- `integration.xml`: 61 passes and 3 skips before replacing the temporary platform skip with a real static-loopback launch. `static-loopback.xml`: the actual fallback child passed in 4.873 seconds. The earlier rendezvous timeout remains preserved.
- `final-integration.xml`: 66 passes, 2 skips, 7 warnings in 19.539 seconds. All 32 LOCAL-019 cases passed. The two skips are existing seed/config torchrun benchmark-validity tests that require Linux. The warnings disclose the unsupported Darwin benchmark-validity environment.
- `direct-cli-help.json`: four actual direct training CLI `--help` invocations exit 0. This verifies import/argument-parser availability only; it does not execute training.
- `source-validation.json`: all three changed sources parse, scoped `git diff --check` passes, both configuration methods have identical ASTs before/after, and the 61 factory callers remain in the inventory. `ruff-final.txt` records the selected static checks passing.

From repository root, the archived source mechanism can be replayed with:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python docs/audits/2026-08-30/evidence/torchrun-verification/reproduce_archived_surrogate.py
```

From `code/`, the final CPU check is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest tests/test_audit_wave1_torchrun_verification.py tests/test_audit_wave1_harness.py tests/test_seed_config_immutability.py tests/test_audit_wave1_timing_provenance.py -q -rs -p no:cacheprovider
```

## Acceptance boundary

Source/control-plane disposition: the false generic verification path is withdrawn and specification failures cannot silently launch the fallback. The intentional unsupported result is not benchmark success. The local Python 3.12.2 / PyTorch 2.8.0 CPU environment is not the pinned GPU stack. No CUDA, NCCL, distributed training, numerical acceptance, memory claim, or speed claim is established here.

To restore a particular training wrapper as a qualified benchmark requires a child-produced result/state protocol, an independent workload-matched reference, negative controls that reject corrupted and unrelated child results, and actual target training checks with complete provenance. Separate ZeRO training gates remain independent; they do not qualify the generic wrapper or other recipes. Historical training timings are not requalified.
