# Nanochat wave-1 remediation evidence

Source remediation for W1-003, W1-034, W1-035, W1-036, W1-085, and W1-121.
CPU correctness/control checks pass. CUDA replay, stream ordering, asynchronous
host copies, cluster execution, and performance qualification remain **HOLD**.
See [receipt.json](receipt.json) for exact source hashes, commands, and outcomes.

## Behavior and boundaries

- Dense single-token graph decode uses a device position for KV writes and rotary
  lookup, with fixed-capacity K/V and a device-updated mask. Capture and warmup do
  not consume a token. Replay increments device position and host metadata once.
  Captures retain the actual cache and are invalidated by cache reset, storage
  changes, a different cache, or different input shape/dtype/device.
- Graph decode uses **masked full-capacity SDPA**, not unmasked FA3. Its attention
  traffic can exceed eager decode; correctness does not establish a speedup.
  Preallocate prompt plus decode capacity. Capacity/rotary overflow, padded or
  masked inputs, custom resident/clustered kernels, CPU inputs, and capture
  failures raise explicitly. There is no silent eager fallback. Manual graphs
  do not also invoke compiler-managed CUDA graphs.
- The legacy `enable_persistent_decode` flag denotes reusable buffers and a
  dedicated CUDA stream, not a resident kernel. The stream waits for producers,
  rejoins the caller before consumption, and records tensor allocator lifetimes.
  Tiny token D2H copies block before Python reads them. A requested side-stream
  decode on a CPU model raises instead of silently executing eager decode.
- Both timing helpers measure synchronized wall time, including host submission
  and all CUDA streams. Incremental inference routes decode through `Engine` and
  checks the observed execution mode. Its steady-state graph timing excludes
  capture setup; requested but unavailable CUDA/FA3/CTA backends fail visibly.
  Graph result metadata exposes the changed attention path.
- The B200 sweep propagates mode flags to attention modules; changing only the
  config object did not update modules that cached those flags.
- CTA hint tests use identical, nonzero weights and identical inputs; shape,
  finiteness, and numerical parity are asserted. Real cluster/backend and stream
  integration checks have separate CUDA capability gates. The standalone test
  command reports skips and propagates failures with a nonzero exit status.

Adjacent prerequisites discovered here: rounded-cache prefill now copies only
the valid prefix, and the non-flash SDPA preference is passed as an API-supported
list. CPU explicitly selects a supported math path. These are separate discoveries,
not extra original finding IDs. The original W1-121 source already exited nonzero
on raised exceptions; its missing assertions and misleading capability claims
were the reproduced defects.

## Validation

From `code/`, using the existing CPU environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NANOCHAT_DISABLE_COMPILE=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_nanochat_regressions.py labs/nanochat_fullstack/test_new_optimizations.py
NANOCHAT_DISABLE_COMPILE=1 /opt/miniconda3/bin/python -m labs.nanochat_fullstack.test_new_optimizations
/opt/miniconda3/bin/python scripts/linting/check_benchmarks.py labs/nanochat_fullstack
```

The focused suite reports **12 passed, 11 skipped**. The skips require real CUDA:
six FP32/BF16 graph cases, one side-stream/token-copy case, one all-stream timer
case, and three backend/integration cases. CPU device-position tests compare six
successive decode positions for batch sizes 1/2/3 to both eager-cache and complete
sequence references, including every valid KV element and poisoned unused tails.
They are CPU tensor tests, not simulated CUDA graph evidence.

Sixteen existing CPU test functions also passed directly via `runpy`; this narrow
run does not load the unrelated tokenizer-building test conftest and does not
qualify the tokenizer or the entire lab suite. Compilation and the lab benchmark
linter pass (two existing harness entrypoints, no errors/warnings).

The initial red logs include unsupported CPU GQA fused-backend selection and the
tuple/list API mismatch. They are retained as diagnostic attempts, not evidence
that a CUDA race was reproduced. No model checkpoint download, paid API call,
CUDA benchmark, or performance qualification was performed.

## Remaining acceptance gates

1. On the pinned CUDA stack, run the same focused suite with **zero unexpected
   skips**, allowing a cluster-specific skip only if explicitly recorded as an
   unsupported backend. Repeated replay must retain one graph per cache, match
   per-step eager logits/KV in FP32 and BF16, advance both counters, and recapture
   for the second cache without corrupting either sequence.
2. Run the real side-stream ordering/D2H and all-stream timer tests. Validate
   allocator lifetime under repeated producer/consumer work. Run supported
   cluster/backend checks with numerical parity; a hint-only CPU pass is not
   cluster execution evidence.
3. Run representative fixed-seed, identical-weight eager/side-stream/graph arms
   with the [workload spec](benchmark_workload_spec.yaml), clock/topology/load
   receipts, repeated alternating A/B trials, raw timings, and Nsight traces.
   Report capture cost separately and validate the actual attention backend.
   Historical README speedups do not qualify this changed graph path.

Primary protocol reference: [PyTorch 2.8 CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html)
describes stream dependencies/lifetimes, side-stream warmup, graph replay with
fixed storage, and capture limitations. This host runs torch 2.8.0 CPU; it is not
the pinned CUDA qualification environment.
