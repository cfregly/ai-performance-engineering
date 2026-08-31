# Remaining P10 tooling evidence

W1-092, W1-106, and W1-113 are remediated. Scope: exporter wire format,
explicit CLI artifact behavior, and corrected versioned advisory code. No real
multi-GPU telemetry collection or FP8 GPU execution is claimed.

- **W1-092:** the original formatter produces a strict Prometheus parser error,
  `second HELP line`, for two GPU-labelled input series. The replacement emits
  HELP/TYPE once per family and snapshots application metrics under its lock.
  Real `promtool 3.12.0 check metrics --extended --lint=none` accepts the new
  0/1/2/8-GPU input fixtures. This disables naming-style lint, not text parsing;
  the same command rejects the original duplicate metadata. The Python client
  parser checks all supplied labels/sample counts. An actual loopback HTTP scrape
  passes the strict parser. Fixture labels do not assert hardware existence.
- **W1-106:** `aisp profile flame INPUT.json --output OUTPUT.html|OUTPUT.json`
  reads complete Chrome trace events through the existing generator and writes
  the requested artifact. HTML is an offline SVG view with escaped names and
  no external scripts. Invalid, missing, empty, and non-finite/negative-duration
  input fails without a success artifact. The view explicitly reports summed
  event durations grouped by category, not elapsed wall time or reconstructed
  call stacks. Memory/kernel placeholder commands now exit nonzero with an
  explicit unsupported message instead of pretending analysis ran.
- **W1-113:** the original FP8 string raises `TypeError` before entering its
  context. The replacement is explicitly **integration guidance, not a standalone
  executable**: Transformer Engine 2.18.x `te.autocast` with `DelayedScaling` and
  an already-ported TE model. Hardware, shape, quality, and API-version constraints
  are stated. The old blanket throughput/accuracy promise is removed; numerical
  estimates are marked illustrative and unmeasured. The real CUDA+TE numerical
  smoke test is skipped on this CPU host, not counted as a pass.

The focused run reports **17 passed, 1 skipped** including an existing what-if
helper check. The skip requires actual CUDA and Transformer Engine 2.18.x.
See [receipt.json](receipt.json) for combined results, exact commands, hashes,
diagnostic attempts, and limitations.

[cpu_trace.json](cpu_trace.json) is a real torch.profiler CPU capture of three
matrix multiplications and ReLUs. The real CLI produced [cpu_flame.html](cpu_flame.html)
and [cpu_flame.json](cpu_flame.json), containing the observed `audit_cpu_matmul`
scope. It is a functional artifact check, not a GPU or performance benchmark.

From `code/`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_tooling_regressions.py tests/test_core_helpers.py::test_whatif_and_ncu_empty
```

The protocol/API contracts were checked against the official
[Prometheus exposition specification](https://prometheus.io/docs/instrumenting/exposition_formats/)
and the archived [Transformer Engine 2.18 FP8 guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/examples/fp8_primer.html).
The newer `te.autocast(..., recipe=...)` API is documented there;
[`fp8_autocast` is deprecated](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#deprecated-functions).
