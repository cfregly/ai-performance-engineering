# Diagnostic tools validation

Validation date: September 26, 2026.

The [six diagnostic tools](diagnostic_tools.md) have runnable source, CLI and MCP
entrypoints, guides, and focused regression coverage. This record covers source
and local execution checks. GPU, engine, and fabric qualification remains open.

## Source and environment

- Base Git revision: `69bc26d7bdb48d96b26447483be171b75e6b9057`.
- Validation used the modified working tree, including the new tool files.
- Host: macOS without CUDA hardware.
- Python: 3.12.2 in `/tmp/aisp-diagnostics-venv`.
- Process and HTTP dependencies used the repository pins `psutil==7.1.0` and
  `httpx==0.28.1`.

## Focused regression result

**113 tests passed in 32.16 seconds.** This was a focused suite, not the full
repository test suite. The exact command ran from `code/`:

```bash
/tmp/aisp-diagnostics-venv/bin/python -m pytest -q \
  tests/test_network_diagnosis.py \
  tests/test_cross_layer_diagnosis.py \
  tests/test_collective_diagnosis.py \
  tests/test_fabric_diagnostics.py \
  tests/test_transport_diagnostics.py \
  tests/test_serving_comparison.py \
  tests/test_diagnostic_process.py \
  tests/test_diagnostics_integration.py \
  tests/test_fabric_evaluator.py \
  tests/test_tools_cli.py \
  tests/test_mcp_docs.py \
  tests/test_mcp_tools.py::test_expected_tool_registration_matches_catalog \
  tests/test_mcp_tools.py::test_tool_list_protocol_matches_registration \
  tests/test_mcp_tools.py::test_suggest_tools_common_intents
```

Coverage includes the following execution paths:

- All six public CLI help paths, a real network calculation through CLI and MCP,
  and generated MCP documentation parity.
- Retained packet, socket, counter, transport, and collective analysis, including
  missing evidence, counter resets, malformed inputs, and failed correctness.
- Cross-layer joins with clock, workload, host, unit, and interval checks.
- Concurrent artifact writers and refusal to overwrite existing tool outputs.
- A complete four-arm serving matrix over local HTTP/SSE fixture servers, invoked
  through the public CLI, with failures, cancellation, and output checks.
- A local two-upstream test of the vLLM proxy's KV handoff protocol and errors.
- Real child processes, detached descendants, bounded shutdown, and timeout
  escalation. These tests check process ownership and cleanup.
- Native serving profile schemas, launch paths, and example launch digests.

The serving fixtures exercise protocol handling. Their token streams and telemetry
are test data, not engine output or measured GPU performance.

An additional isolated MCP `system_env` envelope test passed in the same virtual
environment. An earlier broader MCP attempt in the system Conda interpreter hit
a subprocess error and hung. It is not counted as a passing suite. Its owned test
processes were stopped, and the affected envelope checks passed in the isolated
environment.

## Static and manual checks

Ruff passed for the new tools and tests. All 30 changed or new Python files
compiled. The seven new guides and overview had no missing local links.
`git diff --check` passed.

The CLI bandwidth-delay calculation produced 250,000,000 bytes for 10 Gbit/s and
200 ms. Live network collection on macOS returned an explicit unsupported result.
Neither result was presented as a network throughput measurement.

## Hardware qualification

No live vLLM/SGLang comparison, GPU collective, or RoCE/InfiniBand workload ran on
this host. A read-only remote GPU inventory attempt stopped at SSH host-key
verification. Existing trust configuration was left intact.

Hardware qualification requires a trusted target connection, the declared CUDA
and fabric capabilities, prepared transport peers, and pinned serving engines.
Run the commands in each tool's guide and retain their raw artifacts. Diagnostic
results remain separate from canonical benchmark speedup claims.
