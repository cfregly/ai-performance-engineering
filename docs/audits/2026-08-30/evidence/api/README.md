# P10 API and parser remediation evidence

This package covers original findings **W1-017, W1-023, W1-059, W1-060, W1-064,
W1-107, and W1-110**. The expired Anthropic default is an adjacent discovery;
it does not change the 128-finding source inventory.

## Changes

- Bare `aisp`, `aisp tui`, and `aisp bench tui` open the existing benchmark
  analysis menu. `bench tui --simple` remains an accepted compatibility flag.
  EOF and Ctrl-C close the menu; unavailable functionality is not blamed on curses.
- Output-token budgets use the caller override, then the configured budget.
  Environment-created configurations use the existing 4096-token dataclass
  default. Nonpositive, Boolean, and noninteger call budgets fail before HTTP.
  The client does not invent provider limits or silently increase/clamp budgets.
- `engine.ai.ask` calls the real book index and returns serializable citations,
  including when no LLM is configured. `engine.ai.explain` extracts two sentences
  and up to five bullet points.
- NVLink status preserves GPU IDs, link IDs, and fractional rates. Inactive-only
  output reports unavailable. An explicitly active link without a reported rate
  has a null rate and null aggregate plus a warning. Aggregate rates sum the
  reported GPU-link endpoints; they are not fabric bisection bandwidth.
- Distributed NVLink discovery reads peer GPU IDs from `nvidia-smi topo -m`.
  Local port IDs cannot identify peers. Invalid/incomplete matrices produce a
  visible warning; GPU count alone no longer establishes NVLink or NVSwitch.
- CUTLASS version parsing reads numeric macro definitions only and reports an
  unknown version with a warning for incomplete or invalid definitions.

## Adjacent Anthropic default correction

The shared default is `claude-sonnet-4-6`, also used by the three legacy advisor
clients. Explicit constructor and environment model overrides are preserved.
Anthropic's [lifecycle documentation](https://platform.claude.com/docs/en/about-claude/model-deprecations)
states that Sonnet 4 retired on June 15, 2026 and names Sonnet 4.6 as its
replacement; Sonnet 4.6 remains available. Checked August 30, 2026.

This is a compatible model-ID correction, not a migration to Sonnet 5. The
[Sonnet 5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide)
documents changes to sampling parameters and thinking/response handling that
require a separate client migration.

The NVLink matrix interpretation follows NVIDIA's
[nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
`NV#` describes a connection traversing a bonded set of NVLinks.

## Reproduction and validation

Exact commands, environment versions, final source hashes, and artifact hashes
are recorded in [receipt.json](receipt.json). Original failing and subsequent
passing logs are retained without replacing earlier attempts.

From `code/`, the final focused command is:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_audit_wave1_api_regressions.py tests/test_audit_wave1_parser_regressions.py tests/test_engine_fail_fast.py tests/test_perf_core_runtime_warnings.py
```

Tests invoke the real CLI, real file-backed book/CUTLASS readers, real subprocess
parsers supplied with declared text fixtures, and real loopback HTTP transport
for OpenAI Chat/Responses, Anthropic, vLLM, and Ollama payloads. Protocol fixtures
do not evaluate models or establish model quality. No paid or external LLM
requests were made, and no private prompts were transmitted.

This verification used an existing CPU Python environment, not the pinned GPU
stack. No CUDA benchmark, GPU correctness result, throughput claim, or live
hosted-provider availability is established here. The actual local NVLink
status invocation correctly reports missing `nvidia-smi`; successful GPU status
parsing is fixture-backed rather than a live NVLink-host receipt.
