# Optimization input adapters

The former automatic optimizer execution path has been removed. It accepted one-off timings, estimated CPU and GPU splits, and could overwrite candidates without the repository's correctness and provenance gates. It must not produce benchmark results or promote code.

Use the trusted benchmark harness to collect repeated control and candidate measurements. Record those measurements with the campaign controller:

```bash
python -m core.optimization.campaign --help
```

The full workflow is documented in [Autoresearch campaigns](../../../docs/autoresearch_campaigns.md).

## Retained adapters

The package keeps source discovery and output adapters for campaign integrations:

```python
from core.optimization.auto import BenchmarkAdapter, FileAdapter, RepoAdapter
```

These adapters read and write code. They do not benchmark, verify, gate, or promote a candidate.

## Fail-closed behavior

Every former module invocation exits with status 2 and prints the supported migration path:

```bash
python -m core.optimization.auto
```

A measured campaign needs all of the following before promotion:

- a frozen workload and environment contract
- repeated control and candidate trials
- explicit correctness evidence
- per-case regression gates
- hashed artifacts and Git provenance
- review through the campaign promotion frontier
