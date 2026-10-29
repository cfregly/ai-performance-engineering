# Post package checks

This directory contains the post-package validator, its local dependencies and
focused tests. It checks content, six audience variants, image and evidence
hashes, reader-visual declarations, review receipts, exact Typefully readback,
and unscheduled draft state.
CPU receipt v2 also recomputes a fixed prospective codec study and verifies its
required telemetry. This snapshot contains no measured study results.

Run from this directory:

```bash
python3 -B -m unittest -q tools.test_check_package tools.test_check_reader_visuals tools.test_piq_experiment_evidence
python3 tools/piq_experiment_evidence.py check examples/piq-diagnostic
python3 tools/check_package.py /absolute/package/path
```

## CPU evidence and qualification

[Qualified CPU evidence](QUALIFIED_CPU_EVIDENCE.md) documents receipt v2, its
schema and Python APIs. It requires an advance design and registration, four
fresh treatment pairs, four neutral controls, complete native run identities,
exact round trips, fixed gain/latency/size rules and independent review records.
The telemetry gate verifies retained summary/raw bytes, source-bundle hashes,
process identities and terminal coverage. Missing required collection blocks
acceptance. Host, cgroup, filesystem and network context retain their stated
limits. GPU, fabric and network workload execution cannot be inferred from CPU
context.

[PIQ diagnostic evidence](PIQ_EXPERIMENT_EVIDENCE.md) remains a separate v1
contract. A diagnostic import does not qualify a performance claim. GPU receipt
v1 keeps its existing requirements. The tests and example bundles are explicitly
synthetic and describe no actual execution.

A passing receipt cannot authenticate execution or reviewer independence, prove
GPU qualification, or publish content. The complete package and platform-preview
requirements still apply.

## Analytical and reader-visual contracts

[Reader-facing checks](READER_STANDARD.md) explain the example, mechanism,
comparison, scope and actual-preview requirements. The reader-visual validator
supports measured/source-reported schema 1 and the separately authorized
[analytical schema 2](ANALYTICAL_PACKAGE.md).

Schema 2 requires an explicit human instruction for one named package, retained
synthetic derivation code/results, independent recalculation and current
copy/chart/preview reviews. It does not establish a measurement, runtime,
speedup or feasible implementation. An assistant proposal cannot supply the
human authorization, and approval of one package does not authorize another or
publication. The test authorization records are synthetic fixtures only.

The package gate rejects a missing required reader declaration. It checks
ordered verbatim copy excerpts, exact saved text and chart attachment identity,
source hashes, screenshots and review bindings. The reviewer still judges the
arithmetic, assumptions, scope and legibility. A hash match cannot issue a
reviewer's PASS.

The PIQ helper also supports bounded indexed-original byte transport. That path
checks the exact file census, sizes and hashes, rejects unsafe paths and keeps a
partial receipt on failure. It does not verify science, qualify performance or
make imported data publishable. The snapshot includes its synthetic transport
tests, with no real imported originals.

## Portable transport test fixtures

The indexed-originals importer rejects symlink ancestors as well as symlink
files. If the default temporary path contains a symlink, use a canonical parent
for the test fixtures. This changes only where the unchanged tests create their
temporary files. It leaves the importer's symlink checks and negative tests in
place. The following command runs the package, reader-visual, diagnostic and
indexed-originals suites together:

```bash
python3 -B - <<'PYTEST'
from pathlib import Path
import tempfile
import unittest

tempfile.tempdir = str(Path(tempfile.gettempdir()).resolve())
suite = unittest.defaultTestLoader.loadTestsFromNames([
    "tools.test_check_package",
    "tools.test_check_reader_visuals",
    "tools.test_piq_experiment_evidence",
    "tools.test_indexed_originals_import",
])
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PYTEST
```

## Source identity

`source-hashes.json` lists the tools, tests, schemas, synthetic examples and
supporting contracts. Entries marked `pipeline_source` match the current
pipeline source byte for byte. The generic README and reader guide are marked
`snapshot_documentation` and maintained here. The hash manifest does not hash
itself. Verify every listed snapshot file from this directory:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for item in json.loads(Path("source-hashes.json").read_text())["files"]:
    assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
print("PASS snapshot hashes")
PY
```

## Clear, runnable examples

Every performance post must link to its runnable example in this repository.
Include the command, runtime requirements, inputs, and observed run evidence.
An image or source-only check does not prove that an example ran on the GPU.
Commit and push the code, tests, scientific images, and shareable captures before
calling a related draft ready. Preserve completed runs and failed controls.

Name comparisons in ordinary language within each standalone post or image.
For the nested-graph study, say “with nested graph-break support” and “without
it”. Keep exact identifiers in code and methods references. Enabling support
does not remove graph breaks from the workload.

The wording check covers public article, social copy, and image captions. It
leaves code examples, raw measurements, profiler exports, and archived sources
intact. Update this snapshot, dependencies, tests and hashes whenever the
pipeline validator changes.

## Scientific profile previews

Declare `scientific_profiles: true` in a package manifest and follow [PROFILE_CONVENTIONS.md](PROFILE_CONVENTIONS.md). Run `python3 research/post-workflow/tools/check_package.py /absolute/package/path`. The preview gate binds every media ID to its source file, checks all document pages and rejects crop or overlay failures. Review the final Typefully editor and phone-width preview after upload. Keep source, final draft, schedule and publication as separate states.

## Configurable shared pipeline

`tools/pipeline_config.py` validates an owner profile from `pipeline.example.json`: private drafts repository, public repository and subdirectory, local research hours and IANA timezone, GPU host and theme. It performs no external actions. The same configuration and profile-preview tools are proposed upstream so authors can use different profiles with common code. `check_public_projection.py` validates an explicit staged public copy against retained sources, including scientific captures, without exporting or publishing files. New private drafts may require the claim-scope template before their final check.
