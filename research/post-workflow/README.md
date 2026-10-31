# Post package and retained-evidence checks

This public source snapshot checks retained files, hashes, declared identities,
review receipts and exact draft readbacks. It includes the two-host producer and
execution-owner evidence gates, their local builders and synthetic fixtures.
It does not independently establish execution on two physical hosts, scientific
qualification, package readiness or publication. No real owner callback or
result is included. The production [admissions anchor](trust/piq-two-host-owner-admissions.json)
is intentionally empty.

## Run the snapshot

Use Python 3.10 or newer. The evidence gates use the standard library and make
no runtime, network or external-account calls. The full suite also covers the
retained document compositor, which needs Pillow, ReportLab and pypdf already
available in the selected Python runtime. Run from this directory:

```sh
python3 -m unittest tools.test_piq_experiment_evidence tools.test_check_package
python3 -m unittest discover -s tools -t . -p 'test_*.py'
python3 tools/piq_experiment_evidence.py check examples/piq-diagnostic
```

The diagnostic example and the owner-facts example are synthetic format and
test inputs. Their host labels, hashes, PIDs, times and measurements describe no
real run. Tests inject admissions only in isolated fixtures. They leave the
production anchor empty.

To check a retained package, use its explicit path:

```sh
python3 tools/check_piq_two_host_functional.py --help
python3 tools/check_package.py --help
python3 tools/check_piq_two_host_functional.py /absolute/package
python3 tools/check_package.py /absolute/package
```

The functional checker distinguishes an absent declaration, producer consistency
and an admitted owner receipt. Producer-only success exits zero while retaining
its package-readiness block. A declared invalid or unadmitted owner receipt
fails. The full package gate also requires the other content, scientific,
review, image and draft-readback evidence. Its manifest loader rejects identical
and conflicting duplicate JSON keys at every depth before schema dispatch.

## Two-host evidence and builders

Read [PIQ_EXPERIMENT_EVIDENCE.md](PIQ_EXPERIMENT_EVIDENCE.md) for the diagnostic
envelope and producer receipt, then
[PIQ_TWO_HOST_OWNER_AUTHORITY.md](PIQ_TWO_HOST_OWNER_AUTHORITY.md) for the
separate owner receipt and admission boundary. The owner contract binds exact
producer, envelope, handback, run and attempt bytes to declared launch, rank,
physical-host, transport and cleanup observations. Hashes check byte integrity.
They do not authenticate origin or inspect live machines.

The [handback builder](tools/build_piq_two_host_owner_handback.py) and
[receipt assembler](tools/assemble_piq_two_host_owner_receipt.py) read explicit
local inputs and refuse to overwrite existing output. They do not collect
observations, infer authority, create admissions or launch a workload. Start
with the incomplete [owner-facts template](templates/two-host-owner-facts.json).
The [synthetic example](examples/piq-two-host-owner-facts/owner-facts.json) shows
the field layout only.

The two-host declarations require package manifest schema 2. This projection
therefore includes the current schema-2 package gate and its reader, source
comment and Typefully workflow dependencies. None of those offline tools saves,
schedules, publishes or changes an external account. Real evidence, private
draft readbacks and source-review admissions stay outside this reusable
snapshot until separately reviewed for release.

## Source identity and citations

The projected source revision is
`0114c3af6692e9407df667ee9805d1c9aa96d84e`. The public base revision is
`b8531a6779775640087649b9914b16bef4620e6d`.
[source-hashes.json](source-hashes.json) records every snapshot file's SHA-256,
its exact source SHA-256 when available and its public SHA-256. Each entry
identifies unchanged source bytes, retained public files or a public adaptation.
The hash manifest excludes itself. The enclosing Git commit and tree identify
the manifest. Use commit-pinned repository links when citing any of these files.

Verify the snapshot hashes from this directory:

```sh
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
for item in json.loads(Path("source-hashes.json").read_text())["files"]:
    actual = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
    assert actual == item["sha256"], item["path"]
print("PASS snapshot hashes")
PY
```

Entries marked `verbatim_source` match the pinned source bytes. Public
adaptations narrow evidence wording, add CLI help, apply plain-language policy
and sanitize private test fixtures. The duplicate-key loader and its source
regressions match the merged source. Additional public tests cover nested
duplicates, clean CLI errors, help and the owner-assertion output boundary.
Three test modules replace private artifact paths or account identifiers with
synthetic values. A package marker enables standard test discovery. The hash
manifest records each adaptation while preserving the original source digest.

Passing these tests verifies the source projection and retained-byte checks.
It supplies no execution receipt, host observation, performance result or
publication receipt. The production admissions anchor remains an unchanged,
empty source copy.

## Other package checks

The package gate retains six audience variants, image and evidence hashes,
independent review receipts, readable scientific previews and unscheduled
Typefully readback requirements. Follow [PROFILE_CONVENTIONS.md](PROFILE_CONVENTIONS.md)
for scientific profiles and [QUALIFIED_CPU_EVIDENCE.md](QUALIFIED_CPU_EVIDENCE.md)
for the separate fixed CPU qualification contract.

Every performance post and related reply still needs its runnable example,
inputs and retained results in this repository. Commit and verify the matching
code and scientific images before calling a draft ready. Source-only checks
and synthetic fixtures cannot substitute for observed results.
