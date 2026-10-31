# Post package and current retained-evidence checks

This public source snapshot contains the current measurement receipt v3 and
PIQ diagnostic envelope v2 checks. It removes the retired GPU receipt v1, CPU
receipt v2, PIQ envelope v1, and separate two-host supplement paths.

The snapshot is synthetic-only. It contains no real run, host, device, owner,
account, admission, private package, performance result, qualification, draft,
or publication record. The 59-file
[GPU training diagnostic example](examples/piq-gpu-training-diagnostic) uses
explicitly synthetic identifiers and values. The repository-owned
[two-host admissions anchor](trust/piq-v2-two-host-admissions.json) is empty.

Passing these checks proves validator behavior and retained-byte consistency
only. It does not prove that a workload ran, that two physical hosts
participated, that a scientific result qualified, or that a package is ready
to publish.

## Active contracts

Read [PIQ_EXPERIMENT_EVIDENCE.md](PIQ_EXPERIMENT_EVIDENCE.md) for the closed
PIQ v2 variants:

- Stage 0 checkpoint-write batching binds both arms to one exact independent
  oracle identity and output.
- CPU two-host functional evidence requires an out-of-band row in the fixed
  repository anchor. A bundle cannot admit itself, and no command-line or
  environment override is accepted.
- Transformer Engine GPU training binds all five execution groups and ten
  attempts. Every raw attempt interval must fit inside its exact assignment
  window.

Read
[QUALIFIED_GPU_TRAINING_EVIDENCE.md](QUALIFIED_GPU_TRAINING_EVIDENCE.md) for
the fixed two-rank Transformer Engine MXFP8-versus-BF16 qualification contract,
its deterministic evaluator, and its allowlisted public summary. The private
package scaffold and any real admission rows are deliberately not projected.

The public projection guard scans every staged file as bounded byte chunks,
including binary captures. It rejects any retained private execution identity,
even when that byte sequence crosses a scan-chunk boundary.

## Run the snapshot

Use Python 3.10 or newer. The current evidence validators use the standard
library and make no runtime, network, PIQ, GPU, or external-account calls. Run
these commands from this directory:

```sh
python3 -m unittest -v \
  tools.test_qualified_gpu_training \
  tools.test_piq_gpu_training_evidence \
  tools.test_gpu_training_public_summary \
  tools.test_check_diagnostic_finding \
  tools.test_check_public_projection \
  tools.test_check_package

python3 tools/piq_gpu_training_evidence.py \
  check examples/piq-gpu-training-diagnostic

python3 -m unittest discover -s tools -t . -p 'test_*.py'
```

The full discovery suite also covers the retained document compositor and may
need Pillow, ReportLab, and pypdf in the selected Python runtime.

To check one retained package, pass its explicit path:

```sh
python3 tools/check_package.py --help
python3 tools/check_package.py /absolute/package
```

The package gate also retains its reader, source-comment, scientific-image,
rendered-preview, and unscheduled-draft dependencies. None of these offline
tools saves, schedules, publishes, or changes an external account.

## Source identity and provenance

The projected private source revision is
`05ca0567277ecf48d5af732c60aad9a0959489f1`. Its source tree is
`d25e5e1044d487e2b84620f1426f5d93eec81c17`. The public base revision is
`0942e572e82395277c08952a1536639b1d5303f4`.

[source-hashes.json](source-hashes.json) records every file staged under
`research/post-workflow/`, except the hash manifest itself. Each entry names
the public staged path, its public SHA-256, the exact source SHA-256 when the
private revision contains that path, and its provenance. Public adaptations
remove private scaffolding and identifiers, preserve public CLI behavior, and
apply the repository's plain-language policy. Unrelated public workflow files
remain bound to the public base revision.

Verify every staged byte from this directory:

```sh
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("source-hashes.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    actual = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
    assert actual == item["sha256"], item["path"]
print("PASS snapshot hashes")
PY
```

Every performance post and related reply still needs runnable code, inputs,
and retained observed results in this repository. Source checks and synthetic
fixtures cannot substitute for execution evidence.
