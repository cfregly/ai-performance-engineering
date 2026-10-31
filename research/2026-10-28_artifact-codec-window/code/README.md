# Public text-artifact input and codec workload

This directory reconstructs a fixed public corpus and carries the standalone
projection of the artifact-codec workload. It supplies source and input
authority. It does not supply a performance result, scientific qualification,
or authorization to run the real corpus through a codec.

The current builder reads only Git objects from [public source commit
`03ff53d`](https://github.com/cfregly/ai-performance-engineering/tree/03ff53d63b5ff91e6ddf556619400c6d277a7d29).
It accepts the closed, case-normalized text extension set in `build_corpus.py`,
deduplicates by Git blob, sorts by descending byte size and ascending path, and
takes the first 4,096 objects. Dirty working-tree bytes are never inputs.

The frozen ledger's `selection` field uses the historical phrase "regular
committed blobs." The exact implementation admits only Git mode `100644`.
Mode `100755` blobs are excluded even though Git also represents them as
regular files. When one blob appears at multiple paths, the builder retains the
first path in `git ls-tree` traversal order before applying the final size and
path sort. The retained ledger enumerates the exact selected path and object ID
for every entry. Do not broaden or reinterpret this closed rule when
reconstructing the corpus.

The builder fails unless it reproduces all of these preregistered values:

- 4,096 source files and 107,929,355 source bytes
- a 906,568-byte canonical ledger with SHA-256
  `c4e456506ba8431a24dd5145622d750d9782b5a8826429062a71b874a9d12406`
- a 112,537,600-byte deterministic PAX tar with SHA-256
  `cf533d52f44ab5c2fef3a79c6d8f4f821083a0a1f0ff961e63981ac8f31a9e4c`
- four parts at exact 30 MiB split boundaries
- an 888-byte canonical input index with SHA-256
  `6a5bc63255247dff8f8398b4eb98d40e3d0dc589971968f59dbe0bbfbb899680`

The maintained workload accepts exactly four parts. Its corpus cap is 128 MiB,
each input part is capped at 30 MiB, the ledger is capped at 2 MiB, and every
archive or retained file is capped at 32 MiB. The 2 MiB difference between an
input part and an archive is a hard expansion guard. The complete-process
runner retains its 512 MiB address-space limit and 15-second arm limit.

The earlier 1,024-file input remains pinned at [commit
`89f3cd4`](https://github.com/cfregly/ai-performance-engineering/tree/89f3cd484097ad40a4c21ef8e36b349add07b98b/research/2026-10-28_artifact-codec-window).
This maintained successor does not alter or relabel that closed cohort.

## No-compression reconstruction and audit

Run from this directory with Python 3.10 or newer and a checkout that contains
the source commit. The source-only suite is verified on Python 3.12, 3.13, and
3.14. Python 3.9 is unsupported because the code uses `zip(..., strict=True)`
and modern union type syntax. Use absolute paths and a new output directory.

```bash
python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache build_corpus.py \
  --repository /absolute/ai-performance-engineering \
  --output /absolute/new-public-text4096-corpus

python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache audit_input.py \
  --repository /absolute/ai-performance-engineering \
  --corpus /absolute/new-public-text4096-corpus \
  --output /absolute/new-public-text4096-corpus/INPUT-BYTE-AUDIT.json
```

`audit_input.py` invokes no codec. In a separate pass it re-reads every selected
Git object and SHA-256, the canonical ledger and index, every part, the aggregate
tar, and the order, metadata, and bytes of all 4,097 tar members. A mismatch is
a hard failure. It is not permission to create a different corpus.

The repository retains the small index, receipt, audit, and ledger under
`../evidence/input/`. The four generated binary parts are intentionally omitted.
Reconstruction must reproduce their exact sizes and hashes from the retained
authority.

## Source-only checks

```bash
python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache \
  -m unittest discover -s . -p 'test_*.py' -v
```

The tests use synthetic identity transforms and tiny control-flow fixtures.
They do not run pigz, zstd, or a native codec comparison.

## Execution boundary

`codec_workload.py` preserves the native commands and the adapter-local
measurement window. A parent process owns the primary complete fresh-process
clock. `run_codec_pair.py` remains available for standalone mechanism review,
but this corpus must not be passed to its `run` command as a pilot, warmup,
diagnostic pair, or benchmark. A prospective study requires separate frozen
authority, fixed scheduling, PIQ telemetry, admission, ownership, and retained
evidence. None of those are established here.
