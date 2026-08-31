# Hosted Linux CPU provenance receipt

[Audit Linux CPU Provenance run 33401585682](https://github.com/cfregly/ai-performance-engineering/actions/runs/33401585682)
passed at source revision `cf801679b0df897e0b558d668ee42d5ac789d633`.
The dispatch-only workflow used a clean GitHub-hosted Linux X64 runner and
CPython 3.12 virtual environment. It did not rerun the repository test suite.

The job installed the reviewed 20-direct-pin, 56-distribution CPU lock in one
`pip --isolated --require-hashes` transaction. It retained the selected origin
URL and SHA-256 for every artifact, ran `pip check`, captured `pip inspect`, and
imported all 20 direct packages under Python isolated mode. Every imported module
resolved inside the fresh virtual environment. Torch reported `2.9.1+cpu`, no
CUDA runtime, and produced the expected `[[11.0]]` CPU tensor result.

The uploaded artifact is
`audit-linux-cpu-provenance-33401585682-1`, ID `9761526418`, with verified
archive digest
`sha256:69fc36ebab042fcd1753ff22f6627cb35e8a564bb8c6d67e82e4f33b453c3bf9`.
Its complete retained contents are in
[`vendor/run-33401585682`](vendor/run-33401585682), and the structured summary
is [`receipt.json`](receipt.json). The `vendor/` segment marks this immutable
external artifact as audit evidence so dependency discovery does not treat its
checksum records as live project manifests.

This closes only the reviewed W1-005 Linux CPU provenance sub-gate. The current
Benchmark Validation environment separately installs CMake 3.31.10 and
prometheus-client 0.21.0, which are outside this 56-distribution lock. The
90-direct-specification, 327-package Linux/CUDA graph still requires an actual
supported target installation, CUDA imports and native builds. This receipt has
no GPU, numerical, sanitizer, profiler, or performance acceptance value.

The earlier run `33401209446` was canceled before validation after review found
that import isolation and origin allowlisting needed hardening. Its artifact is
diagnostic only and receives no acceptance credit.
