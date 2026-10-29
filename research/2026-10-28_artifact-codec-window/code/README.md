# Standalone artifact codec pair

These Linux commands reproduce the codec mechanism on a deterministic public corpus and retain standalone observations. This source supplies no performance result or scientific qualification.

The corpus builder reads committed blobs from [public source commit 03ff53d](https://github.com/cfregly/ai-performance-engineering/tree/03ff53d63b5ff91e6ddf556619400c6d277a7d29/code). It selects 1,024 distinct files and retains a ledger, ordered parts, and their hashes. The repository checkout must contain that commit. Dirty working-tree bytes are not inputs.

Run from this directory with an existing Python interpreter. Use absolute paths and new output directories. The workload uses the native `pigz` and `zstd` commands. Nothing installs or downloads them.

```bash
python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache build_corpus.py \
  --repository /absolute/ai-performance-engineering \
  --output /absolute/new-corpus

python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache run_codec_pair.py pin \
  --pigz /usr/bin/pigz --zstd /usr/bin/zstd \
  --output /absolute/new-toolchain

python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache run_codec_pair.py run \
  --input-index /absolute/new-corpus/input-index.json \
  --toolchain /absolute/new-toolchain/toolchain.json \
  --output /absolute/new-pair --cpus 0,1 --order AB
```

Choose two distinct CPU IDs from the calling process's allowed affinity. `A` is pigz gzip with `-6 -p 2 -n -c`. `B` is zstd with `-3 -T2 --no-progress -q -c`. Both receive stdin and verify decompression. Use `--order BA` to reverse the one-pair order. Each arm starts one fresh interpreter with `-I -B -X pycache_prefix=/var/empty/piq-python-cache` and a closed JSON request. The request contains only `input_index`, `output_directory`, `codec`, `toolchain_json`, and `timeout_seconds`. The cache prefix must be absent. Do not create it. Its parent directory may also be absent because bytecode writes are disabled.

`pin` resolves the explicit codec paths to regular files, captures original version and `ldd` stdout/stderr, and hashes the resolved executables and loader/library closure. Metadata commands have a three-second bound and a 64 KiB combined output cap. Missing libraries, unknown `ldd` lines, absent loader, or more than 16 resolved libraries fail. Inspect the retained metadata before trusting a pin. This command invokes the explicitly supplied native programs for metadata. It does not install tools or establish trust in an untrusted executable.

`run` uses a sanitized environment, exactly two allowed CPUs, a 512 MiB address-space limit per process, an 8 MiB file limit, and a 15-second arm budget. It reserves the final quarter-second for termination and leader reaping. These are process limits, not an aggregate cgroup memory reservation. Output is exclusive. The driver retains original child stdout/stderr, request, three workload JSON files, archives, and process records. A failure stops the pair without retry and preserves the later arm as `NOT_STARTED`. `handback.json` is COMPLETE only when both child processes and output checks pass. It otherwise remains PARTIAL.

The driver's `complete_process_wall_ns` starts before `Popen` and ends after owned process-group cleanup and leader reaping, including interpreter startup and exit. `process_exit_observed_ns` separately records the parent's monotonic timestamp when it observes an exited child before cleanup. It remains null if the arm reaches its deadline before that observation. Polling and operating-system scheduling can delay this observation. The delay is not bounded by the polling interval. Parent output validation, archive hashing, and toolchain pinning occur outside the complete-process clock. A timed-out arm records the elapsed failed attempt through cleanup instead of a successful workload time. The child `metric_value` covers its declared adapter-local window. The driver signals its owned process group before reaping the session leader, so it does not target a recycled reaped PGID. It records cleanup actions and leader reaping. It does not independently prove an empty cgroup or prevent descendants from deliberately escaping the process group.

Standalone observations omit PIQ's required telemetry, independent admission, scope ownership evidence, and observer overhead. They are not equivalent to the four treatment pairs and four neutral groups in the prospective PIQ protocol. A single AB or BA pair cannot establish repeatability, a speedup claim, or scientific qualification. `observation_mode` remains `standalone_workload` and `qualification` remains `not_assessed`.

Run focused source checks without compressing the real corpus:

```bash
python3 -I -B -X pycache_prefix=/var/empty/piq-python-cache -m unittest discover -s . -p 'test_run_codec_pair.py' -v
```

Control-flow fixtures are explicitly synthetic. A tiny executor checks the actual process path only on Linux. The tests do not run a native codec study.
