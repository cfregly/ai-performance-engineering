# PyTorch profile extraction follow-up

This adjacent P11 discovery is an offline extraction bug, separate from the original 128 audit findings. The documented helper passed help parsing but raised `ValueError: Invalid suffix` for a real matching profile directory. It also rejected absolute paths/globs and treated unrelated empty directories as captures.

The helper now appends metadata/operator filename suffixes to the requested prefix (including dotted prefixes), accepts direct/absolute/glob capture paths, and ignores unrelated directories. It does not modify source captures or infer CUDA qualification from CPU data. Metadata error fields are preserved when extracting failed capture metadata.

`initial-failure/` preserves the first CLI reproducer and its fixture. `original-extract_pytorch_profile.py.txt` preserves the source before this correction. Four regression controls first failed (`before.txt`) and then passed (`after.txt`, 4 passed in 3.43 seconds). They create a real CPU torch.profiler capture using the production profiling runner with CUDA visibility disabled, exercise relative/absolute/glob selectors and dotted prefixes, and compare extracted operator fields with the real capture JSON. The fourth control rejects an unrelated directory. No GPU or Nsight capture was run.

`root-cpu-receipt.json` contains the exact command used to re-extract the parent's completed CPU capture. All 16 operator names/counts/timing/memory values exactly match the source `key_averages_full.json`; the metadata CSV correctly records CUDA unavailable and no capture error. All source capture hashes were checked unchanged. Byte-identical copies of its six input artifacts are under `actual-cpu-inputs/`, outside the ignored `profiles/` path. The original source capture remains untouched.

`root-cpu-smoke_metadata.csv` and `root-cpu-smoke_operators.csv` are the extracted outputs. `validation-receipts.json` hashes both changed code/test paths and all retained artifacts. GPU profiling, Nsight selectors on an actual target, kernel correctness and performance remain separate HOLD gates.
