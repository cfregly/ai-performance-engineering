# Inclusive prefix scan

Compare PyTorch's native inclusive int32 scan with hierarchical Triton scan and
a CUDA decoupled-look-back variant. The workload contains 1,048,579 values,
including a ragged final tile; arithmetic wraps modulo 2^32 in every arm.

The [hierarchical path](benchmarks.py) scans tile totals recursively and propagates
carries with ordered launches. [scan_lookback.cu](scan_lookback.cu) uses resident
workers, dynamic tile assignment, and acquire/release publication of aggregate
and prefix state. State reset is inside each measured invocation. All variants
verify the complete result against an int64 accumulation converted to int32.

From `code/`:

```bash
python -m cli.aisp bench run --targets labs/prefix_scan:prefix_scan --profile minimal
python -m cli.aisp bench run --targets labs/prefix_scan:prefix_scan_lookback --profile minimal
python -m pytest tests/test_parallel_primitives.py -q
```

GPU execution needs CUDA and Triton; the look-back variant also needs the CUDA
toolkit. CPU tests cover the independent scan reference and unsupported-host
diagnostics. CUDA tests exercise ragged tiles, overflow, and replayed inputs.
No GPU correctness or performance claim is derived from the development Mac.

Online normalization belongs to [Chapter 9](../../ch09/README.md#online-softmax-normalization).
Source: [decoupled look-back](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf).
