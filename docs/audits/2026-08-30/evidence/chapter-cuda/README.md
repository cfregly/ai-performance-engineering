# P03 chapter CUDA source remediation

This receipt covers the 18 original wave-one findings W1-008/009/010/011/012/013/014/015/045/046/048/049/050/051/102/103/104/105. The original audit inventory remains 128 findings. Additional source defects discovered while preparing the real-kernel checks are listed separately in `source-observations.json`.

**Source/host gates pass; CUDA compilation, device linking, native extension import, sanitizer execution, and GPU numerical/timing acceptance remain HOLD.** This macOS arm64 host has no nvcc, compute-sanitizer, or CUDA GPU. There is no performance qualification and no measured speedup in this receipt.

| Findings | Source change | Device acceptance prepared |
|---|---|---|
| W1-008, W1-015 | Record both events on the stream containing the measured launches. Reset graph inputs after warmup. | Real threshold kernels and both graph entrypoints; full graph outputs for identical six-pipeline workloads. Production event placement also has a source regression check. |
| W1-009 | Drain the final cp.async group before reading its tile. Declare the staging buffer's required alignment. | One, two, three, four and multiple tile counts, tails, full outputs, racecheck and synccheck. |
| W1-010 | Partial N groups use their actual width; no wrapping/duplicate tiles. Publish initialized barriers; reject empty/mixed-device operands; preserve accepted whole-tile shapes. | Seven complete asymmetric GEMMs, including small grids, N groups below/at/above eight, short K and multiple parity wraps; four invalid shapes rejected. |
| W1-011, W1-012, W1-048 | Process every chunk across three stream lanes. Connect allocation, H2D, all three compute passes, D2H and buffer reuse with events. Timing fans out to workers and joins final copies. | All outputs for 24 size/chunk configurations, including fewer elements than lanes and ragged chunks. |
| W1-013, W1-049, W1-050, W1-051 | Equal input/reset/warmup/launch counts for scalar/vector/shared variants. Shared kernel uses its own four-element launch grid. Separate copy/compute streams and ordered D2H. Both pipeline modes process the same jobs. | Four actual kernels on eleven sizes plus three complete production overlap comparisons; every output checked. |
| W1-014 | Fan timing start to all streams and join their completion events before stop. Verify both passes and the final timed feedback buffers; propagate numerical failure. | Production demo on one, three and eleven streams with eight ragged batches. |
| W1-045 | Occupancy callback reports the dynamic shared-memory bytes actually launched. Partial-block threads all reach the barrier. | Real occupancy API and six full-output ragged sizes. |
| W1-046 | Validate all eight coefficient patterns over every output, reject nonfinite values. Relabel the time ratio because the two kernels compute different arithmetic. | Scalar and Float8 kernels on eight sizes, including partial vectors and grid-stride reuse. |
| W1-102 | Sequential H2D+compute+D2H bandwidth is not compared to the full-duplex sum or asserted to establish the bottleneck. | Corrected bounded reference prose; actual processing kernel also has three numerical cases. |
| W1-103 | Obtain the 1.0 FP8 byte from CUDA's e4m3 constructor. | Six full-output FP8 cases plus exact initialization readback. |
| W1-104 | Only assert aligned size when byte count and input pointer permit it; use an ordinary-size copy for tails/offset views. | Three production pipeline specializations, ragged counts and three unaligned input views. |
| W1-105 | Describe four CTAs and one DSMEM atomic per CTA. Also guard the newly discovered partial float4 load. | Six cluster reductions, including one-element inputs and partial vectors/clusters. |

The actual CUDA gate compiles eleven separate binaries that include the production source, not substitute kernels. It prepares **195 full-output checks** and runs memcheck, racecheck and synccheck by default. Scalar guards are checked around generic outputs and cluster outputs; FP8 and other exact allocations also rely on memcheck. The separate tcgen05 gate compiles the actual extension in a fresh receipt directory and compares every output against a CPU float64 matrix product rounded to FP16. Exact equality is used only for the fixed dyadic test-input family, whose products and partial sums fit exactly in FP32. It uses a noncontiguous input case and records the selected Torch ABI, CUTLASS header hashes and module hash when a supported target is available.

The code does not claim that shared staging or `cuda::memcpy_async` proves tensor TMA instructions were emitted. Those instruction/performance claims require inspection on the actual compiler and target.

## Recorded commands

From the repository root:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider code/tests/test_chapter_cuda_contracts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider code/tests/test_chapter_cuda_contracts.py code/tests/test_build_entrypoints.py code/tests/test_dual_arch_make_contract.py
make -n -C code/ch12 ARCH=sm_100 PYTHON=/usr/bin/false baseline_cuda_graphs_sm100 optimized_cuda_graphs_sm100
```

- `before.txt`: ten regression checks failed against the original behavior, including a real host C++ execution showing duplicate/missing tiles. The original tile expression was first moved, unchanged, into the production host/device helper so that the test exercised production logic before correction.
- `after.txt`: thirteen checks passed. The two printed graph mismatches are intentional negative controls: last-element corruption and NaN must fail the production host verifier.
- `combined-host-tests.txt`: 179 checks passed, including the existing make/build contracts. Source assertions are not GPU tests.
- `post-review-host-tests.txt`: 180 checks passed, including the added real CPU descendant-termination negative control for build/run timeouts.
- `independent-source-review.txt`: stream/graph and GEMM source review, the bounded-build follow-up, and explicit CUDA HOLD.
- `graph-make-dry-run.txt`: the graph binaries resolve to their .cu sources and architecture flags; no compilation occurred.
- `chapter-gpu-preflight/report.json` and `tcgen05-gpu-preflight/report.json`: preserve the earlier no-nvcc attempts, exit 3. Final-attempt receipt paths are listed in `validation-receipts.json`. Earlier reports intentionally retain their earlier source hashes.

## Required target commands (not executed here)

Choose the exact target architecture and use a fresh output directory for each attempt. The tcgen05 kernel requires the matching SM100a or SM103a target; SM120/121 are not substitutes. Other chapter cases remain subject to their real feature checks (for example cluster launch).

```sh
python code/tests/cuda/run_chapter_cuda_validation.py --arch sm_100a --output-dir /tmp/p03-chapter-gpu-ATTEMPT
python code/tests/cuda/run_ch10_warp_specialized_validation.py --arch sm_100a --cutlass-include /absolute/compatible/cutlass/include --output-dir /tmp/p03-tcgen05-gpu-ATTEMPT
```

Both commands require the sanitizer tools by default. `--sanitizers none` is a diagnostic run and reports `PASS_WITHOUT_SANITIZERS`, not full acceptance. A missing tool/target returns 3/HOLD. Compilation, runtime, missing success marker or timeout is a failure; logs and exact commands are retained. Build/import and device runs are bounded to 300 seconds, with their owned process group terminated on timeout. A successful numerical run still does not qualify a performance comparison. Hardware acceptance must additionally review actual event timing, emitted code, relevant resource usage and any reported sanitizer diagnostics.

## Primary source checks

- [NVIDIA CUDA 13 programming guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html): nonblocking stream dependencies, events, cooperative barriers, and alignment proofs for asynchronous copies.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/): pending cp.async groups and barrier publication requirements.
- [NVIDIA occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html): variable shared-memory occupancy callback.
- [NVIDIA e4m3 type API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html): host/device conversion and public storage member.
- [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): PCIe Gen5 x16 bandwidth in each direction versus the full-duplex sum.
