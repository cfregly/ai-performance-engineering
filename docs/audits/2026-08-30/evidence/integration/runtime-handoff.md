# Runtime acceptance handoff — audit waves 1 and 2

Prepared 2026-08-30 and updated 2026-08-31 from the current plan, ledger, and retained hosted receipts. **No B200 command below was executed while preparing or updating this handoff.** This is an operational checklist, not a completed goal, hardware lease, installation approval, CI dispatch, or permission to promote results.

## Authority, environment and evidence requirements

- [HANDOFF.md](../../../../../HANDOFF.md), especially its GPU ownership status, remains authoritative: another task owns both B200 GPUs. Do not launch, resume, profile, isolate, probe or use remote hardware until ownership is explicitly returned. Do not revive the historical paused sweep or mix its results with this revision. Do not kill unrelated jobs. Use one owner per allocated runtime; never pool qualification across different targets.
- After custody returns, use an isolated, reviewed source revision and the pinned Linux stack. The target requirements include Python 3.12, PyTorch 2.9.1+cu130, CUDA 13 and the package-specific dependencies below; verify actual installed versions rather than inferring them from configuration. Current macOS/PyTorch 2.8 CPU results do not establish pinned Linux CI or CUDA compatibility.
- Canonical/publish-grade performance requires bare metal. A virtualized current-host rerun separately requires explicit approval, locked clocks, recorded provenance and a virtualized/noncanonical label. Do not set clocks outside existing harness/control contexts. Numerical tests below are not performance acceptance, even when they measure a duration.
- Every command uses the selected target environment's `python`. **Unless stated otherwise, run from `code/`.** `/absolute/new/...`, `/absolute/pinned/...` and `/absolute/reviewed/...` are paths the runtime owner must choose, not existing artifacts. Select only allocated devices through the launch environment; do not copy somebody else's device IDs. Every output/build directory or JSON destination must be fresh.
- Preserve exact source/submodule hashes, compiler/driver/tool versions, ABI, GPU identities/capabilities, device visibility, topology, clocks, policy file/hash/reviewer, workload/seeds/dtypes/layouts, complete stdout/stderr, JUnit/full tensors, timeouts and failures. Hash final artifacts. Keep skipped/unsupported/failed checks explicit; a pytest exit code of zero with required cases skipped is not acceptance. Never weaken a failing numerical bound merely to obtain a pass.
- Use the runners' process deadlines where implemented. The tcgen05 pipeline verifier and external multihost worker additionally need an owner-specified job deadline and cleanup of only that job's children. A listed bounded workload is not necessarily a self-containing scheduler job.
- The [plan](../../../../../AUDIT_REMEDIATION_PLAN.md) and [ledger](../../remediation-ledger.json) retain all 128 original findings and all 141 delivered Wave 2 rows. Wave 2 source reconciliation is complete; 48 rows remain `awaiting_runtime`. Final-source hosted Linux CPU CI has executed once and is retained, so do not rerun that full pass merely for this handoff. The full target Linux/CUDA dependency graph and applicable GPU cells remain pending.

## Current retained-evidence checkpoint

- Source revision `3316e0efe985040745ffd926c5f76a6bd4436aff` is the final code epoch; later published commits through the current reconciliation change CI, documentation, and evidence only.
- Hosted CPU run `33391774956` records 4,346 passed, 461 explicit skips, and zero failures/errors on Ubuntu 24.04. It verifies seven exact Wave 1 host/configuration contracts and a bounded CPU regression subgate for all 48 Wave 2 runtime rows. The 76-row Wave 1 local matrix is **7 verified and 69 pending**.
- Hosted CUDA 13 compile run `33391774950` supplies bounded four-target compiler evidence for 17 pending Wave 1 rows and nine Wave 2 rows. `W2-078` has only a partial header-through-consumer result. The job had no GPU.
- Focused Linux CPU provenance run `33401585682` verifies the bounded 20-pin/56-distribution CPU lock. It does not install or qualify the full 90-specification/327-package Linux/CUDA graph.
- B200 custody is unavailable. The next session must re-read `HANDOFF.md` and obtain an explicit custody return before any probe, launch, profiler, sanitizer, or runner action.

See the [non-GPU reconciliation](hosted-non-gpu-runtime-closure/receipt.json), [compile receipt](hosted-cuda-compile-closure/receipt.json), and [remaining matrix](pinned-linux-integration/runtime-update.md).

## P01 — bootstrap and build architecture

**Ready build entrypoints; complete installation acceptance remains open.** See [bootstrap receipt](../bootstrap/README.md) and [build receipt](../build-entrypoints/README.md). A complete `setup.sh` invocation provisions the system; it is not a safe read-only preflight and requires separately authorized isolated Linux provisioning. The full requirements graph, torch pin preservation through TE installation, torchao native CUDA import, real FP8 forward/backward and installed/system-vs-bundled cuDNN still need evidence. There is no standalone script here that certifies all of bootstrap.

Existing build commands, after provisioning/custody:

```bash
bash core/scripts/build_tma_demos.sh --arch sm_100
PYTHON=/absolute/pinned/bin/python BUILD_DIR=/absolute/new/cutlass-sm100 CMAKE_CUDA_ARCHITECTURES=100a bash labs/custom_vs_cublas/cutlass_gemm/build.sh
```

Use `sm_103` / `103a` on an independently allocated SM103 target, with a separate build/output. The CUTLASS module needs CMake 3.31.8+ and CUDA 13+. Inspect verbose compile/device-link and emitted device images, then import `cutlass_blackwell_gemm` from that build directory in the same Python/torch ABI environment; compilation alone is not import or GEMM acceptance. The build script prints its actual import recipe. The retained CUDA 13 compare job now proves real four-target chapter builds and exact alias/compare configuration behavior. It did not build every lab/extension consumer, import `cutlass_blackwell_gemm`, or run a device; retain those explicit remainders from the compile receipt.

On **exact SM121/GB10**, the existing diagnostic is:

```bash
python core/verification/verify_tma_sm121.py
```

Unsupported instructions/toolchain return an explicit non-pass. This diagnostic cannot establish TMA instruction lowering merely from a successful copy. SM120/121 cannot substitute for the SM100a/103a tcgen05 acceptance below. GPU architecture does not determine host CPU architecture.

**ARM gap:** the exact torchao 0.15.0+cu130 aarch64 wheel was unavailable at intake. A source-built v0.15.0 artifact at commit `9338966da58ec44b60f0e0b173cabab08f942ed0` needs its own reviewed build, pinned submodules, wheel hash, native import and CUDA tests. Setup does not yet consume such a qualified artifact. Hardware alone does not close that packaging gap.

## P02 — harness, protections and CI

**Hosted Linux CPU CI is retained; supported-GPU checks remain pending.** Active workflow source contains these commands (from `code/`):

```bash
python -m pytest tests -q -ra -o timeout=120 --junitxml=artifacts/pytest-cpu.xml
python -m pytest tests -q -ra -o timeout=600 --junitxml=artifacts/pytest-gpu.xml
```

The first command already completed in hosted run `33391774956`; its JUnit is retained. Do not rerun it unless a later source/dependency change invalidates that evidence. The second belongs on the attested Tier-1 B200 runner, with the existing exact-device/MIG/environment checks and `TIER1_EXPECTED_GPU_NAME` contract intact. Use a fresh run/worktree so artifact names cannot overwrite previous attempts. This is not an instruction to dispatch the GPU workflow before custody returns.

For the actual clock-lock integration, the existing selector is:

```bash
python -m pytest -q -rs -p no:cacheprovider tests/test_anti_cheat_protections.py::TestEnvironmentProtections::test_frequency_boost_clock_locking
```

It must observe real NVML/application clocks and CUDA work. A local permission/capability skip is not a lock pass; the attested runner contract requires failure when that capability is unavailable. The protection receipt's 35 CUDA skips remain runtime work; its 61 unsupported/obsolete policy skips are **not** 61 implemented guards or implicit new implementation tasks. See [clock/protection disposition](../validation/clock-lock-followup/README.md).

## P03 — chapter CUDA, TMA and timing

**Ready standalone correctness/build/sanitizer gates.** Each command creates a new record. These are the actual production-kernel validators, not CPU translations. See [TMA](../tma/README.md) and [chapter CUDA](../chapter-cuda/README.md).

```bash
python tests/cuda/run_tma_2d_layout_validation.py --arch sm_100a --output-dir /absolute/new/p03-tma-sm100 --compute-sanitizer
python tests/cuda/run_chapter_cuda_validation.py --arch sm_100a --output-dir /absolute/new/p03-chapter-sm100
python tests/cuda/run_ch10_warp_specialized_validation.py --arch sm_100a --cutlass-include /absolute/pinned/cutlass/include --output-dir /absolute/new/p03-warp-sm100
```

Repeat the matching `sm_103a` commands on SM103, never on the SM100 device. TMA prepares 42 full-output/canary cases and optional memcheck is explicitly enabled above. The chapter gate prepares 195 checks across eleven binaries; it and the separate ch10 extension gate require memcheck/racecheck/synccheck by default. `--sanitizers none` is diagnostic `PASS_WITHOUT_SANITIZERS`, not acceptance. The ch10 extension uses a bounded independent CPU FP64 reference and fixed dyadic inputs; that exact-equality family is not a general error budget.

The chapter runner exposes `sm_120`/`sm_121` too, but real feature/capability checks still apply. The separate tcgen05/TMA runners accept only their explicit SM100a/SM103a targets. Missing tool/device exits HOLD; compile failures, runtime mismatches and timeouts remain failures. Event dependency/timing, instruction attribution and performance need separate actual-target review after correctness.

## P04 — Blackwell kernels and feature labels

**Ready standalone gates, with separate architecture/feature contracts.** See [stage lifetime receipt](../validation/cuda-stage-lifetime-receipt.json) and [feature CUDA](../feature-cuda/README.md).

```bash
python -m labs.custom_vs_cublas.verify_tcgen05_pipeline --output /absolute/new/p04-pipeline.json
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --target-processes all --error-exitcode=1 python -m labs.custom_vs_cublas.verify_tcgen05_pipeline --output /absolute/new/p04-pipeline-memcheck.json
python tests/cuda/run_feature_cuda_validation.py --arch sm_100a --output-dir /absolute/new/p04-features-sm100
python tests/cuda/run_feature_extensions_validation.py --case bias_silu --arch sm_100a --cutlass-include /absolute/pinned/cutlass/include --output-dir /absolute/new/p04-bias-sm100
python tests/cuda/run_feature_extensions_validation.py --case grace_tma --arch sm_90 --output-dir /absolute/new/p04-tma-sm90
```

The first verifier requires actual CC10.0 or CC10.3 and checks all seven variants × 30 shapes × two seeds × four repeats = **1,680 full outputs**, input immutability and supplementary canaries. Its JSON explicitly does not grant sanitizer qualification; retain the separate sanitizer output. Pipeline race/synchronization diagnostics and emitted multicast/TMA instructions remain additional target work.

The standalone feature runner checks 49 scalar/FP8/cluster cases, five launches each. The bias extension checks whole supported tiles, nonzero column bias, FP16/FP32 bias, noncontiguous inputs and invalid inputs. The genuine TMA extension's `sm_90` command is for **Hopper**, not a B200 command. Both runners use all three sanitizer modes by default; their CLI also exposes separate supported target architectures. `--multi-device-controls` on the extension runner is only appropriate with **two allocated visible devices**; otherwise that part remains HOLD. Never replace an unavailable requested backend with library GEMM. Retired incomplete experimental files remain retired, not hardware-ready candidates.

## P05 — Nanochat graph replay and stream ownership

**Ready focused real-CUDA tests, not a standalone all-gates acceptance runner.** On the pinned CUDA environment, from `code/`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NANOCHAT_DISABLE_COMPILE=1 python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_nanochat_regressions.py labs/nanochat_fullstack/test_new_optimizations.py --junitxml=/absolute/new/p05-nanochat.xml
```

Review all eleven formerly CUDA-skipped cases: six FP32/BF16 replay cases, side-stream/D2H, all-stream timing and three backend/integration cases. Require successive eager/graph logits and KV equality, correct device/host positions, capture reuse/recapture, real stream lifetimes and host-copy ordering. A cluster-specific unsupported backend may be recorded as HOLD for that cell; it is not complete package acceptance. Graph decode uses masked full-capacity SDPA, and the legacy persistent flag means a reusable-buffer side stream, not a resident kernel.

Timing/trace qualification is still a separate campaign using [the fixed workload specification](../../../../../code/labs/nanochat_fullstack/audit_wave1/benchmark_workload_spec.yaml) and [package receipt](../../../../../code/labs/nanochat_fullstack/audit_wave1/README.md): identical weights/inputs, alternating repeated arms, capture cost separately, actual backend and all-stream interval. No one-shot gate supplied here establishes a graph speedup.

## P06 — distributed training, MoE and transfer ownership

**Ready bounded numerical/distributed gates where listed; other runtime cells remain open.**

Exactly **two visible allocated NCCL/BF16 GPUs**, using the production ZeRO common path and a separate dense reference:

```bash
python tests/test_audit_wave1_zero2_parity_cuda.py --output-dir /absolute/new/p06-zero2-two-gpu --execute
```

This records full parameters/gradients/loss/optimizer state for four updates, both variants and payload modes. Its per-group and parent timeouts are implemented. See [gate design](../zero2-parity/cuda-gate/README.md) for its explicit numerical budgets. It does not cover default hidden-size 10000 / 12 GiB payload scale, compilation mode, the four CLI main functions, full training throughput or memory claims. That historical receipt's statement about the toy wrapper describes its earlier source epoch; LOCAL-019 subsequently withdrew it as described below.

Actual hybrid EP requires **four allocated CUDA GPUs and NCCL** for the local gate:

```bash
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_hybrid_ep.py::test_four_gpu_nccl_forward_backward_and_reuse_match_serial_reference --basetemp=/absolute/new/p06-hybrid-temp --junitxml=/absolute/new/p06-hybrid.xml
```

The test checks full outputs, loss, gradients, replica updates, empty routing and buffer reuse in FP32 with TF32 disabled against a serial CPU FP64 oracle. Two logical node groups on one host are **not physical inter-node fabric or BF16 acceptance**. An optional worker entrypoint in the same file accepts `--nccl-worker --output-dir` under torchrun, and requires four ranks/two local ranks per host. There is no complete reviewed site-specific rendezvous/scheduler command here; the runtime owner must define and bound that separately, preserving each host/rank. Do not launch the worker as an ordinary standalone process.

Ready active Triton and KV checks:

```bash
python tests/cuda/run_moe_triton_validation.py --output-dir /absolute/new/p06-moe-triton
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_kv_overlap.py tests/test_audit_wave1_kv_handoff.py --basetemp=/absolute/new/p06-kv-temp --junitxml=/absolute/new/p06-kv.xml
```

The Triton runner requires two CUDA/BF16-capable GPUs, actual Triton and compute-sanitizer, with all 30 required GPU cases passing in plain and memcheck modes. It validates the active activation and explicit PyTorch reference only; retired fused FFNs remain unavailable. KV overlap needs real CUDA streams; the handoff case needs two NCCL GPUs and complete batched-cache readback. CPU/Gloo results do not cover those mechanisms. The router's actual vLLM cumulative-output streaming and latency/token telemetry still need an authorized model/service workload; deterministic payload fixtures are not a live vLLM gate, and no such standalone workload is supplied here.

Native FP8 and MoE PTX forward **first need actual calibration and external policy review**:

```bash
python -m labs.moe_optimization_journey.calibrate_native_fp8 --output /absolute/new/p06-native-fp8-calibration.json
```

From repository root:

```bash
python docs/audits/2026-08-30/evidence/moe-ptx/calibrate_layer_accuracy.py --output /absolute/new/p06-moe-ptx-calibration.json
```

Both preserve failures/HOLD and do not create acceptance bounds. Three amplitudes at one seed/routing are only bounded diagnostics: expand relevant dtype, shape, input/weight scale, seed and routing cells before review. With reviewed policies, from `code/`:

```bash
AISP_NATIVE_FP8_ACCURACY_POLICY=/absolute/reviewed/native-fp8.json python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_native_fp8.py::test_native_fp8_complete_production_path_with_reviewed_policy
AISP_MOE_PTX_LAYER_ACCURACY_POLICY=/absolute/reviewed/moe-ptx.json python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_moe_ptx.py::test_real_cuda_full_layer_with_reviewed_policy
```

Native FP8 uses actual scaled GEMMs and independent unquantized BF16 reference; its test requires CC8.9+ and the intended backend. MoE PTX tests BF16/FP16 and balanced/skewed routing through both actual full-output paths. They do not upgrade backward-slice or unrelated grouped-GEMM verification. CUDA forward prepacking occurs outside timing while the baseline masks routes inside forward; keep that work boundary explicit in any later comparison.

### LOCAL-019: all 61 generic training wrappers remain unavailable

The parent-side Linear/meta surrogate is gone. All generic wrapper execution/verification/launch-spec requests fail before child launch. Factories/configuration remain discoverable; direct training CLIs remain available but are not qualification. **Hardware alone is insufficient to restore these 61 wrappers.** Re-enabling any wrapper requires an actual child-produced result/state protocol, an independent workload-matched reference, complete numerical comparison, corrupted/unrelated/stale-child-result negative controls and target training evidence. The ZeRO standalone gate is not a generic integration protocol. See [LOCAL-019 receipt](../torchrun-verification/README.md).

## P07 — attention, decode, input and DMA lifetimes

The old `../attention/run_cuda_acceptance.py` is historical and correctly refuses the current source epoch: four persistent-decode files no longer match its preserved manifest. Do not overwrite that old manifest or suppress the check.

The [current-source replacement](attention-final-epoch/README.md) requires **all 25 original real-CUDA attention cases plus seven new full-output prefill cases**, exact identities rather than a relaxed minimum count. From repository root:

```bash
python docs/audits/2026-08-30/evidence/integration/attention-final-epoch/run_cuda_acceptance.py --output-dir /absolute/new/p07-attention-current-epoch
```

Its 115-entry manifest matched every current file when this handoff was checked. The real CPU preflight returned HOLD, with zero CUDA cases dispatched. The driver uses a fresh directory, per-case commands/logs/JUnit/artifacts, a 600-second process-group deadline for each case, and source checks before/between/after cases. The combined cell needs CUDA13/nvcc, CC10.0/10.3/12.0/12.1, actual Triton/FlashAttention CuTe/cuDNN and all requested providers. A missing backend cannot silently select another implementation. All 32 cases must pass without errors or skips. Its bounded correctness status is not sanitizer, full-size, performance or numerical-policy calibration acceptance; inherited decode tolerances remain uncalibrated and the new exact prefill checks use fixed dyadic FP32 inputs.

The current seven-case prefill test is independently selectable:

```bash
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_prefill_full_output.py::test_real_cuda_prefill_decode_full_outputs_and_changed_inputs --basetemp=/absolute/new/p07-prefill-temp --junitxml=/absolute/new/p07-prefill.xml
```

It requires CC10.0/10.3/12.0/12.1, CUDA13 and nvcc and exercises complete changed outputs plus copy/graph modes. Passing it alone does not replace the other 25 cases. The low-level test file's CLI without `--worker` deliberately produces HOLD; use its pytest driver for all seven cases.

After combined correctness, these real-test sanitizer invocations cover the shared reduction/FlashMLA and tail/DMA/copy paths; preserve the sanitizer output as well as JUnit:

```bash
compute-sanitizer --tool racecheck --target-processes all --error-exitcode=9 python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_attention_regressions.py::test_real_cuda_reduction_repeated_multwarp_reuse tests/test_audit_wave1_attention_regressions.py::test_real_flashmla_book_kernel_stable_full_dot --junitxml=/absolute/new/p07-racecheck.xml
compute-sanitizer --tool memcheck --target-processes all --error-exitcode=9 python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_attention_regressions.py::test_real_triton_attention_tail_and_causal tests/test_audit_wave1_attention_regressions.py::test_real_pinned_stage_dma_reuse tests/test_audit_wave1_attention_regressions.py::test_real_thread_async_copy_eager_and_graph_current_stream --junitxml=/absolute/new/p07-memcheck.xml
```

The same reduction/FlashMLA selectors require a separate `--tool synccheck` run with a new artifact name. The selected cases must actually execute; unsupported instrumentation is HOLD, not a clean report. Real cuDNN profiler attribution, emitted copy/TMA instructions and fair full-size timing remain separate gates. Historical TFLOP rates using the old causal numerator and historic speedups remain unqualified. FA4's recorded performance gate is 1.05x, not permission to accept a wrong or unavailable backend.

## P08 — KV quantization, FP4 evaluation and Ozaki

**Calibration CLIs are ready; policy and runtime acceptance are not supplied.** Requires the pinned TE2.18/CUDA stack and a target that actually supports the requested FP8/NVFP4 path.

```bash
python -m labs.kv_cache_compression.calibrate_accuracy --variant fp8 --seed 42 --output /absolute/new/p08-kv-fp8.json
python -m labs.kv_cache_compression.calibrate_accuracy --variant nvfp4 --seed 42 --output /absolute/new/p08-kv-nvfp4.json
```

These collect full K/V errors against an independent BF16 projection reference. Unlike the native-FP8/MoE-PTX collectors, this CLI does not reserve the destination or persist a failure JSON: the operator must enforce a fresh destination and save complete failed stdout/stderr plus external source/environment metadata. Repeat relevant seeds/workloads before policy review. Cache storage remains BF16 (ratio 1.0); compute format is not cache compression. Only after review:

```bash
AISP_KV_CACHE_ACCURACY_POLICY=/absolute/reviewed/kv-cache.json python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
```

NVFP4 candidate/evaluator entrypoints (retain full output and official reference provenance; no `--no-verify`):

```bash
python -m labs.nvfp4_gemm.local_eval_submission --submission-file labs/nvfp4_gemm/optimized_submission.py --reference-file labs/nvfp4_gemm/reference_submission.py --verify --json
python -m labs.nvfp4_gemv.local_eval --submission-file labs/nvfp4_gemv/optimized_submission.py --official-root /absolute/pinned/reference-kernels/problems/nvidia --json
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_numerics_regressions.py::test_group_real_cuda_custom_kernel_against_independent_reference
python -m cli.aisp bench run --targets labs/nvfp4_group_gemm --profile minimal
```

These require actual block-scaled CUDA support; the custom grouped kernel gate is CC10-family-specific. The first GEMM evaluator uses the local frozen reference and limited `--verify-count` samples per case, not proof that the official upstream competition evaluator ran every required input. GEMV needs a separately provisioned, pinned upstream evaluator; this handoff does not download it. Grouped GEMM's one focused CUDA test does not qualify all five benchmark shapes, fused/eager/graph routes, multiple seeds or concurrent-stream/allocator scale-cache lifetime. Its existing full-output FP16 bound is `rtol=atol=1e-3`, not a newly calibrated tolerance. Each route must pass the independent decoded-E2M1/FP64 reference before accepting timing.

Ozaki on the target with the actual cuBLAS fixed-point emulation API:

```bash
make -C labs/ozaki_scheme ARCH=sm_100 all
labs/ozaki_scheme/optimized_ozaki_scheme_dynamic_sm100 --m 4096 --n 4096 --k 4096 --seed 2026 --input-scale 0.001 --dynamic-max-bits 16 --dynamic-offset -56 --accuracy-measure-only
labs/ozaki_scheme/optimized_ozaki_scheme_fixed_sm100 --m 4096 --n 4096 --k 4096 --seed 2026 --input-scale 0.001 --fixed-bits 12 --accuracy-measure-only
```

Measurement-only exit **2** / `MEASUREMENT_ONLY_NOT_ACCEPTED` is intentional and supplies no accepted TIME_MS/TFLOPS/checksum. Sweep retained bits, input scales and seeds, then independently review bounds. After review:

```bash
AISP_OZAKI_ACCURACY_POLICY=/absolute/reviewed/ozaki.json python labs/ozaki_scheme/run_lab.py --skip-build
```

Every accepted output must pass the actual full-array C++ comparator against separate native FP64 storage. Historical checksums/timings do not qualify the new verifier. See [P08 receipt](../numerics/README.md).

## Required external numerical-policy review

No example thresholds are provided. All schemas require `schema_version: 1`; policy existence is not measured accuracy. Preserve who approved which source/workload/hardware cell and the calibration/negative-control evidence.

| Environment variable | Required JSON object(s) and fields | Additional boundary |
| --- | --- | --- |
| `AISP_KV_CACHE_ACCURACY_POLICY` | `fp8`, `nvfp4`: `relative_l2`, `normalized_max_abs`, `pairwise_rtol`, `pairwise_atol` | First three finite in `[0,1)`; last finite/nonnegative; full K and V against BF16 reference. |
| `AISP_NATIVE_FP8_ACCURACY_POLICY` | `native_fp8`: same four fields | Same bounds; independent unquantized BF16 reference, full unsorted top-k output. |
| `AISP_MOE_PTX_LAYER_ACCURACY_POLICY` | `moe_layer_forward`: `relative_l2`, `normalized_max_abs` | Both finite in `[0,1)`; full forward reference. Existing pairwise `.02/.02` is secondary and cannot accept zero by itself. |
| `AISP_OZAKI_ACCURACY_POLICY` | `dynamic`, `fixed`: `relative_l2`, `normalized_max_abs`, `checksum_rtol`, `checksum_atol` | First three finite in `[0,1)`; last finite/nonnegative; checksum is secondary to full native-FP64 comparison. |

Zero, dropped/tail/canceling corruption, NaN, aliasing and shape errors must remain rejected through the real comparator. A calibration result that exceeds the eventual policy is a failure/HOLD for that cell, not a reason to auto-generate a looser policy. Other bounded test tolerances are their stated source contracts, not evidence of workload-wide calibration.

## P09 — metric math and measured peaks

**Five focused target tests are ready; full performance/FP4 accuracy qualification remains open.** Requires two allocated peer-capable GPUs for all five, native FP8 and actual TE NVFP4 on the selected Blackwell device:

```bash
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_peak_metrics.py -k test_real_gpu --junitxml=/absolute/new/p09-peaks.xml
```

The tests check FP8 quantized-operand output, actual FP4 recipe execution, copy byte/timing provenance, a noncurrent device and peer payload accounting. The FP4 recipe/timing test is **not full FP4 numerical calibration**. Peer access/throughput is not proof of NVLink transport; L2 timing is not proof of residency. Actual topology, instruction/provider inspection, representative sizes and repeated controlled measurements are needed before peak/SoL claims. Unknown SKUs cannot inherit B200 peaks; B200, B300, GB200, GB300, RTX12.x and GB10 remain separate cells. See [peak-metrics receipt](../validation/peak-metrics-receipt.json).

## P10 — CLI/API and monitoring

Source/CPU, real CLI, parser, local HTTP/Prometheus and flame-artifact checks have receipts. No private prompt transmission or paid provider call is required by this handoff. The remaining real FP8 guidance smoke test is:

```bash
python -m pytest -q -rs -p no:cacheprovider tests/test_audit_wave1_tooling_regressions.py::test_fp8_template_executes_real_transformer_engine_linear
```

It requires TE2.18.x and Hopper-or-newer CUDA; a dependency/version skip leaves it pending. Its coarse numerical smoke budget does not approve arbitrary production FP8 accuracy. Complete only the remaining target Linux/CUDA dependency and native-runtime cells; the retained final-source hosted CPU suite does not need a redundant rerun. See [API](../api/README.md) and [tooling](../tooling/README.md) receipts.

## P11 — documentation and profiler workflows

Documentation synchronization and offline/CPU extraction are source evidence, not GPU captures. After authorized target provisioning, these existing [documented smoke commands](../../../../../docs/tooling-and-profiling.md) can check actual CUDA capture (from `code/`, with new output names):

```bash
python -m cli.aisp profile torch tests/fixtures/mcp_torch_profile_target.py --mode full --output-name docs-smoke-new-attempt --timeout 60
nsys profile --trace=cuda,nvtx,osrt --output=/absolute/new/timeline python tests/fixtures/mcp_torch_profile_target.py
ncu --set basic --export=/absolute/new/kernel python tests/fixtures/mcp_torch_profile_target.py
```

First check the installed NVIDIA tool's supported options/metrics in the authorized target session. The fixture falls back to CPU when CUDA is absent; inspect actual CUDA activity, never classify a CPU trace as GPU evidence. These commands verify capture plumbing only, not workload correctness, new speedups, or canonical publication. Run actual package workloads for instruction/backend/timing claims after their correctness gates. Keep historical tables/aggregate speedups explicitly unqualified, and refresh generator/docs/manifests only against the final reviewed source epoch.

## Final acceptance sequence

1. Obtain explicit target custody/provisioning authority; reconcile a fresh source epoch and all reviewed manifests without rewriting old receipts.
2. Preserve the completed hosted CPU evidence; complete the still-missing full Linux/CUDA installation, ABI/native build/import gates, and target-specific numerical/policy review.
3. Run the applicable exact standalone/test gates above, rejecting missing/skipped required cases, then sanitizer and profiler attribution where still outstanding.
4. Only afterward assess comparable work and repeated performance/memory behavior, with clocks/topology/source and raw outputs retained. Retired implementations, generic training protocol gaps and unsupported cases remain explicit.
5. Reconcile new hardware receipts against all 128 Wave 1, 141 Wave 2, and 52 adjacent rows. Wave 2 is already captured; an independent final review, evidence completeness, and the remaining goal requirements still govern completion.

This handoff itself executes none of these steps and transfers no operational authority.

## Current attention gate identity

Read-only verified while preparing this handoff; do not edit older manifests to force these to match another revision.

| File under `evidence/integration/attention-final-epoch/` | SHA256 |
| --- | --- |
| `run_cuda_acceptance.py` | `4b8346d15cf102d611706931fab8012ab4e49cb9dd6348e92ec93f82faab5270` |
| `source_manifest.json` | `fa136e6942d8af98e5a235cf7868de00e9662cafd9907a6406ce58fa1be98a4e` |
| `expected_cuda_cases.json` | `c91c3803f4cdbf1f3ea68148f3bfdd9e5cb17e5187185a89eb68d8a110ae385f` |
| `receipt.json` | `7321e0b1f8d4f557d5cd83fa5b68e87e714350f96ae7601c4716e621f80cf1a7` |

The [frozen current-epoch receipt](attention-final-epoch/receipt.json) records 16 CPU gate-control passes, exact 25+7 collection and an actual CPU HOLD preflight, not GPU acceptance. All 115 source-manifest entries were rechecked read-only after the package freeze and matched.

The immutable original wave-1 capture remains SHA256 `474779ab49b67c5c888e5f689b2400204b6d0f46f304219cedd0773694b6e1ba`. All commands here must be reconciled with the final source revision and applicable receipts before an authorized runtime run; this document does not transfer results between source epochs.
