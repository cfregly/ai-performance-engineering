# Core to chapter/lab import backedge review

## Scope and method

This is a read-only review that began from the working-tree epoch at
`b57e4c6a9` recorded in `initial-checker-output.txt`. The only files written by
this reviewer are this evidence package. No source was edited, no package was
installed, and no GPU or remote command was run.

The current checker command is:

```text
python3 code/core/scripts/check_import_edges.py --root code
```

At the review epoch it exited 1 with exactly 11 core-to-chapter/lab import
violations. The review followed every imported symbol to its use and searched
for callers of each core entrypoint. All 11 are real ownership violations. None
needs a checker waiver.

Concurrent source edits began after the 11-row observation. The six-row
intermediate epoch is preserved separately in `intermediate-checker-output.txt`;
it does not replace or erase the initial 11-row inventory. At that epoch,
finding 1 had been redirected to a duplicated MoE plan and findings 4, 7, 8,
and 9 left core compatibility modules that imported the relocated application
scripts. Those interim implementations were subsequently removed: the generic
comparator now has no MoE-specific branch, all three compatibility modules are
absent, and a repository search finds no remaining old module/path references.

The latest `checker-output.txt` is a later working-tree epoch. Static checker
removal is only one part of acceptance. The concurrent implementation status
section below records where an edge is checker-clean but the reviewed behavior
or provider seam is still incomplete. These are follow-up observations, not
source changes made by this reviewer.

## Disposition summary

| Disposition | Edges | Decision |
|---|---:|---|
| Core-owned interface/provider or registration | 7 | Keep the reusable core entrypoint, but remove knowledge of the chapter/lab implementation. |
| Operational composition outside `core` | 4 | Move the chapter-composing command under `code/scripts/` and update its path callers. |
| Narrow checker exception | 0 | No evidence supports an exception. |

## Concurrent implementation status

This table evaluates the concurrent source changes against the required end
state. "Checker-clean" means only that the static import edge disappeared; it
does not imply the behavioral tests or GPU acceptance gate passed.

| Original finding | Latest observed state | Required end state still outstanding |
|---:|---|---|
| 1 | Checker-clean and structurally complete. The generic comparator now requires only `get_benchmark()` and contains no MoE plan branch/import. | Add the focused negative-control test that imports/runs with the lab plan blocked. |
| 2 | Checker-clean but semantically incomplete. `deep_profiling_report.derive_roofline()` still constructs `RooflineAnalyzer()` without artifact specs; `core.analysis.kernel_roofline.get_architecture_specs()` detects the current host through Torch/CUDA. | Supply immutable specs from artifact metadata or an explicit CLI input. Missing specs must be explicit unknown, and offline report generation must not inspect local CUDA. |
| 3 | Checker-clean but broader than the reviewed seam. The entire Torch-dependent capture/context class was promoted to `core.profiling.nsight_systems`; the offline parser was not separated. | Put the stdlib `nsys stats` runner/parser/ranker in a small core module. Keep Torch/NVTX capture behavior separate, or at least lazily import Torch so offline report parsing does not require it. |
| 4 | Checker-clean and structurally complete. The Python command and coupled shell driver now live under `code/scripts/`; no core wrapper or old reference remains. | Run the focused entrypoint/mode-result tests; GPU power sampling is a separate runtime gate. |
| 5 | Still checker-visible in the latest captured epoch. The core analysis method imports the chapter provider and calls affinity-changing `setup_grace_affinity()`. | Inject a pure core-owned analysis provider and return current/suggested affinity without calling `sched_setaffinity`; keep mutation in an explicit chapter operation. |
| 6 | Checker-clean through the same promoted Torch-dependent class as finding 3. | Same small offline parser and negative-control tests as finding 3. |
| 7 | Checker-clean and structurally complete. The operational command is under `scripts.utilities`, all known callers use the new module/path, and no core wrapper remains. | Keep the probe labeled as the chapter 12 graph-bandwidth extension; run the focused caller/entrypoint tests. |
| 8 | Checker-clean and structurally complete as an ownership move; the command remains the explicit chapter composition root. | Preserve all four chapter 6 attempts independently and make import/loader failures affect the command result truthfully. |
| 9 | Checker-clean and structurally complete as the same ownership move. | Preserve the chapter 12 attempt separately from every chapter 6 result. |
| 10 | Checker-clean and materially improved: core owns registrations and calls each registered loader. | Replace `(bool, message)` with an explicit `passed`/`failed`/`skipped` result: an unsupported target is currently recorded as `(True, "Skipped ...")` and counted as success. Wire the operational chapter inventory through the registry or a supplied spec list. Add failure/isolation/no-chapter-import negative controls. |
| 11 | Checker-clean through the same registry work as finding 10. | Same explicit status, operational inventory wiring, and per-owner isolation tests as finding 10. |

The focused pytest command could not run in this shell because the available
`python3` is Python 3.14 without `pytest` installed. No package was installed,
consistent with this task's constraints. Syntax compilation of the promoted
profiling modules had passed before this status capture; that is not behavioral
acceptance.

## Per-violation findings

| # | Current edge | Actual use and callsite evidence | Disposition | Minimal truthful fix |
|---:|---|---|---|---|
| 1 | `core/analysis/compare_benchmark_pairs.py:18` -> `labs.moe_parallelism.plan.PlanEvaluator` | The module calls itself generic. The lab type is used only in the conditional branch at lines 42-47. No discovered `baseline*.py` or `optimized*.py` currently exposes the required `build_plan`, `CLUSTER`, and `MODEL` trio, while `labs/moe_parallelism/compare_pairs.py` already owns the real lab-specific comparison. | Core interface | Remove the dead MoE branch and its import; keep `get_benchmark()` as the generic protocol. If analytic comparisons are later needed, add a core-owned result hook implemented by the target module. Do not merely swap to `core.common.moe_parallelism_plan`: that copy has already drifted from the lab's corrected network and DP/EP math. |
| 2 | `core/analysis/deep_profiling_report.py:41` -> `ch08.roofline.RooflineAnalyzer` | `derive_roofline()` constructs the chapter analyzer and reads its detected current-host peaks. The reporter's own contract says it analyzes offline artifacts without GPU access. Therefore a report for one GPU can silently use ceilings detected on a different host. | Core interface/provider | Add a core-owned pure `RooflineSpecs`/calculator interface. Resolve specs from artifact metadata or an explicit CLI argument and require an explicit unknown status when the source artifact lacks them. Let `ch08.roofline` import or wrap that core calculator for its teaching demo. Do not substitute `core.analysis.roofline_automation.RooflineAnalyzer`; it has a different API and different peak conventions. |
| 3 | `core/analysis/deep_profiling_report.py:42` -> `ch17.blackwell_profiling_guide.NsightSystemsProfiler` | The core report only calls the class's stdlib `nsys stats` summary path at lines 557-571, but importing the chapter guide also imports `torch` and `torch.profiler` and exposes unrelated capture/demo behavior. | Core interface | Extract the report parsing/ranking operations into `core.profiling.nsys_stats` (or equivalent) with a small core result type. Have both the deep report and chapter 17 use it. |
| 4 | `core/benchmark/precision_power_sweep.py:26` -> `ch16.gpt_large_benchmark` | The script explicitly says it runs the chapter 16 benchmark and imports its config, workload, runner, and Transformer Engine status functions. Its history is operational: `code/tools/benchmarking/precision_power_sweep.py` -> `code/benchmark/` -> `code/core/benchmark/`. The only in-repo driver is `core/scripts/run_power_efficiency_sweeps.sh`. | Operational script | Move it to `code/scripts/benchmarking/precision_power_sweep.py` (or into `ch16` if chapter-local ownership is preferred) and update the shell driver/documentation. Keep generic `PowerSampler` in core. |
| 5 | `core/perf_core_base.py:2550` -> `ch04.gb200_grace_numa_optimization` | This is reachable from the public CLI, engine, and MCP `analyze_dataloader` entrypoints. It calls `setup_grace_affinity()`, which changes process CPU affinity, even though the endpoint is described as analysis. Broad exception handling then converts all provider/import failures into an untyped result. | Core provider/interface | Split pure detection/recommendation from explicit affinity application. `PerformanceCoreBase` should depend on a core-owned dataloader/NUMA analysis provider and return an explicit unavailable status. A Grace implementation can be registered by a composition root; the chapter demo can wrap the core provider. An analysis request must not mutate the server/CLI process. |
| 6 | `core/profiling/nsys_summary.py:17` -> `ch17.blackwell_profiling_guide.NsightSystemsProfiler` | This core CLI is entirely a wrapper around the same chapter-owned `summarize_report()` used by the deep report. | Core interface | Use the core `nsys_stats` extraction from finding 3. Chapter 17 should consume the same core parser. |
| 7 | `core/scripts/utilities/dump_hardware_capabilities.py:172` -> `ch12.cuda_extensions.load_graph_bandwidth_extension` | The command is operational and its CUDA-extension section tests one chapter 12 workload while labeling the result as generic extension compilation support. History shows it originally lived under `tools/utilities` and then `scripts/utilities`. It is invoked by the harness, tool registry, and profiling bundle by path/module. | Operational script | Move it back to `code/scripts/utilities/`, update the three path/module callers, and label the chapter probe explicitly. A future core capability probe should compile a core-owned minimal fixture rather than treat a chapter workload as a universal capability. |
| 8 | `core/scripts/utilities/precompile_cuda_extensions.py:39` -> `ch06.cuda_extensions` | The command is an all-chapter build orchestrator, not a reusable core module. History shows it moved from `tools/utilities` to `scripts/utilities` and then into core. | Operational script | Move it back to `code/scripts/utilities/`. Keep the chapter imports in this composition root and make its inventory/result statuses explicit. |
| 9 | `core/scripts/utilities/precompile_cuda_extensions.py:70` -> `ch12.cuda_extensions` | Same composition command as finding 8; this edge selects the chapter 12 loader. | Operational script | Resolve with the same move as finding 8; preserve a separate result row for this chapter 12 attempt. |
| 10 | `core/utils/extension_prewarm.py:289` -> `ch06.cuda_extensions` | `arch_config.configure_optimizations()` can invoke this core runtime path. `_do_prewarm()` also hardcodes the chapter module as a string. Importing `ch06.cuda_extensions` only defines loader functions; it does not compile any extension, so the current prewarm success can be false. The health check verifies function presence despite claiming it loads extensions. | Core registration/interface | Define a core `ExtensionBuildSpec` registry or accept an explicit list of loader callables. Core defaults must contain only core-owned loaders. Chapter 6 registers its real loaders when selected, and the operational all-chapter command supplies the complete inventory. Prewarm must call the registered loader and preserve build/skip/failure per extension. |
| 11 | `core/utils/extension_prewarm.py:308` -> `ch12.cuda_extensions` | Same runtime and false-prewarm mechanism as finding 10, for chapter 12. | Core registration/interface | Resolve with the same loader registry as finding 10 and retain separate chapter 12 results; do not collapse chapter 6 and 12 outcomes. |

## Exact destination and provider seams

| Fix group | Exact ownership seam | Focused CPU/source tests before GPU acceptance |
|---|---|---|
| Generic benchmark protocol (#1) | Keep `core/analysis/compare_benchmark_pairs.py` and delete lines 18 and 42-47. Its only execution contract is target-module `get_benchmark()`. Keep MoE analytic comparison in `labs/moe_parallelism/compare_pairs.py`; do not add a core import of either MoE plan copy. | Add `tests/test_compare_benchmark_pairs.py`: a subprocess import with `labs.moe_parallelism.plan` blocked must succeed; a temporary target exposing only `get_benchmark()` must reach the harness; a target exposing only the old plan trio must fail with the documented missing-`get_benchmark` error. Retain the import-edge gate. |
| Artifact-specific roofline (#2) | Add `core/analysis/roofline_model.py` containing a frozen `RooflineSpecs`, a pure `analyze_kernel(specs, duration_ms, flops, bytes_transferred, precision)` function, and explicit unknown-spec handling. Change `derive_roofline(metrics, specs)` in `deep_profiling_report.py`; add an explicit spec/metadata CLI input instead of consulting local CUDA. Make `ch08/roofline.py` consume the core calculator while keeping its runnable teaching/demo layer. | Add `tests/test_deep_profiling_roofline.py`: fixed metrics plus fixed specs produce exact peaks/ridge/utilization; patching local CUDA detection to raise does not affect offline output; missing artifact specs yield an explicit unknown result rather than CPU/B200 defaults; two different supplied specs produce different ceilings without changing achieved work. |
| Core Nsight stats parser (#3, #6) | Add `core/profiling/nsys_stats.py` with `summarize_report(report_path, *, kernel_regex=None, top_k=5, runner=subprocess.run)` and the CSV parse/rank helpers now embedded in chapter 17. `deep_profiling_report.py` and `nsys_summary.py` call it directly. `ch17/blackwell_profiling_guide.py` delegates its summary method to it; capture/context-manager teaching code stays in the chapter. | Add `tests/test_nsys_stats.py`: fixture CSV ranks by numeric time percentage, regex and `top_k` work, missing `nsys` and nonzero exit are explicit, malformed sections do not become successes, and both core callers import/run with `ch17` and `torch.profiler` blocked. |
| Chapter 16 power composition (#4) | Move the Python command to `scripts/benchmarking/precision_power_sweep.py` and the coupled shell driver to `scripts/benchmarking/run_power_efficiency_sweeps.sh`. `core/utils/power_sampling.py` remains shared core. Update the shell invocation and any entrypoint inventory; do not leave a compatibility module under core that reimports chapter 16. | Move/update the Python-entrypoint test; add a CPU test for `--help`; keep unit fixtures for mode parsing and cost math; mock the chapter runner and sampler to prove each requested mode keeps an independent result/skip/failure. The import-edge checker must see no old core module. |
| DataLoader analysis provider (#5) | Add `core/diagnostics/data_loading.py` with a pure `DataLoadingAnalysisProvider` protocol/result and a default sysfs/OS-backed analyzer. Inject it into `PerformanceCoreBase` (constructor or method argument). Return current/suggested affinity only. Keep the mutating `setup_grace_affinity()` operation in `ch04/gb200_grace_numa_optimization.py`, which may consume core detection helpers but is never called by `analyze_dataloader`. | Add `tests/test_data_loading_analysis.py`: patch `os.sched_setaffinity` to fail if called; unknown NUMA/GPU locality remains `None`; injected provider output reaches CLI/engine/MCP unchanged; provider failure returns explicit unavailable/error status; repeated analysis does not change process affinity. |
| Capability dump relocation (#7) | Move to `scripts/utilities/dump_hardware_capabilities.py`. Update `core/harness/run_benchmarks.py`, `core/tools/tools_commands.py`, `core/scripts/profiling/perf_triage_bundle.sh`, `tests/test_run_benchmarks_config_merge.py`, and `tests/test_python_entrypoints.py`. Keep its chapter 12 check explicitly named as such. A separate generic check, if desired, must use a core-owned minimal extension fixture. | Assert all five callers resolve the new path/module; run `python -m scripts.utilities.dump_hardware_capabilities --help`; mock the chapter loader so its success/failure is labeled `ch12.graph_bandwidth`, not generic CUDA-extension support; retain the import checker. |
| Extension precompile relocation (#8, #9) | Move to `scripts/utilities/precompile_cuda_extensions.py`. Update `core/harness/run_benchmarks.py`, its config-merge tests, and the Python-entrypoint inventory. The operational command is the composition root allowed to import both chapter packages. | With fake loaders, assert four chapter 6 and one chapter 12 attempts each receive a distinct status; one failure makes the command nonzero without erasing successes; no-CUDA is explicit skipped/unsupported rather than success; all path-resolution tests point to `scripts.utilities`. |
| Extension loader registry (#10, #11) | Add `core/utils/extension_registry.py` with immutable `ExtensionBuildSpec(owner, name, loader, capability_check)` and an explicit status enum (`passed`, `failed`, `skipped`). Change `extension_prewarm.prewarm_extensions(..., specs=...)` and `health_check(..., specs=...)` to consume specs and call `spec.loader()`. The no-argument core path includes only genuinely core-owned specs. The operational precompile command constructs the chapter 6/12 list and passes it in. Remove both static imports and the two dynamic chapter module strings from core. | Extend `tests/test_firstparty_correctness_regressions.py`: a fake loader is called exactly once; import alone cannot pass; raised loader is `failed`; false capability is `skipped`; chapter 6 and chapter 12 keys cannot satisfy each other; parallel results preserve owner/name; importing and running the default core path never imports `ch06`/`ch12`. GPU build/load for every real spec is a later per-target gate. |

## Why no exception is justified

The corrected checker is applying its documented rule to files that are
actually under `code/core`. Each edge can be resolved through a smaller,
clearer ownership change without making the rule path-sensitive or symbol-
sensitive. A waiver would conceal concrete behavior problems:

- a generic comparator importing an unused lab implementation;
- offline analysis using the current host's roofline specification;
- pure Nsight report parsing coupled to a chapter module and eager Torch imports;
- an analysis endpoint changing CPU affinity;
- operational all-chapter commands placed under core; and
- a prewarm path reporting package import as extension compilation.

String module names and repository paths used as declarative discovery data do
not need a waiver because they are not imports. Runtime dynamic imports from
those strings still need the same registration/composition ownership. The
current checker does not detect those dynamic edges, so deleting only the two
static health-check imports from `extension_prewarm.py` would be incomplete.

## Smallest coherent implementation order

1. Remove the unused MoE-specific branch from the generic pair comparator.
2. Extract one core Nsight stats parser and a pure, artifact-specific roofline
   specification/calculator; make chapters 8 and 17 depend on those core APIs.
3. Introduce the pure dataloader-analysis provider and separate any affinity
   application into an explicitly mutating operation.
4. Move the precision sweep and the two utility commands to `code/scripts/`,
   updating all known path and module callers.
5. Replace hardcoded extension imports/strings in core prewarm with registered
   loader callables and truthful per-loader results.
6. Re-run the import checker and focused CPU tests. GPU compile/load validation
   remains a separate acceptance gate and must preserve every extension result.

## Verification boundary

This review does not claim implementation or runtime acceptance. A future
source change should at minimum verify:

- the import checker exits 0 with no ignored paths;
- generic pair comparison imports without `labs.moe_parallelism` installed;
- offline deep-report fixtures use an explicit artifact spec and never inspect
  the local GPU;
- both core profiling callers share the same Nsight parser fixtures;
- `analyze_dataloader` cannot call affinity-changing functions;
- relocated script entrypoints and every internal path caller work; and
- extension prewarm negative controls prove that a loader was called, a failing
  loader is recorded as failed, and one chapter's result cannot satisfy another.

Actual CUDA extension compilation/loading remains HOLD until run on the intended
GPU/toolchain with each attempt reported separately.
