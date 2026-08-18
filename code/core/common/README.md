# Common Infrastructure

## Summary
Shared headers, CUDA build flags, and Python utilities that keep every chapter and lab on the same benchmarking, profiling, and build rails.

## Learning Goals
- Reuse the benchmark harness, logging, and artifact plumbing instead of rebuilding them per chapter.
- Target the right GPU features (TMA, pipeline API, SDPA backends) by querying capabilities up front.
- Plug new CUDA/Triton kernels into the harness through `core/utils/chapter_compare_template.py`.
- Keep builds reproducible on Blackwell/Grace-Blackwell by leaning on the common Makefile fragments and env defaults.

## Directory Layout
| Path | Description |
| --- | --- |
| `cuda_arch.mk`, `cuda13_demo_runner.cuh` | Makefile includes and helper headers for SM100, SM103, SM120, or SM121 builds and CUDA 13 samples. |
| `headers/arch_detection.cuh`, `headers/tma_helpers.cuh` | Device feature probes plus TMA helpers shared by CUDA benchmarks and extensions. |
| `tcgen05/` | SM100 (tcgen05/TMEM) kernel loaders and wrappers used by the tcgen05 benchmarks. |
| `async_input_pipeline.py`, `moe_parallelism_plan.py`, `device_utils.py` | Shared Python helpers pulled into both core and labs. |

## Running / Usage
- **Build system**: include `../core/common/cuda_arch.mk` from chapter Makefiles to pick up architecture flags and helper rules.
- **Environment**: `from core.env import apply_env_defaults; apply_env_defaults()` before running benchmarks to set CUDA paths, allocator knobs, and cache locations.
- **Harness**: standard pattern inside chapter scripts:
  ```python
  from pathlib import Path
  from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
  from core.utils.chapter_compare_template import discover_benchmarks, load_benchmark

  harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=10, warmup=3))
  for baseline, optimized_list, _ in discover_benchmarks(Path("ch01")):
      bench = load_benchmark(baseline, optimized_list[0])
      result = harness.benchmark(bench)
      print(result.timing.mean_ms)
  ```
- **CUDA headers**: include `../../core/common/headers/arch_detection.cuh` to select tiles and query limits; include `../../core/common/headers/tma_helpers.cuh` to encode tensor maps for `cp.async.bulk.tensor` kernels.

## Validation Checklist
- `python - <<'PY'\nfrom core.env import dump_environment_and_capabilities\ndump_environment_and_capabilities()\nPY` prints CUDA paths, NCCL preload, and TMA/pipeline support.
- `python - <<'PY'\nfrom pathlib import Path\nfrom core.utils.chapter_compare_template import discover_benchmarks\nprint(len(discover_benchmarks(Path(\"ch01\"))))\nPY` confirms harness discovery works end-to-end.
- Build each configured `ARCH` separately after including `cuda_arch.mk`. The hosted compare workflow verifies `sm_100`, `sm_103`, `sm_120`, and `sm_121` in four independent builds.

## Notes
- Env defaults create `.torch_extensions/` and `.torch_inductor/` under the current workspace to avoid `/tmp` contention during repeated runs.
- Nsight/Proton helpers are optional; imports degrade gracefully when tools are missing so chapter scripts remain runnable on developer laptops.
