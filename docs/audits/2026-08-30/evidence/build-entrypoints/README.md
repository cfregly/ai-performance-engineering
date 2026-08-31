# P01 build entry-point source evidence

Source/orchestration status: **PASS** for the recorded host checks. CUDA compile,
device-link, Python extension import, and GPU correctness: **HOLD / not run**.
This macOS arm64 host has no `nvcc` or supported CUDA GPU. An actual CMake
configure stopped with `Failed to find nvcc`; that failure is retained below.

## Findings addressed

| Finding | Source correction |
| --- | --- |
| W1-007 | All 82 hardware aliases in ch01–ch20 and the two affected labs recurse with explicit ARCH before architecture selection. Clean and build run sequentially. Alias/compare/clean dispatch can parse without a GPU; ordinary builds still require a detected or explicit target. Mixed alias goals are rejected to prevent a parallel clean/build race. |
| W1-025 | CUTLASS CMake uses one `CMAKE_CUDA_ARCHITECTURES` policy before `project()`: default `100a;103a`, with supported caller overrides retained. The target enables separable compilation and device-symbol resolution. Conflicting manual gencode/plain-100 settings are removed. |
| W1-052 | ch16/ch18/ch19/ch20 compare loops stop on the first failed sub-build; the CI compare script and existing negative-control tests now include these chapters. |
| W1-055 / W1-057 | Reporting distinguishes 10.0 B200/GB200, 10.3 B300/GB300, 12.0 RTX Blackwell, and 12.1 GB10/DGX Spark. Metadata no longer folds all 12.x devices into 12.1 or invents an SM count from compute capability. Existing functional `SM_MAP`/legacy fallback behavior is retained; unsupported make targets still fail. |
| W1-067 | The plain GEMM lab Makefile uses shared architecture selection and honors all four supported repository target values. Its `kernels.cu` is the non-tensor-core baseline; this Makefile is not the tcgen05 extension build. |
| W1-111 | The helper resolves `code/`, builds existing suffixed targets, and reports their actual paths. The obsolete `async_prefetch_tma.cu` no longer exists, so the helper selects the existing `async_prefetch_2d_demo.cu` and labels it as a 2D copy. `--dry-run` invokes real make `-n` without claiming binaries were built. |
| W1-112 | The warp-specialization wrapper resolves its test script from `code/`; a real `--help` invocation from outside that directory passes. |
| W1-115 | CMake queries the selected Python's actual `torch.compiled_with_cxx11_abi()`, uses its TorchConfig/include/library locations, and applies the reported ABI instead of hardcoding zero. |

GPU architecture does not determine host CPU architecture. The shared make
configuration no longer adds `-mcpu=native` solely because the GPU is SM120 or
SM121; explicit `HOST_ARCH_FLAGS` remains available. GPU target gencode values
are unchanged: repository `sm_100`/`sm_103` select `100a`/`103a`, while
`sm_120`/`sm_121` select generic `120`/`121`. The distinct `120a` target is not a
substitute for this CUTLASS SM100 tcgen05 implementation.

## Adjacent prerequisites fixed within owned files

- CMake now defines the pybind module name used by `TORCH_EXTENSION_NAME` and
  locates/links `torch_python` from the selected torch installation. The build
  script prints the matching Python module import, rather than suggesting that
  loading the library registers torch operators. This still requires an actual
  target build/import check.
- The SM121 verification helper now resolves the existing TMA sample from the
  code root, uses a unique temporary binary, and propagates compile/run errors.
  It uses the direct TMA sample, which reports unsupported runtime/hardware
  instead of falling back to a manual-copy kernel. The verifier requires exact
  SM121 hardware, treats missing tools as unsupported, and no longer presents
  torch.compile correctness or sample execution as TMA-instruction proof.
- Bootstrap evidence migration created 17 byte-identical `.txt` copies of
  ignored `.log` files. Original logs and pre-migration receipt JSON remain
  intact. See `../bootstrap/log-migration.json`; no check was rerun by migration.

## Executed checks and their limits

`before.txt` records the initial **114 failed / 46 passed** run. It includes the
new CMake policy checks before that helper existed, as well as reproduced alias,
compare-loop, metadata and wrapper failures; it is not a count of audit issues.
`after.txt` records **218 passed** across:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_build_entrypoints.py tests/test_dual_arch_make_contract.py tests/test_repository_configuration.py tests/test_ch10_makefile_contract.py tests/test_tma_2d_layout.py
```

The checks use real recursive make dry runs, real shell control flow and
negative command controls, real CMake policy execution, and the actual installed
torch metadata. GNU make `-n` propagates into recursive calls; no repository
clean/build recipe was executed. The four compare-loop controls override clean
and return an intentional nonzero subcommand status. No CUDA success is mocked.

Additional receipts retain Bash syntax and scoped whitespace checks, CMake
4.3.4 version output, actual torch 2.8.0 CPU metadata (reported CXX11 ABI=1),
the real CMake missing-nvcc failure, and SM121 helper unsupported exit 3.
The host torch is not the pinned Linux torch 2.9.1+cu130 stack.

`validation-receipts.json` contains exact commands, outputs, exit codes, source
hashes, platform, branch, and Git base. `primary-sources.json` contains retrieval
URLs and hashes where available. The direct PyTorch documentation fetch returned
HTTP 403; the same API documentation was readable through the web tool.

## Remaining target acceptance

Use a fresh build directory for each selected Python/toolchain environment.
CMake 3.31.8+ and CUDA 13.0+ are now explicit module prerequisites; no toolchain
was installed or upgraded here. On an assigned supported target:

1. Capture the exact compiler, CMake, torch/CUDA versions, selected ABI, CUTLASS
   source/submodule commits, GPU identity, and source hashes.
2. Run the chapter/lab builds and CUTLASS verbose build. Confirm both compilation
   and device linking contain the selected `100a`/`103a` targets without a
   conflicting plain-100 pass. Inspect the resulting device images.
3. Import `cutlass_blackwell_gemm` in that same Python environment, then perform
   independent full-output GPU correctness checks on each selected hardware
   target. A successful host metadata query cannot close ABI import acceptance.
4. Run the SM121 verifier only on an assigned SM121 system with a compatible
   toolkit. Its sample check is bounded; TMA instruction inspection and full
   output validation remain separate. No SM121 qualification was obtained here.

## Primary contracts

- [NVIDIA GPU compute-capability table](https://developer.nvidia.com/cuda/gpus)
  supports the distinct device labels.
- [CUDA 13 compiler target documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch)
  distinguishes generic, family, and architecture-specific targets; `a` targets
  are restricted to their corresponding architecture.
- [CMake CUDA_ARCHITECTURES](https://cmake.org/cmake/help/v3.31/prop_tgt/CUDA_ARCHITECTURES.html)
  defines target device-code generation, and the
  [CMake 3.31.8 validator](https://github.com/Kitware/CMake/blob/v3.31.8/Modules/Internal/CMakeCUDAArchitecturesValidate.cmake)
  accepts architecture-specific suffixes.
- [PyTorch's ABI query](https://docs.pytorch.org/docs/2.9/generated/torch.compiled_with_cxx11_abi.html)
  reports the actual build flag. The
  [PyTorch 2.9.1 extension builder](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/cpp_extension.py)
  supplies the extension name and links the Python binding library.
