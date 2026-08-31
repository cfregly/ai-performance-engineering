"""Compile the production ch10 tcgen05 kernel and verify complete asymmetric GEMMs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from validation_process import run_command

SHAPES = ((256, 512, 64), (384, 256, 192), (128, 1792, 256),
          (256, 2048, 320), (384, 2304, 576), (256, 4352, 1088), (128, 256, 64))
MODULE_NAME = "audit_ch10_warp_specialized"


def worker(library: Path, arch: str) -> int:
    import torch
    expected_cc = (10, 0) if arch == "sm_100a" else (10, 3)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != expected_cc:
        print("UNSUPPORTED: exact requested tcgen05 target unavailable")
        return 3
    spec = importlib.util.spec_from_file_location(MODULE_NAME, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for index, (m, n, k) in enumerate(SHAPES):
        a = (((torch.arange(m * k, dtype=torch.float64) * 7) % 31) / 32 - .125).reshape(m, k).half()
        b = (((torch.arange(n * k, dtype=torch.float64) * 11) % 37) / 32 - .1875).reshape(n, k).half()
        reference = (a.double() @ b.double().T).half()
        da, db = a.cuda(), b.cuda()
        if index == len(SHAPES) - 1:
            # The production entrypoint promises to make noncontiguous inputs contiguous.
            da = torch.stack((da, da), dim=-1)[:, :, 0]
            db = torch.stack((db, db), dim=-1)[:, :, 0]
        output = module.matmul_tcgen05_warp_specialized(da, db).cpu()
        # Inputs/products are dyadic (multiples of 1/32 and 1/1024).
        # K<=1088 keeps every FP32 partial sum exactly representable; the
        # final FP16 rounding must match. A tolerance could hide wrong tiles.
        torch.testing.assert_close(output, reference, atol=0, rtol=0, equal_nan=False)
        print(f"shape={m},{n},{k} full_output=PASS max_abs={(output.float() - reference.float()).abs().max().item():.9g}", flush=True)
    for m, n, k in ((127, 256, 64), (128, 257, 64), (128, 256, 63), (0, 256, 64)):
        try:
            module.matmul_tcgen05_warp_specialized(torch.empty((m, k), device="cuda", dtype=torch.float16),
                                                   torch.empty((n, k), device="cuda", dtype=torch.float16))
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsupported dimensions accepted: {m},{n},{k}")
    torch.cuda.synchronize()
    print(f"CH10_TCGEN05_PASS full_outputs={len(SHAPES)} invalid_shapes=4")
    return 0


def compile_worker(output: Path, arch: str, cutlass_include: Path) -> int:
    from torch.utils.cpp_extension import load
    code = Path(__file__).resolve().parents[2]
    target = arch.removeprefix("sm_")
    flags = ["-O2", "-std=c++20", f"-gencode=arch=compute_{target},code=sm_{target}"]
    module = load(name=MODULE_NAME, sources=[str(code / "ch10/tcgen05_warp_specialized.cu")],
                  extra_include_paths=[str(cutlass_include.resolve())], extra_cuda_cflags=flags,
                  extra_cflags=["-std=c++20"], extra_ldflags=["-lcuda"],
                  build_directory=str((output / "build").resolve()), verbose=True)
    library = Path(module.__file__).resolve()
    (output / "build-result.json").write_text(json.dumps({"library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("sm_100a", "sm_103a"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cutlass-include", type=Path)
    parser.add_argument("--sanitizers", choices=("all", "none"), default="all")
    parser.add_argument("--worker-library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compile-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compile_worker:
        if args.output_dir is None or args.cutlass_include is None:
            parser.error("compile worker requires output directory and CUTLASS include")
        return compile_worker(args.output_dir, args.arch, args.cutlass_include)
    if args.worker_library:
        return worker(args.worker_library, args.arch)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[2]
    sources = [code / "ch10/tcgen05_warp_specialized.cu", code / "ch10/grouped_tile_schedule.cuh", Path(__file__).resolve(), Path(__file__).with_name("validation_process.py").resolve()]
    report = {"status": "PENDING", "arch": args.arch, "checks": [], "sanitizers": args.sanitizers,
              "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "source_sha256": {str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}

    def finish(status: str, code: int, reason: str) -> int:
        report.update(status=status, reason=reason)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{status}: {reason}", flush=True)
        return code

    nvcc = shutil.which("nvcc")
    sanitizer = shutil.which("compute-sanitizer")
    if nvcc is None:
        return finish("HOLD", 3, "nvcc unavailable; no CUDA compilation or execution occurred")
    if args.sanitizers == "all" and sanitizer is None:
        return finish("HOLD", 3, "compute-sanitizer unavailable")
    if args.cutlass_include is None or not (args.cutlass_include / "cute/tensor.hpp").is_file():
        return finish("HOLD", 3, "explicit compatible --cutlass-include is required")
    import torch
    if not torch.cuda.is_available():
        return finish("HOLD", 3, "CUDA unavailable in selected PyTorch")
    cc = torch.cuda.get_device_capability()
    if cc != ((10, 0) if args.arch == "sm_100a" else (10, 3)):
        return finish("HOLD", 3, f"requested {args.arch}, actual CC {cc}")
    report.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                  cxx11_abi=torch.compiled_with_cxx11_abi(), gpu=torch.cuda.get_device_name(),
                  cutlass_include=str(args.cutlass_include.resolve()))
    header_manifest = {}
    for name in ("cute/tensor.hpp", "cutlass/version.h", "cutlass/arch/barrier.h", "cute/atom/mma_traits_sm100.hpp"):
        header = args.cutlass_include / name
        if not header.is_file():
            return finish("HOLD", 3, f"missing CUTLASS header: {header}")
        header_manifest[name] = hashlib.sha256(header.read_bytes()).hexdigest()
    report["cutlass_header_sha256"] = header_manifest
    def run(label: str, command: list[str]) -> subprocess.CompletedProcess:
        started = time.monotonic()
        result = run_command(command, timeout=300)
        log = args.output_dir / f"{label}.txt"
        log.write_text(result.stdout + result.stderr)
        report["checks"].append({"label": label, "command": command, "exit_code": result.returncode,
                                 "log": log.name, "elapsed_seconds": time.monotonic() - started})
        return result

    build = args.output_dir / "build"
    build.mkdir()
    target = args.arch.removeprefix("sm_")
    report["build"] = {"source": str(sources[0]), "directory": str(build.resolve()),
                       "cuda_flags": ["-O2", "-std=c++20", f"-gencode=arch=compute_{target},code=sm_{target}"]}
    command = [sys.executable, str(Path(__file__).resolve()), "--arch", args.arch,
               "--output-dir", str(args.output_dir.resolve()), "--cutlass-include", str(args.cutlass_include.resolve()),
               "--compile-worker"]
    if run("compile", command).returncode:
        return finish("FAIL", 1, "production CUDA extension compilation/import failed or timed out; see compile.txt")
    build_result = args.output_dir / "build-result.json"
    if not build_result.is_file():
        return finish("FAIL", 1, "compile worker did not produce a module receipt")
    result = json.loads(build_result.read_text())
    library = Path(result["library"]).resolve()
    if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != result["library_sha256"]:
        return finish("FAIL", 1, "compiled library is missing or changed before validation")
    report["library_sha256"] = result["library_sha256"]
    expected_marker = f"CH10_TCGEN05_PASS full_outputs={len(SHAPES)} invalid_shapes=4"
    for mode in ("run", "memcheck", "racecheck", "synccheck") if args.sanitizers == "all" else ("run",):
        command = [sys.executable, str(Path(__file__).resolve()), "--arch", args.arch, "--worker-library", str(library)]
        if mode != "run":
            command = [sanitizer, "--tool", mode, "--error-exitcode", "1"] + command
        result = run(mode, command)
        if result.returncode == 3:
            return finish("HOLD", 3, "requested target unavailable to worker")
        if result.returncode or expected_marker not in result.stdout:
            return finish("FAIL", 1, f"actual GPU {mode} failed")
    return finish("PASS" if args.sanitizers == "all" else "PASS_WITHOUT_SANITIZERS", 0,
                  "complete production GEMM outputs checked; no performance qualification")


if __name__ == "__main__":
    raise SystemExit(main())
