"""Bounded production bias+SiLU / Hopper-TMA extension acceptance, with no fallback."""
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

from run_feature_cuda_validation import source_hashes
from validation_process import run_command

TARGETS = {"sm_90": (9, 0), "sm_100a": (10, 0), "sm_103a": (10, 3),
           "sm_120": (12, 0), "sm_121": (12, 1)}
SOURCES = {"bias_silu": "labs/custom_vs_cublas/tcgen05_gemm.cu",
           "grace_tma": "labs/blackwell_matmul/grace_blackwell_kernels.cu"}
BIAS_SHAPES = ((128, 256, 64), (256, 768, 192), (384, 512, 320), (128, 1792, 1088))
TMA_SHAPES = ((1, 1, 8), (83, 67, 24), (159, 129, 72), (81, 193, 136), (160, 64, 256))


def operands(m, n, k):
    import torch
    a = ((((torch.arange(m*k, dtype=torch.float64) * 7) % 31) - 15) / 32).reshape(m, k).half()
    b = ((((torch.arange(n*k, dtype=torch.float64) * 11) % 37) - 18) / 32).reshape(n, k).half()
    a[0] = 0  # Every output column in row zero isolates the bias epilogue.
    return a, b


def must_reject(fn, *args):
    try:
        fn(*args)
    except RuntimeError:
        return
    raise AssertionError("invalid input was accepted")


def validate_bias(module, multi_device_controls=False):
    import torch
    checks = 0
    for index, (m, n, k) in enumerate(BIAS_SHAPES):
        a, b = operands(m, n, k)
        accumulator = a.double() @ b.double().T
        for dtype in (torch.float32, torch.float16):
            bias = torch.linspace(-4, 4, n, dtype=torch.float64).to(dtype)
            # The matrix products are exact dyadics; bias addition and SiLU are
            # evaluated independently on CPU. FP16 output can differ by one ULP.
            reference = torch.nn.functional.silu(accumulator + bias.double()).half()
            da, db, dz = a.cuda(), b.cuda(), bias.cuda()
            if index == 1:
                da = torch.stack((da, da), dim=-1)[:, :, 0]
                db = torch.stack((db, db), dim=-1)[:, :, 0]
                dz = torch.stack((dz, dz), dim=-1)[:, 0]
            # Exercise a nondefault current stream, including allocation and
            # contiguous conversions; completion must be joined before host reads.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                result = module.matmul_tcgen05_bias_silu(da, db, dz)
            stream.synchronize()
            output = result.cpu()
            torch.testing.assert_close(output, reference, atol=1e-4, rtol=1e-3, equal_nan=False)
            print(f"bias shape={m},{n},{k} dtype={dtype} full_output=PASS", flush=True)
            checks += 1
        # Check the shared non-fused code path after changing its epilogue copy.
        plain = module.matmul_tcgen05(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(plain, accumulator.half(), atol=0, rtol=0, equal_nan=False)
        checks += 1
    a = torch.zeros((128, 64), device="cuda", dtype=torch.float16)
    b = torch.zeros((256, 64), device="cuda", dtype=torch.float16)
    bias = torch.zeros(256, device="cuda")
    fn = module.matmul_tcgen05_bias_silu
    must_reject(fn, a, b, bias[:-1])
    must_reject(fn, a, b, bias.reshape(1, -1))
    must_reject(fn, a, b, bias.long())
    must_reject(fn, a, b, bias.cpu())
    must_reject(fn, a.float(), b, bias)
    for m, n, k in ((127, 256, 64), (128, 257, 64), (128, 256, 63), (0, 256, 64)):
        must_reject(fn, torch.empty((m, k), device="cuda", dtype=torch.float16),
                    torch.empty((n, k), device="cuda", dtype=torch.float16), torch.zeros(n, device="cuda"))
    if multi_device_controls and torch.cuda.device_count() > 1:
        must_reject(fn, a, b, bias.to("cuda:1"))
        must_reject(fn, a, b.to("cuda:1"), bias)
        print("multi-device rejection controls=PASS", flush=True)
    else:
        print("multi-device rejection controls=HOLD (requires --multi-device-controls and two allocated visible devices)", flush=True)
    return checks, 9


def validate_tma(module):
    import torch
    if not module.tma_supported():
        raise AssertionError("production capability query rejected this Hopper-or-newer exact target")
    print(f"production cluster_launch_supported={module.cluster_launch_supported()}", flush=True)
    checks = 0
    for index, (m, n, k) in enumerate(TMA_SHAPES):
        a, b = operands(m, n, k)
        reference = (a.double() @ b.double().T).half()
        da, db = a.cuda(), b.cuda()
        if index == 2:
            da = torch.stack((da, da), dim=-1)[:, :, 0]
            db = torch.stack((db, db), dim=-1)[:, :, 0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = module.optimized_blackwell_matmul_tma(da, db)
        stream.synchronize()
        output = result.cpu()
        torch.testing.assert_close(output, reference, atol=0, rtol=0, equal_nan=False)
        print(f"TMA shape={m},{n},{k} full_output=PASS", flush=True)
        checks += 1
    fn = module.optimized_blackwell_matmul_tma
    must_reject(fn, torch.empty((0, 8), device="cuda", dtype=torch.float16),
                torch.empty((8, 8), device="cuda", dtype=torch.float16))
    must_reject(fn, torch.zeros((8, 8), device="cuda"), torch.zeros((8, 8), device="cuda"))
    must_reject(fn, torch.zeros((8, 8), dtype=torch.float16), torch.zeros((8, 8), dtype=torch.float16))
    return checks, 3


def worker(library: Path, arch: str, case: str, multi_device_controls=False) -> int:
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != TARGETS[arch]:
        print("UNSUPPORTED: exact requested CUDA target unavailable")
        return 3
    spec = importlib.util.spec_from_file_location("audit_feature_" + case, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks, invalid = validate_bias(module, multi_device_controls) if case == "bias_silu" else validate_tma(module)
    torch.cuda.synchronize()
    print(f"FEATURE_EXTENSION_PASS case={case} full_outputs={checks} invalid_inputs={invalid}")
    return 0


def compile_worker(args) -> int:
    from torch.utils.cpp_extension import load
    code = Path(__file__).resolve().parents[2]
    target = args.arch.removeprefix("sm_")
    flags = ["-O2", "-std=c++20", f"-gencode=arch=compute_{target},code=sm_{target}"]
    includes = [str(args.cutlass_include.resolve())] if args.case == "bias_silu" else []
    module = load(name="audit_feature_" + args.case, sources=[str(code / SOURCES[args.case])],
                  extra_include_paths=includes, extra_cuda_cflags=flags, extra_cflags=["-std=c++20"],
                  extra_ldflags=["-lcuda"], build_directory=str((args.output_dir / "build").resolve()), verbose=True)
    library = Path(module.__file__).resolve()
    (args.output_dir / "build-result.json").write_text(json.dumps({"library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(SOURCES), required=True)
    parser.add_argument("--arch", choices=tuple(TARGETS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cutlass-include", type=Path)
    parser.add_argument("--sanitizers", choices=("all", "none"), default="all")
    parser.add_argument("--multi-device-controls", action="store_true", help="Use cuda:1 only when both visible devices are explicitly allocated for this validation")
    parser.add_argument("--worker-library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compile-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case == "bias_silu" and args.arch not in ("sm_100a", "sm_103a"):
        parser.error("tcgen05 bias kernel requires exact sm_100a or sm_103a")
    if args.compile_worker:
        return compile_worker(args)
    if args.worker_library:
        return worker(args.worker_library, args.arch, args.case, args.multi_device_controls)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parents[2]
    manifest = source_hashes(code / SOURCES[args.case], code)
    for name in (Path(__file__).name, "validation_process.py", "run_feature_cuda_validation.py"):
        path = Path(__file__).with_name(name)
        manifest[str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"status": "PENDING", "case": args.case, "arch": args.arch, "checks": [],
              "source_sha256": manifest, "sanitizers": args.sanitizers,
              "multi_device_controls_requested": args.multi_device_controls,
              "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def finish(status, exit_code, reason):
        report.update(status=status, reason=reason)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{status}: {reason}", flush=True)
        return exit_code
    nvcc = shutil.which("nvcc")
    sanitizer = shutil.which("compute-sanitizer")
    if nvcc is None:
        return finish("HOLD", 3, "nvcc unavailable; no CUDA compilation or device execution occurred")
    if args.sanitizers == "all" and sanitizer is None:
        return finish("HOLD", 3, "compute-sanitizer unavailable")
    if args.case == "bias_silu" and (args.cutlass_include is None or not (args.cutlass_include / "cute/tensor.hpp").is_file()):
        return finish("HOLD", 3, "compatible explicit --cutlass-include is required for tcgen05")
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != TARGETS[args.arch]:
        return finish("HOLD", 3, "selected PyTorch lacks the exact requested CUDA target")
    report.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                  cxx11_abi=torch.compiled_with_cxx11_abi(), gpu=torch.cuda.get_device_name())
    if args.case == "bias_silu":
        report["cutlass_header_sha256"] = {}
        for name in ("cute/tensor.hpp", "cutlass/version.h", "cutlass/arch/barrier.h", "cute/atom/mma_traits_sm100.hpp"):
            header = args.cutlass_include / name
            if not header.is_file():
                return finish("HOLD", 3, f"missing CUTLASS header: {header}")
            report["cutlass_header_sha256"][name] = hashlib.sha256(header.read_bytes()).hexdigest()
    def run(label, command):
        started = time.monotonic()
        result = run_command(command, timeout=300)
        log = args.output_dir / f"{label}.txt"
        log.write_text(result.stdout + result.stderr)
        report["checks"].append({"label": label, "command": command, "exit_code": result.returncode,
                                 "log": log.name, "elapsed_seconds": time.monotonic() - started})
        return result
    if run("nvcc-version", [nvcc, "--version"]).returncode:
        return finish("FAIL", 1, "nvcc version command failed")
    inventory = shutil.which("nvidia-smi")
    if inventory:
        run("gpu-inventory", [inventory, "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv"])
    (args.output_dir / "build").mkdir()
    command = [sys.executable, str(Path(__file__).resolve()), "--case", args.case, "--arch", args.arch,
               "--output-dir", str(args.output_dir.resolve()), "--compile-worker"]
    if args.cutlass_include:
        command += ["--cutlass-include", str(args.cutlass_include.resolve())]
    if run("compile", command).returncode:
        return finish("FAIL", 1, "production extension compilation/import failed or timed out")
    build_result = args.output_dir / "build-result.json"
    if not build_result.is_file():
        return finish("FAIL", 1, "compile worker did not produce a module receipt")
    result = json.loads(build_result.read_text())
    library = Path(result["library"])
    if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != result["library_sha256"]:
        return finish("FAIL", 1, "compiled module missing or changed before validation")
    report["library_sha256"] = result["library_sha256"]
    count, invalid = (12, 9) if args.case == "bias_silu" else (5, 3)
    marker = f"FEATURE_EXTENSION_PASS case={args.case} full_outputs={count} invalid_inputs={invalid}"
    for mode in ("run", "memcheck", "racecheck", "synccheck") if args.sanitizers == "all" else ("run",):
        command = [sys.executable, str(Path(__file__).resolve()), "--case", args.case,
                   "--arch", args.arch, "--worker-library", str(library)]
        if args.multi_device_controls:
            command.append("--multi-device-controls")
        if mode != "run":
            command = [sanitizer, "--tool", mode, "--error-exitcode", "1"] + command
        result = run(mode, command)
        if result.returncode == 3:
            return finish("HOLD", 3, "worker target unavailable")
        if result.returncode or marker not in result.stdout:
            return finish("FAIL", 1, f"actual CUDA {mode} failed")
    return finish("PASS" if args.sanitizers == "all" else "PASS_WITHOUT_SANITIZERS", 0,
                  "production full outputs checked; multi-device controls separately reported; no performance qualification")


if __name__ == "__main__":
    raise SystemExit(main())
