"""Compile real chapter kernels and run full-output + sanitizer acceptance.

No GPU is simulated. Exit 3 is unsupported/HOLD, never a successful test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from validation_process import run_command

CASES = (
    ("occupancy", 6), ("ilp", 16), ("threshold", 78),
    ("overlap", 47), ("ordered", 24), ("warp_streams", 3),
    ("graph_baseline", 3), ("graph_replay", 3), ("fp8", 6),
    ("dsmem", 6), ("pcie_kernel", 3),
)


def source_hashes(source: Path, code: Path) -> dict[str, str]:
    result = {}
    pending = [source]
    while pending:
        path = pending.pop().resolve()
        key = str(path.relative_to(code))
        if key in result:
            continue
        content = path.read_bytes()
        result[key] = hashlib.sha256(content).hexdigest()
        for include in re.findall(r'^\s*#include\s+"([^\"]+)"', content.decode(), re.MULTILINE):
            child = (path.parent / include).resolve()
            if child.is_file() and child.is_relative_to(code):
                pending.append(child)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("sm_100a", "sm_103a", "sm_120", "sm_121"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sanitizers", choices=("all", "none"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("chapter_cuda_validation.cu").resolve()
    code = source.parents[2]
    report = {"arch": args.arch, "status": "PENDING", "checks": [],
              "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "source_sha256": source_hashes(source, code), "sanitizers": args.sanitizers}
    report["source_sha256"][str(Path(__file__).resolve().relative_to(code))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    process_helper = Path(__file__).with_name("validation_process.py").resolve()
    report["source_sha256"][str(process_helper.relative_to(code))] = hashlib.sha256(process_helper.read_bytes()).hexdigest()

    def finish(status: str, exit_code: int, reason: str) -> int:
        report.update(status=status, reason=reason)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{status}: {reason}", flush=True)
        return exit_code

    nvcc = shutil.which("nvcc")
    sanitizer = shutil.which("compute-sanitizer")
    if nvcc is None:
        return finish("HOLD", 3, "nvcc unavailable; no CUDA compilation or device execution occurred")
    if args.sanitizers == "all" and sanitizer is None:
        return finish("HOLD", 3, "compute-sanitizer unavailable; required acceptance tools are missing")

    def run(label: str, command: list[str]) -> subprocess.CompletedProcess:
        start = time.monotonic()
        result = run_command(command, timeout=300)
        log = args.output_dir / f"{label}.txt"
        log.write_text(result.stdout + result.stderr)
        report["checks"].append({"label": label, "command": command, "exit_code": result.returncode,
                                 "elapsed_seconds": time.monotonic() - start, "log": log.name})
        print(f"{label}: exit {result.returncode}", flush=True)
        return result

    if run("nvcc-version", [nvcc, "--version"]).returncode:
        return finish("FAIL", 1, "nvcc version command failed")
    smi = shutil.which("nvidia-smi")
    if smi:
        run("gpu-inventory", [smi, "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv"])
    target = args.arch.removeprefix("sm_").removesuffix("a")
    for case, (name, count) in enumerate(CASES):
        binary = (args.output_dir / name).resolve()
        command = [nvcc, "-O2", "-std=c++17", "-rdc=true", f"-arch={args.arch}",
                   f"-DCHAPTER_VALIDATION_CASE={case}", f"-DCHAPTER_VALIDATION_TARGET={target}",
                   str(source), "-lcuda", "-o", str(binary)]
        if run(f"{name}-compile", command).returncode:
            return finish("FAIL", 1, f"CUDA compilation failed for {name}")
        marker = f"CHAPTER_CUDA_PASS case={case} checks={count} full_output=checked"
        for mode in ("run", "memcheck", "racecheck", "synccheck") if args.sanitizers == "all" else ("run",):
            command = [str(binary)] if mode == "run" else [sanitizer, "--tool", mode, "--error-exitcode", "1", str(binary)]
            result = run(f"{name}-{mode}", command)
            if result.returncode == 3:
                return finish("HOLD", 3, f"Target/device feature unavailable for {name}")
            if result.returncode or marker not in result.stdout:
                return finish("FAIL", 1, f"Actual CUDA {mode} failed for {name}")
    status = "PASS" if args.sanitizers == "all" else "PASS_WITHOUT_SANITIZERS"
    return finish(status, 0, f"{sum(count for _, count in CASES)} actual full-output checks; no performance qualification")


if __name__ == "__main__":
    raise SystemExit(main())
