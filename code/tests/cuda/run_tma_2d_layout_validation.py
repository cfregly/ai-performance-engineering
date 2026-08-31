"""Build and run real TMA callers with full output checks on an explicit GPU target."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

CASES = (
    "async_prefetch_2d_demo",
    "optimized_tma_bulk_tensor_2d",
    "optimized_tma_copy",
    "tma_2d_pipeline_blackwell",
    "cuda13_demos",
    "tma_multicast_cluster",
    "tma_multicast_baseline",
)
# Three rectangular inputs per production template configuration. The pipeline
# selects six configurations, the shared demo selects three, other callers one.
SHAPE_COUNTS = (3, 3, 3, 18, 9, 3, 3)


def source_hashes(source: Path, code_root: Path) -> dict[str, str]:
    """Record repository sources, including quoted transitive header includes."""
    result = {}
    pending = [source]
    while pending:
        path = pending.pop().resolve()
        relative = str(path.relative_to(code_root))
        if relative in result:
            continue
        content = path.read_bytes()
        result[relative] = hashlib.sha256(content).hexdigest()
        for include in re.findall(r'^\s*#include\s+"([^\"]+)"', content.decode(), re.MULTILINE):
            dependency = (path.parent / include).resolve()
            if dependency.is_file() and dependency.is_relative_to(code_root):
                pending.append(dependency)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("sm_100a", "sm_103a"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-sanitizer", action="store_true", help="Require and run memcheck too")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("tma_2d_layout_validation.cu").resolve()
    report = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": args.arch,
        "source_sha256": source_hashes(source, source.parents[2]),
        "status": "PENDING",
        "checks": [],
    }

    def finish(status: str, code: int, reason: str = "") -> int:
        report.update(status=status, reason=reason)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{status}: {reason or args.output_dir}")
        return code

    nvcc = shutil.which("nvcc")
    sanitizer = shutil.which("compute-sanitizer")
    if nvcc is None:
        return finish("SKIPPED", 3, "nvcc is unavailable; no CUDA compile or execution occurred")
    if args.compute_sanitizer and sanitizer is None:
        return finish("SKIPPED", 3, "compute-sanitizer was requested but is unavailable")

    def run(label: str, command: list[str]) -> subprocess.CompletedProcess[str]:
        started = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
            result = subprocess.CompletedProcess(command, 124, stdout, stderr + "\nTimed out after 300 seconds\n")
        log = args.output_dir / f"{label}.log"
        log.write_text(result.stdout + result.stderr)
        report["checks"].append({"label": label, "command": command,
                                 "exit_code": result.returncode, "log": log.name,
                                 "elapsed_seconds": time.monotonic() - started})
        print(f"{label}: exit {result.returncode}", flush=True)
        return result

    run("nvcc-version", [nvcc, "--version"])
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        run("gpu-inventory", [nvidia_smi, "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv"])
    feature_target = args.arch.removeprefix("sm_").removesuffix("a")
    for case_id, name in enumerate(CASES):
        binary = (args.output_dir / name).resolve()
        command = [nvcc, "-O2", "-std=c++17", "-rdc=true", f"-arch={args.arch}",
                   f"-DTMA_MULTICAST_TARGET={feature_target}", f"-DTMA_VALIDATION_CASE={case_id}",
                   str(source), "-lcuda", "-o", str(binary)]
        if run(f"{name}-compile", command).returncode != 0:
            return finish("FAIL", 1, f"CUDA compilation failed for {name}")
        execution = run(f"{name}-run", [str(binary)])
        if execution.returncode == 3:
            return finish("SKIPPED", 3, f"The selected CUDA target is unavailable for {name}")
        marker = f"TMA_LAYOUT_PASS case={case_id} shapes={SHAPE_COUNTS[case_id]} full_output_and_canaries=checked"
        if execution.returncode != 0 or marker not in execution.stdout:
            return finish("FAIL", 1, f"Full-output CUDA verification failed for {name}")
        if args.compute_sanitizer:
            result = run(f"{name}-memcheck", [sanitizer, "--tool", "memcheck", "--error-exitcode", "1", str(binary)])
            if result.returncode != 0 or marker not in result.stdout:
                return finish("FAIL", 1, f"CUDA memcheck failed for {name}")
    return finish("PASS", 0, f"All seven real callers passed {sum(SHAPE_COUNTS)} full-output and canary cases")


if __name__ == "__main__":
    raise SystemExit(main())
