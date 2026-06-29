"""Occupancy tuning sweep tool (chapter utility, not a benchmark pair).

This tool runs the `occupancy_tuning` CUDA binary with a small set of presets to
illustrate how block size, shared memory, unrolling, and maxrregcount interact.

Usage:
  python ch08/occupancy_tuning_tool.py
  python ch08/occupancy_tuning_tool.py --preset baseline --preset optimized
  python ch08/occupancy_tuning_tool.py --preset maxrreg32 --runs 5
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Preset:
    name: str
    block_size: int
    smem_bytes: int
    unroll: int
    inner_iters: int
    reps: int
    maxrregcount: Optional[int] = None


PRESETS: dict[str, Preset] = {
    "baseline": Preset(
        name="baseline",
        block_size=32,
        smem_bytes=45_000,
        unroll=1,
        inner_iters=1,
        reps=60,
    ),
    "optimized": Preset(
        name="optimized",
        block_size=256,
        smem_bytes=0,
        unroll=8,
        inner_iters=1,
        reps=60,
    ),
    "bs64": Preset(
        name="bs64",
        block_size=64,
        smem_bytes=0,
        unroll=8,
        inner_iters=1,
        reps=60,
    ),
    "bs128": Preset(
        name="bs128",
        block_size=128,
        smem_bytes=0,
        unroll=8,
        inner_iters=1,
        reps=60,
    ),
    "maxrreg32": Preset(
        name="maxrreg32",
        block_size=256,
        smem_bytes=0,
        unroll=8,
        inner_iters=1,
        reps=60,
        maxrregcount=32,
    ),
}


AVG_KERNEL_RE = re.compile(r"avg_kernel_ms=([0-9]+\.?[0-9]*)")


def _detect_arch_suffix() -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for occupancy tuning tool") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run occupancy tuning tool")

    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if capability >= 121:
        return "sm_121", "_sm121"
    if capability >= 103:
        return "sm_103", "_sm103"
    if capability >= 100:
        return "sm_100", "_sm100"
    raise RuntimeError(f"Unsupported compute capability {major}.{minor}")


def _build_binary(chapter_dir: Path, arch: str, suffix: str, maxrregcount: Optional[int]) -> Path:
    env = os.environ.copy()
    if maxrregcount is None:
        env.pop("MAXRREGCOUNT", None)
    else:
        env["MAXRREGCOUNT"] = str(maxrregcount)

    target = f"occupancy_tuning{suffix}"
    subprocess.run(
        ["make", f"ARCH={arch}", target],
        cwd=chapter_dir,
        env=env,
        check=True,
        capture_output=False,
        text=True,
    )
    exe = chapter_dir / target
    if not exe.exists():
        raise FileNotFoundError(f"Built binary not found at {exe}")
    return exe


def _run_once(exe: Path, preset: Preset) -> float:
    cmd = [
        str(exe),
        "--block-size",
        str(preset.block_size),
        "--smem-bytes",
        str(preset.smem_bytes),
        "--unroll",
        str(preset.unroll),
        "--inner-iters",
        str(preset.inner_iters),
        "--reps",
        str(preset.reps),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    match = AVG_KERNEL_RE.search(completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse avg_kernel_ms from stdout:\n{completed.stdout}")
    return float(match.group(1))


def _mean_runtime_ms(exe: Path, preset: Preset, runs: int) -> float:
    count = max(runs, 1)
    total_ms = 0.0
    for _ in range(count):
        total_ms += _run_once(exe, preset)
    return total_ms / count


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="Preset to run (repeatable). Default: baseline + optimized + maxrreg32.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of outer runs per preset (the binary itself also loops via --reps).",
    )
    args = parser.parse_args(argv)

    selected = args.preset or ["baseline", "optimized", "maxrreg32"]
    presets = [PRESETS[name] for name in selected]

    chapter_dir = Path(__file__).resolve().parent
    arch, suffix = _detect_arch_suffix()

    by_maxrreg: dict[Optional[int], list[Preset]] = {}
    for preset in presets:
        by_maxrreg.setdefault(preset.maxrregcount, []).append(preset)

    baseline_ms: Optional[float] = None
    default_group = by_maxrreg.pop(None, [])

    exe_default = _build_binary(chapter_dir, arch, suffix, maxrregcount=None)
    restore_path: Optional[Path] = None
    tempdir: Optional[tempfile.TemporaryDirectory] = None
    if by_maxrreg:
        tempdir = tempfile.TemporaryDirectory()
        restore_path = Path(tempdir.name) / exe_default.name
        shutil.copy2(exe_default, restore_path)

    try:
        for preset in default_group:
            avg_ms = _mean_runtime_ms(exe_default, preset, args.runs)
            if preset.name == "baseline":
                baseline_ms = avg_ms
            speedup = (baseline_ms / avg_ms) if baseline_ms else None
            if speedup is None:
                print(f"{preset.name:>10} (maxrreg={'default':>7}): {avg_ms:8.4f} ms")
            else:
                print(f"{preset.name:>10} (maxrreg={'default':>7}): {avg_ms:8.4f} ms  ({speedup:5.2f}x)")

        for maxrregcount, group in sorted(by_maxrreg.items(), key=lambda item: item[0] or 0):
            exe = _build_binary(chapter_dir, arch, suffix, maxrregcount=maxrregcount)
            for preset in group:
                avg_ms = _mean_runtime_ms(exe, preset, args.runs)
                speedup = (baseline_ms / avg_ms) if baseline_ms else None
                maxrreg_str = str(maxrregcount)
                if speedup is None:
                    print(f"{preset.name:>10} (maxrreg={maxrreg_str:>7}): {avg_ms:8.4f} ms")
                else:
                    print(f"{preset.name:>10} (maxrreg={maxrreg_str:>7}): {avg_ms:8.4f} ms  ({speedup:5.2f}x)")

            if restore_path is not None:
                shutil.copy2(restore_path, exe_default)
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
