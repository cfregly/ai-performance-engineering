"""Shared helpers for the Ozaki scheme lab scripts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.benchmark.cuda_binary_benchmark import ARCH_SUFFIX, detect_supported_arch

MetricValue = int | float | str

_METRIC_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "variant": re.compile(r"VARIANT:\s*(\S+)"),
    "time_ms": re.compile(r"TIME_MS:\s*([0-9.eE+-]+)"),
    "tflops": re.compile(r"TFLOPS:\s*([0-9.eE+-]+)"),
    "retained_bits": re.compile(r"RETAINED_BITS:\s*(-?\d+)"),
    "emulation_used": re.compile(r"EMULATION_USED:\s*(\d+)"),
    "relative_l2_error": re.compile(r"RELATIVE_L2_ERROR:\s*([0-9.eE+-]+)"),
    "normalized_max_abs_error": re.compile(r"NORMALIZED_MAX_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "max_abs_error": re.compile(r"MAX_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "mean_abs_error": re.compile(r"MEAN_ABS_ERROR:\s*([0-9.eE+-]+)"),
    "checksum": re.compile(r"RESULT_CHECKSUM:\s*([0-9.eE+-]+)"),
    "emulation_strategy": re.compile(r"EMULATION_STRATEGY:\s*(\S+)"),
}


def detect_binary_suffix() -> str:
    return ARCH_SUFFIX[detect_supported_arch()]


def parse_metrics(stdout: str) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(stdout)
        if not match:
            continue
        value = match.group(1)
        if key in {"variant", "emulation_strategy"}:
            metrics[key] = value
        elif key in {"retained_bits", "emulation_used"}:
            metrics[key] = int(value)
        else:
            metrics[key] = float(value)
    return metrics


def run_binary(binary: str, args: list[str]) -> dict[str, MetricValue]:
    completed = subprocess.run(
        [str(LAB_DIR / binary), *args],
        cwd=LAB_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{binary} failed with rc={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    metrics = parse_metrics(completed.stdout)
    metrics["stdout"] = completed.stdout
    return metrics


def build_lab() -> None:
    subprocess.run(["make", "all"], cwd=LAB_DIR, check=True)


def common_args_from_values(
    *,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
    seed: int,
    input_scale: float,
    emulation_strategy: str,
) -> list[str]:
    return [
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--input-scale",
        str(input_scale),
        "--emulation-strategy",
        emulation_strategy,
    ]


def speedup_vs_baseline(baseline_ms: float, candidate_ms: float) -> float:
    return baseline_ms / candidate_ms if candidate_ms > 0.0 else 0.0


def format_result_row(label: str, metrics: dict[str, MetricValue], baseline_ms: float) -> str:
    time_ms = float(metrics.get("time_ms", 0.0))
    return (
        f"| {label} | {time_ms:.3f} | {float(metrics.get('tflops', 0.0)):.3f} | "
        f"{speedup_vs_baseline(baseline_ms, time_ms):.2f}x | {metrics.get('retained_bits', '-')} | "
        f"{metrics.get('emulation_used', 0)} | {float(metrics.get('max_abs_error', 0.0)):.3e} | "
        f"{float(metrics.get('mean_abs_error', 0.0)):.3e} |"
    )


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_csv(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def summarize_reproducibility(records: list[dict[str, MetricValue]]) -> dict[str, bool | int]:
    checksums = {record.get("checksum") for record in records}
    retained_bits = {record.get("retained_bits") for record in records}
    emulation_used = {record.get("emulation_used") for record in records}
    return {
        "run_count": len(records),
        "checksum_stable": len(checksums) == 1,
        "retained_bits_stable": len(retained_bits) == 1,
        "emulation_used_stable": len(emulation_used) == 1,
    }
