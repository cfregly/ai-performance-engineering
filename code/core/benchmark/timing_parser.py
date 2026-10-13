"""Shared utilities for parsing kernel timing from CUDA executable stdout.

This module provides consistent timing extraction across:
- core/benchmark/cuda_binary_benchmark.py (CudaBinaryBenchmark class)
- core/harness/run_benchmarks.py (benchmark_cuda_executable function)

CUDA executables typically print timing like:
- "Host-staged GEMM (baseline): 2.3074 ms"
- "cuBLASLt batched GEMM (optimized): 0.008146 ms"
- "Kernel execution: 1234.56 us"
"""

from __future__ import annotations

import math
import re
from typing import Optional, List, Tuple

# Match the complete numeric token: truncating an exponent changes the timing
# by orders of magnitude (e.g. Ozaki prints TIME_MS in scientific notation).
_TIME_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TIME_TOKEN = rf"(?<![\w.+-])({_TIME_NUMBER})"

# Regex patterns to parse timing from stdout (supports ms, us, s units)
# Each tuple: (compiled_pattern, multiplier_to_ms)
_TIME_PATTERNS: List[Tuple[re.Pattern, float]] = [
    # TIME_MS format: "TIME_MS: 0.9026" or "TIME_MS:0.9026" (common in CUDA benchmarks)
    (re.compile(rf'TIME_MS:\s*({_TIME_NUMBER})(?![\w.+-])', re.IGNORECASE), 1.0),
    # Milliseconds: "2.3074 ms", "0.008 ms"
    (re.compile(rf'{_TIME_TOKEN}\s*ms\b', re.IGNORECASE), 1.0),
    # Microseconds: "1234 us", "500 μs"
    (re.compile(rf'{_TIME_TOKEN}\s*[uμ]s\b', re.IGNORECASE), 0.001),
    # Seconds: "1.5 s" (but not "ms" or "us" - negative lookbehind)
    (re.compile(rf'{_TIME_TOKEN}\s*s\b(?![uμm])', re.IGNORECASE), 1000.0),
]


def _parse_custom_time_ms(stdout: str, custom_regex: str) -> float:
    """Return the time captured by custom_regex, or raise ValueError.

    The default patterns pick the last time-like value in stdout, which can
    belong to a different kernel than the one a wrapper declares. A custom
    regex therefore never falls back to them.
    """
    try:
        pattern = re.compile(custom_regex)
    except re.error as exc:
        raise ValueError(f"time_regex {custom_regex!r} does not compile: {exc}") from exc
    if pattern.groups != 1:
        raise ValueError(
            f"time_regex {custom_regex!r} must have exactly one capture group for the "
            f"time value, but it has {pattern.groups}."
        )
    match = pattern.search(stdout)
    if match is None:
        raise ValueError(
            f"time_regex {custom_regex!r} did not match the binary output.\n"
            f"stdout:\n{stdout}"
        )
    captured = match.group(1)
    if captured is None:
        raise ValueError(
            f"time_regex {custom_regex!r} matched {match.group(0)!r}, but its capture "
            "group did not take part in the match."
        )
    try:
        time_ms = float(captured)
    except ValueError as exc:
        raise ValueError(
            f"time_regex {custom_regex!r} captured {captured!r}, which is not a number."
        ) from exc
    if not math.isfinite(time_ms):
        raise ValueError(
            f"time_regex {custom_regex!r} captured {captured!r}, which is not a finite number."
        )
    return time_ms


def parse_kernel_time_ms(stdout: str, custom_regex: Optional[str] = None) -> Optional[float]:
    """Parse kernel timing from CUDA executable stdout.

    Without custom_regex, supports multiple time units (ms, us/μs, s) and returns
    the value in milliseconds. Returns the LAST matching time value, as executables
    often print multiple lines and the final timing is usually the benchmark result.

    Args:
        stdout: The stdout text from a CUDA executable
        custom_regex: Optional regex with exactly one capture group for the time value,
                     in milliseconds. When it is set, it is the only pattern used and
                     its first match is returned. If it does not compile, does not have
                     exactly one capture group, does not match, or captures a value that
                     is not a finite number, ValueError is raised. It never falls back
                     to the default patterns.
                     Example: r"TIME_MS:\\s*([0-9.]+)"

    Returns:
        Parsed time in milliseconds. Without custom_regex, None if no timing is found.

    Raises:
        ValueError: custom_regex is set and does not yield a finite time.

    Examples:
        >>> parse_kernel_time_ms("Kernel: 2.3074 ms")
        2.3074
        >>> parse_kernel_time_ms("Time: 1500 us")
        1.5
        >>> parse_kernel_time_ms("Warmup: 100 ms\\nResult: 2.5 ms")
        2.5
        >>> parse_kernel_time_ms("TIME_MS: 1.234", r"TIME_MS:\\s*([0-9.]+)")
        1.234
    """
    if custom_regex is not None:
        return _parse_custom_time_ms(stdout, custom_regex)

    matches: List[Tuple[int, float]] = []
    
    for pattern, multiplier in _TIME_PATTERNS:
        for match in pattern.finditer(stdout):
            try:
                value = float(match.group(1))
                time_ms = value * multiplier
                if not math.isfinite(time_ms):
                    continue
                matches.append((match.start(), time_ms))
            except (ValueError, IndexError):
                continue
    
    if not matches:
        return None
    
    # Sort by position and return the last match
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]
