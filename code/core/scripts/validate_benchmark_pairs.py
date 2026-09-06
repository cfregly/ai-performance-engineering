#!/usr/bin/env python3
"""Validate that baseline/optimized benchmark pairs have matching signatures.

This script discovers all baseline_*.py and optimized_*.py pairs and verifies
that their input signatures match, ensuring fair performance comparisons.
For VerificationPayloadMixin-backed benchmarks, it will execute setup() and a
single benchmark_fn() as needed to populate the verification payload before
extracting a validated InputSignature.
Payloads that require a distributed worker callback are explicitly skipped by
this standalone signature tool; validate those pairs through ``aisp bench run``.

Usage:
    # Validate all pairs
    python -m core.scripts.validate_benchmark_pairs
    
    # Validate specific chapter
    python -m core.scripts.validate_benchmark_pairs --chapter ch07
    
    # Generate JSON report
    python -m core.scripts.validate_benchmark_pairs --report artifacts/pair_validation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.benchmark.verification import (
    InputSignature,
    SignatureEquivalenceSpec,
    coerce_input_signature,
    get_signature_equivalence_spec,
    signature_workload_dict,
)
from core.discovery import discover_all_chapters, discover_benchmarks

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SignatureMismatch:
    """Details of a signature mismatch between baseline and optimized."""
    key: str
    baseline_value: Any
    optimized_value: Any


@dataclass
class PairValidationResult:
    """Result of validating a single benchmark pair."""
    chapter: str
    example_name: str
    baseline_path: str
    optimized_path: str
    
    # Status
    valid: bool = False
    error: Optional[str] = None
    skipped: bool = False
    
    # Signature comparison
    baseline_has_signature: bool = False
    optimized_has_signature: bool = False
    signatures_match: bool = False
    mismatches: List[SignatureMismatch] = field(default_factory=list)
    
    # Extra keys
    baseline_only_keys: Set[str] = field(default_factory=set)
    optimized_only_keys: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "example_name": self.example_name,
            "baseline_path": self.baseline_path,
            "optimized_path": self.optimized_path,
            "valid": self.valid,
            "error": self.error,
            "skipped": self.skipped,
            "baseline_has_signature": self.baseline_has_signature,
            "optimized_has_signature": self.optimized_has_signature,
            "signatures_match": self.signatures_match,
            "mismatches": [
                {
                    "key": mismatch.key,
                    "baseline_value": mismatch.baseline_value,
                    "optimized_value": mismatch.optimized_value,
                }
                for mismatch in self.mismatches
            ],
            "baseline_only_keys": sorted(self.baseline_only_keys),
            "optimized_only_keys": sorted(self.optimized_only_keys),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PairValidationResult":
        return cls(
            chapter=payload["chapter"],
            example_name=payload["example_name"],
            baseline_path=payload["baseline_path"],
            optimized_path=payload["optimized_path"],
            valid=bool(payload.get("valid", False)),
            error=payload.get("error"),
            skipped=bool(payload.get("skipped", False)),
            baseline_has_signature=bool(payload.get("baseline_has_signature", False)),
            optimized_has_signature=bool(payload.get("optimized_has_signature", False)),
            signatures_match=bool(payload.get("signatures_match", False)),
            mismatches=[
                SignatureMismatch(
                    key=item["key"],
                    baseline_value=item.get("baseline_value"),
                    optimized_value=item.get("optimized_value"),
                )
                for item in payload.get("mismatches", [])
            ],
            baseline_only_keys=set(payload.get("baseline_only_keys", [])),
            optimized_only_keys=set(payload.get("optimized_only_keys", [])),
        )


@dataclass
class ValidationReport:
    """Complete pair validation report."""
    timestamp: str
    total_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    missing_signature_pairs: int = 0
    signature_mismatch_pairs: int = 0
    skipped_pairs: int = 0
    error_pairs: int = 0
    results: List[PairValidationResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_pairs": self.total_pairs,
                "valid_pairs": self.valid_pairs,
                "invalid_pairs": self.invalid_pairs,
                "missing_signature_pairs": self.missing_signature_pairs,
                "signature_mismatch_pairs": self.signature_mismatch_pairs,
                "skipped_pairs": self.skipped_pairs,
                "error_pairs": self.error_pairs,
            },
            "results": [
                {
                    **r.to_dict(),
                    "mismatches": [
                        {"key": m.key, "baseline": m.baseline_value, "optimized": m.optimized_value}
                        for m in r.mismatches
                    ],
                }
                for r in self.results
            ],
        }


# =============================================================================
# Benchmark Loading
# =============================================================================

def load_benchmark_class(file_path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """Dynamically load and instantiate a benchmark from a file."""
    try:
        spec = importlib.util.spec_from_file_location("benchmark_module", file_path)
        if spec is None or spec.loader is None:
            return None, "Could not create import spec"
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark_module"] = module
        spec.loader.exec_module(module)
        
        # Look for get_benchmark factory function
        if hasattr(module, "get_benchmark"):
            return module.get_benchmark(), None
        
        # Look for Benchmark class
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, "benchmark_fn") and 
                name != "BaseBenchmark"):
                return obj(), None
        
        return None, "No get_benchmark() factory or benchmark class found"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        if "benchmark_module" in sys.modules:
            del sys.modules["benchmark_module"]


def _run_signature_capture_path(benchmark: Any) -> None:
    """Populate a VerificationPayload-backed signature by executing the benchmark once."""
    if not hasattr(benchmark, "setup") or not callable(getattr(benchmark, "setup")):
        raise RuntimeError("Benchmark is missing setup(); cannot capture verification payload for signature")
    if not hasattr(benchmark, "benchmark_fn") or not callable(getattr(benchmark, "benchmark_fn")):
        raise RuntimeError("Benchmark is missing benchmark_fn(); cannot capture verification payload for signature")
    if not hasattr(benchmark, "capture_verification_payload") or not callable(
        getattr(benchmark, "capture_verification_payload")
    ):
        raise RuntimeError(
            "Benchmark is missing capture_verification_payload(); cannot capture verification payload for signature"
        )

    benchmark.setup()
    # Some benchmarks can capture verification payload immediately after setup.
    try:
        benchmark.capture_verification_payload()
        return
    except Exception:
        pass

    # A torchrun wrapper's benchmark_fn does not execute its distributed worker
    # or consume the full-rank result callback. Do not substitute parent-side
    # setup state for that missing signature.
    get_config = getattr(benchmark, "get_config", None)
    if callable(get_config):
        config = get_config()
        launch_via = getattr(config, "launch_via", None)
        if getattr(launch_via, "value", launch_via) == "torchrun":
            raise RuntimeError(
                "SKIPPED: signature capture requires a torchrun worker result; "
                "validate this pair with aisp bench run"
            )

    benchmark.benchmark_fn()
    benchmark.capture_verification_payload()


def get_input_signature_safe(benchmark: Any) -> Tuple[Optional[InputSignature], Optional[str]]:
    """Safely get a validated InputSignature from a benchmark instance.

    For payload-backed benchmarks (VerificationPayloadMixin), this executes:
    setup() + benchmark_fn() + capture_verification_payload() as needed.
    """
    if not hasattr(benchmark, "get_input_signature") or not callable(getattr(benchmark, "get_input_signature")):
        return None, "Method not implemented"

    attempted_execution_path = False
    try:
        sig_raw = benchmark.get_input_signature()
        if sig_raw is None:
            return None, "Method returned None"
        return coerce_input_signature(sig_raw), None
    except (RuntimeError, AttributeError, ValueError):
        # Most commonly: payload-backed benchmarks before capture_verification_payload().
        try:
            attempted_execution_path = True
            _run_signature_capture_path(benchmark)
            sig_raw = benchmark.get_input_signature()
            if sig_raw is None:
                return None, "Method returned None after capture_verification_payload()"
            return coerce_input_signature(sig_raw), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if attempted_execution_path and hasattr(benchmark, "teardown") and callable(getattr(benchmark, "teardown")):
            try:
                benchmark.teardown()
            except Exception:
                # Best-effort cleanup; surface original validation error.
                pass


def _is_skip_message(message: Optional[str]) -> bool:
    return bool(message and "SKIPPED" in message.upper())


def _validator_env() -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Best-effort kill of a process group started with start_new_session=True."""
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)
            return
    except (OSError, ProcessLookupError, AttributeError):
        pass

    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2)
        except Exception:
            pass


def _validate_pair_in_subprocess(
    chapter: str,
    example_name: str,
    baseline_path: Path,
    optimized_path: Path,
    *,
    timeout_seconds: Optional[int],
    root_dir: Path,
) -> PairValidationResult:
    """Validate a pair inside an isolated subprocess so timeouts can kill descendants."""
    timeout_limit = float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None
    fd, output_path_str = tempfile.mkstemp(prefix="pair_validation_", suffix=".json")
    os.close(fd)
    output_path = Path(output_path_str)
    command = [
        sys.executable,
        "-m",
        "core.scripts.validate_benchmark_pairs",
        "--_single-pair-output",
        str(output_path),
        "--_single-pair-chapter",
        chapter,
        "--_single-pair-example",
        example_name,
        "--_single-pair-baseline",
        str(baseline_path),
        "--_single-pair-optimized",
        str(optimized_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(root_dir if (root_dir / "core").exists() else REPO_ROOT),
        env=_validator_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    stdout = ""
    stderr = ""
    try:
        stdout, stderr = process.communicate(timeout=timeout_limit)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except (subprocess.TimeoutExpired, ValueError):
            stdout = stdout or ""
            stderr = stderr or ""
        return PairValidationResult(
            chapter=chapter,
            example_name=example_name,
            baseline_path=str(baseline_path),
            optimized_path=str(optimized_path),
            error=f"Validation timeout after {int(timeout_limit)} seconds",
        )
    finally:
        if process.poll() is None:
            _terminate_process_group(process)

    try:
        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"pair validator exited with code {process.returncode}"
            return PairValidationResult(
                chapter=chapter,
                example_name=example_name,
                baseline_path=str(baseline_path),
                optimized_path=str(optimized_path),
                error=f"Isolated pair validation failed: {detail}",
            )
        if not output_path.exists():
            detail = stderr.strip() or stdout.strip() or "missing result payload"
            return PairValidationResult(
                chapter=chapter,
                example_name=example_name,
                baseline_path=str(baseline_path),
                optimized_path=str(optimized_path),
                error=f"Isolated pair validation failed: {detail}",
            )
        return PairValidationResult.from_dict(json.loads(output_path.read_text(encoding="utf-8")))
    except Exception as exc:
        detail = stderr.strip() or stdout.strip()
        suffix = f" ({detail})" if detail else ""
        return PairValidationResult(
            chapter=chapter,
            example_name=example_name,
            baseline_path=str(baseline_path),
            optimized_path=str(optimized_path),
            error=f"Failed to decode isolated pair validation result: {type(exc).__name__}: {exc}{suffix}",
        )
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass


# =============================================================================
# Pair Discovery
# =============================================================================

def discover_benchmark_pairs(
    root_dir: Path,
    chapter: Optional[str] = None,
    allow_unvalidated: Optional[bool] = None,
) -> Dict[str, Dict[str, Path]]:
    """Discover baseline/optimized benchmark pairs.
    
    Returns:
        Dict mapping (chapter, example_name) to {baseline: path, optimized: path}
    """
    pairs: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)
    if allow_unvalidated is None:
        allow_unvalidated = not (root_dir / "core").exists()

    if chapter:
        chapter_dir = root_dir / chapter
        if not chapter_dir.exists():
            raise FileNotFoundError(f"Chapter directory not found: {chapter_dir}")
        chapter_dirs = [chapter_dir]
    else:
        chapter_dirs = discover_all_chapters(root_dir)

    for chapter_dir in chapter_dirs:
        chapter_name = str(chapter_dir.relative_to(root_dir))
        if allow_unvalidated:
            baseline_files = sorted(chapter_dir.glob("baseline_*.py"))
            for baseline_path in baseline_files:
                example_name = baseline_path.stem.replace("baseline_", "", 1)
                optimized_paths: List[Path] = []
                for opt_path in sorted(chapter_dir.glob(f"optimized_{example_name}_*.py")):
                    optimized_paths.append(opt_path)
                opt_exact = chapter_dir / f"optimized_{example_name}.py"
                if opt_exact.exists():
                    optimized_paths.append(opt_exact)
                if not optimized_paths:
                    continue
                for optimized_path in optimized_paths:
                    key = optimized_path.stem.replace("optimized_", "", 1)
                    pairs[(chapter_name, key)]["baseline"] = baseline_path
                    pairs[(chapter_name, key)]["optimized"] = optimized_path
        else:
            discovered = discover_benchmarks(chapter_dir, validate=False, warn_missing=False)
            for baseline_path, optimized_paths, _example_name in discovered:
                for optimized_path in optimized_paths:
                    # Align with harness pairing: baseline_<name>.py matches optimized_<name>.py and
                    # optimized_<name>_*.py variants; the pair key is derived from the optimized stem.
                    key = optimized_path.stem.replace("optimized_", "", 1)
                    pairs[(chapter_name, key)]["baseline"] = baseline_path
                    pairs[(chapter_name, key)]["optimized"] = optimized_path

    return {f"{ch}:{name}": paths for (ch, name), paths in pairs.items()}


# =============================================================================
# Signature Comparison
# =============================================================================

def compare_signatures(
    baseline_sig: Dict[str, Any],
    optimized_sig: Dict[str, Any],
) -> Tuple[bool, List[SignatureMismatch], Set[str], Set[str]]:
    """Compare two input signatures.
    
    Returns:
        Tuple of (match, mismatches, baseline_only_keys, optimized_only_keys)
    """
    mismatches: List[SignatureMismatch] = []
    
    baseline_keys = set(baseline_sig.keys())
    optimized_keys = set(optimized_sig.keys())
    
    baseline_only = baseline_keys - optimized_keys
    optimized_only = optimized_keys - baseline_keys
    common_keys = baseline_keys & optimized_keys
    
    # Compare common keys
    for key in common_keys:
        baseline_val = baseline_sig[key]
        optimized_val = optimized_sig[key]
        
        if baseline_val != optimized_val:
            mismatches.append(SignatureMismatch(
                key=key,
                baseline_value=baseline_val,
                optimized_value=optimized_val,
            ))
    
    match = len(mismatches) == 0 and len(baseline_only) == 0 and len(optimized_only) == 0
    return match, mismatches, baseline_only, optimized_only


def _format_equivalence_spec(spec: Optional[SignatureEquivalenceSpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {"group": spec.group, "ignore_fields": list(spec.ignore_fields)}


# =============================================================================
# Validation
# =============================================================================

def validate_pair(
    chapter: str,
    example_name: str,
    baseline_path: Path,
    optimized_path: Path,
) -> PairValidationResult:
    """Validate a single baseline/optimized pair."""
    result = PairValidationResult(
        chapter=chapter,
        example_name=example_name,
        baseline_path=str(baseline_path),
        optimized_path=str(optimized_path),
    )
    
    # Load benchmarks
    baseline_benchmark, baseline_load_error = load_benchmark_class(baseline_path)
    if baseline_benchmark is None:
        if _is_skip_message(baseline_load_error):
            result.skipped = True
            result.error = baseline_load_error
            return result
        result.error = f"Failed to load baseline benchmark: {baseline_load_error}"
        return result
    
    optimized_benchmark, optimized_load_error = load_benchmark_class(optimized_path)
    if optimized_benchmark is None:
        if _is_skip_message(optimized_load_error):
            result.skipped = True
            result.error = optimized_load_error
            return result
        result.error = f"Failed to load optimized benchmark: {optimized_load_error}"
        return result
    
    # Get signatures
    baseline_sig, baseline_err = get_input_signature_safe(baseline_benchmark)
    optimized_sig, optimized_err = get_input_signature_safe(optimized_benchmark)
    
    result.baseline_has_signature = baseline_sig is not None
    result.optimized_has_signature = optimized_sig is not None

    if _is_skip_message(baseline_err) or _is_skip_message(optimized_err):
        result.skipped = True
        details: list[str] = []
        if baseline_err:
            details.append(f"baseline_error={baseline_err!r}")
        if optimized_err:
            details.append(f"optimized_error={optimized_err!r}")
        result.error = "SKIPPED: " + ", ".join(details)
        return result
    
    if baseline_sig is None and optimized_sig is None:
        if baseline_err and optimized_err and "SKIPPED" in baseline_err and "SKIPPED" in optimized_err:
            result.skipped = True
            result.error = (
                "SKIPPED: both benchmarks reported SKIPPED. "
                f"baseline_error={baseline_err!r}, optimized_error={optimized_err!r}"
            )
            return result
        result.error = (
            "Failed to extract input signatures for both benchmarks. "
            f"baseline_error={baseline_err!r}, optimized_error={optimized_err!r}"
        )
        return result
    
    if baseline_sig is None:
        result.error = f"Baseline missing signature: {baseline_err}"
        return result
    
    if optimized_sig is None:
        result.error = f"Optimized missing signature: {optimized_err}"
        return result
    
    # Compare signature-equivalence specs (must match for comparable pairs)
    try:
        baseline_equiv = get_signature_equivalence_spec(baseline_benchmark)
        optimized_equiv = get_signature_equivalence_spec(optimized_benchmark)
    except Exception as exc:
        result.error = f"Failed to read signature equivalence metadata: {type(exc).__name__}: {exc}"
        return result

    baseline_workload = signature_workload_dict(baseline_sig, equivalence=baseline_equiv)
    optimized_workload = signature_workload_dict(optimized_sig, equivalence=optimized_equiv)

    # Compare workload dicts (with any allowed ignore_fields removed)
    match, mismatches, baseline_only, optimized_only = compare_signatures(
        baseline_workload, optimized_workload
    )

    equiv_match = baseline_equiv == optimized_equiv
    if not equiv_match:
        mismatches.append(
            SignatureMismatch(
                key="signature_equivalence",
                baseline_value=_format_equivalence_spec(baseline_equiv),
                optimized_value=_format_equivalence_spec(optimized_equiv),
            )
        )
    
    result.signatures_match = match and equiv_match
    result.mismatches = mismatches
    result.baseline_only_keys = baseline_only
    result.optimized_only_keys = optimized_only
    result.valid = result.signatures_match
    
    return result


def validate_all_pairs(
    root_dir: Path,
    chapter: Optional[str] = None,
    pair_timeout_seconds: Optional[int] = 120,
) -> ValidationReport:
    """Validate all benchmark pairs."""
    report = ValidationReport(timestamp=datetime.now().isoformat())
    
    # Discover pairs
    pairs = discover_benchmark_pairs(root_dir, chapter)
    report.total_pairs = len(pairs)
    
    print(f"Found {len(pairs)} benchmark pairs to validate")
    print()
    
    for pair_name, paths in sorted(pairs.items()):
        chapter_name, example_name = pair_name.split(":", 1)
        
        # Check if we have both baseline and optimized
        if "baseline" not in paths:
            result = PairValidationResult(
                chapter=chapter_name,
                example_name=example_name,
                baseline_path="",
                optimized_path=str(paths.get("optimized", "")),
                error="Missing baseline file",
            )
            report.results.append(result)
            report.error_pairs += 1
            continue
        
        if "optimized" not in paths:
            result = PairValidationResult(
                chapter=chapter_name,
                example_name=example_name,
                baseline_path=str(paths.get("baseline", "")),
                optimized_path="",
                error="Missing optimized file",
            )
            report.results.append(result)
            report.error_pairs += 1
            continue
        
        # Validate the pair
        result = _validate_pair_in_subprocess(
            chapter_name,
            example_name,
            paths["baseline"],
            paths["optimized"],
            timeout_seconds=pair_timeout_seconds,
            root_dir=root_dir,
        )
        report.results.append(result)
        
        if result.valid:
            report.valid_pairs += 1
            print(f"  ✓ {pair_name}")
        elif result.skipped:
            report.skipped_pairs += 1
            print(f"  ○ {pair_name} - {result.error}")
        elif result.error:
            if "missing signature" in result.error.lower() or "not implemented" in result.error.lower():
                report.missing_signature_pairs += 1
                print(f"  ○ {pair_name} - {result.error}")
            else:
                report.error_pairs += 1
                print(f"  ✗ {pair_name} - {result.error}")
        else:
            report.signature_mismatch_pairs += 1
            report.invalid_pairs += 1
            print(f"  ✗ {pair_name} - signature mismatch:")
            for m in result.mismatches:
                print(f"      {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")
            if result.baseline_only_keys:
                print(f"      baseline-only keys: {result.baseline_only_keys}")
            if result.optimized_only_keys:
                print(f"      optimized-only keys: {result.optimized_only_keys}")
    
    return report


# =============================================================================
# Reporting
# =============================================================================

def print_summary(report: ValidationReport) -> None:
    """Print summary of validation results."""
    print()
    print("=" * 60)
    print("PAIR VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs:              {report.total_pairs}")
    print(f"Valid pairs:              {report.valid_pairs}")
    print(f"Signature mismatches:     {report.signature_mismatch_pairs}")
    print(f"Missing signatures:       {report.missing_signature_pairs}")
    print(f"Skipped pairs:            {report.skipped_pairs}")
    print(f"Errors:                   {report.error_pairs}")
    
    if report.signature_mismatch_pairs > 0:
        print()
        print("PAIRS WITH SIGNATURE MISMATCHES:")
        for r in report.results:
            if r.mismatches:
                print(f"  {r.chapter}:{r.example_name}")
                for m in r.mismatches:
                    print(f"    {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")


def generate_fix_suggestions(report: ValidationReport) -> List[str]:
    """Generate fix suggestions for mismatched pairs."""
    suggestions = []
    
    for r in report.results:
        if r.mismatches:
            suggestions.append(f"# {r.chapter}:{r.example_name}")
            suggestions.append(f"# Baseline: {r.baseline_path}")
            suggestions.append(f"# Optimized: {r.optimized_path}")
            suggestions.append("# Mismatches:")
            for m in r.mismatches:
                suggestions.append(f"#   {m.key}: baseline={m.baseline_value}, optimized={m.optimized_value}")
            suggestions.append("")
    
    return suggestions


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate baseline/optimized benchmark pairs have matching signatures"
    )
    parser.add_argument(
        "--chapter", "-c",
        help="Validate only a specific chapter (e.g., ch07, labs/moe_cuda)"
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Path to write JSON validation report"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=".",
        help="Root directory (default: current directory)"
    )
    parser.add_argument(
        "--pair-timeout-seconds",
        type=int,
        default=120,
        help="Maximum time per pair validation before failing that pair (default: 120)",
    )
    parser.add_argument("--_single-pair-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_single-pair-chapter", help=argparse.SUPPRESS)
    parser.add_argument("--_single-pair-example", help=argparse.SUPPRESS)
    parser.add_argument("--_single-pair-baseline", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_single-pair-optimized", type=Path, help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    if args._single_pair_output:
        if not (
            args._single_pair_chapter
            and args._single_pair_example
            and args._single_pair_baseline
            and args._single_pair_optimized
        ):
            raise SystemExit("single-pair validator invocation is missing required arguments")
        result = validate_pair(
            args._single_pair_chapter,
            args._single_pair_example,
            args._single_pair_baseline.resolve(),
            args._single_pair_optimized.resolve(),
        )
        args._single_pair_output.parent.mkdir(parents=True, exist_ok=True)
        args._single_pair_output.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        return 0

    root_dir = args.root.resolve()
    
    # Run validation
    report = validate_all_pairs(
        root_dir,
        args.chapter,
        pair_timeout_seconds=args.pair_timeout_seconds,
    )
    
    # Print summary
    print_summary(report)
    
    # Write report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written to {args.report}")
    
    # Return non-zero if there are mismatches
    return 1 if report.signature_mismatch_pairs > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
