"""Strict B200 diagnostics for effective HBM bandwidth and BF16 GEMM throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .formulas import formula_report, gemm_logical_cost

_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_ROOT.parents[2]


class DiagnosticGateError(RuntimeError):
    """Raised when a required hardware or validity gate is not satisfied."""


def _git_output(*arguments: str) -> str:
    """Run one fixed Git query without exposing command output on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DiagnosticGateError("Git source provenance could not be resolved") from exc
    return result.stdout.strip()


def _validate_live_source_commit(source_commit: str) -> str:
    """Require the declared revision to be the exact clean live checkout."""
    _validate_source_commit(source_commit)
    head = _git_output("rev-parse", "--verify", "HEAD")
    if source_commit != head:
        raise DiagnosticGateError("source_commit must exactly match the live Git HEAD")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise DiagnosticGateError("Canonical measurement requires a clean Git source tree")
    return head


@contextmanager
def _bf16_fp32_accumulation_policy(torch: Any) -> Iterator[dict[str, bool]]:
    """Disable reduced-precision BF16 reductions and restore the prior policy."""
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    field = "allow_bf16_reduced_precision_reduction"
    if matmul is None or not hasattr(matmul, field):
        raise DiagnosticGateError("PyTorch does not expose the required BF16 reduction policy")
    previous = bool(getattr(matmul, field))
    try:
        __import__("core.harness.benchmark_harness")
        setattr(matmul, field, False)
        if bool(getattr(matmul, field)):
            raise DiagnosticGateError("PyTorch did not disable reduced-precision BF16 reductions")
        yield {field: False}
    finally:
        setattr(matmul, field, previous)


def _require_bf16_fp32_accumulation_policy(torch: Any) -> None:
    """Fail if a caller bypasses the required BF16 accumulation context."""
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    field = "allow_bf16_reduced_precision_reduction"
    if matmul is None or not hasattr(matmul, field) or bool(getattr(matmul, field)):
        raise DiagnosticGateError("Reduced-precision BF16 reductions must be disabled")


def _preflight_output(output: Path | None) -> None:
    """Reject an existing report before doing expensive measurement work."""
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable report: {output}")


def _publication_status(
    profile_artifacts: dict[str, list[str]],
    *,
    virtualized: bool,
) -> dict[str, Any]:
    """Return a fail-closed publication decision for unparsed profiler files."""
    blockers = [
        "profiler_artifacts_are_declared_but_not_parsed_or_bound_to_this_run",
        "operator_review_is_not_recorded_by_this_tool",
    ]
    if virtualized:
        blockers.append("virtualized_run_is_noncanonical")
    if not profile_artifacts["ncu"]:
        blockers.append("ncu_artifact_not_declared")
    if not profile_artifacts["nsys"]:
        blockers.append("nsys_artifact_not_declared")
    return {"publication_eligible": False, "publication_blockers": blockers}


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _package_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in _PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".cpp", ".cu", ".cuh", ".md", ".py", ".yaml"}
    )
    for path in paths:
        relative = path.relative_to(_PACKAGE_ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def _sample_summary(values: Sequence[float], unit: str) -> dict[str, Any]:
    if not values:
        raise ValueError("At least one timing sample is required")
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "unit": unit,
        "samples": samples,
        "count": len(samples),
        "min": min(samples),
        "p50": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "max": max(samples),
        "mean": mean,
        "stdev": stdev,
        "cv_pct": stdev / mean * 100.0 if mean > 0 else None,
    }


@contextmanager
def _nvtx_range(torch: Any, name: str) -> Iterator[None]:
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _event_samples_ms(
    torch: Any,
    *,
    device: Any,
    repeats: int,
    range_name: str,
    operation: Any,
) -> list[float]:
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(repeats)
    ]
    stream = torch.cuda.current_stream(device)
    with _nvtx_range(torch, range_name):
        for sample_index, (start, end) in enumerate(events):
            start.record(stream)
            operation(sample_index)
            end.record(stream)
    torch.cuda.synchronize(device)
    samples = [float(start.elapsed_time(end)) for start, end in events]
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError(f"Invalid CUDA event timing samples for {range_name}: {samples}")
    return samples


def _measure_hbm_copy(
    torch: Any,
    *,
    device: Any,
    copy_size_mib: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    from core.benchmark.metrics import compute_copy_bandwidth_metrics

    _positive_int("copy_size_mib", copy_size_mib)
    payload_bytes = copy_size_mib * 1024 * 1024
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    source = torch.randint(
        0,
        256,
        (payload_bytes,),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    destination = torch.zeros_like(source)

    for _ in range(warmups):
        destination.copy_(source)
    torch.cuda.synchronize(device)

    samples_ms = _event_samples_ms(
        torch,
        device=device,
        repeats=repeats,
        range_name="ai_b200_hbm_copy",
        operation=lambda _sample_index: destination.copy_(source),
    )
    output_matches = bool(torch.equal(destination, source))
    if not output_matches:
        raise RuntimeError("HBM copy output did not match the full source tensor")

    bandwidth_samples = [
        float(
            compute_copy_bandwidth_metrics(
                payload_bytes,
                1,
                sample_ms,
                traffic="read_plus_write",
            )["peak_bandwidth_gbs"]
        )
        for sample_ms in samples_ms
    ]
    result = {
        "name": "device_to_device_hbm_copy",
        "timing": _sample_summary(samples_ms, "ms"),
        "effective_bandwidth": _sample_summary(bandwidth_samples, "decimal_GB_per_s"),
        "warmups": warmups,
        "repeats": repeats,
        "payload_bytes": payload_bytes,
        "working_set_bytes": payload_bytes * 2,
        "input": {
            "distribution": "torch_randint_uniform_uint8_0_255",
            "seed": seed,
            "generation_timed": False,
        },
        "logical_requested_bytes_per_iteration": payload_bytes * 2,
        "byte_accounting": "one_source_read_plus_one_destination_write",
        "physical_hbm_counter_bytes": None,
        "physical_hbm_counter_source": "requires_ncu_dram_read_write_counters",
        "correctness": {
            "method": "full_tensor_torch_equal_after_measured_copy",
            "output_elements_checked": destination.numel(),
            "passed": output_matches,
        },
    }
    return result


def _rotation_count(*, weight_bytes: int, target_rotation_mib: int) -> int:
    _positive_int("weight_bytes", weight_bytes)
    _positive_int("target_rotation_mib", target_rotation_mib)
    target_bytes = target_rotation_mib * 1024 * 1024
    return max(2, math.ceil(target_bytes / weight_bytes))


def _measure_bf16_gemm(
    torch: Any,
    *,
    device: Any,
    m: int,
    n: int,
    k: int,
    label: str,
    warmups: int,
    repeats: int,
    rotation_mib: int,
    seed: int,
    l2_cache_bytes: int | None,
) -> dict[str, Any]:
    logical = gemm_logical_cost(m=m, n=n, k=k, beta_zero=True)
    weight_bytes = k * n * 2
    buffer_count = _rotation_count(
        weight_bytes=weight_bytes,
        target_rotation_mib=rotation_mib,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_scale = k**-0.25
    a = torch.randn(
        (m, k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(input_scale)
    base_b = torch.randn(
        (k, n),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(input_scale)
    weights = [base_b]
    weights.extend(base_b.clone() for _ in range(buffer_count - 1))
    output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

    for warmup_index in range(warmups):
        torch.mm(a, weights[warmup_index % buffer_count], out=output)
    torch.cuda.synchronize(device)

    samples_ms = _event_samples_ms(
        torch,
        device=device,
        repeats=repeats,
        range_name=f"ai_b200_gemm_{label}",
        operation=lambda sample_index: torch.mm(
            a,
            weights[sample_index % buffer_count],
            out=output,
        ),
    )
    last_weight = weights[(repeats - 1) % buffer_count]
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        reference = torch.mm(a.float(), last_weight.float())
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    actual = output.float()
    difference = (actual - reference).abs()
    rtol = 0.03
    atol = 0.03
    passed = bool(torch.allclose(actual, reference, rtol=rtol, atol=atol))
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    if not passed:
        raise RuntimeError(
            f"BF16 GEMM {label} failed full-output correctness against FP32 reference: "
            f"max_abs_error={max_abs_error}, mean_abs_error={mean_abs_error}"
        )

    flops = int(logical["flops"])
    tflops_samples = [flops / (sample_ms / 1e3) / 1e12 for sample_ms in samples_ms]
    rotation_bytes = buffer_count * weight_bytes
    result = {
        "name": f"bf16_gemm_{label}",
        "shape": {"m": m, "n": n, "k": k},
        "timing": _sample_summary(samples_ms, "ms"),
        "throughput": _sample_summary(tflops_samples, "TFLOP_per_s"),
        "warmups": warmups,
        "repeats": repeats,
        "logical_model": logical,
        "cache_policy": {
            "policy": "rotating_equal_value_weight_buffers",
            "weight_buffer_count": buffer_count,
            "weight_bytes_per_buffer": weight_bytes,
            "weight_rotation_bytes": rotation_bytes,
            "requested_rotation_mib": rotation_mib,
            "device_l2_cache_bytes": l2_cache_bytes,
            "rotation_exceeds_reported_l2": (
                rotation_bytes > l2_cache_bytes
                if l2_cache_bytes is not None and l2_cache_bytes > 0
                else None
            ),
            "physical_hbm_counter_bytes": None,
            "physical_hbm_counter_source": "requires_ncu_dram_read_write_counters",
        },
        "correctness": {
            "method": "full_output_against_fp32_matmul_of_the_measured_bf16_operands",
            "reference_float32_matmul_precision": "highest",
            "output_elements_checked": output.numel(),
            "rtol": rtol,
            "atol": atol,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "passed": passed,
        },
    }
    return result


def _driver_version() -> str:
    try:
        import pynvml
    except ImportError as exc:
        raise DiagnosticGateError("pynvml is required for driver provenance") from exc
    pynvml.nvmlInit()
    try:
        value = pynvml.nvmlSystemGetDriverVersion()
    finally:
        pynvml.nvmlShutdown()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _runtime_provenance(
    torch: Any,
    *,
    device: Any,
    source_commit: str,
    run_id: str,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "lab_tree_sha256": _package_sha256(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "precision_policy": {
            "allow_bf16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
        "nvidia_driver": _driver_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": {
            "logical_index": int(device.index or 0),
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "sm_count": int(properties.multi_processor_count),
            "l2_cache_bytes": (
                int(properties.l2_cache_size)
                if getattr(properties, "l2_cache_size", None) is not None
                else None
            ),
        },
    }


def _validate_source_commit(source_commit: str) -> None:
    if not _COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a 7 to 64 character lowercase hex Git commit")


def _environment_payload(validation: Any) -> dict[str, Any]:
    return {
        "validity_profile": "strict",
        "is_valid": bool(validation.is_valid),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "notices": list(validation.notices),
        "details": dict(validation.details),
    }


def run_gpu_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run the strict, clock-locked single-GPU diagnostic."""
    _positive_int("warmups", args.warmups)
    _positive_int("repeats", args.repeats)
    _positive_int("large_size", args.large_size)
    _positive_int("rotation_mib", args.rotation_mib)
    _validate_live_source_commit(args.source_commit)

    import torch

    _require_bf16_fp32_accumulation_policy(torch)
    if not torch.cuda.is_available():
        raise DiagnosticGateError("CUDA is required for the live B200 diagnostic")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise DiagnosticGateError(
            f"Requested CUDA device {args.device}, but {torch.cuda.device_count()} are visible"
        )
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    capability = (int(properties.major), int(properties.minor))

    from core.benchmark.metrics import hardware_specs_for_device
    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.harness.validity_checks import validate_environment
    from core.profiling.gpu_telemetry import query_gpu_telemetry

    profile = hardware_specs_for_device(properties.name, capability)
    reviewed = formula_report()["hardware_profile"]
    if float(profile.tensor_tflops) != float(reviewed["dense_bf16_tflops"]):
        raise DiagnosticGateError("Detected B200 profile does not match reviewed BF16 peak")
    if float(profile.hbm_bandwidth_gbps) != float(reviewed["hbm_bandwidth_decimal_gb_per_s"]):
        raise DiagnosticGateError("Detected B200 profile does not match reviewed HBM peak")
    if not torch.cuda.is_bf16_supported():
        raise DiagnosticGateError("The selected CUDA device does not support BF16")

    validation = validate_environment(
        device=device,
        allow_virtualization=False,
        allow_foreign_gpu_processes=False,
        expected_device_uuid=str(properties.uuid),
        expected_compute_capability=capability,
    )
    environment = _environment_payload(validation)
    if not validation.is_valid:
        raise DiagnosticGateError(
            "Strict environment validation failed: " + " | ".join(validation.errors)
        )

    provenance = _runtime_provenance(
        torch,
        device=device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    l2_cache_bytes = provenance["device"]["l2_cache_bytes"]
    measurements: dict[str, Any] = {}

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        ramp_gpu_clocks(device=args.device)
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        if args.phase in {"all", "copy"}:
            measurements["hbm_copy"] = _measure_hbm_copy(
                torch,
                device=device,
                copy_size_mib=args.copy_size_mib,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
            )
            torch.cuda.empty_cache()
        gemm_cases = {
            "gemm-m1": (1, 4096, 4096, "m1_k4096_n4096"),
            "gemm-m64": (64, 4096, 4096, "m64_k4096_n4096"),
            "gemm-large": (
                args.large_size,
                args.large_size,
                args.large_size,
                f"m{args.large_size}_k{args.large_size}_n{args.large_size}",
            ),
        }
        selected = (
            gemm_cases
            if args.phase == "all"
            else {args.phase: gemm_cases[args.phase]} if args.phase in gemm_cases else {}
        )
        for case_index, (case_name, (m, n, k, label)) in enumerate(selected.items()):
            measurements[case_name.replace("-", "_")] = _measure_bf16_gemm(
                torch,
                device=device,
                m=m,
                n=n,
                k=k,
                label=label,
                warmups=args.warmups,
                repeats=args.repeats,
                rotation_mib=args.rotation_mib,
                seed=args.seed + case_index,
                l2_cache_bytes=l2_cache_bytes,
            )
            torch.cuda.empty_cache()
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    required_clock_fields = (
        "applications_clock_sm_mhz",
        "applications_clock_memory_mhz",
    )
    for field in required_clock_fields:
        if telemetry_before.get(field) is None or telemetry_after.get(field) is None:
            raise DiagnosticGateError(f"Required application-clock telemetry is missing: {field}")

    virtualized = bool(environment["details"].get("virtualized"))
    profile_artifacts = {
        "ncu": [str(Path(path).resolve()) for path in args.ncu_artifact],
        "nsys": [str(Path(path).resolve()) for path in args.nsys_artifact],
    }
    for artifact_paths in profile_artifacts.values():
        invalid = [
            path
            for path in artifact_paths
            if not Path(path).is_file() or Path(path).stat().st_size <= 0
        ]
        if invalid:
            raise DiagnosticGateError("Declared profiler artifacts must be nonempty regular files")

    publication_status = _publication_status(profile_artifacts, virtualized=virtualized)

    return {
        "schema_version": 1,
        "kind": "b200_arithmetic_intensity_live_diagnostic",
        "status": "diagnostic_measurement_complete",
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        **publication_status,
        "provenance": provenance,
        "configuration": {
            "phase": args.phase,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "copy_size_mib": args.copy_size_mib,
            "large_size": args.large_size,
            "rotation_mib": args.rotation_mib,
            "seed": args.seed,
            "clock_lock": "core.harness.benchmark_harness.lock_gpu_clocks",
        },
        "environment_validation": environment,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        },
        "analytical_model": formula_report(),
        "measurements": measurements,
        "profiler_artifacts": profile_artifacts,
        "claim_boundary": {
            "effective_copy_bandwidth_is_physical_counter_bandwidth": False,
            "gemm_logical_bytes_are_physical_hbm_bytes": False,
            "ncu_dram_counters_required_for_physical_hbm_bytes": True,
            "strict_virtualized_run_is_canonical": False if virtualized else None,
        },
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify B200 arithmetic-intensity formulas or run strict live diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    formulas = subparsers.add_parser("formulas", help="Run hardware-independent formula checks.")
    formulas.add_argument("--output", type=Path)

    gpu = subparsers.add_parser("gpu", help="Run the strict, clock-locked B200 diagnostic.")
    gpu.add_argument("--run-id", required=True)
    gpu.add_argument("--source-commit", required=True)
    gpu.add_argument("--output", type=Path)
    gpu.add_argument("--device", type=int, default=0)
    gpu.add_argument(
        "--phase",
        choices=("all", "copy", "gemm-m1", "gemm-m64", "gemm-large"),
        default="all",
    )
    gpu.add_argument("--warmups", type=int, default=5)
    gpu.add_argument("--repeats", type=int, default=10)
    gpu.add_argument("--copy-size-mib", type=int, default=4096)
    gpu.add_argument("--large-size", type=int, default=8192)
    gpu.add_argument("--rotation-mib", type=int, default=512)
    gpu.add_argument("--seed", type=int, default=20261013)
    gpu.add_argument("--ncu-artifact", action="append", default=[])
    gpu.add_argument("--nsys-artifact", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _preflight_output(getattr(args, "output", None))
        if args.command == "formulas":
            report = formula_report()
        else:
            import torch

            with _bf16_fp32_accumulation_policy(torch):
                report = run_gpu_diagnostic(args)
    except (DiagnosticGateError, RuntimeError, ValueError, FileExistsError) as exc:
        failure = {
            "schema_version": 1,
            "kind": "b200_arithmetic_intensity_diagnostic_failure",
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "lab_tree_sha256": _package_sha256(),
        }
        failure_output = None if isinstance(exc, FileExistsError) else getattr(args, "output", None)
        try:
            _write_report(failure, failure_output)
        except FileExistsError:
            _write_report(failure, None)
        return 2
    try:
        _write_report(report, args.output)
    except FileExistsError as exc:
        _write_report(
            {
                "schema_version": 1,
                "kind": "b200_arithmetic_intensity_diagnostic_failure",
                "status": "rejected",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "lab_tree_sha256": _package_sha256(),
            },
            None,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
