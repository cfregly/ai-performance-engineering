"""Bounded sustained HBM and dense BF16 ceiling characterization for one B200."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ceiling_protocol import (
    BandwidthCandidate,
    GemmCandidate,
    bandwidth_candidates,
    bootstrap_median_ci,
    confirmation_invocations_per_sample,
    gemm_candidates,
    parse_layout_csv,
    parse_positive_csv,
)
from .formulas import formula_report, gemm_logical_cost
from .verification_tool import (
    DiagnosticGateError,
    _bf16_fp32_accumulation_policy,
    _environment_payload,
    _package_sha256,
    _preflight_output,
    _require_bf16_fp32_accumulation_policy,
    _require_lab_runtime,
    _runtime_provenance,
    _sample_summary,
    _validate_live_source_commit,
    _validate_source_commit,
    _write_report,
)

_UINT32_MASK = 0xFFFF_FFFF


def _require_positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _progress(event: str, **details: Any) -> None:
    """Emit one compact machine-readable milestone without polluting report stdout."""
    print(
        json.dumps({"event": event, **details}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _blas_backend_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value).rsplit(".", maxsplit=1)[-1].lower()


@contextmanager
def _preferred_blas_backend(torch: Any, requested: str):
    """Apply and restore an explicit PyTorch BLAS preference with readback."""
    if requested not in {"default", "cublas", "cublaslt"}:
        raise ValueError(f"unsupported BLAS backend preference: {requested}")
    preference = torch.backends.cuda.preferred_blas_library
    before = preference()
    setter_result = None
    if requested != "default":
        try:
            setter_result = preference(requested)
        except Exception as exc:
            preference(before)
            raise DiagnosticGateError(
                f"PyTorch rejected BLAS backend preference {requested!r}: {exc}"
            ) from exc
    observed = preference()
    observed_name = _blas_backend_name(observed)
    if requested != "default" and observed_name != requested:
        preference(before)
        raise DiagnosticGateError(
            f"Requested BLAS backend {requested!r}, read back {observed_name!r}"
        )
    metadata = {
        "requested": requested,
        "setter_invoked": requested != "default",
        "before": _blas_backend_name(before),
        "setter_return": (None if setter_result is None else _blas_backend_name(setter_result)),
        "observed_during_measurement": observed_name,
        "cublaslt_log_level": os.environ.get("CUBLASLT_LOG_LEVEL"),
        "preference_is_dispatch_proof": False,
        "dispatch_proof": "requires_cublas_api_trace_or_cublaslt_log",
        "restored": None,
    }
    try:
        yield metadata
    finally:
        preference(before)
        metadata["restored"] = _blas_backend_name(preference())


def _nvtx_range(torch: Any, name: str):
    class _Range:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            torch.cuda.nvtx.range_pop()
            return False

    return _Range()


def _measure_aggregate_operation(
    torch: Any,
    *,
    device: Any,
    operation: Callable[[int], None],
    range_name: str,
    warmups: int,
    repeats: int,
    invocations_per_sample: int,
) -> dict[str, Any]:
    """Time aggregate samples and preserve both aggregate and normalized durations."""
    for name, value in (
        ("warmups", warmups),
        ("repeats", repeats),
        ("invocations_per_sample", invocations_per_sample),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    invocation_id = 0
    for _ in range(warmups):
        for _ in range(invocations_per_sample):
            operation(invocation_id)
            invocation_id += 1
    torch.cuda.synchronize(device)

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    stream = torch.cuda.current_stream(device)
    with _nvtx_range(torch, range_name):
        for start, end in events:
            start.record(stream)
            for _ in range(invocations_per_sample):
                operation(invocation_id)
                invocation_id += 1
            end.record(stream)
    torch.cuda.synchronize(device)
    aggregate_ms = [float(start.elapsed_time(end)) for start, end in events]
    if any(not math.isfinite(value) or value <= 0 for value in aggregate_ms):
        raise RuntimeError(f"Invalid CUDA timing samples for {range_name}: {aggregate_ms}")
    normalized_ms = [value / invocations_per_sample for value in aggregate_ms]
    return {
        "aggregate_timing": _sample_summary(aggregate_ms, "ms_per_aggregate_sample"),
        "normalized_timing": _sample_summary(normalized_ms, "ms_per_operation"),
        "invocations_per_sample": invocations_per_sample,
        "warmup_invocations": warmups * invocations_per_sample,
        "measured_invocations": repeats * invocations_per_sample,
        "measured_gpu_active_ms": sum(aggregate_ms),
        "last_invocation_id": invocation_id - 1,
    }


def _rate_samples(requested_bytes: int, timing_ms: Sequence[float]) -> list[float]:
    return [requested_bytes / (sample / 1e3) / 1e9 for sample in timing_ms]


def _bandwidth_byte_accounting(
    candidate: BandwidthCandidate,
    *,
    payload_bytes: int,
    checksum_output_bytes: int = 0,
) -> dict[str, Any]:
    if candidate.operation == "copy":
        requested_read = payload_bytes
        requested_write = payload_bytes
        primary = requested_read + requested_write
        scope = "effective_requested_read_plus_write"
    elif candidate.operation == "read":
        requested_read = payload_bytes
        requested_write = checksum_output_bytes
        primary = requested_read
        scope = "effective_requested_read_only"
    elif candidate.operation == "write":
        requested_read = 0
        requested_write = payload_bytes
        primary = requested_write
        scope = "effective_requested_write_only"
    else:
        raise ValueError(f"unsupported bandwidth operation: {candidate.operation}")
    return {
        "requested_read_bytes_per_operation": requested_read,
        "requested_write_bytes_per_operation": requested_write,
        "requested_total_bytes_per_operation": requested_read + requested_write,
        "primary_rate_bytes_per_operation": primary,
        "primary_rate_scope": scope,
        "physical_hbm_counter_bytes": None,
        "physical_hbm_counter_source": "requires_ncu_dram_read_write_counters",
    }


def _exact_read_reference(
    torch: Any,
    source: Any,
    *,
    chunk_bytes: int,
    batch_bytes: int = 64 * 1024 * 1024,
) -> Any:
    """Compute exact full-input uint32 chunk sums through independent PyTorch ops."""
    if source.numel() % chunk_bytes != 0:
        raise ValueError("source byte count must be divisible by chunk_bytes")
    words_per_chunk = chunk_bytes // 4
    chunk_count = source.numel() // chunk_bytes
    chunks_per_batch = max(1, batch_bytes // chunk_bytes)
    words = source.view(torch.int32)
    expected = torch.empty(chunk_count, dtype=torch.int64, device=source.device)
    for start_chunk in range(0, chunk_count, chunks_per_batch):
        end_chunk = min(chunk_count, start_chunk + chunks_per_batch)
        start_word = start_chunk * words_per_chunk
        end_word = end_chunk * words_per_chunk
        unsigned_words = words[start_word:end_word].to(torch.int64)
        unsigned_words.bitwise_and_(_UINT32_MASK)
        expected[start_chunk:end_chunk] = unsigned_words.view(
            end_chunk - start_chunk,
            words_per_chunk,
        ).sum(dim=1, dtype=torch.int64)
    return expected


def _candidate_blocks(candidate: BandwidthCandidate, sm_count: int) -> int | None:
    if candidate.blocks_per_sm is None:
        return None
    return candidate.blocks_per_sm * sm_count


def _run_bandwidth_case(
    torch: Any,
    *,
    extension: Any,
    device: Any,
    source: Any,
    destination: Any,
    candidate: BandwidthCandidate,
    sm_count: int,
    warmups: int,
    repeats: int,
    invocations_per_sample: int,
    seed: int,
    exact_read_reference: Any | None = None,
) -> dict[str, Any]:
    payload_bytes = int(source.numel())
    if destination.numel() != payload_bytes:
        raise ValueError("source and destination byte counts must match")
    blocks = _candidate_blocks(candidate, sm_count)
    sink = None

    if candidate.strategy == "torch_copy":

        def operation(_invocation_id: int) -> None:
            destination.copy_(source)

    elif candidate.strategy == "native_scalar":
        assert candidate.threads is not None and blocks is not None

        def operation(_invocation_id: int) -> None:
            extension.copy_scalar(
                source,
                destination,
                candidate.threads,
                blocks,
            )

    elif candidate.strategy == "native_uint4":
        assert candidate.threads is not None and blocks is not None

        def operation(_invocation_id: int) -> None:
            extension.copy_uint4(
                source,
                destination,
                candidate.threads,
                blocks,
            )

    elif candidate.strategy == "native_chunk_sum":
        assert candidate.threads is not None and candidate.chunk_kib is not None
        chunk_bytes = candidate.chunk_kib * 1024
        if payload_bytes % chunk_bytes != 0:
            raise ValueError("payload must be divisible by the checksum chunk size")
        sink = torch.empty(
            payload_bytes // chunk_bytes,
            dtype=torch.int64,
            device=device,
        )

        def operation(_invocation_id: int) -> None:
            extension.read_chunk_sums(
                source,
                sink,
                candidate.threads,
                chunk_bytes,
            )

    elif candidate.strategy == "native_high_entropy_uint4":
        assert candidate.threads is not None and blocks is not None

        def operation(invocation_id: int) -> None:
            extension.write_uint4(
                destination,
                candidate.threads,
                blocks,
                (seed + invocation_id) & _UINT32_MASK,
            )

    else:
        raise ValueError(f"unsupported bandwidth strategy: {candidate.strategy}")

    range_name = f"ai_b200_peak_{candidate.candidate_id}"
    timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=operation,
        range_name=range_name,
        warmups=warmups,
        repeats=repeats,
        invocations_per_sample=invocations_per_sample,
    )

    if candidate.operation == "copy":
        passed = bool(torch.equal(destination, source))
        correctness = {
            "method": "full_tensor_torch_equal",
            "output_elements_checked": payload_bytes,
            "passed": passed,
        }
    elif candidate.operation == "read":
        if sink is None or exact_read_reference is None:
            raise RuntimeError("read cases require a full exact reference")
        passed = bool(torch.equal(sink, exact_read_reference))
        correctness = {
            "method": "full_per_chunk_output_against_independent_pytorch_full_input_reference",
            "input_bytes_covered": payload_bytes,
            "output_chunks_checked": int(sink.numel()),
            "chunk_bytes": candidate.chunk_kib * 1024,
            "passed": passed,
        }
    else:
        assert candidate.threads is not None and blocks is not None
        final_seed = (seed + timing["last_invocation_id"]) & _UINT32_MASK
        mismatches = int(
            extension.validate_write_uint4(
                destination,
                candidate.threads,
                blocks,
                final_seed,
            ).item()
        )
        passed = mismatches == 0
        correctness = {
            "method": "full_tensor_deterministic_word_validation",
            "output_words_checked": payload_bytes // 4,
            "mismatch_count": mismatches,
            "final_seed": final_seed,
            "passed": passed,
        }
    if not passed:
        raise RuntimeError(f"bandwidth correctness failed for {candidate.candidate_id}")

    accounting = _bandwidth_byte_accounting(
        candidate,
        payload_bytes=payload_bytes,
        checksum_output_bytes=0 if sink is None else int(sink.numel()) * 8,
    )
    normalized_ms = timing["normalized_timing"]["samples"]
    rates = _rate_samples(accounting["primary_rate_bytes_per_operation"], normalized_ms)
    return {
        **candidate.to_dict(),
        "nvtx_range": range_name,
        "payload_bytes": payload_bytes,
        "blocks": blocks,
        "timing": timing,
        "byte_accounting": accounting,
        "effective_requested_bandwidth": _sample_summary(rates, "decimal_GB_per_s"),
        "correctness": correctness,
    }


def _screen_bandwidth(
    torch: Any,
    *,
    extension: Any,
    device: Any,
    sm_count: int,
    candidates: Sequence[BandwidthCandidate],
    warmups: int,
    repeats: int,
    seed: int,
    maximum_cv_pct: float,
) -> dict[str, Any]:
    maximum_payload = max(candidate.payload_mib for candidate in candidates) * 1024 * 1024
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    source_storage = torch.randint(
        0,
        256,
        (maximum_payload,),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    destination_storage = torch.empty_like(source_storage)
    reference_cache: dict[tuple[int, int], Any] = {}
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        payload_bytes = candidate.payload_mib * 1024 * 1024
        source = source_storage[:payload_bytes]
        destination = destination_storage[:payload_bytes]
        exact_reference = None
        if candidate.operation == "read":
            assert candidate.chunk_kib is not None
            key = (payload_bytes, candidate.chunk_kib)
            if key not in reference_cache:
                reference_cache[key] = _exact_read_reference(
                    torch,
                    source,
                    chunk_bytes=candidate.chunk_kib * 1024,
                )
            exact_reference = reference_cache[key]
        result = _run_bandwidth_case(
            torch,
            extension=extension,
            device=device,
            source=source,
            destination=destination,
            candidate=candidate,
            sm_count=sm_count,
            warmups=warmups,
            repeats=repeats,
            invocations_per_sample=1,
            seed=seed + index * 1009,
            exact_read_reference=exact_reference,
        )
        reasons = []
        if result["effective_requested_bandwidth"]["cv_pct"] > maximum_cv_pct:
            reasons.append(f"screen_cv_pct_exceeds_{maximum_cv_pct:g}")
        result["ranking_eligible"] = not reasons
        result["screen_rejection_reasons"] = reasons
        results.append(result)
    del reference_cache, destination_storage, source_storage
    torch.cuda.empty_cache()
    return {
        "input_distribution": "seeded_uniform_random_uint8_0_255",
        "seed": seed,
        "generation_timed": False,
        "warmups": warmups,
        "repeats": repeats,
        "maximum_cv_pct": maximum_cv_pct,
        "candidates": results,
    }


def _select_bandwidth_finalists(screen: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    limits = {"copy": 2, "read": 1, "write": 1}
    for operation, limit in limits.items():
        eligible = [
            result
            for result in screen["candidates"]
            if result["operation"] == operation and result["ranking_eligible"]
        ]
        eligible.sort(
            key=lambda result: result["effective_requested_bandwidth"]["p50"],
            reverse=True,
        )
        selected.extend(eligible[:limit])
    return selected


def _candidate_from_result(result: dict[str, Any]) -> BandwidthCandidate:
    return BandwidthCandidate(
        operation=result["operation"],
        strategy=result["strategy"],
        payload_mib=int(result["payload_mib"]),
        threads=result["threads"],
        blocks_per_sm=result["blocks_per_sm"],
        chunk_kib=result["chunk_kib"],
    )


def _confirm_bandwidth_candidate(
    torch: Any,
    *,
    extension: Any,
    device: Any,
    sm_count: int,
    screen_result: dict[str, Any],
    windows: int,
    warmups: int,
    repeats: int,
    minimum_window_gpu_ms: float,
    precondition_ms: float,
    seed: int,
    maximum_cv_pct: float,
    ramp_gpu_clocks: Callable[..., None],
    query_gpu_telemetry: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    candidate = _candidate_from_result(screen_result)
    initial_invocations = confirmation_invocations_per_sample(
        screen_result["timing"]["normalized_timing"]["p50"],
        repeats=repeats,
        minimum_window_gpu_ms=minimum_window_gpu_ms,
    )
    window_results = []
    for window_index in range(windows):
        window_seed = seed + window_index * 1_000_003
        payload_bytes = candidate.payload_mib * 1024 * 1024
        generator = torch.Generator(device=device)
        generator.manual_seed(window_seed)
        source = torch.randint(
            0,
            256,
            (payload_bytes,),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        destination = torch.empty_like(source)
        exact_reference = None
        if candidate.operation == "read":
            assert candidate.chunk_kib is not None
            exact_reference = _exact_read_reference(
                torch,
                source,
                chunk_bytes=candidate.chunk_kib * 1024,
            )
        ramp_gpu_clocks(device=int(device.index or 0), duration_ms=precondition_ms)
        telemetry_before = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        invocations = initial_invocations
        result = _run_bandwidth_case(
            torch,
            extension=extension,
            device=device,
            source=source,
            destination=destination,
            candidate=candidate,
            sm_count=sm_count,
            warmups=warmups,
            repeats=repeats,
            invocations_per_sample=invocations,
            seed=window_seed,
            exact_read_reference=exact_reference,
        )
        repair = None
        active_ms = result["timing"]["measured_gpu_active_ms"]
        if active_ms < minimum_window_gpu_ms:
            repaired_invocations = max(
                invocations + 1,
                math.ceil(invocations * minimum_window_gpu_ms * 1.05 / active_ms),
            )
            repair = {
                "reason": "initial_aggregate_duration_below_sustained_window_gate",
                "initial_invocations_per_sample": invocations,
                "initial_measured_gpu_active_ms": active_ms,
                "repaired_invocations_per_sample": repaired_invocations,
            }
            result = _run_bandwidth_case(
                torch,
                extension=extension,
                device=device,
                source=source,
                destination=destination,
                candidate=candidate,
                sm_count=sm_count,
                warmups=warmups,
                repeats=repeats,
                invocations_per_sample=repaired_invocations,
                seed=window_seed + 7_919,
                exact_read_reference=exact_reference,
            )
        telemetry_after = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        result["window_index"] = window_index
        result["window_seed"] = window_seed
        result["precondition_ms"] = precondition_ms
        result["bounded_duration_repair"] = repair
        result["telemetry_before"] = telemetry_before
        result["telemetry_after"] = telemetry_after
        result["sustained_window_gate_passed"] = (
            result["timing"]["measured_gpu_active_ms"] >= minimum_window_gpu_ms
        )
        window_results.append(result)
        del destination, source, exact_reference
        torch.cuda.empty_cache()

    rate_samples = [
        sample
        for window in window_results
        for sample in window["effective_requested_bandwidth"]["samples"]
    ]
    timing_samples = [
        sample
        for window in window_results
        for sample in window["timing"]["normalized_timing"]["samples"]
    ]
    window_medians = [window["effective_requested_bandwidth"]["p50"] for window in window_results]
    rate_summary = _sample_summary(rate_samples, "decimal_GB_per_s")
    all_correct = all(window["correctness"]["passed"] for window in window_results)
    all_sustained = all(window["sustained_window_gate_passed"] for window in window_results)
    rejection_reasons = []
    if not all_correct:
        rejection_reasons.append("confirmation_correctness_failed")
    if not all_sustained:
        rejection_reasons.append("confirmation_window_below_sustained_duration_gate")
    if rate_summary["cv_pct"] > maximum_cv_pct:
        rejection_reasons.append(f"confirmation_cv_pct_exceeds_{maximum_cv_pct:g}")
    return {
        **candidate.to_dict(),
        "nvtx_range": window_results[0]["nvtx_range"],
        "byte_accounting": window_results[0]["byte_accounting"],
        "screen_p50_decimal_GB_per_s": screen_result["effective_requested_bandwidth"]["p50"],
        "window_count": windows,
        "minimum_gpu_active_ms_per_window": minimum_window_gpu_ms,
        "windows": window_results,
        "normalized_timing": _sample_summary(timing_samples, "ms_per_operation"),
        "effective_requested_bandwidth": rate_summary,
        "window_median_bandwidth": _sample_summary(
            window_medians,
            "decimal_GB_per_s",
        ),
        "window_median_bootstrap_95_ci": bootstrap_median_ci(
            window_medians,
            seed=seed + 31,
        ),
        "stable": not rejection_reasons,
        "confirmation_rejection_reasons": rejection_reasons,
    }


def _make_logical_gemm_operands(torch: Any, *, device: Any, size: int, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = size**-0.25
    a = torch.randn(
        (size, size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(scale)
    b = torch.randn(
        (size, size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(scale)
    return a, b


def _physical_layout(logical: Any, *, transposed: bool):
    if not transposed:
        return logical
    storage = logical.transpose(0, 1).contiguous()
    return storage.transpose(0, 1)


def _fp32_reference(torch: Any, a: Any, b: Any):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        a_fp32 = a.float()
        b_fp32 = b.float()
        reference = torch.mm(a_fp32, b_fp32)
        del a_fp32, b_fp32
        return reference
    finally:
        torch.set_float32_matmul_precision(previous)


def _gemm_correctness(torch: Any, output: Any, reference: Any) -> dict[str, Any]:
    actual = output.float()
    difference = (actual - reference).abs()
    rtol = 0.03
    atol = 0.03
    passed = bool(torch.allclose(actual, reference, rtol=rtol, atol=atol))
    result = {
        "method": "full_output_against_fp32_matmul_of_exact_bf16_operands",
        "output_elements_checked": int(output.numel()),
        "reference_float32_matmul_precision": "highest",
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "passed": passed,
    }
    del actual, difference
    return result


def _run_gemm_case(
    torch: Any,
    *,
    device: Any,
    candidate: GemmCandidate,
    a_logical: Any,
    b_logical: Any,
    reference: Any,
    warmups: int,
    repeats: int,
    invocations_per_sample: int,
) -> dict[str, Any]:
    a = _physical_layout(a_logical, transposed=candidate.layout[0] == "t")
    b = _physical_layout(b_logical, transposed=candidate.layout[1] == "t")
    output = torch.empty(
        (candidate.size, candidate.size),
        dtype=torch.bfloat16,
        device=device,
    )
    range_name = f"ai_b200_peak_{candidate.candidate_id}"
    timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=lambda _invocation_id: torch.mm(a, b, out=output),
        range_name=range_name,
        warmups=warmups,
        repeats=repeats,
        invocations_per_sample=invocations_per_sample,
    )
    correctness = _gemm_correctness(torch, output, reference)
    if not correctness["passed"]:
        raise RuntimeError(f"GEMM correctness failed for {candidate.candidate_id}")
    logical = gemm_logical_cost(
        m=candidate.size,
        n=candidate.size,
        k=candidate.size,
        beta_zero=True,
    )
    tflops = [
        logical["flops"] / (sample_ms / 1e3) / 1e12
        for sample_ms in timing["normalized_timing"]["samples"]
    ]
    return {
        **candidate.to_dict(),
        "nvtx_range": range_name,
        "api": "torch.mm_out_cublas_or_cublaslt_dispatch",
        "dtype": "bf16",
        "beta_zero": True,
        "timing": timing,
        "throughput": _sample_summary(tflops, "TFLOP_per_s"),
        "logical_model": logical,
        "correctness": correctness,
    }


def _screen_gemm(
    torch: Any,
    *,
    device: Any,
    candidates: Sequence[GemmCandidate],
    warmups: int,
    repeats: int,
    seed: int,
    maximum_cv_pct: float,
) -> dict[str, Any]:
    results = []
    candidates_by_size: dict[int, list[GemmCandidate]] = {}
    for candidate in candidates:
        candidates_by_size.setdefault(candidate.size, []).append(candidate)
    for size_index, (size, size_candidates) in enumerate(candidates_by_size.items()):
        a, b = _make_logical_gemm_operands(
            torch,
            device=device,
            size=size,
            seed=seed + size_index * 1009,
        )
        reference = _fp32_reference(torch, a, b)
        for candidate in size_candidates:
            result = _run_gemm_case(
                torch,
                device=device,
                candidate=candidate,
                a_logical=a,
                b_logical=b,
                reference=reference,
                warmups=warmups,
                repeats=repeats,
                invocations_per_sample=1,
            )
            reasons = []
            if result["throughput"]["cv_pct"] > maximum_cv_pct:
                reasons.append(f"screen_cv_pct_exceeds_{maximum_cv_pct:g}")
            result["ranking_eligible"] = not reasons
            result["screen_rejection_reasons"] = reasons
            results.append(result)
        del reference, a, b
        torch.cuda.empty_cache()
    return {
        "input_distribution": "seeded_scaled_normal_bf16",
        "seed": seed,
        "generation_timed": False,
        "warmups": warmups,
        "repeats": repeats,
        "maximum_cv_pct": maximum_cv_pct,
        "candidates": results,
    }


def _select_gemm_finalists(screen: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = [result for result in screen["candidates"] if result["ranking_eligible"]]
    eligible.sort(key=lambda result: result["throughput"]["p50"], reverse=True)
    return eligible[:2]


def _gemm_candidate_from_result(result: dict[str, Any]) -> GemmCandidate:
    return GemmCandidate(size=int(result["size"]), layout=str(result["layout"]))


def _confirm_gemm_candidate(
    torch: Any,
    *,
    device: Any,
    screen_result: dict[str, Any],
    windows: int,
    warmups: int,
    repeats: int,
    minimum_window_gpu_ms: float,
    precondition_ms: float,
    seed: int,
    maximum_cv_pct: float,
    ramp_gpu_clocks: Callable[..., None],
    query_gpu_telemetry: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    candidate = _gemm_candidate_from_result(screen_result)
    initial_invocations = confirmation_invocations_per_sample(
        screen_result["timing"]["normalized_timing"]["p50"],
        repeats=repeats,
        minimum_window_gpu_ms=minimum_window_gpu_ms,
    )
    window_results = []
    for window_index in range(windows):
        window_seed = seed + window_index * 1_000_003
        a, b = _make_logical_gemm_operands(
            torch,
            device=device,
            size=candidate.size,
            seed=window_seed,
        )
        reference = _fp32_reference(torch, a, b)
        ramp_gpu_clocks(device=int(device.index or 0), duration_ms=precondition_ms)
        telemetry_before = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        invocations = initial_invocations
        result = _run_gemm_case(
            torch,
            device=device,
            candidate=candidate,
            a_logical=a,
            b_logical=b,
            reference=reference,
            warmups=warmups,
            repeats=repeats,
            invocations_per_sample=invocations,
        )
        repair = None
        active_ms = result["timing"]["measured_gpu_active_ms"]
        if active_ms < minimum_window_gpu_ms:
            repaired_invocations = max(
                invocations + 1,
                math.ceil(invocations * minimum_window_gpu_ms * 1.05 / active_ms),
            )
            repair = {
                "reason": "initial_aggregate_duration_below_sustained_window_gate",
                "initial_invocations_per_sample": invocations,
                "initial_measured_gpu_active_ms": active_ms,
                "repaired_invocations_per_sample": repaired_invocations,
            }
            result = _run_gemm_case(
                torch,
                device=device,
                candidate=candidate,
                a_logical=a,
                b_logical=b,
                reference=reference,
                warmups=warmups,
                repeats=repeats,
                invocations_per_sample=repaired_invocations,
            )
        telemetry_after = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        result["window_index"] = window_index
        result["window_seed"] = window_seed
        result["precondition_ms"] = precondition_ms
        result["bounded_duration_repair"] = repair
        result["telemetry_before"] = telemetry_before
        result["telemetry_after"] = telemetry_after
        result["sustained_window_gate_passed"] = (
            result["timing"]["measured_gpu_active_ms"] >= minimum_window_gpu_ms
        )
        window_results.append(result)
        del reference, a, b
        torch.cuda.empty_cache()

    throughput_samples = [
        sample for window in window_results for sample in window["throughput"]["samples"]
    ]
    timing_samples = [
        sample
        for window in window_results
        for sample in window["timing"]["normalized_timing"]["samples"]
    ]
    window_medians = [window["throughput"]["p50"] for window in window_results]
    throughput_summary = _sample_summary(throughput_samples, "TFLOP_per_s")
    all_correct = all(window["correctness"]["passed"] for window in window_results)
    all_sustained = all(window["sustained_window_gate_passed"] for window in window_results)
    rejection_reasons = []
    if not all_correct:
        rejection_reasons.append("confirmation_correctness_failed")
    if not all_sustained:
        rejection_reasons.append("confirmation_window_below_sustained_duration_gate")
    if throughput_summary["cv_pct"] > maximum_cv_pct:
        rejection_reasons.append(f"confirmation_cv_pct_exceeds_{maximum_cv_pct:g}")
    return {
        **candidate.to_dict(),
        "nvtx_range": window_results[0]["nvtx_range"],
        "logical_model": window_results[0]["logical_model"],
        "screen_p50_TFLOP_per_s": screen_result["throughput"]["p50"],
        "window_count": windows,
        "minimum_gpu_active_ms_per_window": minimum_window_gpu_ms,
        "windows": window_results,
        "normalized_timing": _sample_summary(timing_samples, "ms_per_operation"),
        "throughput": throughput_summary,
        "window_median_throughput": _sample_summary(
            window_medians,
            "TFLOP_per_s",
        ),
        "window_median_bootstrap_95_ci": bootstrap_median_ci(
            window_medians,
            seed=seed + 47,
        ),
        "stable": not rejection_reasons,
        "confirmation_rejection_reasons": rejection_reasons,
    }


def _strict_runtime(torch: Any, *, device_index: int, source_commit: str, run_id: str):
    _validate_live_source_commit(source_commit)
    lab_runtime_contract = _require_lab_runtime(torch)
    _require_bf16_fp32_accumulation_policy(torch)
    if not torch.cuda.is_available():
        raise DiagnosticGateError("CUDA is required for B200 ceiling characterization")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise DiagnosticGateError(
            f"Requested CUDA device {device_index}, but {torch.cuda.device_count()} are visible"
        )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    capability = (int(properties.major), int(properties.minor))
    if capability != (10, 0) or "B200" not in properties.name:
        raise DiagnosticGateError(
            f"Expected NVIDIA B200 compute capability 10.0, got {properties.name} {capability}"
        )
    if not torch.cuda.is_bf16_supported():
        raise DiagnosticGateError("The selected CUDA device does not support BF16")

    from core.benchmark.metrics import hardware_specs_for_device
    from core.harness.validity_checks import validate_environment

    reviewed = formula_report()["hardware_profile"]
    profile = hardware_specs_for_device(properties.name, capability)
    if float(profile.tensor_tflops) != float(reviewed["dense_bf16_tflops"]):
        raise DiagnosticGateError("Detected B200 profile does not match reviewed BF16 peak")
    if float(profile.hbm_bandwidth_gbps) != float(reviewed["hbm_bandwidth_decimal_gb_per_s"]):
        raise DiagnosticGateError("Detected B200 profile does not match reviewed HBM peak")
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
        source_commit=source_commit,
        run_id=run_id,
        lab_runtime_contract=lab_runtime_contract,
    )
    return device, properties, environment, provenance, reviewed


def run_characterization(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed screen and bounded sustained confirmation protocol."""
    _validate_source_commit(args.source_commit)
    for name in (
        "screen_warmups",
        "screen_repeats",
        "confirmation_windows",
        "confirmation_warmups",
        "confirmation_repeats",
    ):
        _require_positive(name, getattr(args, name))
    for name in (
        "minimum_window_gpu_ms",
        "precondition_ms",
        "screen_maximum_cv_pct",
        "confirmation_maximum_cv_pct",
    ):
        _require_positive(name, getattr(args, name))
    payloads = parse_positive_csv(args.payload_mib, name="payload_mib")
    sizes = parse_positive_csv(args.gemm_sizes, name="gemm_sizes")
    layouts = parse_layout_csv(args.layouts)

    import torch

    device, properties, environment, provenance, reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry

    from .ceiling_extension import load_ceiling_extension

    extension = load_ceiling_extension()
    bandwidth_search = bandwidth_candidates(payloads)
    gemm_search = gemm_candidates(sizes, layouts)
    telemetry: dict[str, Any] = {}
    with (
        _preferred_blas_backend(torch, args.blas_backend) as blas_backend,
        lock_gpu_clocks(device=args.device) as lock_peaks,
    ):
        ramp_gpu_clocks(device=args.device, duration_ms=args.precondition_ms)
        telemetry["before_screen"] = query_gpu_telemetry(args.device, force_refresh=True)
        _progress(
            "gates_passed_and_screen_started",
            gpu_name=properties.name,
            gpu_uuid=str(properties.uuid),
            lab_tree_sha256=_package_sha256(),
            bandwidth_candidate_count=len(bandwidth_search),
            gemm_candidate_count=len(gemm_search),
        )
        bandwidth_screen = _screen_bandwidth(
            torch,
            extension=extension,
            device=device,
            sm_count=int(properties.multi_processor_count),
            candidates=bandwidth_search,
            warmups=args.screen_warmups,
            repeats=args.screen_repeats,
            seed=args.seed,
            maximum_cv_pct=args.screen_maximum_cv_pct,
        )
        telemetry["after_bandwidth_screen"] = query_gpu_telemetry(
            args.device,
            force_refresh=True,
        )
        bandwidth_finalists = _select_bandwidth_finalists(bandwidth_screen)
        if len(bandwidth_finalists) < 4:
            raise RuntimeError(
                "Bandwidth screen did not produce two copy, one read, and one write finalist"
            )
        _progress(
            "bandwidth_screen_complete",
            finalists=[
                {
                    "candidate_id": result["candidate_id"],
                    "operation": result["operation"],
                    "p50_decimal_GB_per_s": result["effective_requested_bandwidth"]["p50"],
                }
                for result in bandwidth_finalists
            ],
        )
        bandwidth_confirmations = [
            _confirm_bandwidth_candidate(
                torch,
                extension=extension,
                device=device,
                sm_count=int(properties.multi_processor_count),
                screen_result=result,
                windows=args.confirmation_windows,
                warmups=args.confirmation_warmups,
                repeats=args.confirmation_repeats,
                minimum_window_gpu_ms=args.minimum_window_gpu_ms,
                precondition_ms=args.precondition_ms,
                seed=args.seed + 10_000_019 + index * 100_003,
                maximum_cv_pct=args.confirmation_maximum_cv_pct,
                ramp_gpu_clocks=ramp_gpu_clocks,
                query_gpu_telemetry=query_gpu_telemetry,
            )
            for index, result in enumerate(bandwidth_finalists)
        ]
        telemetry["after_bandwidth_confirmation"] = query_gpu_telemetry(
            args.device,
            force_refresh=True,
        )
        _progress(
            "bandwidth_confirmation_complete",
            confirmations=[
                {
                    "candidate_id": result["candidate_id"],
                    "stable": result["stable"],
                    "p50_decimal_GB_per_s": result["effective_requested_bandwidth"]["p50"],
                    "cv_pct": result["effective_requested_bandwidth"]["cv_pct"],
                }
                for result in bandwidth_confirmations
            ],
        )

        gemm_screen = _screen_gemm(
            torch,
            device=device,
            candidates=gemm_search,
            warmups=args.screen_warmups,
            repeats=args.screen_repeats,
            seed=args.seed + 20_000_033,
            maximum_cv_pct=args.screen_maximum_cv_pct,
        )
        telemetry["after_gemm_screen"] = query_gpu_telemetry(args.device, force_refresh=True)
        gemm_finalists = _select_gemm_finalists(gemm_screen)
        if len(gemm_finalists) < 2:
            raise RuntimeError("GEMM screen did not produce two stable finalists")
        _progress(
            "gemm_screen_complete",
            finalists=[
                {
                    "candidate_id": result["candidate_id"],
                    "p50_TFLOP_per_s": result["throughput"]["p50"],
                }
                for result in gemm_finalists
            ],
        )
        gemm_confirmations = [
            _confirm_gemm_candidate(
                torch,
                device=device,
                screen_result=result,
                windows=args.confirmation_windows,
                warmups=args.confirmation_warmups,
                repeats=args.confirmation_repeats,
                minimum_window_gpu_ms=args.minimum_window_gpu_ms,
                precondition_ms=args.precondition_ms,
                seed=args.seed + 30_000_047 + index * 100_003,
                maximum_cv_pct=args.confirmation_maximum_cv_pct,
                ramp_gpu_clocks=ramp_gpu_clocks,
                query_gpu_telemetry=query_gpu_telemetry,
            )
            for index, result in enumerate(gemm_finalists)
        ]
        telemetry["after_gemm_confirmation"] = query_gpu_telemetry(
            args.device,
            force_refresh=True,
        )
        _progress(
            "gemm_confirmation_complete",
            confirmations=[
                {
                    "candidate_id": result["candidate_id"],
                    "stable": result["stable"],
                    "p50_TFLOP_per_s": result["throughput"]["p50"],
                    "cv_pct": result["throughput"]["cv_pct"],
                }
                for result in gemm_confirmations
            ],
        )
        telemetry["after_all"] = query_gpu_telemetry(args.device, force_refresh=True)

    copy_winners = [
        result
        for result in bandwidth_confirmations
        if result["operation"] == "copy" and result["stable"]
    ]
    if not copy_winners:
        raise RuntimeError("No sustained copy finalist passed the confirmation gates")
    hbm_ceiling = max(
        copy_winners,
        key=lambda result: result["effective_requested_bandwidth"]["p50"],
    )
    stable_gemm = [result for result in gemm_confirmations if result["stable"]]
    if not stable_gemm:
        raise RuntimeError("No sustained BF16 GEMM finalist passed the confirmation gates")
    bf16_ceiling = max(stable_gemm, key=lambda result: result["throughput"]["p50"])
    hbm_gbps = hbm_ceiling["effective_requested_bandwidth"]["p50"]
    bf16_tflops = bf16_ceiling["throughput"]["p50"]
    measured_ridge = bf16_tflops * 1000.0 / hbm_gbps
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_sustained_ceiling_characterization",
        "status": "diagnostic_characterization_complete",
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "provenance": provenance,
        "configuration": {
            "payload_mib": payloads,
            "gemm_sizes": sizes,
            "layouts": layouts,
            "screen_warmups": args.screen_warmups,
            "screen_repeats": args.screen_repeats,
            "confirmation_windows": args.confirmation_windows,
            "confirmation_warmups": args.confirmation_warmups,
            "confirmation_repeats": args.confirmation_repeats,
            "minimum_gpu_active_ms_per_confirmation_window": args.minimum_window_gpu_ms,
            "precondition_ms": args.precondition_ms,
            "screen_maximum_cv_pct": args.screen_maximum_cv_pct,
            "confirmation_maximum_cv_pct": args.confirmation_maximum_cv_pct,
            "seed": args.seed,
            "clock_lock": "core.harness.benchmark_harness.lock_gpu_clocks",
            "blas_backend_preference": blas_backend,
        },
        "environment_validation": environment,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry": telemetry,
        },
        "bandwidth": {
            "screen": bandwidth_screen,
            "finalist_ids": [result["candidate_id"] for result in bandwidth_finalists],
            "confirmations": bandwidth_confirmations,
            "attained_copy_ceiling": hbm_ceiling,
        },
        "dense_bf16": {
            "screen": gemm_screen,
            "finalist_ids": [result["candidate_id"] for result in gemm_finalists],
            "confirmations": gemm_confirmations,
            "attained_ceiling": bf16_ceiling,
        },
        "roofline_inputs": {
            "nominal": {
                "dense_bf16_TFLOP_per_s": reviewed["dense_bf16_tflops"],
                "hbm_decimal_GB_per_s": reviewed["hbm_bandwidth_decimal_gb_per_s"],
                "ridge_FLOP_per_byte": reviewed["ridge_flop_per_byte"],
                "source": reviewed["source"],
                "measured_on_active_gpu": False,
            },
            "measured_protocol_ceiling": {
                "dense_bf16_TFLOP_per_s": bf16_tflops,
                "hbm_effective_requested_read_plus_write_decimal_GB_per_s": hbm_gbps,
                "ridge_FLOP_per_effective_requested_byte": measured_ridge,
                "bf16_fraction_of_nominal": bf16_tflops / reviewed["dense_bf16_tflops"],
                "hbm_fraction_of_nominal": hbm_gbps / reviewed["hbm_bandwidth_decimal_gb_per_s"],
                "hbm_label": "effective requested read plus write bandwidth",
                "physical_counter_validation": "pending_exact_range_ncu_capture",
            },
        },
        "profiler_artifacts": {"ncu": [], "nsys": []},
        "claim_boundary": {
            "highest_sustained_under_this_bounded_protocol_is_vendor_peak": False,
            "effective_requested_bandwidth_is_raw_ncu_dram_peak": False,
            "read_or_write_supporting_diagnostic_defines_dense_gemm_roof": False,
            "ncu_required_to_validate_requested_to_physical_byte_ratio": True,
            "requested_ai_and_physical_counter_ai_are_separate": True,
            "strict_virtualized_run_is_canonical": False if virtualized else None,
        },
    }


def run_gemm_backend_characterization(args: argparse.Namespace) -> dict[str, Any]:
    """Screen and confirm one explicit PyTorch BLAS preference."""
    _validate_source_commit(args.source_commit)
    if args.blas_backend == "default":
        raise ValueError("gemm-backend requires --blas-backend cublas or cublaslt")
    for name in (
        "screen_warmups",
        "screen_repeats",
        "confirmation_windows",
        "confirmation_warmups",
        "confirmation_repeats",
    ):
        _require_positive(name, getattr(args, name))
    for name in (
        "minimum_window_gpu_ms",
        "precondition_ms",
        "screen_maximum_cv_pct",
        "confirmation_maximum_cv_pct",
    ):
        _require_positive(name, getattr(args, name))
    sizes = parse_positive_csv(args.gemm_sizes, name="gemm_sizes")
    layouts = parse_layout_csv(args.layouts)

    import torch

    device, properties, environment, provenance, reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry

    candidates = gemm_candidates(sizes, layouts)
    telemetry: dict[str, Any] = {}
    with (
        _preferred_blas_backend(torch, args.blas_backend) as blas_backend,
        lock_gpu_clocks(device=args.device) as lock_peaks,
    ):
        ramp_gpu_clocks(device=args.device, duration_ms=args.precondition_ms)
        telemetry["before_screen"] = query_gpu_telemetry(args.device, force_refresh=True)
        _progress(
            "gemm_backend_gates_passed_and_screen_started",
            backend=args.blas_backend,
            gpu_name=properties.name,
            gpu_uuid=str(properties.uuid),
            lab_tree_sha256=_package_sha256(),
            gemm_candidate_count=len(candidates),
        )
        screen = _screen_gemm(
            torch,
            device=device,
            candidates=candidates,
            warmups=args.screen_warmups,
            repeats=args.screen_repeats,
            seed=args.seed,
            maximum_cv_pct=args.screen_maximum_cv_pct,
        )
        telemetry["after_screen"] = query_gpu_telemetry(args.device, force_refresh=True)
        finalists = _select_gemm_finalists(screen)
        if len(finalists) < 2:
            raise RuntimeError("Explicit-backend screen did not produce two stable finalists")
        _progress(
            "gemm_backend_screen_complete",
            backend=args.blas_backend,
            finalists=[
                {
                    "candidate_id": result["candidate_id"],
                    "p50_TFLOP_per_s": result["throughput"]["p50"],
                }
                for result in finalists
            ],
        )
        confirmations = [
            _confirm_gemm_candidate(
                torch,
                device=device,
                screen_result=result,
                windows=args.confirmation_windows,
                warmups=args.confirmation_warmups,
                repeats=args.confirmation_repeats,
                minimum_window_gpu_ms=args.minimum_window_gpu_ms,
                precondition_ms=args.precondition_ms,
                seed=args.seed + 10_000_019 + index * 100_003,
                maximum_cv_pct=args.confirmation_maximum_cv_pct,
                ramp_gpu_clocks=ramp_gpu_clocks,
                query_gpu_telemetry=query_gpu_telemetry,
            )
            for index, result in enumerate(finalists)
        ]
        telemetry["after_confirmation"] = query_gpu_telemetry(
            args.device,
            force_refresh=True,
        )
        _progress(
            "gemm_backend_confirmation_complete",
            backend=args.blas_backend,
            confirmations=[
                {
                    "candidate_id": result["candidate_id"],
                    "stable": result["stable"],
                    "p50_TFLOP_per_s": result["throughput"]["p50"],
                    "cv_pct": result["throughput"]["cv_pct"],
                }
                for result in confirmations
            ],
        )

    stable = [result for result in confirmations if result["stable"]]
    if not stable:
        raise RuntimeError("No explicit-backend BF16 finalist passed confirmation gates")
    winner = max(stable, key=lambda result: result["throughput"]["p50"])
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_explicit_blas_backend_characterization",
        "status": "diagnostic_characterization_complete",
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "provenance": provenance,
        "configuration": {
            "gemm_sizes": sizes,
            "layouts": layouts,
            "screen_warmups": args.screen_warmups,
            "screen_repeats": args.screen_repeats,
            "confirmation_windows": args.confirmation_windows,
            "confirmation_warmups": args.confirmation_warmups,
            "confirmation_repeats": args.confirmation_repeats,
            "minimum_gpu_active_ms_per_confirmation_window": args.minimum_window_gpu_ms,
            "precondition_ms": args.precondition_ms,
            "screen_maximum_cv_pct": args.screen_maximum_cv_pct,
            "confirmation_maximum_cv_pct": args.confirmation_maximum_cv_pct,
            "seed": args.seed,
            "blas_backend_preference": blas_backend,
            "clock_lock": "core.harness.benchmark_harness.lock_gpu_clocks",
        },
        "environment_validation": environment,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry": telemetry,
        },
        "dense_bf16": {
            "screen": screen,
            "finalist_ids": [result["candidate_id"] for result in finalists],
            "confirmations": confirmations,
            "attained_ceiling": winner,
        },
        "nominal_comparison": {
            "reviewed_dense_bf16_TFLOP_per_s": reviewed["dense_bf16_tflops"],
            "attained_dense_bf16_TFLOP_per_s": winner["throughput"]["p50"],
            "attained_fraction_of_nominal": (
                winner["throughput"]["p50"] / reviewed["dense_bf16_tflops"]
            ),
            "nominal_source": reviewed["source"],
        },
        "profiler_artifacts": {"ncu": [], "nsys": []},
        "claim_boundary": {
            "blas_preference_readback_proves_actual_dispatch": False,
            "actual_dispatch_requires_external_trace_or_log": True,
            "highest_sustained_under_this_backend_protocol_is_vendor_peak": False,
            "strict_virtualized_run_is_canonical": False if virtualized else None,
        },
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Run a narrow exact-range profile replay for selected winners."""
    _validate_source_commit(args.source_commit)
    import torch

    device, properties, environment, provenance, _reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry

    measurements: dict[str, Any] = {}
    with (
        _preferred_blas_backend(torch, args.blas_backend) as blas_backend,
        lock_gpu_clocks(device=args.device) as lock_peaks,
    ):
        ramp_gpu_clocks(device=args.device, duration_ms=args.precondition_ms)
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        if args.profile_target in {"bandwidth", "both"}:
            from .ceiling_extension import load_ceiling_extension

            extension = load_ceiling_extension()
            candidate = BandwidthCandidate(
                operation=args.bandwidth_operation,
                strategy=args.bandwidth_strategy,
                payload_mib=args.profile_payload_mib,
                threads=(None if args.bandwidth_strategy == "torch_copy" else args.profile_threads),
                blocks_per_sm=(
                    args.profile_blocks_per_sm
                    if args.bandwidth_strategy
                    in {"native_scalar", "native_uint4", "native_high_entropy_uint4"}
                    else None
                ),
                chunk_kib=(
                    args.profile_chunk_kib
                    if args.bandwidth_strategy == "native_chunk_sum"
                    else None
                ),
            )
            valid_strategies = {
                "copy": {"torch_copy", "native_scalar", "native_uint4"},
                "read": {"native_chunk_sum"},
                "write": {"native_high_entropy_uint4"},
            }
            if candidate.strategy not in valid_strategies[candidate.operation]:
                raise ValueError(
                    f"strategy {candidate.strategy!r} is not valid for "
                    f"operation {candidate.operation!r}"
                )
            payload_bytes = candidate.payload_mib * 1024 * 1024
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)
            source = torch.randint(
                0,
                256,
                (payload_bytes,),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            destination = torch.empty_like(source)
            reference = None
            if candidate.operation == "read":
                if candidate.chunk_kib is None:
                    raise ValueError("read profiling requires profile_chunk_kib")
                reference = _exact_read_reference(
                    torch,
                    source,
                    chunk_bytes=candidate.chunk_kib * 1024,
                )
            measurements["bandwidth"] = _run_bandwidth_case(
                torch,
                extension=extension,
                device=device,
                source=source,
                destination=destination,
                candidate=candidate,
                sm_count=int(properties.multi_processor_count),
                warmups=args.profile_warmups,
                repeats=args.profile_repeats,
                invocations_per_sample=args.profile_invocations_per_sample,
                seed=args.seed,
                exact_read_reference=reference,
            )
        if args.profile_target in {"gemm", "both"}:
            candidate = GemmCandidate(size=args.profile_gemm_size, layout=args.profile_gemm_layout)
            a, b = _make_logical_gemm_operands(
                torch,
                device=device,
                size=candidate.size,
                seed=args.seed + 1,
            )
            reference = _fp32_reference(torch, a, b)
            measurements["gemm"] = _run_gemm_case(
                torch,
                device=device,
                candidate=candidate,
                a_logical=a,
                b_logical=b,
                reference=reference,
                warmups=args.profile_warmups,
                repeats=args.profile_repeats,
                invocations_per_sample=args.profile_invocations_per_sample,
            )
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_sustained_ceiling_profile_replay",
        "status": "profile_replay_complete",
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "provenance": provenance,
        "configuration": {
            "profile_target": args.profile_target,
            "warmups": args.profile_warmups,
            "repeats": args.profile_repeats,
            "invocations_per_sample": args.profile_invocations_per_sample,
            "seed": args.seed,
            "blas_backend_preference": blas_backend,
        },
        "environment_validation": environment,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        },
        "measurements": measurements,
        "claim_boundary": {
            "profiler_timing_is_primary_confirmation_timing": False,
            "requested_bytes_are_physical_hbm_counter_bytes": False,
            "strict_virtualized_run_is_canonical": False if virtualized else None,
        },
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--precondition-ms", type=float, default=250.0)
    parser.add_argument(
        "--blas-backend",
        choices=("default", "cublas", "cublaslt"),
        default="default",
        help="PyTorch preferred BLAS library. This preference may fall back internally.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Characterize sustained requested-byte HBM and dense BF16 ceilings on B200."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    characterize = subparsers.add_parser("characterize")
    _add_common_arguments(characterize)
    characterize.add_argument("--payload-mib", default="1024,2048,4096,8192")
    characterize.add_argument("--gemm-sizes", default="4096,8192,12288,16384")
    characterize.add_argument("--layouts", default="nn,nt,tn,tt")
    characterize.add_argument("--screen-warmups", type=int, default=3)
    characterize.add_argument("--screen-repeats", type=int, default=7)
    characterize.add_argument("--confirmation-windows", type=int, default=5)
    characterize.add_argument("--confirmation-warmups", type=int, default=3)
    characterize.add_argument("--confirmation-repeats", type=int, default=15)
    characterize.add_argument("--minimum-window-gpu-ms", type=float, default=1000.0)
    characterize.add_argument("--screen-maximum-cv-pct", type=float, default=5.0)
    characterize.add_argument("--confirmation-maximum-cv-pct", type=float, default=3.0)

    gemm_backend = subparsers.add_parser("gemm-backend")
    _add_common_arguments(gemm_backend)
    gemm_backend.add_argument("--gemm-sizes", default="4096,8192,12288,16384")
    gemm_backend.add_argument("--layouts", default="nn,nt,tn,tt")
    gemm_backend.add_argument("--screen-warmups", type=int, default=3)
    gemm_backend.add_argument("--screen-repeats", type=int, default=7)
    gemm_backend.add_argument("--confirmation-windows", type=int, default=5)
    gemm_backend.add_argument("--confirmation-warmups", type=int, default=3)
    gemm_backend.add_argument("--confirmation-repeats", type=int, default=15)
    gemm_backend.add_argument("--minimum-window-gpu-ms", type=float, default=1000.0)
    gemm_backend.add_argument("--screen-maximum-cv-pct", type=float, default=5.0)
    gemm_backend.add_argument("--confirmation-maximum-cv-pct", type=float, default=3.0)

    profile = subparsers.add_parser("profile")
    _add_common_arguments(profile)
    profile.add_argument("--profile-target", choices=("bandwidth", "gemm", "both"), required=True)
    profile.add_argument("--profile-warmups", type=int, default=1)
    profile.add_argument("--profile-repeats", type=int, default=1)
    profile.add_argument("--profile-invocations-per-sample", type=int, default=1)
    profile.add_argument("--bandwidth-operation", choices=("copy", "read", "write"), default="copy")
    profile.add_argument(
        "--bandwidth-strategy",
        choices=(
            "torch_copy",
            "native_scalar",
            "native_uint4",
            "native_chunk_sum",
            "native_high_entropy_uint4",
        ),
        default="torch_copy",
    )
    profile.add_argument("--profile-payload-mib", type=int, default=8192)
    profile.add_argument("--profile-threads", type=int, default=256)
    profile.add_argument("--profile-blocks-per-sm", type=int, default=8)
    profile.add_argument("--profile-chunk-kib", type=int, default=1024)
    profile.add_argument("--profile-gemm-size", type=int, default=16384)
    profile.add_argument("--profile-gemm-layout", choices=("nn", "nt", "tn", "tt"), default="nn")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _preflight_output(getattr(args, "output", None))
        runners = {
            "characterize": run_characterization,
            "gemm-backend": run_gemm_backend_characterization,
            "profile": run_profile,
        }
        import torch

        with _bf16_fp32_accumulation_policy(torch):
            report = runners[args.command](args)
    except (DiagnosticGateError, RuntimeError, ValueError, FileExistsError) as exc:
        failure = {
            "schema_version": 1,
            "kind": "b200_sustained_ceiling_failure",
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
                "kind": "b200_sustained_ceiling_failure",
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
