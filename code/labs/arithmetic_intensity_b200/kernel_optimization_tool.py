"""Bounded B200 kernel tuner and interleaved sustained confirmation tool."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from labs.arithmetic_intensity_b200.ceiling_protocol import (
    bootstrap_median_ci,
    confirmation_invocations_per_sample,
)
from labs.arithmetic_intensity_b200.formulas import gemm_logical_cost
from labs.arithmetic_intensity_b200.optimization_benchmarks import (
    BASELINE_COPY_BLOCKS_PER_SM,
    BASELINE_COPY_THREADS,
    COPY_PAYLOAD_BYTES,
    GEMM_MAX_ALGORITHMS,
    GEMM_SIZE,
    GEMM_WORKSPACE_BYTES,
    SELECTED_COPY_BLOCKS_PER_SM,
    SELECTED_COPY_THREADS,
    SELECTED_GEMM_ALGORITHM_SIGNATURE,
)
from labs.arithmetic_intensity_b200.optimization_protocol import (
    alternating_pair_order,
    copy_launch_candidates,
    integrate_power_samples_over_window,
    observer_overhead_summary,
    paired_window_summary,
    select_algorithm_index,
    summarize_nominal_gap_telemetry,
)
from labs.arithmetic_intensity_b200.verification_tool import (
    DiagnosticGateError,
    _bf16_fp32_accumulation_policy,
    _package_sha256,
    _preflight_output,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


def _validate_screen_for_confirmation(
    screen: dict[str, Any],
    *,
    expected_screen_lab_tree_sha256: str,
    source_commit: str,
    current_provenance: dict[str, Any],
) -> dict[str, Any]:
    if screen.get("schema_version") != 1:
        raise ValueError("confirmation requires screen schema version 1")
    if screen.get("kind") != "b200_kernel_optimization_bounded_screen":
        raise ValueError("confirmation requires a B200 kernel-optimization screen")
    if screen.get("status") != "screen_complete":
        raise ValueError("confirmation requires a completed bounded screen")
    screen_provenance = screen.get("provenance", {})
    actual_screen_hash = str(screen_provenance.get("lab_tree_sha256", ""))
    if actual_screen_hash != expected_screen_lab_tree_sha256:
        raise ValueError("screen lab-tree SHA-256 does not match the explicitly frozen digest")
    if str(screen_provenance.get("source_commit")) != source_commit:
        raise ValueError("screen and confirmation source commits differ")
    screen_device = screen_provenance.get("device", {})
    current_device = current_provenance.get("device", {})
    for field in ("uuid", "name", "compute_capability"):
        if screen_device.get(field) != current_device.get(field):
            raise ValueError(f"screen and confirmation device {field} differ")
    for field in ("pytorch", "pytorch_cuda", "precision_policy"):
        if screen_provenance.get(field) != current_provenance.get(field):
            raise ValueError(f"screen and confirmation {field} versions differ")

    gemm = screen.get("gemm", {})
    if gemm.get("shape") != {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE}:
        raise ValueError("screen GEMM shape differs from the frozen workload")
    if gemm.get("layout") != "row_major_nn" or gemm.get("dtype") != "bf16":
        raise ValueError("screen GEMM layout or dtype differs from the frozen workload")
    if gemm.get("compute_type") != "fp32_accumulation" or gemm.get("beta_zero") is not True:
        raise ValueError("screen GEMM precision or beta semantics differ")
    if gemm.get("selected_signature") != SELECTED_GEMM_ALGORITHM_SIGNATURE:
        raise ValueError("screen GEMM choice differs from the frozen source signature")
    gemm_matches = [
        item
        for item in gemm.get("candidates", [])
        if item.get("signature") == SELECTED_GEMM_ALGORITHM_SIGNATURE
    ]
    if len(gemm_matches) != 1:
        raise ValueError("screen must contain the frozen GEMM choice exactly once")
    if (
        gemm_matches[0].get("ranking_eligible") is not True
        or gemm_matches[0].get("correctness", {}).get("passed") is not True
    ):
        raise ValueError("screen frozen GEMM choice was not correct and ranking eligible")

    copy = screen.get("copy", {})
    if int(copy.get("payload_bytes", -1)) != COPY_PAYLOAD_BYTES:
        raise ValueError("screen copy payload differs from the frozen 8 GiB workload")
    if int(copy.get("selected_threads", -1)) != SELECTED_COPY_THREADS:
        raise ValueError("screen copy thread choice differs from frozen source")
    if int(copy.get("selected_blocks_per_sm", -1)) != SELECTED_COPY_BLOCKS_PER_SM:
        raise ValueError("screen copy block choice differs from frozen source")
    selected_copy_id = copy.get("selected_candidate_id")
    copy_matches = [
        item for item in copy.get("candidates", []) if item.get("candidate_id") == selected_copy_id
    ]
    if len(copy_matches) != 1:
        raise ValueError("screen must contain the frozen copy choice exactly once")
    copy_choice = copy_matches[0]
    if (
        int(copy_choice.get("threads", -1)) != SELECTED_COPY_THREADS
        or int(copy_choice.get("blocks_per_sm", -1)) != SELECTED_COPY_BLOCKS_PER_SM
    ):
        raise ValueError("screen selected-copy metadata differs from frozen source")
    if (
        copy_choice.get("ranking_eligible") is not True
        or copy_choice.get("correctness", {}).get("passed") is not True
    ):
        raise ValueError("screen frozen copy choice was not correct and ranking eligible")
    if copy.get("byte_accounting", {}).get("scope") != (
        "requested source read plus requested destination write"
    ):
        raise ValueError("screen copy byte-accounting boundary differs")
    if copy.get("byte_accounting", {}).get("is_raw_physical_hbm_bandwidth") is not False:
        raise ValueError("screen copy must not claim raw physical HBM bandwidth")

    return {
        "screen_lab_tree_sha256": actual_screen_hash,
        "confirmation_lab_tree_sha256": current_provenance["lab_tree_sha256"],
        "source_commit_matched": True,
        "device_identity_matched": True,
        "runtime_versions_matched": True,
        "workload_contract_matched": True,
        "frozen_source_selections_matched": True,
        "expected_screen_hash_supplied": True,
        "reason_hashes_differ": (
            "the screen selected supported candidates before their exact signatures and "
            "launch values were frozen in source"
        ),
    }


def _tflops(logical_flops: int, timing_ms: Sequence[float]) -> list[float]:
    return [logical_flops / (sample / 1e3) / 1e12 for sample in timing_ms]


def _bandwidth_tbps(requested_bytes: int, timing_ms: Sequence[float]) -> list[float]:
    return [requested_bytes / (sample / 1e3) / 1e12 for sample in timing_ms]


def _arm_summary(
    *,
    windows: Sequence[dict[str, Any]],
    metric_key: str,
    unit: str,
    seed: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _sample_summary

    samples = [sample for window in windows for sample in window[metric_key]["samples"]]
    medians = [float(window[metric_key]["p50"]) for window in windows]
    return {
        "all_samples": _sample_summary(samples, unit),
        "window_medians": _sample_summary(medians, unit),
        "window_median_bootstrap_95_ci": bootstrap_median_ci(medians, seed=seed),
    }


def _screen_gemm(
    torch: Any,
    *,
    device: Any,
    extension: Any,
    warmups: int,
    repeats: int,
    seed: int,
    maximum_cv_pct: float,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _fp32_reference,
        _gemm_correctness,
        _make_logical_gemm_operands,
        _measure_aggregate_operation,
        _sample_summary,
    )

    a, b = _make_logical_gemm_operands(
        torch,
        device=device,
        size=GEMM_SIZE,
        seed=seed,
    )
    reference = _fp32_reference(torch, a, b)
    baseline_output = torch.empty_like(a)
    candidate_output = torch.empty_like(a)
    plan = extension.Bf16LtMatmulPlan(
        a,
        b,
        candidate_output,
        GEMM_WORKSPACE_BYTES,
        GEMM_MAX_ALGORITHMS,
    )
    algorithms = [dict(item) for item in plan.algorithms()]
    logical = gemm_logical_cost(m=GEMM_SIZE, n=GEMM_SIZE, k=GEMM_SIZE, beta_zero=True)
    baseline_call = partial(torch.mm, a, b, out=baseline_output)
    baseline_timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=lambda _invocation, call=baseline_call: call(),
        range_name="ai_b200_optimization_screen_baseline_gemm",
        warmups=warmups,
        repeats=repeats,
        invocations_per_sample=1,
    )
    baseline_correctness = _gemm_correctness(torch, baseline_output, reference)
    baseline_rates = _tflops(
        int(logical["flops"]),
        baseline_timing["normalized_timing"]["samples"],
    )
    baseline = {
        "implementation_id": "torch_mm_out_default_cublas",
        "timing": baseline_timing,
        "throughput": _sample_summary(baseline_rates, "TFLOP_per_s"),
        "correctness": baseline_correctness,
    }
    results: list[dict[str, Any]] = []
    for metadata in algorithms:
        index = int(metadata["heuristic_index"])
        try:
            candidate_call = partial(plan.run, a, b, candidate_output, index)
            timing = _measure_aggregate_operation(
                torch,
                device=device,
                operation=lambda _invocation, call=candidate_call: call(),
                range_name=f"ai_b200_optimization_screen_cublaslt_{index}",
                warmups=warmups,
                repeats=repeats,
                invocations_per_sample=1,
            )
            correctness = _gemm_correctness(torch, candidate_output, reference)
            rates = _tflops(
                int(logical["flops"]),
                timing["normalized_timing"]["samples"],
            )
            throughput = _sample_summary(rates, "TFLOP_per_s")
            reasons = []
            if not correctness["passed"]:
                reasons.append("full_output_correctness_failed")
            if throughput["cv_pct"] > maximum_cv_pct:
                reasons.append(f"screen_cv_pct_exceeds_{maximum_cv_pct:g}")
            results.append(
                {
                    **metadata,
                    "status": "measured",
                    "timing": timing,
                    "throughput": throughput,
                    "correctness": correctness,
                    "ranking_eligible": not reasons,
                    "rejection_reasons": reasons,
                }
            )
        except RuntimeError as error:
            results.append(
                {
                    **metadata,
                    "status": "rejected_runtime_error",
                    "ranking_eligible": False,
                    "rejection_reasons": [str(error)],
                }
            )
    eligible = [item for item in results if item["ranking_eligible"]]
    eligible.sort(key=lambda item: item["throughput"]["p50"], reverse=True)
    return {
        "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
        "layout": "row_major_nn",
        "dtype": "bf16",
        "compute_type": "fp32_accumulation",
        "beta_zero": True,
        "logical_model": logical,
        "input_distribution": "seeded_scaled_normal_bf16",
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "maximum_cv_pct": maximum_cv_pct,
        "workspace_bytes": GEMM_WORKSPACE_BYTES,
        "baseline": baseline,
        "candidates": results,
        "selected_signature": eligible[0]["signature"] if eligible else None,
        "selected_heuristic_index": eligible[0]["heuristic_index"] if eligible else None,
        "eligible_candidate_count": len(eligible),
    }


def _screen_copy(
    torch: Any,
    *,
    device: Any,
    ceiling_extension: Any,
    optimization_extension: Any,
    sm_count: int,
    payload_bytes: int,
    warmups: int,
    repeats: int,
    seed: int,
    maximum_cv_pct: float,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _measure_aggregate_operation,
        _sample_summary,
    )

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
    destination = torch.empty_like(source)
    requested_bytes = 2 * payload_bytes

    def measure(
        *,
        candidate_id: str,
        operation: Callable[[], None],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timing = _measure_aggregate_operation(
            torch,
            device=device,
            operation=lambda _invocation: operation(),
            range_name=f"ai_b200_optimization_screen_{candidate_id}",
            warmups=warmups,
            repeats=repeats,
            invocations_per_sample=1,
        )
        passed = bool(torch.equal(destination, source))
        rates = _bandwidth_tbps(
            requested_bytes,
            timing["normalized_timing"]["samples"],
        )
        throughput = _sample_summary(rates, "effective_requested_TB_per_s")
        reasons = []
        if not passed:
            reasons.append("full_tensor_equality_failed")
        if throughput["cv_pct"] > maximum_cv_pct:
            reasons.append(f"screen_cv_pct_exceeds_{maximum_cv_pct:g}")
        return {
            "candidate_id": candidate_id,
            **metadata,
            "timing": timing,
            "throughput": throughput,
            "correctness": {
                "method": "full_tensor_torch_equal",
                "output_elements_checked": payload_bytes,
                "passed": passed,
            },
            "ranking_eligible": not reasons,
            "rejection_reasons": reasons,
        }

    baseline_blocks = BASELINE_COPY_BLOCKS_PER_SM * sm_count
    baseline = measure(
        candidate_id="native_uint4_grid_stride",
        operation=partial(
            ceiling_extension.copy_uint4,
            source,
            destination,
            BASELINE_COPY_THREADS,
            baseline_blocks,
        ),
        metadata={
            "threads": BASELINE_COPY_THREADS,
            "blocks_per_sm": BASELINE_COPY_BLOCKS_PER_SM,
            "blocks": baseline_blocks,
        },
    )
    copy_engine = measure(
        candidate_id="torch_copy_engine",
        operation=partial(destination.copy_, source),
        metadata={"threads": None, "blocks_per_sm": None, "blocks": None},
    )
    candidates = []
    for candidate in copy_launch_candidates():
        threads = int(candidate["threads"])
        blocks_per_sm = int(candidate["blocks_per_sm"])
        blocks = blocks_per_sm * sm_count
        candidates.append(
            measure(
                candidate_id=str(candidate["candidate_id"]),
                operation=partial(
                    optimization_extension.copy_uint4_cg_unroll4,
                    source,
                    destination,
                    threads,
                    blocks,
                ),
                metadata={
                    "threads": threads,
                    "blocks_per_sm": blocks_per_sm,
                    "blocks": blocks,
                },
            )
        )
    eligible = [item for item in candidates if item["ranking_eligible"]]
    eligible.sort(key=lambda item: item["throughput"]["p50"], reverse=True)
    return {
        "payload_bytes": payload_bytes,
        "input_distribution": "seeded_uniform_random_uint8_0_255",
        "seed": seed,
        "warmups": warmups,
        "repeats": repeats,
        "maximum_cv_pct": maximum_cv_pct,
        "byte_accounting": {
            "requested_read_bytes_per_operation": payload_bytes,
            "requested_write_bytes_per_operation": payload_bytes,
            "requested_total_bytes_per_operation": requested_bytes,
            "scope": "requested source read plus requested destination write",
            "is_raw_physical_hbm_bandwidth": False,
        },
        "baseline": baseline,
        "copy_engine_reference": copy_engine,
        "candidates": candidates,
        "selected_candidate_id": eligible[0]["candidate_id"] if eligible else None,
        "selected_threads": eligible[0]["threads"] if eligible else None,
        "selected_blocks_per_sm": eligible[0]["blocks_per_sm"] if eligible else None,
        "eligible_candidate_count": len(eligible),
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_extension import load_ceiling_extension
    from labs.arithmetic_intensity_b200.ceiling_tool import _strict_runtime
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    if args.device != 0:
        raise ValueError("This optimization protocol is authorized for GPU0 only")
    device, properties, environment, provenance, reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    ceiling_extension = load_ceiling_extension()
    optimization_extension = load_optimization_extension()
    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        ramp_gpu_clocks(device=args.device, duration_ms=args.precondition_ms)
        gemm = _screen_gemm(
            torch,
            device=device,
            extension=optimization_extension,
            warmups=args.screen_warmups,
            repeats=args.screen_repeats,
            seed=args.seed,
            maximum_cv_pct=args.screen_maximum_cv_pct,
        )
        ramp_gpu_clocks(device=args.device, duration_ms=args.precondition_ms)
        copy = _screen_copy(
            torch,
            device=device,
            ceiling_extension=ceiling_extension,
            optimization_extension=optimization_extension,
            sm_count=int(properties.multi_processor_count),
            payload_bytes=args.copy_payload_mib * 1024 * 1024,
            warmups=args.screen_warmups,
            repeats=args.screen_repeats,
            seed=args.seed + 10_000_019,
            maximum_cv_pct=args.screen_maximum_cv_pct,
        )
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_kernel_optimization_bounded_screen",
        "status": "screen_complete",
        "timestamp_utc": _timestamp(),
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        },
        "configuration": {
            "screen_warmups": args.screen_warmups,
            "screen_repeats": args.screen_repeats,
            "precondition_ms": args.precondition_ms,
            "copy_payload_mib": args.copy_payload_mib,
        },
        "gemm": gemm,
        "copy": copy,
    }


def _measure_gemm_arm(
    torch: Any,
    *,
    device: Any,
    operation: Callable[[], None],
    output: Any,
    reference: Any,
    logical_flops: int,
    label: str,
    warmups: int,
    repeats: int,
    invocations_per_sample: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _gemm_correctness,
        _measure_aggregate_operation,
        _sample_summary,
    )

    timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=lambda _invocation: operation(),
        range_name=label,
        warmups=warmups,
        repeats=repeats,
        invocations_per_sample=invocations_per_sample,
    )
    correctness = _gemm_correctness(torch, output, reference)
    rates = _tflops(logical_flops, timing["normalized_timing"]["samples"])
    return {
        "timing": timing,
        "throughput": _sample_summary(rates, "TFLOP_per_s"),
        "correctness": correctness,
    }


def _confirm_gemm(
    torch: Any,
    *,
    device: Any,
    optimization_extension: Any,
    selected_signature: str,
    screen: dict[str, Any],
    windows: int,
    warmups: int,
    repeats: int,
    minimum_window_ms: float,
    precondition_ms: float,
    seed: int,
    ramp_gpu_clocks: Callable[..., None],
    query_gpu_telemetry: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _fp32_reference,
        _make_logical_gemm_operands,
    )

    selected_screen = next(
        item for item in screen["candidates"] if item.get("signature") == selected_signature
    )
    baseline_screen_ms = float(screen["baseline"]["timing"]["normalized_timing"]["p50"])
    candidate_screen_ms = float(selected_screen["timing"]["normalized_timing"]["p50"])
    invocations = max(
        confirmation_invocations_per_sample(
            baseline_screen_ms,
            repeats=repeats,
            minimum_window_gpu_ms=minimum_window_ms,
        ),
        confirmation_invocations_per_sample(
            candidate_screen_ms,
            repeats=repeats,
            minimum_window_gpu_ms=minimum_window_ms,
        ),
    )
    logical = gemm_logical_cost(m=GEMM_SIZE, n=GEMM_SIZE, k=GEMM_SIZE, beta_zero=True)
    results = []
    for window_index in range(windows):
        window_seed = seed + window_index * 1_000_003
        a, b = _make_logical_gemm_operands(
            torch,
            device=device,
            size=GEMM_SIZE,
            seed=window_seed,
        )
        reference = _fp32_reference(torch, a, b)
        baseline_output = torch.empty_like(a)
        candidate_output = torch.empty_like(a)
        plan = optimization_extension.Bf16LtMatmulPlan(
            a,
            b,
            candidate_output,
            GEMM_WORKSPACE_BYTES,
            GEMM_MAX_ALGORITHMS,
        )
        algorithms = [dict(item) for item in plan.algorithms()]
        algorithm_index = select_algorithm_index(algorithms, selected_signature)
        ramp_gpu_clocks(device=int(device.index or 0), duration_ms=precondition_ms)
        telemetry_before = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        order = alternating_pair_order(window_index)
        arms: dict[str, Any] = {}
        for arm in order:
            if arm == "baseline":
                operation = partial(torch.mm, a, b, out=baseline_output)
                arms[arm] = _measure_gemm_arm(
                    torch,
                    device=device,
                    operation=operation,
                    output=baseline_output,
                    reference=reference,
                    logical_flops=int(logical["flops"]),
                    label=f"ai_b200_optimization_confirm_gemm_baseline_w{window_index}",
                    warmups=warmups,
                    repeats=repeats,
                    invocations_per_sample=invocations,
                )
            else:
                operation = partial(plan.run, a, b, candidate_output, algorithm_index)
                arms[arm] = _measure_gemm_arm(
                    torch,
                    device=device,
                    operation=operation,
                    output=candidate_output,
                    reference=reference,
                    logical_flops=int(logical["flops"]),
                    label=f"ai_b200_optimization_confirm_gemm_candidate_w{window_index}",
                    warmups=warmups,
                    repeats=repeats,
                    invocations_per_sample=invocations,
                )
        telemetry_after = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        for arm in ("baseline", "candidate"):
            arms[arm]["sustained_window_gate_passed"] = (
                arms[arm]["timing"]["measured_gpu_active_ms"] >= minimum_window_ms
            )
        results.append(
            {
                "window_index": window_index,
                "window_seed": window_seed,
                "order": list(order),
                "invocations_per_sample": invocations,
                "telemetry_before": telemetry_before,
                "telemetry_after": telemetry_after,
                **arms,
            }
        )
        del operation, plan, candidate_output, baseline_output, reference, a, b
        torch.cuda.empty_cache()
    baseline_windows = [window["baseline"] for window in results]
    candidate_windows = [window["candidate"] for window in results]
    baseline_medians = [item["throughput"]["p50"] for item in baseline_windows]
    candidate_medians = [item["throughput"]["p50"] for item in candidate_windows]
    paired = paired_window_summary(baseline_medians, candidate_medians, seed=seed + 91)
    return {
        "selected_algorithm_signature": selected_signature,
        "selected_algorithm_metadata": next(
            item for item in algorithms if item["signature"] == selected_signature
        ),
        "window_count": windows,
        "minimum_gpu_active_ms_per_arm_per_window": minimum_window_ms,
        "windows": results,
        "baseline": _arm_summary(
            windows=baseline_windows,
            metric_key="throughput",
            unit="TFLOP_per_s",
            seed=seed + 11,
        ),
        "candidate": _arm_summary(
            windows=candidate_windows,
            metric_key="throughput",
            unit="TFLOP_per_s",
            seed=seed + 17,
        ),
        "paired_interleaved": paired,
        "all_correct": all(
            window[arm]["correctness"]["passed"]
            for window in results
            for arm in ("baseline", "candidate")
        ),
        "all_sustained": all(
            window[arm]["sustained_window_gate_passed"]
            for window in results
            for arm in ("baseline", "candidate")
        ),
        "logical_model": logical,
    }


def _measure_copy_arm(
    torch: Any,
    *,
    device: Any,
    operation: Callable[[], None],
    output: Any,
    source: Any,
    requested_bytes: int,
    label: str,
    warmups: int,
    repeats: int,
    invocations_per_sample: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _measure_aggregate_operation,
        _sample_summary,
    )

    timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=lambda _invocation: operation(),
        range_name=label,
        warmups=warmups,
        repeats=repeats,
        invocations_per_sample=invocations_per_sample,
    )
    passed = bool(torch.equal(output, source))
    rates = _bandwidth_tbps(requested_bytes, timing["normalized_timing"]["samples"])
    return {
        "timing": timing,
        "throughput": _sample_summary(rates, "effective_requested_TB_per_s"),
        "correctness": {
            "method": "full_tensor_torch_equal",
            "output_elements_checked": int(source.numel()),
            "passed": passed,
        },
    }


def _confirm_copy(
    torch: Any,
    *,
    device: Any,
    ceiling_extension: Any,
    optimization_extension: Any,
    sm_count: int,
    selected_candidate: dict[str, Any],
    screen: dict[str, Any],
    windows: int,
    warmups: int,
    repeats: int,
    minimum_window_ms: float,
    precondition_ms: float,
    seed: int,
    ramp_gpu_clocks: Callable[..., None],
    query_gpu_telemetry: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    payload_bytes = COPY_PAYLOAD_BYTES
    requested_bytes = 2 * payload_bytes
    threads = int(selected_candidate["threads"])
    blocks_per_sm = int(selected_candidate["blocks_per_sm"])
    candidate_blocks = blocks_per_sm * sm_count
    baseline_blocks = BASELINE_COPY_BLOCKS_PER_SM * sm_count
    baseline_screen_ms = float(screen["baseline"]["timing"]["normalized_timing"]["p50"])
    candidate_screen_ms = float(selected_candidate["timing"]["normalized_timing"]["p50"])
    copy_engine_screen_ms = float(
        screen["copy_engine_reference"]["timing"]["normalized_timing"]["p50"]
    )
    pair_invocations = max(
        confirmation_invocations_per_sample(
            baseline_screen_ms,
            repeats=repeats,
            minimum_window_gpu_ms=minimum_window_ms,
        ),
        confirmation_invocations_per_sample(
            candidate_screen_ms,
            repeats=repeats,
            minimum_window_gpu_ms=minimum_window_ms,
        ),
    )
    copy_engine_invocations = confirmation_invocations_per_sample(
        copy_engine_screen_ms,
        repeats=repeats,
        minimum_window_gpu_ms=minimum_window_ms,
    )
    results = []
    for window_index in range(windows):
        window_seed = seed + window_index * 1_000_003
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
        baseline_output = torch.empty_like(source)
        candidate_output = torch.empty_like(source)
        copy_engine_output = torch.empty_like(source)
        pair_order = alternating_pair_order(window_index)
        execution_order = (
            [*pair_order, "copy_engine"] if window_index % 2 == 0 else ["copy_engine", *pair_order]
        )
        ramp_gpu_clocks(device=int(device.index or 0), duration_ms=precondition_ms)
        telemetry_before = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        arms: dict[str, Any] = {}
        for arm in execution_order:
            if arm == "baseline":
                operation = partial(
                    ceiling_extension.copy_uint4,
                    source,
                    baseline_output,
                    BASELINE_COPY_THREADS,
                    baseline_blocks,
                )
                output = baseline_output
                invocations = pair_invocations
            elif arm == "candidate":
                operation = partial(
                    optimization_extension.copy_uint4_cg_unroll4,
                    source,
                    candidate_output,
                    threads,
                    candidate_blocks,
                )
                output = candidate_output
                invocations = pair_invocations
            else:
                operation = partial(copy_engine_output.copy_, source)
                output = copy_engine_output
                invocations = copy_engine_invocations
            arms[arm] = _measure_copy_arm(
                torch,
                device=device,
                operation=operation,
                output=output,
                source=source,
                requested_bytes=requested_bytes,
                label=f"ai_b200_optimization_confirm_copy_{arm}_w{window_index}",
                warmups=warmups,
                repeats=repeats,
                invocations_per_sample=invocations,
            )
            arms[arm]["sustained_window_gate_passed"] = (
                arms[arm]["timing"]["measured_gpu_active_ms"] >= minimum_window_ms
            )
        telemetry_after = query_gpu_telemetry(int(device.index or 0), force_refresh=True)
        results.append(
            {
                "window_index": window_index,
                "window_seed": window_seed,
                "pair_order": list(pair_order),
                "execution_order": execution_order,
                "pair_invocations_per_sample": pair_invocations,
                "copy_engine_invocations_per_sample": copy_engine_invocations,
                "telemetry_before": telemetry_before,
                "telemetry_after": telemetry_after,
                **arms,
            }
        )
        del operation, copy_engine_output, candidate_output, baseline_output, source
        torch.cuda.empty_cache()
    baseline_windows = [window["baseline"] for window in results]
    candidate_windows = [window["candidate"] for window in results]
    copy_engine_windows = [window["copy_engine"] for window in results]
    baseline_medians = [item["throughput"]["p50"] for item in baseline_windows]
    candidate_medians = [item["throughput"]["p50"] for item in candidate_windows]
    paired = paired_window_summary(baseline_medians, candidate_medians, seed=seed + 97)
    return {
        "payload_bytes": payload_bytes,
        "requested_bytes_per_operation": requested_bytes,
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_threads": threads,
        "selected_blocks_per_sm": blocks_per_sm,
        "window_count": windows,
        "minimum_gpu_active_ms_per_arm_per_window": minimum_window_ms,
        "windows": results,
        "baseline": _arm_summary(
            windows=baseline_windows,
            metric_key="throughput",
            unit="effective_requested_TB_per_s",
            seed=seed + 13,
        ),
        "candidate": _arm_summary(
            windows=candidate_windows,
            metric_key="throughput",
            unit="effective_requested_TB_per_s",
            seed=seed + 19,
        ),
        "copy_engine_reference": _arm_summary(
            windows=copy_engine_windows,
            metric_key="throughput",
            unit="effective_requested_TB_per_s",
            seed=seed + 23,
        ),
        "paired_interleaved": paired,
        "all_correct": all(
            window[arm]["correctness"]["passed"]
            for window in results
            for arm in ("baseline", "candidate", "copy_engine")
        ),
        "all_sustained": all(
            window[arm]["sustained_window_gate_passed"]
            for window in results
            for arm in ("baseline", "candidate", "copy_engine")
        ),
        "byte_accounting": {
            "scope": "requested source read plus requested destination write",
            "is_raw_physical_hbm_bandwidth": False,
            "physical_hbm_counter_bytes": None,
        },
    }


def run_confirm(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_extension import load_ceiling_extension
    from labs.arithmetic_intensity_b200.ceiling_tool import _strict_runtime
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    if args.device != 0:
        raise ValueError("This optimization protocol is authorized for GPU0 only")
    if args.confirmation_windows < 5:
        raise ValueError("confirmation requires at least five independent windows")
    if args.confirmation_repeats < 1:
        raise ValueError("confirmation repeats must be positive")
    if args.minimum_window_gpu_ms < 1_000.0:
        raise ValueError("confirmation requires at least 1000 GPU-active ms per arm")
    screen_report = json.loads(args.screen.read_text())
    device, properties, environment, provenance, reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    lineage = _validate_screen_for_confirmation(
        screen_report,
        expected_screen_lab_tree_sha256=args.expected_screen_lab_tree_sha256,
        source_commit=args.source_commit,
        current_provenance=provenance,
    )
    selected_signature = str(screen_report["gemm"]["selected_signature"])
    selected_copy_id = str(screen_report["copy"]["selected_candidate_id"])
    selected_copy = next(
        item
        for item in screen_report["copy"]["candidates"]
        if item["candidate_id"] == selected_copy_id
    )
    ceiling_extension = load_ceiling_extension()
    optimization_extension = load_optimization_extension()
    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        gemm = _confirm_gemm(
            torch,
            device=device,
            optimization_extension=optimization_extension,
            selected_signature=selected_signature,
            screen=screen_report["gemm"],
            windows=args.confirmation_windows,
            warmups=args.confirmation_warmups,
            repeats=args.confirmation_repeats,
            minimum_window_ms=args.minimum_window_gpu_ms,
            precondition_ms=args.precondition_ms,
            seed=args.seed,
            ramp_gpu_clocks=ramp_gpu_clocks,
            query_gpu_telemetry=query_gpu_telemetry,
        )
        copy = _confirm_copy(
            torch,
            device=device,
            ceiling_extension=ceiling_extension,
            optimization_extension=optimization_extension,
            sm_count=int(properties.multi_processor_count),
            selected_candidate=selected_copy,
            screen=screen_report["copy"],
            windows=args.confirmation_windows,
            warmups=args.confirmation_warmups,
            repeats=args.confirmation_repeats,
            minimum_window_ms=args.minimum_window_gpu_ms,
            precondition_ms=args.precondition_ms,
            seed=args.seed + 10_000_019,
            ramp_gpu_clocks=ramp_gpu_clocks,
            query_gpu_telemetry=query_gpu_telemetry,
        )
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)
    virtualized = bool(environment["details"].get("virtualized"))
    qualified = (
        gemm["all_correct"]
        and gemm["all_sustained"]
        and copy["all_correct"]
        and copy["all_sustained"]
    )
    minimum_effect_size_pct = 5.0
    gemm_harness_eligible = (
        gemm["paired_interleaved"]["verdict"] == "improved"
        and gemm["paired_interleaved"]["candidate_vs_baseline_pct"] >= minimum_effect_size_pct
    )
    copy_harness_eligible = (
        copy["paired_interleaved"]["verdict"] == "improved"
        and copy["paired_interleaved"]["candidate_vs_baseline_pct"] >= minimum_effect_size_pct
    )
    return {
        "schema_version": 1,
        "kind": "b200_kernel_optimization_interleaved_confirmation",
        "status": "confirmation_complete" if qualified else "confirmation_disqualified",
        "timestamp_utc": _timestamp(),
        "screen_run_id": screen_report["provenance"]["run_id"],
        "lineage_reconciliation": lineage,
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "clock_lock": {
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        },
        "configuration": {
            "confirmation_windows": args.confirmation_windows,
            "confirmation_warmups": args.confirmation_warmups,
            "confirmation_repeats": args.confirmation_repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "precondition_ms": args.precondition_ms,
        },
        "gemm": gemm,
        "copy": copy,
        "optimization_decision": {
            "measurement_complete": qualified,
            "minimum_effect_size_pct": minimum_effect_size_pct,
            "baseline_replacement_requires_qualified_harness_pair": True,
            "gemm": {
                "paired_verdict": gemm["paired_interleaved"]["verdict"],
                "eligible_for_harness_promotion": gemm_harness_eligible,
                "accepted_for_baseline_replacement": False,
            },
            "copy": {
                "paired_verdict": copy["paired_interleaved"]["verdict"],
                "eligible_for_harness_promotion": copy_harness_eligible,
                "accepted_for_baseline_replacement": False,
            },
            "any_candidate_accepted": False,
        },
        "claim_boundaries": {
            "attained_is_absolute_hardware_max": False,
            "copy_is_raw_physical_hbm_bandwidth": False,
            "requested_and_physical_ai_are_separate": True,
            "profiled_timing_is_primary": False,
        },
    }


def _nominal_gap_progress(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), file=sys.stderr, flush=True)


def _validate_nominal_gap_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("The nominal-gap diagnostic is authorized for GPU0 only")
    integer_bounds = {
        "short_warmups": (1, 10),
        "short_repeats": (3, 20),
        "calibration_rounds": (8, 8),
        "calibration_repeats": (3, 30),
        "sustained_windows": (3, 5),
        "sustained_repeats": (5, 30),
        "minimum_in_window_telemetry_samples": (8, 100),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    numeric_bounds = {
        "settling_minimum_gpu_ms": (3000.0, 10_000.0),
        "calibration_minimum_gpu_ms": (1000.0, 3000.0),
        "minimum_window_gpu_ms": (1000.0, 3000.0),
        "telemetry_interval_ms": (50.0, 250.0),
        "maximum_telemetry_gap_ms": (100.0, 250.0),
        "observer_overhead_limit_pct": (0.0, 5.0),
    }
    for name, (minimum, maximum) in numeric_bounds.items():
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be finite and in [{minimum:g}, {maximum:g}]")
    if args.maximum_telemetry_gap_ms < args.telemetry_interval_ms:
        raise ValueError("maximum_telemetry_gap_ms must be at least telemetry_interval_ms")
    if args.maximum_telemetry_gap_ms > 2.5 * args.telemetry_interval_ms:
        raise ValueError("maximum_telemetry_gap_ms must be at most 2.5 times telemetry_interval_ms")
    for window_name in ("calibration_minimum_gpu_ms", "minimum_window_gpu_ms"):
        conservative_complete_query_slots = (
            math.floor(getattr(args, window_name) / args.telemetry_interval_ms) - 1
        )
        if conservative_complete_query_slots < args.minimum_in_window_telemetry_samples:
            raise ValueError(
                f"{window_name} and telemetry_interval_ms cannot conservatively fit "
                "minimum_in_window_telemetry_samples"
            )


def _operand_identity(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": int(tensor.data_ptr()),
        "tensor_version": int(tensor._version),
    }


def _realized_operand_summary(torch: Any, tensor: Any) -> dict[str, Any]:
    """Summarize every realized BF16 element outside the measured region."""
    values = tensor.float()
    element_count = int(values.numel())
    rms = float(torch.linalg.vector_norm(values).item()) / math.sqrt(element_count)
    summary = {
        "scope": "all_realized_elements",
        "element_count": element_count,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "rms": rms,
    }
    del values
    return summary


def _make_nominal_gap_operands(
    torch: Any,
    *,
    device: Any,
    seed: int,
    fill_mode: str,
) -> tuple[Any, Any, dict[str, Any]]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _make_logical_gemm_operands

    if fill_mode == "scaled_normal":
        a, b = _make_logical_gemm_operands(
            torch,
            device=device,
            size=GEMM_SIZE,
            seed=seed,
        )
        generator_contract = {
            "distribution": "torch.randn generated directly as BF16",
            "scale": GEMM_SIZE**-0.25,
            "scale_application": "in-place BF16 multiplication after BF16 generation",
            "expected_pre_quantization_rms": GEMM_SIZE**-0.25,
            "expected_rms_ratio_to_scaled_normal": 1.0,
            "amplitude_confounded_relative_to_primary": False,
        }
    elif fill_mode == "uniform_minus1_plus1":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        a = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        b = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        a.uniform_(-1.0, 1.0, generator=generator)
        b.uniform_(-1.0, 1.0, generator=generator)
        generator_contract = {
            "distribution": "uniform[-1,1] generated directly as BF16",
            "scale": 1.0,
            "scale_application": "none",
            "expected_pre_quantization_rms": 1.0 / math.sqrt(3.0),
            "expected_rms_ratio_to_scaled_normal": (GEMM_SIZE**0.25 / math.sqrt(3.0)),
            "amplitude_confounded_relative_to_primary": True,
            "amplitude_confounding_limitation": (
                "Uniform[-1,1] changes both the distribution and operand RMS relative "
                "to scaled-normal inputs, so it cannot isolate distribution alone."
            ),
        }
    elif fill_mode == "uniform_rms_matched_to_scaled_normal":
        target_rms = GEMM_SIZE**-0.25
        uniform_bound = math.sqrt(3.0) * target_rms
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        a = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        b = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        a.uniform_(-uniform_bound, uniform_bound, generator=generator)
        b.uniform_(-uniform_bound, uniform_bound, generator=generator)
        generator_contract = {
            "distribution": "RMS-matched symmetric uniform generated directly as BF16",
            "uniform_bound": uniform_bound,
            "target_rms": target_rms,
            "scale_application": "uniform bounds chosen before BF16 generation",
            "expected_pre_quantization_rms": target_rms,
            "expected_rms_ratio_to_scaled_normal": 1.0,
            "amplitude_confounded_relative_to_primary": False,
            "distribution_differs_from_primary": True,
        }
    elif fill_mode == "zero_power_switching_calibration":
        a = torch.zeros((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        b = torch.zeros((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
        generator_contract = {
            "distribution": "all zero BF16 power-switching calibration",
            "scale": 1.0,
            "scale_application": "none",
            "expected_pre_quantization_rms": 0.0,
            "expected_rms_ratio_to_scaled_normal": 0.0,
            "amplitude_confounded_relative_to_primary": True,
            "amplitude_confounding_limitation": (
                "All-zero operands intentionally remove value switching and are only a "
                "power-switching calibration."
            ),
        }
    else:
        raise ValueError(f"Unsupported nominal-gap fill mode: {fill_mode}")
    return (
        a,
        b,
        {
            "fill_mode": fill_mode,
            "seed": seed,
            "generator_contract": generator_contract,
            "a_initial_identity": _operand_identity(a),
            "b_initial_identity": _operand_identity(b),
            "resident_tensors_reused_across_all_arms": True,
        },
    )


class _NominalGapObserver:
    """Run exact telemetry and coarse energy collectors beside one timed arm."""

    def __init__(
        self,
        *,
        logical_device_index: int,
        expected_device_uuid: str,
        interval_s: float,
    ) -> None:
        from core.profiling.gpu_telemetry import (
            _resolve_physical_gpu_index,
            normalize_gpu_uuid,
            query_gpu_telemetry,
        )
        from core.utils import power_sampling

        self.logical_device_index = logical_device_index
        self.physical_device_index = _resolve_physical_gpu_index(logical_device_index)
        normalized_expected_uuid = normalize_gpu_uuid(expected_device_uuid)
        if normalized_expected_uuid is None:
            raise RuntimeError("CUDA device UUID is required for PowerSampler identity")
        power_sampling.ensure_nvml_initialized()
        power_handle = power_sampling.pynvml.nvmlDeviceGetHandleByIndex(self.physical_device_index)
        normalized_power_uuid = normalize_gpu_uuid(
            power_sampling.pynvml.nvmlDeviceGetUUID(power_handle)
        )
        if normalized_power_uuid is None or normalized_power_uuid != normalized_expected_uuid:
            raise RuntimeError(
                "PowerSampler NVML device UUID does not match the selected CUDA device UUID"
            )
        self.expected_cuda_device_uuid = normalized_expected_uuid
        self.power_sampler_nvml_device_uuid = normalized_power_uuid
        self.interval_s = interval_s
        self._query_gpu_telemetry = query_gpu_telemetry
        self._power_sampler = power_sampling.PowerSampler(
            [self.physical_device_index],
            interval=interval_s,
        )
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._error: BaseException | None = None

    def _query_loop(self) -> None:
        sequence = 0
        deadline = time.monotonic()
        try:
            while not self._stop_event.is_set():
                query_start_monotonic_s = time.monotonic()
                query_start_timestamp_utc = _timestamp()
                telemetry = self._query_gpu_telemetry(
                    self.logical_device_index,
                    force_refresh=True,
                )
                query_end_monotonic_s = time.monotonic()
                query_end_timestamp_utc = _timestamp()
                telemetry_timestamp = telemetry.pop("timestamp", None)
                self._samples.append(
                    {
                        "sequence": sequence,
                        "query_start_monotonic_s": query_start_monotonic_s,
                        "query_end_monotonic_s": query_end_monotonic_s,
                        "query_start_timestamp_utc": query_start_timestamp_utc,
                        "query_end_timestamp_utc": query_end_timestamp_utc,
                        "telemetry_timestamp_utc": telemetry_timestamp,
                        **telemetry,
                    }
                )
                sequence += 1
                self._ready_event.set()
                deadline += self.interval_s
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    deadline = time.monotonic()
                    continue
                if self._stop_event.wait(remaining):
                    break
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()
            self._stop_event.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Nominal-gap observer is already running")
        self._samples = []
        self._error = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._power_sampler.start()
        self._thread = threading.Thread(target=self._query_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=max(5.0, self.interval_s * 20.0)):
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            self._power_sampler.stop()
            raise RuntimeError("Timed out waiting for the first telemetry query")
        if self._error is not None:
            self.stop_without_summary()
            raise RuntimeError("Initial GPU telemetry query failed") from self._error

    def stop_without_summary(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._power_sampler.stop()

    def stop(
        self,
        *,
        window_start_monotonic_s: float,
        window_end_monotonic_s: float,
        window_start_epoch_s: float,
        window_end_epoch_s: float,
        minimum_samples: int,
        maximum_gap_s: float,
    ) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is None:
            raise RuntimeError("Nominal-gap observer was not started")
        self._thread.join()
        self._thread = None
        power_capture = self._power_sampler.stop()
        if self._error is not None:
            raise RuntimeError("GPU telemetry sampling failed") from self._error
        power_samples = [
            {
                "timestamp_epoch_s": float(timestamp),
                "physical_gpu_indices": [self.physical_device_index],
                "per_device_power_w": [float(value) for value in per_device],
                "total_power_w": float(total),
            }
            for timestamp, per_device, total in self._power_sampler.samples
        ]
        timeline = summarize_nominal_gap_telemetry(
            self._samples,
            window_start_monotonic_s=window_start_monotonic_s,
            window_end_monotonic_s=window_end_monotonic_s,
            minimum_samples=minimum_samples,
            expected_gpu_index=self.logical_device_index,
            requested_interval_s=self.interval_s,
            maximum_gap_s=maximum_gap_s,
        )
        return {
            "logical_device_index": self.logical_device_index,
            "power_sampler_physical_device_index": self.physical_device_index,
            "device_mapping": (
                "numeric CUDA_VISIBLE_DEVICES mapping resolved by repository helper and "
                "verified against the exact CUDA device UUID"
            ),
            "power_sampler_device_identity": {
                "expected_cuda_device_uuid": self.expected_cuda_device_uuid,
                "observed_nvml_device_uuid": self.power_sampler_nvml_device_uuid,
                "exact_uuid_match": True,
            },
            "all_telemetry_query_samples": self._samples,
            "in_window_telemetry": timeline,
            "power_sampler": {
                "all_samples": power_samples,
                "capture_envelope_metrics": power_capture,
                "accepted_host_phase_energy": integrate_power_samples_over_window(
                    power_samples,
                    window_start_epoch_s=window_start_epoch_s,
                    window_end_epoch_s=window_end_epoch_s,
                ),
                "capture_envelope_avg_watts_is_primary": False,
            },
        }


def _run_fixed_settling_block(
    torch: Any,
    *,
    device: Any,
    operation: Callable[[], None],
    invocations: int,
    minimum_gpu_ms: float,
    range_name: str,
    telemetry_sampling_active: bool,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _nvtx_range

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    torch.cuda.synchronize(device)
    host_start_monotonic_s = time.monotonic()
    host_start_epoch_s = time.time()
    host_start_timestamp_utc = _timestamp()
    with _nvtx_range(torch, range_name):
        start.record(stream)
        for _ in range(invocations):
            operation()
        end.record(stream)
    torch.cuda.synchronize(device)
    elapsed_ms = float(start.elapsed_time(end))
    host_end_monotonic_s = time.monotonic()
    host_end_epoch_s = time.time()
    host_end_timestamp_utc = _timestamp()
    if not math.isfinite(elapsed_ms) or elapsed_ms < minimum_gpu_ms:
        raise RuntimeError(
            f"Exact-work settling block {range_name} recorded {elapsed_ms:.3f}ms, "
            f"below the required {minimum_gpu_ms:.3f}ms"
        )
    return {
        "nvtx_range": range_name,
        "invocations": invocations,
        "gpu_active_ms": elapsed_ms,
        "completion_monotonic_s": host_end_monotonic_s,
        "completion_timestamp_utc": host_end_timestamp_utc,
        "host_timing_envelope": {
            "start_monotonic_s": host_start_monotonic_s,
            "end_monotonic_s": host_end_monotonic_s,
            "start_epoch_s": host_start_epoch_s,
            "end_epoch_s": host_end_epoch_s,
            "start_timestamp_utc": host_start_timestamp_utc,
            "end_timestamp_utc": host_end_timestamp_utc,
            "limitation": (
                "This conservative host envelope brackets event recording, exact-work "
                "submission, and final synchronization. It does not give CUDA events an "
                "absolute host timestamp."
            ),
        },
        "continuous_submission": True,
        "telemetry_sampling_active": telemetry_sampling_active,
    }


def _measure_nominal_gap_window(
    torch: Any,
    *,
    device: Any,
    logical_device_index: int,
    expected_device_uuid: str,
    operation: Callable[[], None],
    settling_operation: Callable[[], None],
    output: Any,
    reference: Any,
    logical_flops: int,
    settling_invocations: int,
    settling_minimum_gpu_ms: float,
    settling_range_name: str,
    repeats: int,
    invocations_per_sample: int,
    minimum_gpu_active_ms: float,
    range_name: str,
    observe: bool,
    telemetry_interval_s: float,
    maximum_telemetry_gap_s: float,
    minimum_telemetry_samples: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _gemm_correctness,
        _nvtx_range,
        _sample_summary,
    )

    output.fill_(float("nan"))
    torch.cuda.synchronize(device)
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    stream = torch.cuda.current_stream(device)
    observer = (
        _NominalGapObserver(
            logical_device_index=logical_device_index,
            expected_device_uuid=expected_device_uuid,
            interval_s=telemetry_interval_s,
        )
        if observe
        else None
    )
    if observer is not None:
        observer.start()
    try:
        settling = _run_fixed_settling_block(
            torch,
            device=device,
            operation=settling_operation,
            invocations=settling_invocations,
            minimum_gpu_ms=settling_minimum_gpu_ms,
            range_name=settling_range_name,
            telemetry_sampling_active=observe,
        )
        window_start_monotonic_s = time.monotonic()
        window_start_epoch_s = time.time()
        window_start_timestamp_utc = _timestamp()
        settle_to_timing_host_gap_ms = (
            window_start_monotonic_s - settling["completion_monotonic_s"]
        ) * 1000.0
        if settle_to_timing_host_gap_ms > 20.0:
            raise RuntimeError(
                f"Host gap after exact-work settling was {settle_to_timing_host_gap_ms:.3f}ms, "
                "above the 20ms limit"
            )
        with _nvtx_range(torch, range_name):
            for start, end in events:
                start.record(stream)
                for _ in range(invocations_per_sample):
                    operation()
                end.record(stream)
        torch.cuda.synchronize(device)
    except BaseException:
        if observer is not None:
            observer.stop_without_summary()
        raise
    window_end_monotonic_s = time.monotonic()
    window_end_epoch_s = time.time()
    window_end_timestamp_utc = _timestamp()
    observation = (
        observer.stop(
            window_start_monotonic_s=window_start_monotonic_s,
            window_end_monotonic_s=window_end_monotonic_s,
            window_start_epoch_s=window_start_epoch_s,
            window_end_epoch_s=window_end_epoch_s,
            minimum_samples=minimum_telemetry_samples,
            maximum_gap_s=maximum_telemetry_gap_s,
        )
        if observer is not None
        else None
    )
    aggregate_ms = [float(start.elapsed_time(end)) for start, end in events]
    if any(not math.isfinite(value) or value <= 0 for value in aggregate_ms):
        raise RuntimeError(f"Invalid CUDA timings for {range_name}: {aggregate_ms}")
    normalized_ms = [value / invocations_per_sample for value in aggregate_ms]
    measured_gpu_active_ms = sum(aggregate_ms)
    if measured_gpu_active_ms < minimum_gpu_active_ms:
        raise RuntimeError(
            f"Timed window {range_name} recorded {measured_gpu_active_ms:.3f}ms, "
            f"below the required {minimum_gpu_active_ms:.3f}ms"
        )
    correctness = _gemm_correctness(torch, output, reference)
    if not correctness["passed"]:
        raise RuntimeError(f"Full-output correctness failed for {range_name}")
    throughput = [logical_flops / (sample / 1e3) / 1e12 for sample in normalized_ms]
    return {
        "nvtx_range": range_name,
        "observer_enabled": observe,
        "preceding_exact_work_settling": settling,
        "settle_to_timing_host_gap_ms": settle_to_timing_host_gap_ms,
        "maximum_settle_to_timing_host_gap_ms": 20.0,
        "host_timing_envelope": {
            "start_monotonic_s": window_start_monotonic_s,
            "end_monotonic_s": window_end_monotonic_s,
            "start_epoch_s": window_start_epoch_s,
            "end_epoch_s": window_end_epoch_s,
            "start_timestamp_utc": window_start_timestamp_utc,
            "end_timestamp_utc": window_end_timestamp_utc,
            "limitation": (
                "CUDA events have no absolute host timestamp. The envelope contains event "
                "submission and final synchronization and does not imply point alignment."
            ),
        },
        "aggregate_timing": _sample_summary(aggregate_ms, "ms_per_aggregate_sample"),
        "normalized_timing": _sample_summary(normalized_ms, "ms_per_operation"),
        "throughput": _sample_summary(throughput, "TFLOP_per_s"),
        "invocations_per_sample": invocations_per_sample,
        "measured_invocations": repeats * invocations_per_sample,
        "measured_gpu_active_ms": measured_gpu_active_ms,
        "pre_timing_output_poison": "all NaN",
        "correctness": correctness,
        "observation": observation,
    }


def _nominal_gap_summary(
    windows: Sequence[dict[str, Any]],
    *,
    nominal_tflop_per_s: float,
    seed: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _sample_summary

    all_rates = [float(sample) for window in windows for sample in window["throughput"]["samples"]]
    window_medians = [float(window["throughput"]["p50"]) for window in windows]
    p50 = statistics.median(window_medians)
    return {
        "statistic": "median of independent sustained-window medians",
        "all_samples": _sample_summary(all_rates, "TFLOP_per_s"),
        "window_medians": _sample_summary(window_medians, "TFLOP_per_s"),
        "window_median_bootstrap_95_ci": bootstrap_median_ci(
            window_medians,
            seed=seed,
        ),
        "p50_tflop_per_s": p50,
        "fraction_of_book_dense_bf16_nominal": p50 / nominal_tflop_per_s,
        "nominal_denominator_tflop_per_s": nominal_tflop_per_s,
        "clock_scaled_nominal_used": False,
    }


def _nominal_gap_steady_state_qualification(
    windows: Sequence[dict[str, Any]],
    *,
    expected_window_count: int,
) -> dict[str, Any]:
    """Fail closed unless every expected sustained window passes the declared gate."""
    if (
        isinstance(expected_window_count, bool)
        or not isinstance(expected_window_count, int)
        or expected_window_count <= 0
    ):
        raise ValueError("expected_window_count must be a positive integer")
    window_results = []
    for sequence, window in enumerate(windows):
        try:
            passed = window["measurement"]["observation"]["in_window_telemetry"][
                "steady_state_diagnostics"
            ]["passes_declared_steady_state_heuristic"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Sustained window {sequence} lacks the required steady-state gate result"
            ) from exc
        if not isinstance(passed, bool):
            raise RuntimeError(
                f"Sustained window {sequence} has a non-boolean steady-state gate result"
            )
        window_results.append(
            {
                "window": int(window.get("window", sequence)),
                "passes_declared_steady_state_gate": passed,
            }
        )
    all_expected_windows_present = len(windows) == expected_window_count
    all_windows_pass = bool(window_results) and all(
        result["passes_declared_steady_state_gate"] for result in window_results
    )
    qualified = all_expected_windows_present and all_windows_pass
    return {
        "expected_window_count": expected_window_count,
        "observed_window_count": len(windows),
        "window_results": window_results,
        "all_expected_windows_present": all_expected_windows_present,
        "all_windows_pass_declared_steady_state_gate": all_windows_pass,
        "sustained_kpi_qualified": qualified,
        "unsettled_windows_retained_for_diagnosis": bool(window_results) and not qualified,
        "acceptance_rule": (
            "every expected sustained window must pass the declared same-window clock "
            "and power no-trend and span gate"
        ),
    }


def run_nominal_gap_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Measure clock and power residency for the fixed dense BF16 workload."""
    _validate_nominal_gap_args(args)
    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _fp32_reference,
        _measure_aggregate_operation,
        _sample_summary,
        _strict_runtime,
    )
    from labs.arithmetic_intensity_b200.verification_tool import _validate_source_commit

    _validate_source_commit(args.source_commit)
    device, properties, environment, provenance, reviewed = _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )
    logical_model = gemm_logical_cost(
        m=GEMM_SIZE,
        n=GEMM_SIZE,
        k=GEMM_SIZE,
        beta_zero=True,
    )
    a, b, operand_provenance = _make_nominal_gap_operands(
        torch,
        device=device,
        seed=args.seed,
        fill_mode=args.fill_mode,
    )
    operand_provenance["realized_full_tensor_summaries"] = {
        "a": _realized_operand_summary(torch, a),
        "b": _realized_operand_summary(torch, b),
    }
    reference = _fp32_reference(torch, a, b)
    output = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
    settling_output = torch.empty_like(output)
    operation = partial(torch.mm, a, b, out=output)
    settling_operation = partial(torch.mm, a, b, out=settling_output)
    telemetry_interval_s = args.telemetry_interval_ms / 1000.0
    maximum_telemetry_gap_s = args.maximum_telemetry_gap_ms / 1000.0
    logical_flops = int(logical_model["flops"])
    virtualized = bool(environment["details"].get("virtualized"))

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        _nominal_gap_progress("short_diagnostic_start", fill_mode=args.fill_mode)
        short_timing = _measure_aggregate_operation(
            torch,
            device=device,
            operation=lambda _invocation: operation(),
            range_name=f"ai_b200_nominal_gap_short_{args.fill_mode}",
            warmups=args.short_warmups,
            repeats=args.short_repeats,
            invocations_per_sample=1,
        )
        short_normalized_ms = short_timing["normalized_timing"]["samples"]
        short_rates = [
            logical_flops / (float(sample) / 1e3) / 1e12 for sample in short_normalized_ms
        ]
        short_diagnostic = {
            "classification": "short warm diagnostic, excluded from sustained KPI",
            "timing": short_timing,
            "throughput": _sample_summary(short_rates, "TFLOP_per_s"),
        }
        short_p50_ms = float(short_timing["normalized_timing"]["p50"])
        settling_invocations = max(
            1,
            math.ceil(args.settling_minimum_gpu_ms * 1.50 / short_p50_ms),
        )
        calibration_invocations = confirmation_invocations_per_sample(
            short_p50_ms,
            repeats=args.calibration_repeats,
            minimum_window_gpu_ms=args.calibration_minimum_gpu_ms,
            margin=1.20,
        )
        sustained_invocations = confirmation_invocations_per_sample(
            short_p50_ms,
            repeats=args.sustained_repeats,
            minimum_window_gpu_ms=args.minimum_window_gpu_ms,
            margin=1.20,
        )

        calibration_rounds = []
        off_medians = []
        on_medians = []
        for round_index in range(args.calibration_rounds):
            order = ("off", "on") if round_index % 2 == 0 else ("on", "off")
            arms: dict[str, Any] = {}
            for arm in order:
                _nominal_gap_progress(
                    "observer_calibration_arm_start",
                    round=round_index,
                    arm=arm,
                )
                measured = _measure_nominal_gap_window(
                    torch,
                    device=device,
                    logical_device_index=args.device,
                    expected_device_uuid=str(properties.uuid),
                    operation=operation,
                    settling_operation=settling_operation,
                    output=output,
                    reference=reference,
                    logical_flops=logical_flops,
                    settling_invocations=settling_invocations,
                    settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                    settling_range_name=(
                        f"ai_b200_nominal_gap_settle_calibration_{round_index}_{arm}"
                    ),
                    repeats=args.calibration_repeats,
                    invocations_per_sample=calibration_invocations,
                    minimum_gpu_active_ms=args.calibration_minimum_gpu_ms,
                    range_name=(f"ai_b200_nominal_gap_observer_calibration_{round_index}_{arm}"),
                    observe=arm == "on",
                    telemetry_interval_s=telemetry_interval_s,
                    maximum_telemetry_gap_s=maximum_telemetry_gap_s,
                    minimum_telemetry_samples=args.minimum_in_window_telemetry_samples,
                )
                arms[arm] = {"measurement": measured}
            off_medians.append(float(arms["off"]["measurement"]["normalized_timing"]["p50"]))
            on_medians.append(float(arms["on"]["measurement"]["normalized_timing"]["p50"]))
            calibration_rounds.append({"round": round_index, "order": list(order), "arms": arms})
        observer_overhead = observer_overhead_summary(
            off_medians,
            on_medians,
            limit_pct=args.observer_overhead_limit_pct,
            seed=args.seed + 71,
        )

        sustained_windows = []
        if observer_overhead["within_observer_overhead_limit"]:
            for window_index in range(args.sustained_windows):
                _nominal_gap_progress(
                    "sustained_window_start",
                    window=window_index,
                    fill_mode=args.fill_mode,
                )
                measured = _measure_nominal_gap_window(
                    torch,
                    device=device,
                    logical_device_index=args.device,
                    expected_device_uuid=str(properties.uuid),
                    operation=operation,
                    settling_operation=settling_operation,
                    output=output,
                    reference=reference,
                    logical_flops=logical_flops,
                    settling_invocations=settling_invocations,
                    settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                    settling_range_name=(f"ai_b200_nominal_gap_settle_window_{window_index}"),
                    repeats=args.sustained_repeats,
                    invocations_per_sample=sustained_invocations,
                    minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                    range_name=(
                        f"ai_b200_nominal_gap_steady_{args.fill_mode}_window_{window_index}"
                    ),
                    observe=True,
                    telemetry_interval_s=telemetry_interval_s,
                    maximum_telemetry_gap_s=maximum_telemetry_gap_s,
                    minimum_telemetry_samples=args.minimum_in_window_telemetry_samples,
                )
                sustained_windows.append({"window": window_index, "measurement": measured})
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    final_operand_identity = {
        "a_final_identity": _operand_identity(a),
        "b_final_identity": _operand_identity(b),
    }
    operands_unchanged = (
        final_operand_identity["a_final_identity"] == operand_provenance["a_initial_identity"]
        and final_operand_identity["b_final_identity"] == operand_provenance["b_initial_identity"]
    )
    if not operands_unchanged:
        raise RuntimeError("The fixed BF16 operands changed during the diagnostic")
    steady_state_qualification = _nominal_gap_steady_state_qualification(
        sustained_windows,
        expected_window_count=args.sustained_windows,
    )
    sustained_kpi_qualified = bool(
        observer_overhead["within_observer_overhead_limit"]
        and steady_state_qualification["sustained_kpi_qualified"]
    )
    sustained_summary = (
        _nominal_gap_summary(
            [window["measurement"] for window in sustained_windows],
            nominal_tflop_per_s=float(reviewed["dense_bf16_tflops"]),
            seed=args.seed + 101,
        )
        if sustained_kpi_qualified
        else None
    )
    if not observer_overhead["within_observer_overhead_limit"]:
        status = "diagnostic_disqualified_observer_overhead"
        sustained_summary_suppression_reason = "observer_overhead_gate_failed"
    elif not steady_state_qualification["sustained_kpi_qualified"]:
        status = "diagnostic_unsettled_no_sustained_kpi"
        sustained_summary_suppression_reason = "steady_state_gate_failed"
    else:
        status = "diagnostic_complete"
        sustained_summary_suppression_reason = None
    fill_is_primary = args.fill_mode == "scaled_normal"
    return {
        "schema_version": 1,
        "kind": "b200_dense_bf16_nominal_gap_timeline_diagnostic",
        "status": status,
        "timestamp_utc": _timestamp(),
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "architectural_model_inputs": {
            "usable_sm_count_from_cuda_device_properties": int(properties.multi_processor_count),
            "dense_bf16_fp32_fma_per_sm_cycle": 4096,
            "dense_bf16_fp32_flop_per_sm_cycle": 8192,
            "mode_boundaries": (
                "native dense BF16 times BF16 with FP32 accumulation, without sparsity, "
                "TF32, FP8, or reduced accumulation"
            ),
            "target_scope": "single exposed GPU0 CUDA context and one exact NVTX kernel range",
            "exact_same_range_cycle_metric": "gpc__cycles_elapsed.avg.per_second",
            "exact_same_range_cycle_metric_value": None,
            "exact_same_range_cycle_metric_unit": None,
            "exact_same_range_clock_definition": None,
            "exact_ncu_model_ready": False,
            "model_formula": (
                "usable SM count times 8192 FLOP per SM cycle times exact same-range "
                "NCU cycles per second"
            ),
            "sampled_nvml_clock_model_computed": False,
            "book_nominal_scaled_by_clock": False,
        },
        "clock_lock": {
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "repository_context_may_set_and_restores_application_clocks": True,
            "power_limit_change_requested": False,
        },
        "configuration": {
            "fill_mode": args.fill_mode,
            "short_warmups": args.short_warmups,
            "short_repeats": args.short_repeats,
            "settling_minimum_gpu_ms": args.settling_minimum_gpu_ms,
            "settling_invocations": settling_invocations,
            "settling_invocation_margin": 1.50,
            "calibration_rounds": args.calibration_rounds,
            "calibration_repeats": args.calibration_repeats,
            "calibration_minimum_gpu_ms": args.calibration_minimum_gpu_ms,
            "calibration_invocations_per_sample": calibration_invocations,
            "timed_window_invocation_margin": 1.20,
            "sustained_windows": args.sustained_windows,
            "sustained_repeats": args.sustained_repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "sustained_invocations_per_sample": sustained_invocations,
            "telemetry_interval_ms": args.telemetry_interval_ms,
            "maximum_telemetry_gap_ms": args.maximum_telemetry_gap_ms,
            "minimum_in_window_telemetry_samples": (args.minimum_in_window_telemetry_samples),
        },
        "workload": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "implementation_id": "torch_mm_default_pytorch_dispatch",
            "kernel_identity": None,
            "kernel_identity_source": "requires matching Nsight Systems trace",
            "logical_model": logical_model,
            "semantic_minimum_requested_bytes": int(logical_model["logical_requested_bytes"]),
            "operands": {
                **operand_provenance,
                **final_operand_identity,
                "identities_unchanged": operands_unchanged,
            },
            "reference": {
                "method": "full output against FP32 matmul of the exact BF16 operands",
                "float32_matmul_precision": "highest",
            },
        },
        "short_diagnostic": short_diagnostic,
        "observer_calibration": {
            "paired_order": "alternating OFF/ON then ON/OFF",
            "rounds": calibration_rounds,
            "summary": observer_overhead,
        },
        "sustained": {
            "classification": (
                "primary fixed-workload diagnostic"
                if fill_is_primary
                else "fill-sensitivity diagnostic excluded from the primary result"
            ),
            "windows": sustained_windows,
            "steady_state_qualification": steady_state_qualification,
            "summary": sustained_summary,
            "summary_suppression_reason": sustained_summary_suppression_reason,
        },
        "claim_boundaries": {
            "replaces_prior_primary_result": False,
            "accepted_optimization_claim": False,
            "fill_is_primary_scaled_normal_workload": fill_is_primary,
            "sustained_kpi_qualified": sustained_kpi_qualified,
            "unsettled_windows_may_support_a_sustained_nominal_fraction": False,
            "short_diagnostic_is_sustained_kpi": False,
            "profiled_timing_is_primary": False,
            "nominal_clock_scaling_performed": False,
            "telemetry_queries_are_exactly_aligned_to_cuda_events": False,
            "power_sampler_is_a_causal_timeline_clock": False,
            "software_power_cap_bit_is_a_universal_stability_gate": False,
            "ncu_tensor_pipeline_counters_collected": False,
            "kernel_identity_resolved": False,
        },
        "limitations": [
            "Host and NVML timestamps bound queries but do not give CUDA events an absolute host time.",
            "The observer is accepted only when its balanced paired calibration stays within the declared limit.",
            "PowerSampler supplies coarse host-phase energy and is not used as the clock timeline.",
            "NCU tensor-pipeline and issue-occupancy counters require a later exact-range profiler run.",
            "One deterministic seed does not establish input-distribution invariance.",
            *(
                [
                    "Uniform[-1,1] changes both distribution and RMS amplitude relative "
                    "to the scaled-normal primary, so it cannot isolate distribution alone."
                ]
                if args.fill_mode == "uniform_minus1_plus1"
                else []
            ),
            *(
                [
                    "RMS matching uses the pre-quantization distribution expectation. "
                    "The full realized-input summaries define the actual BF16 amplitudes."
                ]
                if args.fill_mode == "uniform_rms_matched_to_scaled_normal"
                else []
            ),
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("screen", "confirm", "diagnose-nominal-gap"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--source-commit", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", type=int, default=0)
        command.add_argument("--seed", type=int, default=20_261_014)
    for name in ("screen", "confirm"):
        command = subparsers.choices[name]
        command.add_argument("--precondition-ms", type=float, default=250.0)
    screen = subparsers.choices["screen"]
    screen.add_argument("--screen-warmups", type=int, default=3)
    screen.add_argument("--screen-repeats", type=int, default=7)
    screen.add_argument("--screen-maximum-cv-pct", type=float, default=5.0)
    screen.add_argument("--copy-payload-mib", type=int, default=8192)
    confirm = subparsers.choices["confirm"]
    confirm.add_argument("--screen", type=Path, required=True)
    confirm.add_argument("--expected-screen-lab-tree-sha256", required=True)
    confirm.add_argument("--confirmation-windows", type=int, default=5)
    confirm.add_argument("--confirmation-warmups", type=int, default=3)
    confirm.add_argument("--confirmation-repeats", type=int, default=15)
    confirm.add_argument("--minimum-window-gpu-ms", type=float, default=1000.0)
    nominal_gap = subparsers.choices["diagnose-nominal-gap"]
    nominal_gap.add_argument(
        "--fill-mode",
        choices=(
            "scaled_normal",
            "uniform_minus1_plus1",
            "uniform_rms_matched_to_scaled_normal",
            "zero_power_switching_calibration",
        ),
        default="scaled_normal",
    )
    nominal_gap.add_argument("--short-warmups", type=int, default=3)
    nominal_gap.add_argument("--short-repeats", type=int, default=7)
    nominal_gap.add_argument("--settling-minimum-gpu-ms", type=float, default=3000.0)
    nominal_gap.add_argument("--calibration-rounds", type=int, default=8)
    nominal_gap.add_argument("--calibration-repeats", type=int, default=5)
    nominal_gap.add_argument("--calibration-minimum-gpu-ms", type=float, default=1000.0)
    nominal_gap.add_argument("--sustained-windows", type=int, default=3)
    nominal_gap.add_argument("--sustained-repeats", type=int, default=15)
    nominal_gap.add_argument("--minimum-window-gpu-ms", type=float, default=1000.0)
    nominal_gap.add_argument("--telemetry-interval-ms", type=float, default=100.0)
    nominal_gap.add_argument("--maximum-telemetry-gap-ms", type=float, default=250.0)
    nominal_gap.add_argument(
        "--minimum-in-window-telemetry-samples",
        type=int,
        default=8,
    )
    nominal_gap.add_argument("--observer-overhead-limit-pct", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runners = {
        "screen": run_screen,
        "confirm": run_confirm,
        "diagnose-nominal-gap": run_nominal_gap_diagnostic,
    }
    try:
        _preflight_output(args.output)
        import torch

        with _bf16_fp32_accumulation_policy(torch):
            report = runners[args.command](args)
    except (DiagnosticGateError, RuntimeError, ValueError, FileExistsError) as exc:
        failure = {
            "schema_version": 1,
            "kind": "b200_kernel_optimization_failure",
            "status": "rejected",
            "timestamp_utc": _timestamp(),
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "lab_tree_sha256": _package_sha256(),
        }
        failure_output = None if isinstance(exc, FileExistsError) else args.output
        try:
            _write_report(failure_output, failure)
        except FileExistsError:
            _write_report(None, failure)
        return 2
    try:
        _write_report(args.output, report)
    except FileExistsError as exc:
        _write_report(
            None,
            {
                "schema_version": 1,
                "kind": "b200_kernel_optimization_failure",
                "status": "rejected",
                "timestamp_utc": _timestamp(),
                "command": args.command,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "lab_tree_sha256": _package_sha256(),
            },
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
