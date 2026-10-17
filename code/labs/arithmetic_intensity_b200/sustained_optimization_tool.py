"""Settled cuBLASLt search, confirmation, and peak-calibration controls for B200."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from labs.arithmetic_intensity_b200.ceiling_protocol import (
    bootstrap_median_ci,
    confirmation_invocations_per_sample,
)
from labs.arithmetic_intensity_b200.formulas import gemm_logical_cost
from labs.arithmetic_intensity_b200.kernel_optimization_tool import (
    _arm_summary,
    _make_nominal_gap_operands,
    _measure_nominal_gap_window,
    _nominal_gap_steady_state_qualification,
    _NominalGapObserver,
    _operand_identity,
    _realized_operand_summary,
    _run_fixed_settling_block,
    _timestamp,
    _write_report,
)
from labs.arithmetic_intensity_b200.optimization_benchmarks import GEMM_SIZE
from labs.arithmetic_intensity_b200.optimization_protocol import (
    alternating_pair_order,
    observer_overhead_summary,
    paired_window_summary,
)
from labs.arithmetic_intensity_b200.verification_tool import (
    DiagnosticGateError,
    _bf16_fp32_accumulation_policy,
    _package_sha256,
    _preflight_output,
)

WORKSPACE_CEILINGS_BYTES = (0, 64 << 20, 512 << 20, 2 << 30)
MAX_HEURISTICS_PER_WORKSPACE = 128
DEFAULT_SEARCH_CANDIDATE_CAP = 24
MINIMUM_EFFECT_SIZE_PCT = 5.0
COPY_PAYLOAD_BYTES = 8 << 30
PEAK_PHASE_GPU_ACTIVE_BUDGET_MS = 25.0 * 60.0 * 1000.0

_ALGORITHM_IDENTITY_FIELDS = (
    "heuristic_index",
    "algorithm_id",
    "tile_id",
    "stages_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzle",
    "custom_option",
    "cluster_shape_id",
    "workspace_bytes",
    "waves_count",
    "signature",
    "cluster_shape",
    "cluster_cta_count",
    "cluster_shape_decode_status",
)

PRIOR_FULL_OUTPUT_PASSING_SIGNATURES = (
    "id=66,tile=513,stages=35,split_k=1,reduction=0,swizzle=0,custom=1,cluster=3",
    "id=66,tile=535,stages=35,split_k=1,reduction=0,swizzle=0,custom=3,cluster=6",
    "id=66,tile=447,stages=35,split_k=1,reduction=0,swizzle=0,custom=1,cluster=3",
    "id=66,tile=210,stages=35,split_k=1,reduction=0,swizzle=0,custom=1,cluster=3",
    "id=66,tile=232,stages=35,split_k=1,reduction=0,swizzle=0,custom=3,cluster=6",
    "id=66,tile=587,stages=35,split_k=1,reduction=0,swizzle=0,custom=3,cluster=5",
    "id=66,tile=177,stages=35,split_k=1,reduction=0,swizzle=0,custom=1,cluster=3",
    "id=66,tile=570,stages=35,split_k=1,reduction=0,swizzle=0,custom=3,cluster=5",
)

# CUDA 13.0 cublasLtClusterShape_t values. AUTO has no fixed CTA count.
_CLUSTER_SHAPES = {
    0: ("auto", None),
    2: ("1x1x1", 1),
    3: ("2x1x1", 2),
    4: ("4x1x1", 4),
    5: ("1x2x1", 2),
    6: ("2x2x1", 4),
    7: ("4x2x1", 8),
    8: ("1x4x1", 4),
    9: ("2x4x1", 8),
    10: ("4x4x1", 16),
    11: ("8x1x1", 8),
    12: ("1x8x1", 8),
    13: ("8x2x1", 16),
    14: ("2x8x1", 16),
    15: ("16x1x1", 16),
    16: ("1x16x1", 16),
    17: ("3x1x1", 3),
    18: ("5x1x1", 5),
    19: ("6x1x1", 6),
    20: ("7x1x1", 7),
    21: ("9x1x1", 9),
    22: ("10x1x1", 10),
    23: ("11x1x1", 11),
    24: ("12x1x1", 12),
    25: ("13x1x1", 13),
    26: ("14x1x1", 14),
    27: ("15x1x1", 15),
    28: ("3x2x1", 6),
    29: ("5x2x1", 10),
    30: ("6x2x1", 12),
    31: ("7x2x1", 14),
    32: ("1x3x1", 3),
    33: ("2x3x1", 6),
    34: ("3x3x1", 9),
    35: ("4x3x1", 12),
    36: ("5x3x1", 15),
    37: ("3x4x1", 12),
    38: ("1x5x1", 5),
    39: ("2x5x1", 10),
    40: ("3x5x1", 15),
    41: ("1x6x1", 6),
    42: ("2x6x1", 12),
    43: ("1x7x1", 7),
    44: ("2x7x1", 14),
    45: ("1x9x1", 9),
    46: ("1x10x1", 10),
    47: ("1x11x1", 11),
    48: ("1x12x1", 12),
    49: ("1x13x1", 13),
    50: ("1x14x1", 14),
    51: ("1x15x1", 15),
}


def _cluster_shape_metadata(cluster_shape_id: int) -> dict[str, Any]:
    """Decode the CUDA 13.0 cluster enum without inferring unknown values."""
    shape = _CLUSTER_SHAPES.get(cluster_shape_id)
    if shape is None:
        return {
            "cluster_shape": None,
            "cluster_cta_count": None,
            "cluster_shape_decode_status": "unknown_cuda13_enum_value",
        }
    return {
        "cluster_shape": shape[0],
        "cluster_cta_count": shape[1],
        "cluster_shape_decode_status": "decoded_cuda13_cublaslt_enum",
    }


def _augment_algorithm_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    cluster_shape_id = int(result["cluster_shape_id"])
    result.update(_cluster_shape_metadata(cluster_shape_id))
    return result


def _algorithm_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _ALGORITHM_IDENTITY_FIELDS if field not in metadata]
    if missing:
        raise RuntimeError(f"cuBLASLt algorithm metadata is incomplete: {missing}")
    return {field: metadata[field] for field in _ALGORITHM_IDENTITY_FIELDS}


def _require_algorithm_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected_identity = _algorithm_identity(expected)
    observed_identity = _algorithm_identity(observed)
    if observed_identity != expected_identity:
        raise RuntimeError(f"{context} cuBLASLt algorithm metadata differs from the freeze")
    return observed_identity


def _attach_blas_runtime_identity(
    provenance: dict[str, Any],
    extension: Any,
) -> dict[str, int]:
    identity = dict(extension.blas_runtime_versions())
    expected_fields = {"cublas", "cublaslt"}
    if set(identity) != expected_fields:
        raise RuntimeError("Native extension returned incomplete BLAS runtime identity")
    normalized = {field: int(identity[field]) for field in sorted(expected_fields)}
    if any(value <= 0 for value in normalized.values()):
        raise RuntimeError("BLAS runtime versions must be positive integers")
    provenance["blas_runtime_identity"] = normalized
    return normalized


def _validate_runtime_lineage(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    phase_names: str,
) -> None:
    for field in (
        "python",
        "python_executable",
        "pytorch",
        "pytorch_cuda",
        "precision_policy",
        "nvidia_driver",
        "blas_runtime_identity",
    ):
        if previous.get(field) != current.get(field):
            raise ValueError(f"{phase_names} {field} identities differ")
    for field in (
        "uuid",
        "name",
        "compute_capability",
        "total_memory_bytes",
        "sm_count",
        "l2_cache_bytes",
    ):
        if previous.get("device", {}).get(field) != current.get("device", {}).get(field):
            raise ValueError(f"{phase_names} device {field} identities differ")


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    cta_count = candidate.get("cluster_cta_count")
    if cta_count == 1:
        cta_rank = 0
    elif cta_count == 2:
        cta_rank = 1
    elif isinstance(cta_count, int):
        cta_rank = 2
    else:
        cta_rank = 3
    return (
        cta_rank,
        int(candidate["selected_workspace_ceiling_bytes"]),
        int(candidate["workspace_bytes"]),
        int(candidate["stages_id"]),
        int(candidate["tile_id"]),
        str(candidate["signature"]),
    )


def _bounded_candidate_signatures(
    inventory: Sequence[Mapping[str, Any]],
    *,
    required_signatures: Sequence[str],
    maximum_candidates: int,
) -> list[str]:
    """Retain prior candidates, then fill the cap with 1-CTA and 2-CTA strata first."""
    if maximum_candidates < len(required_signatures):
        raise ValueError("maximum_candidates cannot exclude a required prior candidate")
    by_signature = {str(item["signature"]): item for item in inventory}
    missing = [signature for signature in required_signatures if signature not in by_signature]
    if missing:
        raise RuntimeError(f"Required prior cuBLASLt signatures are unavailable: {missing}")
    selected = list(dict.fromkeys(required_signatures))
    extras = sorted(
        (item for item in inventory if str(item["signature"]) not in selected),
        key=_candidate_sort_key,
    )
    selected.extend(str(item["signature"]) for item in extras[: maximum_candidates - len(selected)])
    return selected


def _parse_workspace_mib(value: str) -> tuple[int, ...]:
    try:
        workspace_mib = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workspace MiB values must be integers") from exc
    if not workspace_mib or any(item < 0 for item in workspace_mib):
        raise argparse.ArgumentTypeError("workspace MiB values must be nonnegative")
    if len(set(workspace_mib)) != len(workspace_mib):
        raise argparse.ArgumentTypeError("workspace MiB values must be unique")
    workspace_bytes = tuple(item << 20 for item in workspace_mib)
    if workspace_bytes != WORKSPACE_CEILINGS_BYTES:
        raise argparse.ArgumentTypeError("workspace union must be exactly 0,64,512,2048 MiB")
    return workspace_bytes


def _parse_fill_modes(value: str) -> tuple[str, ...]:
    allowed = (
        "scaled_normal",
        "uniform_rms_matched_to_scaled_normal",
        "uniform_minus1_plus1",
        "k_order_random_balanced",
        "k_order_positive_then_negative",
        "zero_power_switching_calibration",
    )
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if result != allowed:
        raise argparse.ArgumentTypeError("fill modes and order are frozen as " + ",".join(allowed))
    return result


def _progress(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), file=sys.stderr, flush=True)


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.device != 0:
        raise ValueError("The sustained B200 protocol is authorized for GPU0 only")
    if args.settling_minimum_gpu_ms < 3000.0:
        raise ValueError("Every settled arm requires at least 3000 GPU-active ms")
    if args.minimum_window_gpu_ms < 1000.0:
        raise ValueError("Every timed window requires at least 1000 GPU-active ms")
    if args.repeats < 3:
        raise ValueError("Each timed window requires at least three CUDA-event samples")


def _validate_observer_args(args: argparse.Namespace) -> None:
    if args.calibration_rounds != 8:
        raise ValueError("observer calibration requires exactly eight balanced pairs")
    if not 3 <= args.calibration_repeats <= 30:
        raise ValueError("calibration_repeats must be in [3, 30]")
    if not 8 <= args.minimum_telemetry_samples <= 100:
        raise ValueError("minimum_telemetry_samples must be in [8, 100]")
    if not 50.0 <= args.telemetry_interval_ms <= 250.0:
        raise ValueError("telemetry_interval_ms must be in [50, 250]")
    if not 100.0 <= args.maximum_telemetry_gap_ms <= 250.0:
        raise ValueError("maximum_telemetry_gap_ms must be in [100, 250]")
    if args.maximum_telemetry_gap_ms < args.telemetry_interval_ms:
        raise ValueError("maximum_telemetry_gap_ms must be at least telemetry_interval_ms")
    if args.maximum_telemetry_gap_ms > 2.5 * args.telemetry_interval_ms:
        raise ValueError("maximum_telemetry_gap_ms must be at most 2.5 times the interval")
    conservative_slots = math.floor(args.minimum_window_gpu_ms / args.telemetry_interval_ms) - 1
    if conservative_slots < args.minimum_telemetry_samples:
        raise ValueError("timed window cannot fit the required complete telemetry queries")
    if not 0.0 <= args.observer_overhead_limit_pct <= 5.0:
        raise ValueError("observer_overhead_limit_pct must be in [0, 5]")


def _runtime(args: argparse.Namespace, torch: Any) -> tuple[Any, ...]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _strict_runtime
    from labs.arithmetic_intensity_b200.verification_tool import _validate_source_commit

    _validate_source_commit(args.source_commit)
    return _strict_runtime(
        torch,
        device_index=args.device,
        source_commit=args.source_commit,
        run_id=args.run_id,
    )


def _short_measurement(
    torch: Any,
    *,
    device: Any,
    operation: Callable[[], None],
    output: Any,
    reference: Any,
    logical_flops: int,
    range_name: str,
    repeats: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _gemm_correctness,
        _measure_aggregate_operation,
        _sample_summary,
    )

    output.fill_(float("nan"))
    torch.cuda.synchronize(device)
    timing = _measure_aggregate_operation(
        torch,
        device=device,
        operation=lambda _invocation: operation(),
        range_name=range_name,
        warmups=1,
        repeats=repeats,
        invocations_per_sample=1,
    )
    correctness = _gemm_correctness(torch, output, reference)
    if not correctness["passed"]:
        raise RuntimeError(f"Full-output correctness failed for {range_name}")
    rates = [
        logical_flops / (float(sample) / 1e3) / 1e12
        for sample in timing["normalized_timing"]["samples"]
    ]
    return {
        "classification": "short sizing diagnostic excluded from optimization claims",
        "timing": timing,
        "throughput": _sample_summary(rates, "TFLOP_per_s"),
        "correctness": correctness,
    }


def _short_round_robin_measurement(
    torch: Any,
    *,
    device: Any,
    operation: _RoundRobinOperation,
    bundles: Sequence[Mapping[str, Any]],
    logical_flops: int,
    range_name: str,
    repeats: int,
) -> dict[str, Any]:
    """Size one rotated operation and verify every output written by the diagnostic."""
    from labs.arithmetic_intensity_b200.ceiling_tool import _gemm_correctness

    if 1 + repeats < len(bundles):
        raise ValueError("short sizing diagnostic must write every rotated buffer")
    for bundle in bundles:
        bundle["output"].fill_(float("nan"))
    operation.reset()
    measured = _short_measurement(
        torch,
        device=device,
        operation=operation,
        output=bundles[0]["output"],
        reference=bundles[0]["reference"],
        logical_flops=logical_flops,
        range_name=range_name,
        repeats=repeats,
    )
    per_buffer = [
        {
            "buffer_index": int(bundle["buffer_index"]),
            **_gemm_correctness(torch, bundle["output"], bundle["reference"]),
        }
        for bundle in bundles
    ]
    if not all(item["passed"] for item in per_buffer):
        raise RuntimeError(f"Rotated short sizing correctness failed for {range_name}")
    measured["correctness"] = {
        "method": "full output for every rotated buffer against its independent FP32 oracle",
        "buffer_count": len(bundles),
        "all_buffers_passed": True,
        "per_buffer": per_buffer,
    }
    return measured


def _invocation_counts(
    short_p50_ms: float,
    *,
    repeats: int,
    settling_minimum_gpu_ms: float,
    minimum_window_gpu_ms: float,
) -> tuple[int, int]:
    if not math.isfinite(short_p50_ms) or short_p50_ms <= 0:
        raise ValueError("short_p50_ms must be finite and positive")
    settling_invocations = max(1, math.ceil(settling_minimum_gpu_ms * 1.50 / short_p50_ms))
    timed_invocations = confirmation_invocations_per_sample(
        short_p50_ms,
        repeats=repeats,
        minimum_window_gpu_ms=minimum_window_gpu_ms,
        margin=1.20,
    )
    return settling_invocations, timed_invocations


def _planned_peak_gpu_active_ms(
    sizing_records: Sequence[Mapping[str, Any]],
    *,
    observer_pairs: int,
    observer_repeats: int,
    primary_windows: int,
    primary_repeats: int,
) -> float:
    total_ms = 0.0
    for record in sizing_records:
        short_p50_ms = float(record["short_p50_ms"])
        primary_settling_ms = int(record["settling_invocations"]) * short_p50_ms
        observer_settling_ms = (
            int(record.get("observer_settling_invocations", record["settling_invocations"]))
            * short_p50_ms
        )
        observer_timing_ms = (
            int(record["observer_invocations_per_sample"]) * observer_repeats * short_p50_ms
        )
        primary_timing_ms = (
            int(record["primary_invocations_per_sample"]) * primary_repeats * short_p50_ms
        )
        total_ms += 2 * observer_pairs * (observer_settling_ms + observer_timing_ms)
        total_ms += primary_windows * (primary_settling_ms + primary_timing_ms)
    return total_ms


def _discover_heuristic_union(
    extension: Any,
    *,
    a: Any,
    b: Any,
    output: Any,
    workspace_ceilings: Sequence[int],
    max_heuristics: int,
) -> tuple[dict[int, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plans: dict[int, Any] = {}
    union: dict[str, dict[str, Any]] = {}
    per_workspace = []
    for workspace_bytes in workspace_ceilings:
        plan = extension.Bf16LtMatmulPlan(
            a,
            b,
            output,
            int(workspace_bytes),
            max_heuristics,
        )
        plans[int(workspace_bytes)] = plan
        algorithms = [_augment_algorithm_metadata(item) for item in plan.algorithms()]
        per_workspace.append(
            {
                "workspace_ceiling_bytes": int(workspace_bytes),
                "requested_max_heuristics": max_heuristics,
                "returned_supported_heuristics": len(algorithms),
                "algorithms": algorithms,
            }
        )
        for algorithm in algorithms:
            signature = str(algorithm["signature"])
            availability = {
                "workspace_ceiling_bytes": int(workspace_bytes),
                "heuristic_index": int(algorithm["heuristic_index"]),
            }
            if signature not in union:
                union[signature] = {
                    **algorithm,
                    "available_bindings": [availability],
                    "selected_workspace_ceiling_bytes": int(workspace_bytes),
                    "selected_heuristic_index": int(algorithm["heuristic_index"]),
                }
            else:
                union[signature]["available_bindings"].append(availability)
                current = int(union[signature]["selected_workspace_ceiling_bytes"])
                if int(workspace_bytes) < current:
                    union[signature]["selected_workspace_ceiling_bytes"] = int(workspace_bytes)
                    union[signature]["selected_heuristic_index"] = int(algorithm["heuristic_index"])
    inventory = sorted(union.values(), key=lambda item: str(item["signature"]))
    return plans, inventory, per_workspace


def _algorithm_binding(
    plans: Mapping[int, Any],
    candidate: Mapping[str, Any],
) -> tuple[Any, int]:
    workspace = int(candidate["selected_workspace_ceiling_bytes"])
    return plans[workspace], int(candidate["selected_heuristic_index"])


def _inventory_summary(inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in inventory:
        count = item.get("cluster_cta_count")
        key = "auto_or_unknown" if count is None else str(int(count))
        counts[key] = counts.get(key, 0) + 1
    return {
        "unique_signature_count": len(inventory),
        "cluster_cta_count_histogram": counts,
        "one_cta_signature_count": counts.get("1", 0),
        "two_cta_signature_count": counts.get("2", 0),
        "inventory_scope": (
            "union of every supported heuristic returned by cuBLASLt for the four declared "
            "workspace ceilings, with 128 results requested per ceiling"
        ),
        "exhaustive_supported_algorithm_enumeration": False,
    }


def _search_pair_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changes = []
    for pair in pairs:
        baseline = float(pair["arms"]["baseline"]["throughput"]["p50"])
        candidate = float(pair["arms"]["candidate"]["throughput"]["p50"])
        changes.append((candidate / baseline - 1.0) * 100.0)
    return {
        "statistic": "median of balanced pair candidate/baseline throughput changes",
        "pair_candidate_vs_baseline_pct": changes,
        "median_candidate_vs_baseline_pct": statistics.median(changes),
        "inference_scope": "bounded ranking screen only",
        "confidence_interval": None,
    }


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    """Inventory and rank a bounded union of cuBLASLt heuristics after exact settling."""
    _validate_common_args(args)
    if not 8 <= args.maximum_candidates <= 32:
        raise ValueError("maximum_candidates must be in [8, 32]")
    if args.pair_rounds != 2:
        raise ValueError("the ranking screen requires exactly two balanced pair rounds")
    if args.short_repeats < 3:
        raise ValueError("short_repeats must be at least three")
    if args.max_heuristics != MAX_HEURISTICS_PER_WORKSPACE:
        raise ValueError("max_heuristics must remain 128 for the bounded union")

    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _fp32_reference,
        _make_logical_gemm_operands,
    )
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    device, properties, environment, provenance, reviewed = _runtime(args, torch)
    extension = load_optimization_extension()
    _attach_blas_runtime_identity(provenance, extension)
    logical = gemm_logical_cost(m=GEMM_SIZE, n=GEMM_SIZE, k=GEMM_SIZE, beta_zero=True)
    logical_flops = int(logical["flops"])
    a, b = _make_logical_gemm_operands(torch, device=device, size=GEMM_SIZE, seed=args.seed)
    reference = _fp32_reference(torch, a, b)
    baseline_output = torch.empty_like(a)
    baseline_settling_output = torch.empty_like(a)
    candidate_output = torch.empty_like(a)
    candidate_settling_output = torch.empty_like(a)
    baseline_operation = partial(torch.mm, a, b, out=baseline_output)
    baseline_settling_operation = partial(torch.mm, a, b, out=baseline_settling_output)

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        plans, inventory, per_workspace = _discover_heuristic_union(
            extension,
            a=a,
            b=b,
            output=candidate_output,
            workspace_ceilings=args.workspace_bytes,
            max_heuristics=args.max_heuristics,
        )
        selected_signatures = _bounded_candidate_signatures(
            inventory,
            required_signatures=PRIOR_FULL_OUTPUT_PASSING_SIGNATURES,
            maximum_candidates=args.maximum_candidates,
        )
        by_signature = {str(item["signature"]): item for item in inventory}
        baseline_short = _short_measurement(
            torch,
            device=device,
            operation=baseline_operation,
            output=baseline_output,
            reference=reference,
            logical_flops=logical_flops,
            range_name=f"ai_b200_settled_search_{args.run_id}_baseline_short",
            repeats=args.short_repeats,
        )
        baseline_settling_invocations, baseline_timed_invocations = _invocation_counts(
            float(baseline_short["timing"]["normalized_timing"]["p50"]),
            repeats=args.repeats,
            settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
            minimum_window_gpu_ms=args.minimum_window_gpu_ms,
        )
        candidate_results = []
        for candidate_number, signature in enumerate(selected_signatures):
            metadata = by_signature[signature]
            plan, algorithm_index = _algorithm_binding(plans, metadata)
            candidate_operation = partial(
                plan.run,
                a,
                b,
                candidate_output,
                algorithm_index,
            )
            candidate_settling_operation = partial(
                plan.run,
                a,
                b,
                candidate_settling_output,
                algorithm_index,
            )
            _progress(
                "settled_search_candidate_start",
                candidate=candidate_number,
                candidate_count=len(selected_signatures),
                signature=signature,
            )
            try:
                candidate_short = _short_measurement(
                    torch,
                    device=device,
                    operation=candidate_operation,
                    output=candidate_output,
                    reference=reference,
                    logical_flops=logical_flops,
                    range_name=(
                        f"ai_b200_settled_search_{args.run_id}_candidate_{candidate_number}_short"
                    ),
                    repeats=args.short_repeats,
                )
                candidate_settling_invocations, candidate_timed_invocations = _invocation_counts(
                    float(candidate_short["timing"]["normalized_timing"]["p50"]),
                    repeats=args.repeats,
                    settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                    minimum_window_gpu_ms=args.minimum_window_gpu_ms,
                )
                pairs = []
                for pair_index in range(args.pair_rounds):
                    order = alternating_pair_order(pair_index)
                    arms: dict[str, Any] = {}
                    for arm in order:
                        if arm == "baseline":
                            operation = baseline_operation
                            settling_operation = baseline_settling_operation
                            output = baseline_output
                            settling_invocations = baseline_settling_invocations
                            timed_invocations = baseline_timed_invocations
                        else:
                            operation = candidate_operation
                            settling_operation = candidate_settling_operation
                            output = candidate_output
                            settling_invocations = candidate_settling_invocations
                            timed_invocations = candidate_timed_invocations
                        arms[arm] = _measure_nominal_gap_window(
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
                                f"ai_b200_settled_search_{args.run_id}_c{candidate_number}_"
                                f"p{pair_index}_{arm}_settle"
                            ),
                            repeats=args.repeats,
                            invocations_per_sample=timed_invocations,
                            minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                            range_name=(
                                f"ai_b200_settled_search_{args.run_id}_c{candidate_number}_"
                                f"p{pair_index}_{arm}_timed"
                            ),
                            observe=False,
                            telemetry_interval_s=0.1,
                            maximum_telemetry_gap_s=0.25,
                            minimum_telemetry_samples=8,
                        )
                    pairs.append({"pair": pair_index, "order": list(order), "arms": arms})
                candidate_results.append(
                    {
                        **metadata,
                        "search_index": candidate_number,
                        "status": "measured",
                        "short_sizing_diagnostic": candidate_short,
                        "pairs": pairs,
                        "paired_ranking": _search_pair_summary(pairs),
                        "ranking_eligible": True,
                        "rejection_reasons": [],
                    }
                )
            except RuntimeError as error:
                candidate_results.append(
                    {
                        **metadata,
                        "search_index": candidate_number,
                        "status": "rejected_runtime_error",
                        "ranking_eligible": False,
                        "rejection_reasons": [str(error)],
                    }
                )
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    ranked = [item for item in candidate_results if item["ranking_eligible"]]
    ranked.sort(
        key=lambda item: float(item["paired_ranking"]["median_candidate_vs_baseline_pct"]),
        reverse=True,
    )
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_cublaslt_settled_heuristic_search",
        "status": "search_complete",
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
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "power_limit_change_requested": False,
        },
        "workload": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "input_distribution": "seeded_scaled_normal_bf16",
            "seed": args.seed,
            "logical_model": logical,
            "same_resident_operands_for_every_search_pair": True,
            "cache_policy": "single resident operand set, natural warmed-cache execution",
        },
        "configuration": {
            "workspace_ceilings_bytes": list(args.workspace_bytes),
            "max_heuristics_per_workspace": args.max_heuristics,
            "maximum_candidates": args.maximum_candidates,
            "pair_rounds": args.pair_rounds,
            "repeats": args.repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "settling_minimum_gpu_ms": args.settling_minimum_gpu_ms,
        },
        "heuristic_inventory": {
            "per_workspace": per_workspace,
            "union": inventory,
            "summary": _inventory_summary(inventory),
            "required_prior_signatures": list(PRIOR_FULL_OUTPUT_PASSING_SIGNATURES),
            "all_required_prior_signatures_present": True,
        },
        "baseline_short_sizing_diagnostic": baseline_short,
        "screened_signatures": selected_signatures,
        "candidates": candidate_results,
        "ranking": [str(item["signature"]) for item in ranked],
        "recommended_confirmation_signatures": [str(item["signature"]) for item in ranked[:2]],
        "claim_boundaries": {
            "descriptive_search_only": True,
            "baseline_vs_optimized_speedup_claim": False,
            "all_returned_heuristics_screened": len(selected_signatures) == len(inventory),
            "inventory_is_exhaustive_supported_algorithm_enumeration": False,
            "candidate_selection_uses_zero_or_fill_results": False,
            "observer_enabled_during_primary_search_timing": False,
            "three_second_exact_work_settling_before_every_arm": True,
            "baseline_replacement_accepted": False,
        },
    }


class _RoundRobinOperation:
    def __init__(self, operations: Sequence[Callable[[], None]]) -> None:
        if not operations:
            raise ValueError("round-robin operations must be nonempty")
        self._operations = tuple(operations)
        self._next = 0

    def reset(self) -> None:
        self._next = 0

    def __call__(self) -> None:
        operation = self._operations[self._next]
        self._next = (self._next + 1) % len(self._operations)
        operation()


def _prepare_gemm_bundles(
    torch: Any,
    *,
    device: Any,
    seed: int,
    fill_mode: str,
    buffer_sets: int,
) -> list[dict[str, Any]]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _fp32_reference

    bundles = []
    for buffer_index in range(buffer_sets):
        buffer_seed = seed + buffer_index * 1_000_003
        a, b, operand_provenance = _make_calibration_operands(
            torch,
            device=device,
            seed=buffer_seed,
            fill_mode=fill_mode,
        )
        operand_provenance["realized_full_tensor_summaries"] = {
            "a": _realized_operand_summary(torch, a),
            "b": _realized_operand_summary(torch, b),
        }
        output = torch.empty_like(a)
        settling_output = torch.empty_like(a)
        bundles.append(
            {
                "buffer_index": buffer_index,
                "seed": buffer_seed,
                "a": a,
                "b": b,
                "reference": _fp32_reference(torch, a, b),
                "output": output,
                "settling_output": settling_output,
                "operand_provenance": operand_provenance,
            }
        )
    return bundles


def _two_value_histogram(torch: Any, tensor: Any) -> dict[str, Any]:
    positive_value = float(tensor.abs().max().item())
    positive_count = int(torch.count_nonzero(tensor == positive_value).item())
    negative_count = int(torch.count_nonzero(tensor == -positive_value).item())
    zero_count = int(torch.count_nonzero(tensor == 0).item())
    return {
        "scope": "all_realized_elements",
        "positive_value": positive_value,
        "negative_value": -positive_value,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "total_count": int(tensor.numel()),
        "exactly_two_nonzero_values": positive_count + negative_count == int(tensor.numel()),
    }


def _k_order_control_operands(
    torch: Any,
    *,
    device: Any,
    seed: int,
    permuted: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    magnitude = 1.0 / 16.0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pair_bits = torch.randint(
        0,
        2,
        (GEMM_SIZE // 2,),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    pair_signs = pair_bits.to(torch.bfloat16).mul_(2.0).sub_(1.0)
    k_signs = torch.empty(GEMM_SIZE, dtype=torch.bfloat16, device=device)
    k_signs[0::2].copy_(pair_signs)
    k_signs[1::2].copy_(-pair_signs)
    a_row = k_signs.mul(magnitude)
    a = a_row.view(1, -1).expand(GEMM_SIZE, -1).clone()
    random_bits = torch.randint(
        0,
        2,
        (GEMM_SIZE // 2, GEMM_SIZE),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    paired_rows = random_bits.to(torch.bfloat16).mul_(2.0).sub_(1.0).mul_(magnitude)
    b = torch.empty((GEMM_SIZE, GEMM_SIZE), dtype=torch.bfloat16, device=device)
    b[0::2].copy_(paired_rows)
    b[1::2].copy_(-paired_rows)
    if permuted:
        positive_indices = torch.nonzero(k_signs > 0, as_tuple=False).flatten()
        negative_indices = torch.nonzero(k_signs < 0, as_tuple=False).flatten()
        if positive_indices.numel() != negative_indices.numel():
            raise RuntimeError("K-order control requires exactly balanced A signs")
        permutation = torch.cat((positive_indices, negative_indices))
        a = a[:, permutation].contiguous()
        b = b[permutation, :].contiguous()
        del positive_indices, negative_indices, permutation
    del paired_rows, random_bits, pair_bits, pair_signs, a_row, k_signs
    return (
        a,
        b,
        {
            "distribution": (
                "A repeats one randomly oriented balanced sign-pair K vector and B uses "
                "balanced random-sign row pairs, all at an exact BF16 power-of-two magnitude"
            ),
            "arrangement": (
                "positive A K positions followed by negative A K positions"
                if permuted
                else "randomly oriented adjacent sign pairs"
            ),
            "magnitude": magnitude,
            "common_k_permutation_preserves_exact_matrix_product": True,
            "expected_rms_ratio_to_scaled_normal": magnitude / (GEMM_SIZE**-0.25),
            "amplitude_confounded_relative_to_primary": True,
            "arrangement_pair_isolates_global_value_histogram_and_exact_AB": True,
            "does_not_isolate_operand_from_accumulator_activity": True,
        },
    )


def _make_calibration_operands(
    torch: Any,
    *,
    device: Any,
    seed: int,
    fill_mode: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if fill_mode not in {"k_order_random_balanced", "k_order_positive_then_negative"}:
        return _make_nominal_gap_operands(
            torch,
            device=device,
            seed=seed,
            fill_mode=fill_mode,
        )
    a, b, generator_contract = _k_order_control_operands(
        torch,
        device=device,
        seed=seed,
        permuted=fill_mode == "k_order_positive_then_negative",
    )
    return (
        a,
        b,
        {
            "fill_mode": fill_mode,
            "seed": seed,
            "generator_contract": generator_contract,
            "exact_two_value_histograms": {
                "a": _two_value_histogram(torch, a),
                "b": _two_value_histogram(torch, b),
            },
            "a_initial_identity": _operand_identity(a),
            "b_initial_identity": _operand_identity(b),
            "resident_tensors_reused_across_all_arms": True,
        },
    )


def _resolve_candidate_plan(
    extension: Any,
    *,
    bundle: Mapping[str, Any],
    workspace_ceiling_bytes: int,
    signature: str,
) -> tuple[Any, int, dict[str, Any]]:
    plan = extension.Bf16LtMatmulPlan(
        bundle["a"],
        bundle["b"],
        bundle["output"],
        workspace_ceiling_bytes,
        MAX_HEURISTICS_PER_WORKSPACE,
    )
    algorithms = [_augment_algorithm_metadata(item) for item in plan.algorithms()]
    matches = [item for item in algorithms if item["signature"] == signature]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected candidate signature exactly once at the frozen workspace, found {len(matches)}"
        )
    metadata = matches[0]
    return plan, int(metadata["heuristic_index"]), metadata


def _build_round_robin_operations(
    torch: Any,
    *,
    bundles: Sequence[Mapping[str, Any]],
    implementation: Mapping[str, Any],
    extension: Any | None,
) -> tuple[_RoundRobinOperation, _RoundRobinOperation, dict[str, Any], Any | None]:
    implementation_id = str(implementation["implementation_id"])
    plan = None
    if implementation_id == "torch_mm_out_default_cublas":
        operations = [
            partial(torch.mm, bundle["a"], bundle["b"], out=bundle["output"]) for bundle in bundles
        ]
        settling_operations = [
            partial(
                torch.mm,
                bundle["a"],
                bundle["b"],
                out=bundle["settling_output"],
            )
            for bundle in bundles
        ]
        metadata = {
            "implementation_id": implementation_id,
            "api_path": "torch.mm(..., out=...) with default PyTorch BLAS dispatch",
            "algorithm_signature": None,
            "workspace_ceiling_bytes": 0,
            "kernel_identity": None,
        }
    elif implementation_id == "direct_cublaslt_frozen_algorithm":
        if extension is None:
            raise RuntimeError("The direct cuBLASLt implementation requires the extension")
        signature = str(implementation["algorithm_signature"])
        workspace = int(implementation["workspace_ceiling_bytes"])
        plan, algorithm_index, algorithm = _resolve_candidate_plan(
            extension,
            bundle=bundles[0],
            workspace_ceiling_bytes=workspace,
            signature=signature,
        )
        if "algorithm" in implementation:
            _require_algorithm_identity(
                implementation["algorithm"],
                algorithm,
                context="resolved frozen candidate",
            )
        operations = [
            partial(
                plan.run,
                bundle["a"],
                bundle["b"],
                bundle["output"],
                algorithm_index,
            )
            for bundle in bundles
        ]
        settling_operations = [
            partial(
                plan.run,
                bundle["a"],
                bundle["b"],
                bundle["settling_output"],
                algorithm_index,
            )
            for bundle in bundles
        ]
        metadata = {
            "implementation_id": implementation_id,
            "api_path": "persistent direct cublasLtMatmul plan",
            "algorithm_signature": signature,
            "workspace_ceiling_bytes": workspace,
            "algorithm": algorithm,
            "kernel_identity": None,
        }
    else:
        raise RuntimeError(f"Unsupported frozen implementation: {implementation_id}")
    return (
        _RoundRobinOperation(operations),
        _RoundRobinOperation(settling_operations),
        metadata,
        plan,
    )


def _measure_round_robin_gemm_window(
    torch: Any,
    *,
    device: Any,
    logical_device_index: int,
    expected_device_uuid: str,
    operation: _RoundRobinOperation,
    settling_operation: _RoundRobinOperation,
    bundles: Sequence[Mapping[str, Any]],
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
    from labs.arithmetic_intensity_b200.ceiling_tool import _gemm_correctness

    measured_invocations = repeats * invocations_per_sample
    if measured_invocations < len(bundles):
        raise RuntimeError("Timed window cannot write every rotated output")
    for bundle in bundles:
        bundle["output"].fill_(float("nan"))
    torch.cuda.synchronize(device)
    operation.reset()
    settling_operation.reset()
    measured = _measure_nominal_gap_window(
        torch,
        device=device,
        logical_device_index=logical_device_index,
        expected_device_uuid=expected_device_uuid,
        operation=operation,
        settling_operation=settling_operation,
        output=bundles[0]["output"],
        reference=bundles[0]["reference"],
        logical_flops=logical_flops,
        settling_invocations=settling_invocations,
        settling_minimum_gpu_ms=settling_minimum_gpu_ms,
        settling_range_name=settling_range_name,
        repeats=repeats,
        invocations_per_sample=invocations_per_sample,
        minimum_gpu_active_ms=minimum_gpu_active_ms,
        range_name=range_name,
        observe=observe,
        telemetry_interval_s=telemetry_interval_s,
        maximum_telemetry_gap_s=maximum_telemetry_gap_s,
        minimum_telemetry_samples=minimum_telemetry_samples,
    )
    correctness_per_buffer = [
        {
            "buffer_index": int(bundle["buffer_index"]),
            **_gemm_correctness(torch, bundle["output"], bundle["reference"]),
        }
        for bundle in bundles
    ]
    if not all(item["passed"] for item in correctness_per_buffer):
        raise RuntimeError(f"Full-output correctness failed for a rotated buffer in {range_name}")
    measured["correctness"] = {
        "method": "full output for every timed buffer against its independent FP32 oracle",
        "buffer_count": len(bundles),
        "all_buffers_passed": True,
        "per_buffer": correctness_per_buffer,
    }
    measured["buffer_rotation"] = {
        "buffer_count": len(bundles),
        "round_robin": len(bundles) > 1,
        "every_timed_output_checked": True,
        "natural_warmed_cache_without_explicit_flush": True,
    }
    return measured


def _run_observer_calibration(
    torch: Any,
    *,
    args: argparse.Namespace,
    device: Any,
    properties: Any,
    operation: _RoundRobinOperation,
    settling_operation: _RoundRobinOperation,
    bundles: Sequence[Mapping[str, Any]],
    logical_flops: int,
    settling_invocations: int,
    timed_invocations: int,
    range_prefix: str,
) -> dict[str, Any]:
    rounds = []
    off_medians = []
    on_medians = []
    for round_index in range(args.calibration_rounds):
        order = ("off", "on") if round_index % 2 == 0 else ("on", "off")
        arms: dict[str, Any] = {}
        for arm in order:
            measured = _measure_round_robin_gemm_window(
                torch,
                device=device,
                logical_device_index=args.device,
                expected_device_uuid=str(properties.uuid),
                operation=operation,
                settling_operation=settling_operation,
                bundles=bundles,
                logical_flops=logical_flops,
                settling_invocations=settling_invocations,
                settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                settling_range_name=f"{range_prefix}_observer_{round_index}_{arm}_settle",
                repeats=args.calibration_repeats,
                invocations_per_sample=timed_invocations,
                minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                range_name=f"{range_prefix}_observer_{round_index}_{arm}_timed",
                observe=arm == "on",
                telemetry_interval_s=args.telemetry_interval_ms / 1000.0,
                maximum_telemetry_gap_s=args.maximum_telemetry_gap_ms / 1000.0,
                minimum_telemetry_samples=args.minimum_telemetry_samples,
            )
            arms[arm] = measured
        off_medians.append(float(arms["off"]["normalized_timing"]["p50"]))
        on_medians.append(float(arms["on"]["normalized_timing"]["p50"]))
        rounds.append({"round": round_index, "order": list(order), "arms": arms})
    summary = observer_overhead_summary(
        off_medians,
        on_medians,
        limit_pct=args.observer_overhead_limit_pct,
        seed=args.seed + 71,
    )
    return {"rounds": rounds, "summary": summary}


def _validate_search_lineage(
    search: Mapping[str, Any],
    *,
    expected_lab_tree_sha256: str,
    source_commit: str,
    current_provenance: Mapping[str, Any],
) -> None:
    if search.get("schema_version") != 1:
        raise ValueError("search schema version must be 1")
    if search.get("kind") != "b200_cublaslt_settled_heuristic_search":
        raise ValueError("confirmation requires the settled cuBLASLt search")
    if search.get("status") != "search_complete":
        raise ValueError("confirmation requires a completed search")
    provenance = search.get("provenance", {})
    if provenance.get("lab_tree_sha256") != expected_lab_tree_sha256:
        raise ValueError("search lab-tree SHA-256 differs from the explicit expected digest")
    if provenance.get("lab_tree_sha256") != current_provenance.get("lab_tree_sha256"):
        raise ValueError("search and confirmation lab-tree SHA-256 differ")
    if provenance.get("source_commit") != source_commit:
        raise ValueError("search and confirmation source commits differ")
    _validate_runtime_lineage(
        provenance,
        current_provenance,
        phase_names="search and confirmation",
    )


def run_confirm(args: argparse.Namespace) -> dict[str, Any]:
    """Confirm one search candidate with five fresh-operand, observer-qualified pairs."""
    _validate_common_args(args)
    _validate_observer_args(args)
    if args.windows != 5:
        raise ValueError("final confirmation requires exactly five independent windows")
    if args.calibration_rounds != 8:
        raise ValueError("observer calibration requires exactly eight balanced pairs")

    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    search = json.loads(args.search.read_text())
    device, properties, environment, provenance, reviewed = _runtime(args, torch)
    extension = load_optimization_extension()
    _attach_blas_runtime_identity(provenance, extension)
    _validate_search_lineage(
        search,
        expected_lab_tree_sha256=args.expected_search_lab_tree_sha256,
        source_commit=args.source_commit,
        current_provenance=provenance,
    )
    matches = [
        item
        for item in search.get("candidates", [])
        if item.get("signature") == args.candidate_signature
    ]
    if len(matches) != 1 or matches[0].get("ranking_eligible") is not True:
        raise ValueError("candidate signature must identify one ranking-eligible search result")
    search_candidate = matches[0]
    implementation = {
        "implementation_id": "direct_cublaslt_frozen_algorithm",
        "algorithm_signature": args.candidate_signature,
        "workspace_ceiling_bytes": int(search_candidate["selected_workspace_ceiling_bytes"]),
        "algorithm": _algorithm_identity(search_candidate),
    }
    logical = gemm_logical_cost(m=GEMM_SIZE, n=GEMM_SIZE, k=GEMM_SIZE, beta_zero=True)
    logical_flops = int(logical["flops"])
    baseline_short_ms = float(
        search["baseline_short_sizing_diagnostic"]["timing"]["normalized_timing"]["p50"]
    )
    candidate_short_ms = float(
        search_candidate["short_sizing_diagnostic"]["timing"]["normalized_timing"]["p50"]
    )
    baseline_counts = _invocation_counts(
        baseline_short_ms,
        repeats=args.repeats,
        settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
        minimum_window_gpu_ms=args.minimum_window_gpu_ms,
    )
    candidate_counts = _invocation_counts(
        candidate_short_ms,
        repeats=args.repeats,
        settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
        minimum_window_gpu_ms=args.minimum_window_gpu_ms,
    )
    baseline_observer_counts = _invocation_counts(
        baseline_short_ms,
        repeats=args.calibration_repeats,
        settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
        minimum_window_gpu_ms=args.minimum_window_gpu_ms,
    )

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        calibration_bundles = _prepare_gemm_bundles(
            torch,
            device=device,
            seed=args.seed + 80_000_239,
            fill_mode="scaled_normal",
            buffer_sets=1,
        )
        baseline_calibration_operation, baseline_calibration_settling, _, _ = (
            _build_round_robin_operations(
                torch,
                bundles=calibration_bundles,
                implementation={"implementation_id": "torch_mm_out_default_cublas"},
                extension=None,
            )
        )
        observer_calibration = _run_observer_calibration(
            torch,
            args=args,
            device=device,
            properties=properties,
            operation=baseline_calibration_operation,
            settling_operation=baseline_calibration_settling,
            bundles=calibration_bundles,
            logical_flops=logical_flops,
            settling_invocations=baseline_observer_counts[0],
            timed_invocations=baseline_observer_counts[1],
            range_prefix=f"ai_b200_sustained_confirm_{args.run_id}_calibration",
        )
        if not observer_calibration["summary"]["within_observer_overhead_limit"]:
            raise RuntimeError("Observer overhead calibration exceeded the declared limit")

        windows = []
        for window_index in range(args.windows):
            window_seed = args.seed + window_index * 1_000_003
            bundles = _prepare_gemm_bundles(
                torch,
                device=device,
                seed=window_seed,
                fill_mode="scaled_normal",
                buffer_sets=1,
            )
            baseline_operation, baseline_settling, baseline_metadata, _ = (
                _build_round_robin_operations(
                    torch,
                    bundles=bundles,
                    implementation={"implementation_id": "torch_mm_out_default_cublas"},
                    extension=None,
                )
            )
            candidate_operation, candidate_settling, candidate_metadata, candidate_plan = (
                _build_round_robin_operations(
                    torch,
                    bundles=bundles,
                    implementation=implementation,
                    extension=extension,
                )
            )
            order = alternating_pair_order(window_index)
            arms: dict[str, Any] = {}
            for arm in order:
                if arm == "baseline":
                    operation = baseline_operation
                    settling_operation = baseline_settling
                    counts = baseline_counts
                else:
                    operation = candidate_operation
                    settling_operation = candidate_settling
                    counts = candidate_counts
                arms[arm] = _measure_round_robin_gemm_window(
                    torch,
                    device=device,
                    logical_device_index=args.device,
                    expected_device_uuid=str(properties.uuid),
                    operation=operation,
                    settling_operation=settling_operation,
                    bundles=bundles,
                    logical_flops=logical_flops,
                    settling_invocations=counts[0],
                    settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                    settling_range_name=(
                        f"ai_b200_sustained_confirm_{args.run_id}_w{window_index}_{arm}_settle"
                    ),
                    repeats=args.repeats,
                    invocations_per_sample=counts[1],
                    minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                    range_name=(
                        f"ai_b200_sustained_confirm_{args.run_id}_w{window_index}_{arm}_timed"
                    ),
                    observe=True,
                    telemetry_interval_s=args.telemetry_interval_ms / 1000.0,
                    maximum_telemetry_gap_s=args.maximum_telemetry_gap_ms / 1000.0,
                    minimum_telemetry_samples=args.minimum_telemetry_samples,
                )
            windows.append(
                {
                    "window": window_index,
                    "seed": window_seed,
                    "order": list(order),
                    "arms": arms,
                    "baseline_implementation": baseline_metadata,
                    "candidate_implementation": candidate_metadata,
                }
            )
            del candidate_plan, bundles
            torch.cuda.empty_cache()
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    baseline_windows = [window["arms"]["baseline"] for window in windows]
    candidate_windows = [window["arms"]["candidate"] for window in windows]
    baseline_gate = _nominal_gap_steady_state_qualification(
        [
            {"window": index, "measurement": measurement}
            for index, measurement in enumerate(baseline_windows)
        ],
        expected_window_count=args.windows,
    )
    candidate_gate = _nominal_gap_steady_state_qualification(
        [
            {"window": index, "measurement": measurement}
            for index, measurement in enumerate(candidate_windows)
        ],
        expected_window_count=args.windows,
    )
    measurement_qualified = bool(
        baseline_gate["sustained_kpi_qualified"] and candidate_gate["sustained_kpi_qualified"]
    )
    paired = (
        paired_window_summary(
            [float(item["throughput"]["p50"]) for item in baseline_windows],
            [float(item["throughput"]["p50"]) for item in candidate_windows],
            seed=args.seed + 47,
        )
        if measurement_qualified
        else None
    )
    eligible_for_harness = bool(
        measurement_qualified
        and paired is not None
        and paired["verdict"] == "improved"
        and float(paired["candidate_vs_baseline_pct"]) >= MINIMUM_EFFECT_SIZE_PCT
    )
    frozen_calibration_implementation = (
        (
            implementation
            if eligible_for_harness
            else {"implementation_id": "torch_mm_out_default_cublas"}
        )
        if measurement_qualified
        else None
    )
    baseline_summary = (
        _arm_summary(
            windows=baseline_windows,
            metric_key="throughput",
            unit="TFLOP_per_s",
            seed=args.seed + 101,
        )
        if measurement_qualified
        else None
    )
    candidate_summary = (
        _arm_summary(
            windows=candidate_windows,
            metric_key="throughput",
            unit="TFLOP_per_s",
            seed=args.seed + 103,
        )
        if measurement_qualified
        else None
    )
    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_cublaslt_settled_interleaved_confirmation",
        "status": "confirmation_complete" if measurement_qualified else "confirmation_disqualified",
        "timestamp_utc": _timestamp(),
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "search_lineage": {
            "path": str(args.search),
            "run_id": search["provenance"]["run_id"],
            "lab_tree_sha256": search["provenance"]["lab_tree_sha256"],
            "candidate_signature": args.candidate_signature,
        },
        "clock_lock": {
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "power_limit_change_requested": False,
        },
        "workload": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "input_distribution": "seeded_scaled_normal_bf16",
            "logical_model": logical,
            "fresh_operands_per_window": True,
            "same_operands_within_each_baseline_candidate_pair": True,
            "cache_policy": "single resident operand set per fresh window",
        },
        "configuration": {
            "windows": args.windows,
            "repeats": args.repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "settling_minimum_gpu_ms": args.settling_minimum_gpu_ms,
            "calibration_rounds": args.calibration_rounds,
            "observer_overhead_limit_pct": args.observer_overhead_limit_pct,
            "baseline_primary_invocation_counts": {
                "settling": baseline_counts[0],
                "per_sample": baseline_counts[1],
            },
            "candidate_primary_invocation_counts": {
                "settling": candidate_counts[0],
                "per_sample": candidate_counts[1],
            },
            "baseline_observer_invocation_counts": {
                "settling": baseline_observer_counts[0],
                "per_sample": baseline_observer_counts[1],
            },
        },
        "observer_calibration": observer_calibration,
        "windows": windows,
        "baseline": {
            "implementation": {"implementation_id": "torch_mm_out_default_cublas"},
            "summary": baseline_summary,
            "steady_state_qualification": baseline_gate,
        },
        "candidate": {
            "implementation": implementation,
            "summary": candidate_summary,
            "steady_state_qualification": candidate_gate,
        },
        "paired_interleaved": paired,
        "optimization_decision": {
            "measurement_qualified": measurement_qualified,
            "minimum_effect_size_pct": MINIMUM_EFFECT_SIZE_PCT,
            "eligible_for_harness_promotion": eligible_for_harness,
            "accepted_for_baseline_replacement": False,
            "strict_harness_still_required": True,
            "frozen_implementation_for_peak_calibration": frozen_calibration_implementation,
        },
        "claim_boundaries": {
            "comparison_is_equivalent_work": True,
            "profiled_timing_is_primary": False,
            "baseline_replacement_accepted": False,
            "no_speedup_claim_without_strict_harness": True,
            "nominal_clock_scaling_performed": False,
        },
    }


def _validate_confirmation_lineage(
    confirmation: Mapping[str, Any],
    *,
    expected_lab_tree_sha256: str,
    source_commit: str,
    current_provenance: Mapping[str, Any],
) -> None:
    if confirmation.get("schema_version") != 1:
        raise ValueError("confirmation schema version must be 1")
    if confirmation.get("kind") != "b200_cublaslt_settled_interleaved_confirmation":
        raise ValueError("peak calibration requires the settled interleaved confirmation")
    if confirmation.get("status") != "confirmation_complete":
        raise ValueError("peak calibration requires a complete qualified confirmation")
    provenance = confirmation.get("provenance", {})
    if provenance.get("lab_tree_sha256") != expected_lab_tree_sha256:
        raise ValueError("confirmation lab-tree SHA-256 differs from the expected digest")
    if provenance.get("lab_tree_sha256") != current_provenance.get("lab_tree_sha256"):
        raise ValueError("confirmation and calibration lab-tree SHA-256 differ")
    if provenance.get("source_commit") != source_commit:
        raise ValueError("confirmation and calibration source commits differ")
    _validate_runtime_lineage(
        provenance,
        current_provenance,
        phase_names="confirmation and calibration",
    )


def _confirmed_implementation_timing_ms(
    confirmation: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> float:
    arm = (
        "candidate"
        if implementation["implementation_id"] == "direct_cublaslt_frozen_algorithm"
        else "baseline"
    )
    samples = [
        float(window["arms"][arm]["normalized_timing"]["p50"]) for window in confirmation["windows"]
    ]
    if len(samples) != 5:
        raise ValueError("confirmation must contain exactly five timing windows")
    return statistics.median(samples)


def _timing_only_window_summary(
    windows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _sample_summary

    samples = [
        float(sample) for window in windows for sample in window["normalized_timing"]["samples"]
    ]
    medians = [float(window["normalized_timing"]["p50"]) for window in windows]
    return {
        "statistic": "median of five settled-window latency medians",
        "all_samples": _sample_summary(samples, "ms_per_operation"),
        "window_medians": _sample_summary(medians, "ms_per_operation"),
        "window_median_bootstrap_95_ci": bootstrap_median_ci(medians, seed=seed),
    }


def _balanced_fill_order(fill_modes: Sequence[str], window_index: int) -> list[str]:
    """Return five rows from a six-treatment Williams design.

    Each treatment occupies five distinct ordinal positions. The two K-order
    treatments occur in both directions, three versus two times, across the
    five retained rows.
    """
    if len(fill_modes) != 6:
        raise ValueError("the Williams fill design requires exactly six treatments")
    if isinstance(window_index, bool) or not isinstance(window_index, int):
        raise ValueError("window_index must be an integer")
    if not 0 <= window_index < 5:
        raise ValueError("the frozen Williams subset contains exactly five rows")
    base = (0, 1, 5, 2, 4, 3)
    return [fill_modes[(position + window_index) % 6] for position in base]


def _attach_coarse_energy_per_gemm(measurement: dict[str, Any]) -> None:
    energy = measurement["observation"]["power_sampler"]["accepted_host_phase_energy"]
    measured_invocations = int(measurement["measured_invocations"])
    if measured_invocations <= 0:
        raise RuntimeError("Measured invocation count must be positive for energy attribution")
    measurement["coarse_board_energy"] = {
        "host_phase_joules_per_gemm": float(energy["energy_joules"]) / measured_invocations,
        "time_weighted_host_phase_power_w": float(energy["time_weighted_power_w"]),
        "scope": "coarse board energy over the accepted host timing envelope",
        "is_exact_per_kernel_energy": False,
    }


def run_peak_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run rotated-buffer nonzero and low-switching fills on one frozen implementation."""
    _validate_common_args(args)
    _validate_observer_args(args)
    if args.windows != 5:
        raise ValueError("peak calibration requires exactly five settled windows per fill")
    if args.calibration_rounds != 8:
        raise ValueError("observer calibration requires exactly eight balanced pairs")
    if args.buffer_sets != 2:
        raise ValueError("peak calibration requires exactly two rotated resident buffer sets")

    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    confirmation = json.loads(args.confirmation.read_text())
    device, properties, environment, provenance, reviewed = _runtime(args, torch)
    runtime_extension = load_optimization_extension()
    _attach_blas_runtime_identity(provenance, runtime_extension)
    _validate_confirmation_lineage(
        confirmation,
        expected_lab_tree_sha256=args.expected_confirmation_lab_tree_sha256,
        source_commit=args.source_commit,
        current_provenance=provenance,
    )
    implementation = dict(
        confirmation["optimization_decision"]["frozen_implementation_for_peak_calibration"]
    )
    execution_extension = (
        runtime_extension
        if implementation["implementation_id"] == "direct_cublaslt_frozen_algorithm"
        else None
    )
    confirmed_p50_ms = _confirmed_implementation_timing_ms(confirmation, implementation)
    logical = gemm_logical_cost(m=GEMM_SIZE, n=GEMM_SIZE, k=GEMM_SIZE, beta_zero=True)
    logical_flops = int(logical["flops"])
    fill_results: dict[str, Any] = {}
    fill_states: dict[str, dict[str, Any]] = {}
    interleaved_orders = []

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        arrangement_seed = args.seed + 50_000_101

        # Build all payloads and prove the histogram/output-preserving K permutation
        # before any observer or primary timing is allowed to begin.
        for fill_index, fill_mode in enumerate(args.fill_modes):
            _progress("peak_calibration_fill_prepare", fill_mode=fill_mode)
            fill_seed = (
                arrangement_seed
                if fill_mode in {"k_order_random_balanced", "k_order_positive_then_negative"}
                else args.seed + fill_index * 10_000_019
            )
            bundles = _prepare_gemm_bundles(
                torch,
                device=device,
                seed=fill_seed,
                fill_mode=fill_mode,
                buffer_sets=args.buffer_sets,
            )
            operation, settling_operation, implementation_metadata, plan = (
                _build_round_robin_operations(
                    torch,
                    bundles=bundles,
                    implementation=implementation,
                    extension=execution_extension,
                )
            )
            fill_states[fill_mode] = {
                "seed": fill_seed,
                "bundles": bundles,
                "operation": operation,
                "settling_operation": settling_operation,
                "implementation_metadata": implementation_metadata,
                "plan": plan,
                "measured_logical_flops": (
                    0 if fill_mode == "zero_power_switching_calibration" else logical_flops
                ),
                "windows": [],
            }

        original = fill_states["k_order_random_balanced"]["bundles"]
        permuted = fill_states["k_order_positive_then_negative"]["bundles"]
        arrangement_checks = []
        for buffer_index, (left, right) in enumerate(zip(original, permuted, strict=True)):
            a_histograms_match = (
                left["operand_provenance"]["exact_two_value_histograms"]["a"]
                == right["operand_provenance"]["exact_two_value_histograms"]["a"]
            )
            b_histograms_match = (
                left["operand_provenance"]["exact_two_value_histograms"]["b"]
                == right["operand_provenance"]["exact_two_value_histograms"]["b"]
            )
            fp32_references_equal = bool(torch.equal(left["reference"], right["reference"]))
            shapes_and_strides_match = bool(
                left["a"].shape == right["a"].shape
                and left["b"].shape == right["b"].shape
                and left["a"].stride() == right["a"].stride()
                and left["b"].stride() == right["b"].stride()
            )
            alignment_classes = {
                "a_random_data_ptr_mod_256": int(left["a"].data_ptr()) % 256,
                "a_clustered_data_ptr_mod_256": int(right["a"].data_ptr()) % 256,
                "b_random_data_ptr_mod_256": int(left["b"].data_ptr()) % 256,
                "b_clustered_data_ptr_mod_256": int(right["b"].data_ptr()) % 256,
            }
            alignment_classes_match = bool(
                alignment_classes["a_random_data_ptr_mod_256"]
                == alignment_classes["a_clustered_data_ptr_mod_256"]
                and alignment_classes["b_random_data_ptr_mod_256"]
                == alignment_classes["b_clustered_data_ptr_mod_256"]
            )
            if not all(
                (
                    a_histograms_match,
                    b_histograms_match,
                    fp32_references_equal,
                    shapes_and_strides_match,
                    alignment_classes_match,
                )
            ):
                raise RuntimeError("K-order arrangement controls failed the pre-timing gate")
            arrangement_checks.append(
                {
                    "buffer_index": buffer_index,
                    "a_exact_histograms_match": a_histograms_match,
                    "b_exact_histograms_match": b_histograms_match,
                    "full_fp32_references_equal": fp32_references_equal,
                    "shapes_and_strides_match": shapes_and_strides_match,
                    "allocation_alignment_classes_match": alignment_classes_match,
                    "allocation_alignment_classes": alignment_classes,
                    "output_elements_checked": int(left["reference"].numel()),
                }
            )

        sizing_records = []
        for fill_mode in args.fill_modes:
            state = fill_states[fill_mode]
            short = _short_round_robin_measurement(
                torch,
                device=device,
                operation=state["operation"],
                bundles=state["bundles"],
                logical_flops=int(state["measured_logical_flops"]),
                range_name=f"ai_b200_peak_{args.run_id}_{fill_mode}_short",
                repeats=3,
            )
            short_p50_ms = float(short["timing"]["normalized_timing"]["p50"])
            primary_counts = _invocation_counts(
                short_p50_ms,
                repeats=args.repeats,
                settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                minimum_window_gpu_ms=args.minimum_window_gpu_ms,
            )
            observer_counts = _invocation_counts(
                short_p50_ms,
                repeats=args.calibration_repeats,
                settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                minimum_window_gpu_ms=args.minimum_window_gpu_ms,
            )
            sizing = {
                "fill_mode": fill_mode,
                "short_p50_ms": short_p50_ms,
                "short_sizing_diagnostic": short,
                "settling_invocations": primary_counts[0],
                "primary_invocations_per_sample": primary_counts[1],
                "observer_settling_invocations": observer_counts[0],
                "observer_invocations_per_sample": observer_counts[1],
            }
            state["sizing"] = sizing
            sizing_records.append(sizing)

        planned_gpu_active_ms = _planned_peak_gpu_active_ms(
            sizing_records,
            observer_pairs=args.calibration_rounds,
            observer_repeats=args.calibration_repeats,
            primary_windows=args.windows,
            primary_repeats=args.repeats,
        )
        if planned_gpu_active_ms > PEAK_PHASE_GPU_ACTIVE_BUDGET_MS:
            raise RuntimeError(
                "Planned peak-calibration observer and primary GPU-active time exceeds "
                f"the {PEAK_PHASE_GPU_ACTIVE_BUDGET_MS / 60000.0:.1f}-minute in-phase cap"
            )

        for fill_mode in args.fill_modes:
            state = fill_states[fill_mode]
            sizing = state["sizing"]
            observer_calibration = _run_observer_calibration(
                torch,
                args=args,
                device=device,
                properties=properties,
                operation=state["operation"],
                settling_operation=state["settling_operation"],
                bundles=state["bundles"],
                logical_flops=int(state["measured_logical_flops"]),
                settling_invocations=int(sizing["observer_settling_invocations"]),
                timed_invocations=int(sizing["observer_invocations_per_sample"]),
                range_prefix=f"ai_b200_peak_{args.run_id}_{fill_mode}_calibration",
            )
            if not observer_calibration["summary"]["within_observer_overhead_limit"]:
                raise RuntimeError(f"Observer overhead calibration failed for {fill_mode}")
            state["observer_calibration"] = observer_calibration

        for window_index in range(args.windows):
            order = _balanced_fill_order(args.fill_modes, window_index)
            interleaved_orders.append({"window": window_index, "order": order})
            for fill_mode in order:
                state = fill_states[fill_mode]
                sizing = state["sizing"]
                measurement = _measure_round_robin_gemm_window(
                    torch,
                    device=device,
                    logical_device_index=args.device,
                    expected_device_uuid=str(properties.uuid),
                    operation=state["operation"],
                    settling_operation=state["settling_operation"],
                    bundles=state["bundles"],
                    logical_flops=int(state["measured_logical_flops"]),
                    settling_invocations=int(sizing["settling_invocations"]),
                    settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                    settling_range_name=(
                        f"ai_b200_peak_{args.run_id}_{fill_mode}_w{window_index}_settle"
                    ),
                    repeats=args.repeats,
                    invocations_per_sample=int(sizing["primary_invocations_per_sample"]),
                    minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                    range_name=f"ai_b200_peak_{args.run_id}_{fill_mode}_w{window_index}_timed",
                    observe=True,
                    telemetry_interval_s=args.telemetry_interval_ms / 1000.0,
                    maximum_telemetry_gap_s=args.maximum_telemetry_gap_ms / 1000.0,
                    minimum_telemetry_samples=args.minimum_telemetry_samples,
                )
                _attach_coarse_energy_per_gemm(measurement)
                state["windows"].append(measurement)

        for check, left, right in zip(arrangement_checks, original, permuted, strict=True):
            actual_outputs_bitwise_equal = bool(torch.equal(left["output"], right["output"]))
            original_output_nonzero = bool(torch.count_nonzero(left["output"]).item())
            permuted_output_nonzero = bool(torch.count_nonzero(right["output"]).item())
            if (
                not actual_outputs_bitwise_equal
                or not original_output_nonzero
                or not permuted_output_nonzero
            ):
                raise RuntimeError(
                    "K-order arrangement controls failed the final timed-output equivalence gate"
                )
            check.update(
                {
                    "final_timed_outputs_bitwise_equal": actual_outputs_bitwise_equal,
                    "random_balanced_output_nonzero": original_output_nonzero,
                    "positive_then_negative_output_nonzero": permuted_output_nonzero,
                }
            )

        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    from labs.arithmetic_intensity_b200.ceiling_tool import _sample_summary

    for fill_mode, state in fill_states.items():
        windows = state["windows"]
        fill_seed = int(state["seed"])
        window_wrappers = [
            {"window": index, "measurement": measurement}
            for index, measurement in enumerate(windows)
        ]
        steady_gate = _nominal_gap_steady_state_qualification(
            window_wrappers,
            expected_window_count=args.windows,
        )
        qualified = bool(steady_gate["sustained_kpi_qualified"])
        nonzero_fill = fill_mode != "zero_power_switching_calibration"
        throughput_summary = (
            _arm_summary(
                windows=windows,
                metric_key="throughput",
                unit="TFLOP_per_s",
                seed=fill_seed + 101,
            )
            if qualified and nonzero_fill
            else None
        )
        latency_summary = (
            _timing_only_window_summary(windows, seed=fill_seed + 103) if qualified else None
        )
        energy_summary = None
        if qualified:
            energy_samples = [
                float(window["coarse_board_energy"]["host_phase_joules_per_gemm"])
                for window in windows
            ]
            power_samples = [
                float(window["coarse_board_energy"]["time_weighted_host_phase_power_w"])
                for window in windows
            ]
            energy_summary = {
                "joules_per_gemm": _sample_summary(energy_samples, "J_per_GEMM"),
                "time_weighted_power": _sample_summary(power_samples, "W"),
                "is_exact_per_kernel_energy": False,
            }
        operand_records = []
        operands_unchanged = True
        for bundle in state["bundles"]:
            operand_provenance = bundle["operand_provenance"]
            final_identity = {
                "a_final_identity": _operand_identity(bundle["a"]),
                "b_final_identity": _operand_identity(bundle["b"]),
            }
            unchanged = bool(
                final_identity["a_final_identity"] == operand_provenance["a_initial_identity"]
                and final_identity["b_final_identity"] == operand_provenance["b_initial_identity"]
            )
            operands_unchanged = operands_unchanged and unchanged
            operand_records.append(
                {
                    "buffer_index": int(bundle["buffer_index"]),
                    "seed": int(bundle["seed"]),
                    "operands": {
                        **operand_provenance,
                        **final_identity,
                        "identities_unchanged": unchanged,
                    },
                }
            )
        if not operands_unchanged:
            raise RuntimeError(f"Peak calibration operands changed for {fill_mode}")
        fill_results[fill_mode] = {
            "classification": (
                "low-switching calibration pending exact executed-tensor-work proof"
                if not nonzero_fill
                else (
                    "controlled matrix-value arrangement study"
                    if fill_mode.startswith("k_order_")
                    else "nonzero matrix-value calibration control"
                )
            ),
            "implementation": state["implementation_metadata"],
            "buffers": operand_records,
            "sizing": state["sizing"],
            "observer_calibration": state["observer_calibration"],
            "windows": windows,
            "steady_state_qualification": steady_gate,
            "latency_summary": latency_summary,
            "throughput_summary": throughput_summary,
            "coarse_board_energy_summary": energy_summary,
            "logical_flops_per_launch": logical_flops,
            "executed_dense_tensor_work_verified": False if not nonzero_fill else None,
            "advertised_product_share": (
                float(throughput_summary["window_medians"]["p50"])
                / float(reviewed["dense_bf16_tflops"])
                if throughput_summary is not None
                else None
            ),
            "zero_math_rate_withheld_pending_profiler_work_proof": not nonzero_fill,
            "qualified_raw_timing": qualified,
            "unqualified_cross_window_aggregates_suppressed": not qualified,
        }

    arrangement_pair_qualified = bool(
        fill_results["k_order_random_balanced"]["qualified_raw_timing"]
        and fill_results["k_order_positive_then_negative"]["qualified_raw_timing"]
    )
    arrangement_paired = (
        paired_window_summary(
            [
                float(window["throughput"]["p50"])
                for window in fill_states["k_order_random_balanced"]["windows"]
            ],
            [
                float(window["throughput"]["p50"])
                for window in fill_states["k_order_positive_then_negative"]["windows"]
            ],
            seed=args.seed + 211,
        )
        if arrangement_pair_qualified
        else None
    )

    all_raw_qualified = all(item["qualified_raw_timing"] for item in fill_results.values())
    virtualized = bool(environment["details"].get("virtualized"))
    report = {
        "schema_version": 1,
        "kind": "b200_dense_bf16_rotated_buffer_peak_calibration",
        "status": (
            "raw_calibration_complete" if all_raw_qualified else "raw_calibration_disqualified"
        ),
        "timestamp_utc": _timestamp(),
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "confirmation_lineage": {
            "path": str(args.confirmation),
            "run_id": confirmation["provenance"]["run_id"],
            "lab_tree_sha256": confirmation["provenance"]["lab_tree_sha256"],
        },
        "frozen_implementation": implementation,
        "clock_lock": {
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "power_limit_change_requested": False,
        },
        "workload": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "logical_model": logical,
            "buffer_sets": args.buffer_sets,
            "rotation": "round-robin over two fully preallocated operand and output sets",
            "cache_policy": "natural warmed cache with no explicit cache flush",
            "every_timed_output_full_oracle_checked": True,
        },
        "configuration": {
            "fill_modes": list(args.fill_modes),
            "balanced_interleaved_window_orders": interleaved_orders,
            "fill_order_design": (
                "five-row subset of a six-treatment Williams design; every fill occupies "
                "five distinct ordinal positions"
            ),
            "k_order_pair_direction_counts": {
                "random_before_clustered": 3,
                "clustered_before_random": 2,
            },
            "windows_per_fill": args.windows,
            "repeats": args.repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "settling_minimum_gpu_ms": args.settling_minimum_gpu_ms,
            "calibration_rounds_per_fill": args.calibration_rounds,
            "calibration_repeats": args.calibration_repeats,
            "confirmed_reference_p50_ms_not_used_for_sizing": confirmed_p50_ms,
            "per_fill_sizing": sizing_records,
            "planned_observer_and_primary_gpu_active_ms": planned_gpu_active_ms,
            "peak_phase_gpu_active_budget_ms": PEAK_PHASE_GPU_ACTIVE_BUDGET_MS,
        },
        "k_order_arrangement_equivalence": {
            "common_k_permutation": (
                "positive entries of the repeated A sign vector followed by negative entries"
            ),
            "checks": arrangement_checks,
            "exact_global_histograms_match": True,
            "full_fp32_references_equal": True,
            "final_timed_outputs_bitwise_equal": True,
            "final_timed_outputs_nonzero": True,
            "paired_timing_diagnostic": arrangement_paired,
            "paired_timing_diagnostic_qualified": arrangement_pair_qualified,
            "isolates_operand_and_accumulator_activity": False,
            "identifies_internal_unit_power_causality": False,
        },
        "fills": fill_results,
        "claim_boundaries": {
            "replaces_representative_scaled_normal_result": False,
            "fill_or_cache_result_is_an_optimization_claim": False,
            "uniform_minus1_plus1_is_amplitude_confounded": True,
            "matrix_values_can_change_memory_cache_and_arithmetic_behavior": True,
            "nominal_clock_scaling_performed": False,
            "zero_fill_public_math_rate_ready": False,
            "zero_fill_requires_matching_kernel_and_tensor_instruction_proof": True,
            "profiled_timing_is_primary": False,
            "attained_is_absolute_hardware_max": False,
            "unsettled_fill_cross_window_kpis_are_suppressed": True,
        },
    }
    for state in fill_states.values():
        state.clear()
    torch.cuda.empty_cache()
    return report


def _validate_peak_lineage(
    calibration: Mapping[str, Any],
    *,
    expected_lab_tree_sha256: str,
    source_commit: str,
    current_provenance: Mapping[str, Any],
    fill_mode: str,
) -> None:
    if calibration.get("schema_version") != 1:
        raise ValueError("peak calibration schema version must be 1")
    if calibration.get("kind") != "b200_dense_bf16_rotated_buffer_peak_calibration":
        raise ValueError("profile replay requires the rotated-buffer peak calibration")
    if calibration.get("status") != "raw_calibration_complete":
        raise ValueError("profile replay requires complete qualified raw calibration")
    if fill_mode not in calibration.get("fills", {}):
        raise ValueError("requested profile fill is absent from the peak calibration")
    provenance = calibration.get("provenance", {})
    if provenance.get("lab_tree_sha256") != expected_lab_tree_sha256:
        raise ValueError("peak lab-tree SHA-256 differs from the expected digest")
    if provenance.get("lab_tree_sha256") != current_provenance.get("lab_tree_sha256"):
        raise ValueError("peak calibration and profiler lab-tree SHA-256 differ")
    if provenance.get("source_commit") != source_commit:
        raise ValueError("peak calibration and profiler source commits differ")
    _validate_runtime_lineage(
        provenance,
        current_provenance,
        phase_names="peak calibration and profiler",
    )


def run_profile_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Replay one exact post-settling launch for matching Nsys or NCU capture."""
    if args.device != 0:
        raise ValueError("The profiler replay is authorized for GPU0 only")
    if args.settling_minimum_gpu_ms < 3000.0:
        raise ValueError("profiler replay requires at least 3000 GPU-active settling ms")

    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_tool import _gemm_correctness, _nvtx_range
    from labs.arithmetic_intensity_b200.optimization_extension import (
        load_optimization_extension,
    )

    calibration = json.loads(args.peak_calibration.read_text())
    device, properties, environment, provenance, reviewed = _runtime(args, torch)
    runtime_extension = load_optimization_extension()
    _attach_blas_runtime_identity(provenance, runtime_extension)
    _validate_peak_lineage(
        calibration,
        expected_lab_tree_sha256=args.expected_peak_lab_tree_sha256,
        source_commit=args.source_commit,
        current_provenance=provenance,
        fill_mode=args.fill_mode,
    )
    implementation = dict(calibration["frozen_implementation"])
    extension = (
        runtime_extension
        if implementation["implementation_id"] == "direct_cublaslt_frozen_algorithm"
        else None
    )
    fill = calibration["fills"][args.fill_mode]
    buffer_records = fill["buffers"]
    seeds = [int(item["seed"]) for item in buffer_records]
    if len(seeds) != 2:
        raise ValueError("profile replay requires the two frozen calibration buffer seeds")
    bundles = []
    for seed in seeds:
        bundle = _prepare_gemm_bundles(
            torch,
            device=device,
            seed=seed,
            fill_mode=args.fill_mode,
            buffer_sets=1,
        )[0]
        bundles.append(bundle)
    operation, settling_operation, implementation_metadata, plan = _build_round_robin_operations(
        torch,
        bundles=bundles,
        implementation=implementation,
        extension=extension,
    )
    settling_invocations = int(fill["sizing"]["settling_invocations"])
    exact_range = f"ai_b200_exact_profile_{args.run_id}_{args.fill_mode}"

    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        settling_operation.reset()
        settling = _run_fixed_settling_block(
            torch,
            device=device,
            operation=settling_operation,
            invocations=settling_invocations,
            minimum_gpu_ms=args.settling_minimum_gpu_ms,
            range_name=f"{exact_range}_settle",
            telemetry_sampling_active=False,
        )
        for bundle in bundles:
            bundle["output"].fill_(float("nan"))
        torch.cuda.synchronize(device)
        operation.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        host_start_monotonic_s = time.monotonic()
        host_start_timestamp_utc = _timestamp()
        with _nvtx_range(torch, exact_range):
            start.record(stream)
            operation()
            end.record(stream)
        torch.cuda.synchronize(device)
        host_end_monotonic_s = time.monotonic()
        host_end_timestamp_utc = _timestamp()
        duration_ms = float(start.elapsed_time(end))
        correctness = _gemm_correctness(
            torch,
            bundles[0]["output"],
            bundles[0]["reference"],
        )
        if not correctness["passed"]:
            raise RuntimeError("Exact profiler launch failed full-output correctness")
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_dense_bf16_exact_profile_replay",
        "status": "profile_replay_complete",
        "timestamp_utc": _timestamp(),
        "provenance": provenance,
        "environment_validation": environment,
        "evidence_class": (
            "strict_noncanonical_virtualized" if virtualized else "strict_bare_metal"
        ),
        "publication_eligible": False,
        "publication_blockers": ["operator_review_and_profiler_validation_required"],
        "reviewed_nominal": reviewed,
        "peak_calibration_lineage": {
            "path": str(args.peak_calibration),
            "run_id": calibration["provenance"]["run_id"],
            "lab_tree_sha256": calibration["provenance"]["lab_tree_sha256"],
            "fill_mode": args.fill_mode,
        },
        "clock_lock": {
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "power_limit_change_requested": False,
        },
        "workload": {
            "shape": {"m": GEMM_SIZE, "n": GEMM_SIZE, "k": GEMM_SIZE},
            "layout": "row_major_nn",
            "dtype": "bf16",
            "compute_type": "fp32_accumulation",
            "beta_zero": True,
            "fill_mode": args.fill_mode,
            "logical_model": gemm_logical_cost(
                m=GEMM_SIZE,
                n=GEMM_SIZE,
                k=GEMM_SIZE,
                beta_zero=True,
            ),
            "buffer_rotation_during_settling": True,
            "profiled_launch_uses_buffer_index": 0,
            "cache_policy": "natural warmed cache with no explicit cache flush",
        },
        "implementation": implementation_metadata,
        "exact_range": {
            "nvtx_range": exact_range,
            "launch_count": 1,
            "duration_ms": duration_ms,
            "host_envelope": {
                "start_monotonic_s": host_start_monotonic_s,
                "end_monotonic_s": host_end_monotonic_s,
                "start_timestamp_utc": host_start_timestamp_utc,
                "end_timestamp_utc": host_end_timestamp_utc,
            },
            "correctness": correctness,
            "settling": settling,
        },
        "profiler_contract": {
            "clock_control": "none",
            "cache_control": "none",
            "nvtx_include": f"{exact_range}/",
            "launch_count": 1,
            "timing_role": "mechanism evidence only",
            "same_range_cycle_metric": "gpc__cycles_elapsed.avg.per_second",
            "same_range_duration_metric": "gpu__time_duration.sum",
            "tensor_work_metric": "must be selected from installed NCU metric query",
            "combine_metrics_only_when_same_pass_and_same_range": True,
        },
        "claim_boundaries": {
            "profiled_timing_is_primary": False,
            "zero_executed_tensor_work_proved_by_this_json_alone": False,
            "nominal_clock_scaling_performed": False,
            "profiler_counter_values": None,
        },
    }


def _measure_copy_window(
    torch: Any,
    *,
    device: Any,
    logical_device_index: int,
    expected_device_uuid: str,
    source: Any,
    destination: Any,
    settling_destination: Any,
    settling_invocations: int,
    settling_minimum_gpu_ms: float,
    repeats: int,
    invocations_per_sample: int,
    minimum_gpu_active_ms: float,
    range_prefix: str,
    observe: bool,
    telemetry_interval_s: float,
    maximum_telemetry_gap_s: float,
    minimum_telemetry_samples: int,
) -> dict[str, Any]:
    from labs.arithmetic_intensity_b200.ceiling_tool import _nvtx_range, _sample_summary

    destination.copy_(source)
    destination.bitwise_not_()
    torch.cuda.synchronize(device)
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
            operation=partial(settling_destination.copy_, source),
            invocations=settling_invocations,
            minimum_gpu_ms=settling_minimum_gpu_ms,
            range_name=f"{range_prefix}_settle",
            telemetry_sampling_active=observe,
        )
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats)
        ]
        stream = torch.cuda.current_stream(device)
        window_start_monotonic_s = time.monotonic()
        window_start_epoch_s = time.time()
        window_start_timestamp_utc = _timestamp()
        settle_to_timing_host_gap_ms = (
            window_start_monotonic_s - settling["completion_monotonic_s"]
        ) * 1000.0
        if settle_to_timing_host_gap_ms > 20.0:
            raise RuntimeError(
                f"Copy host gap after settling was {settle_to_timing_host_gap_ms:.3f}ms, "
                "above the 20ms limit"
            )
        with _nvtx_range(torch, f"{range_prefix}_timed"):
            for start, end in events:
                start.record(stream)
                for _ in range(invocations_per_sample):
                    destination.copy_(source)
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
        raise RuntimeError(f"Invalid CUDA timings for {range_prefix}: {aggregate_ms}")
    measured_gpu_active_ms = sum(aggregate_ms)
    if measured_gpu_active_ms < minimum_gpu_active_ms:
        raise RuntimeError("Copy window did not meet the minimum GPU-active duration")
    if not bool(torch.equal(destination, source)):
        raise RuntimeError("Copy window failed full 8 GiB equality")
    normalized_ms = [value / invocations_per_sample for value in aggregate_ms]
    requested_bytes = 2 * int(source.numel())
    rates = [requested_bytes / (value / 1e3) / 1e12 for value in normalized_ms]
    return {
        "nvtx_range": f"{range_prefix}_timed",
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
        "throughput": _sample_summary(rates, "effective_requested_TB_per_s"),
        "invocations_per_sample": invocations_per_sample,
        "measured_invocations": repeats * invocations_per_sample,
        "measured_gpu_active_ms": measured_gpu_active_ms,
        "pre_timing_output_poison": "bitwise complement of the complete random source",
        "correctness": {
            "method": "full_tensor_torch_equal",
            "output_elements_checked": int(source.numel()),
            "passed": True,
        },
        "observation": observation,
    }


def _relative_half_change(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise RuntimeError("steady-state trend requires at least two samples")
    midpoint = len(values) // 2
    first = statistics.median(values[:midpoint])
    second = statistics.median(values[midpoint:])
    if first <= 0:
        raise RuntimeError("steady-state trend denominator must be positive")
    return (second / first - 1.0) * 100.0


def _copy_window_steady_state_qualification(
    measurement: Mapping[str, Any],
    *,
    expected_gpu_index: int,
) -> dict[str, Any]:
    try:
        timeline = measurement["observation"]["in_window_telemetry"]
        samples = timeline["samples"]
        metrics = timeline["metrics"]
        cadence = timeline["cadence"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Copy window lacks required in-window telemetry") from exc
    if len(samples) < int(timeline["minimum_required_samples"]):
        raise RuntimeError("Copy window lacks the required complete telemetry samples")

    memory_clocks = [float(sample["memory_clock_mhz"]) for sample in samples]
    application_memory_clocks = [
        float(sample["applications_clock_memory_mhz"]) for sample in samples
    ]
    power_draws = [float(sample["power_draw_w"]) for sample in samples]
    performance_states = [int(float(sample["performance_state"])) for sample in samples]
    gpu_indices = [int(float(sample["gpu_index"])) for sample in samples]
    if any(value <= 0 for value in [*memory_clocks, *application_memory_clocks, *power_draws]):
        raise RuntimeError("Copy telemetry contains a nonpositive required runtime value")

    no_trend_threshold_pct = 3.0
    span_threshold_pct = 10.0
    memory_half_change_pct = _relative_half_change(memory_clocks)
    power_half_change_pct = _relative_half_change(power_draws)
    memory_metric = metrics["memory_clock_mhz"]
    power_metric = metrics["power_draw_w"]
    memory_span_pct = (
        (float(memory_metric["p90"]) - float(memory_metric["p10"]))
        / float(memory_metric["p50"])
        * 100.0
    )
    power_span_pct = (
        (float(power_metric["p90"]) - float(power_metric["p10"]))
        / float(power_metric["p50"])
        * 100.0
    )
    maximum_allowed_gap_s = float(cadence["maximum_allowed_gap_s"])
    boundary = cadence["boundary_gap_s"]
    boundary_cadence_passed = bool(
        boundary["acceptance_gate"] and float(boundary["maximum_observed"]) <= maximum_allowed_gap_s
    )
    midpoint = cadence["midpoint_gap_s"]
    midpoint_cadence_passed = bool(
        midpoint is None or float(midpoint["max"]) <= maximum_allowed_gap_s
    )
    memory_clock_stable = bool(
        abs(memory_half_change_pct) <= no_trend_threshold_pct
        and memory_span_pct <= span_threshold_pct
    )
    power_stable = bool(
        abs(power_half_change_pct) <= no_trend_threshold_pct
        and power_span_pct <= span_threshold_pct
    )
    application_memory_clock_stable = len(set(application_memory_clocks)) == 1
    performance_state_is_p0 = set(performance_states) == {0}
    expected_gpu_only = set(gpu_indices) == {expected_gpu_index}
    passed = bool(
        boundary_cadence_passed
        and midpoint_cadence_passed
        and memory_clock_stable
        and power_stable
        and application_memory_clock_stable
        and performance_state_is_p0
        and expected_gpu_only
    )
    failed_gates = [
        name
        for name, gate in (
            ("boundary_cadence", boundary_cadence_passed),
            ("midpoint_cadence", midpoint_cadence_passed),
            ("memory_clock_stability", memory_clock_stable),
            ("power_stability", power_stable),
            ("application_memory_clock_stability", application_memory_clock_stable),
            ("performance_state_p0", performance_state_is_p0),
            ("expected_gpu_identity", expected_gpu_only),
        )
        if not gate
    ]
    return {
        "passed": passed,
        "failed_gates": failed_gates,
        "sample_count": len(samples),
        "minimum_required_samples": int(timeline["minimum_required_samples"]),
        "cadence": {
            "boundary_passed": boundary_cadence_passed,
            "midpoint_passed": midpoint_cadence_passed,
            "maximum_allowed_gap_s": maximum_allowed_gap_s,
            "observed_boundary_maximum_gap_s": float(boundary["maximum_observed"]),
            "observed_midpoint_maximum_gap_s": (
                None if midpoint is None else float(midpoint["max"])
            ),
        },
        "memory_clock": {
            "first_to_second_half_change_pct": memory_half_change_pct,
            "p10_to_p90_span_pct_of_p50": memory_span_pct,
            "stable": memory_clock_stable,
            "application_clock_values_mhz": sorted(set(application_memory_clocks)),
            "application_clock_stable": application_memory_clock_stable,
        },
        "power": {
            "first_to_second_half_change_pct": power_half_change_pct,
            "p10_to_p90_span_pct_of_p50": power_span_pct,
            "stable": power_stable,
        },
        "runtime_state": {
            "performance_states": sorted(set(performance_states)),
            "requires_p0": True,
            "p0_passed": performance_state_is_p0,
            "gpu_indices": sorted(set(gpu_indices)),
            "expected_gpu_index": expected_gpu_index,
            "gpu_identity_passed": expected_gpu_only,
        },
        "graphics_clock_role": "diagnostic_only_not_a_copy_acceptance_gate",
        "thresholds": {
            "first_to_second_half_max_abs_change_pct": no_trend_threshold_pct,
            "p10_to_p90_max_span_pct_of_p50": span_threshold_pct,
        },
    }


def _copy_steady_state_qualification(
    windows: Sequence[Mapping[str, Any]],
    *,
    expected_window_count: int,
    expected_gpu_index: int,
) -> dict[str, Any]:
    window_results = [
        {
            "window": index,
            **_copy_window_steady_state_qualification(
                window["measurement"],
                expected_gpu_index=expected_gpu_index,
            ),
        }
        for index, window in enumerate(windows)
    ]
    all_expected_windows_present = len(windows) == expected_window_count
    all_windows_pass = bool(window_results) and all(item["passed"] for item in window_results)
    return {
        "expected_window_count": expected_window_count,
        "observed_window_count": len(windows),
        "all_expected_windows_present": all_expected_windows_present,
        "all_windows_pass": all_windows_pass,
        "sustained_copy_kpi_qualified": all_expected_windows_present and all_windows_pass,
        "window_results": window_results,
        "acceptance_rule": (
            "every expected window must pass complete-query cadence, memory-clock and power "
            "stability, stable application memory clock, P0, and GPU identity gates"
        ),
    }


def _run_copy_observer_calibration(
    torch: Any,
    *,
    args: argparse.Namespace,
    device: Any,
    properties: Any,
    source: Any,
    destination: Any,
    settling_destination: Any,
    settling_invocations: int,
    timed_invocations: int,
) -> dict[str, Any]:
    rounds = []
    off_medians = []
    on_medians = []
    for round_index in range(args.calibration_rounds):
        order = ("off", "on") if round_index % 2 == 0 else ("on", "off")
        arms = {}
        for arm in order:
            arms[arm] = _measure_copy_window(
                torch,
                device=device,
                logical_device_index=args.device,
                expected_device_uuid=str(properties.uuid),
                source=source,
                destination=destination,
                settling_destination=settling_destination,
                settling_invocations=settling_invocations,
                settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                repeats=args.calibration_repeats,
                invocations_per_sample=timed_invocations,
                minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                range_prefix=f"ai_b200_copy_refresh_{args.run_id}_observer_{round_index}_{arm}",
                observe=arm == "on",
                telemetry_interval_s=args.telemetry_interval_ms / 1000.0,
                maximum_telemetry_gap_s=args.maximum_telemetry_gap_ms / 1000.0,
                minimum_telemetry_samples=args.minimum_telemetry_samples,
            )
        off_medians.append(float(arms["off"]["normalized_timing"]["p50"]))
        on_medians.append(float(arms["on"]["normalized_timing"]["p50"]))
        rounds.append({"round": round_index, "order": list(order), "arms": arms})
    return {
        "rounds": rounds,
        "summary": observer_overhead_summary(
            off_medians,
            on_medians,
            limit_pct=args.observer_overhead_limit_pct,
            seed=args.seed + 307,
        ),
    }


def run_copy_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Refresh the 8 GiB requested-byte copy result with in-window qualification."""
    _validate_common_args(args)
    _validate_observer_args(args)
    if args.windows != 5:
        raise ValueError("copy calibration requires exactly five fresh-operand windows")
    if args.payload_bytes != COPY_PAYLOAD_BYTES:
        raise ValueError("copy calibration payload must remain exactly 8 GiB")

    import torch

    from core.harness.benchmark_harness import lock_gpu_clocks
    from core.profiling.gpu_telemetry import query_gpu_telemetry
    from labs.arithmetic_intensity_b200.ceiling_tool import (
        _measure_aggregate_operation,
        _sample_summary,
    )

    device, properties, environment, provenance, reviewed = _runtime(args, torch)
    requested_bytes = 2 * args.payload_bytes
    with lock_gpu_clocks(device=args.device) as lock_peaks:
        telemetry_before = query_gpu_telemetry(args.device, force_refresh=True)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        sizing_source = torch.randint(
            0,
            256,
            (args.payload_bytes,),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        sizing_destination = torch.empty_like(sizing_source)
        sizing_settling_destination = torch.empty_like(sizing_source)
        short = _measure_aggregate_operation(
            torch,
            device=device,
            operation=lambda _invocation, call=partial(
                sizing_destination.copy_, sizing_source
            ): call(),
            range_name=f"ai_b200_copy_refresh_{args.run_id}_short",
            warmups=1,
            repeats=3,
            invocations_per_sample=1,
        )
        if not bool(torch.equal(sizing_destination, sizing_source)):
            raise RuntimeError("Copy sizing diagnostic failed full 8 GiB equality")
        short_p50_ms = float(short["normalized_timing"]["p50"])
        primary_counts = _invocation_counts(
            short_p50_ms,
            repeats=args.repeats,
            settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
            minimum_window_gpu_ms=args.minimum_window_gpu_ms,
        )
        observer_counts = _invocation_counts(
            short_p50_ms,
            repeats=args.calibration_repeats,
            settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
            minimum_window_gpu_ms=args.minimum_window_gpu_ms,
        )
        observer_calibration = _run_copy_observer_calibration(
            torch,
            args=args,
            device=device,
            properties=properties,
            source=sizing_source,
            destination=sizing_destination,
            settling_destination=sizing_settling_destination,
            settling_invocations=observer_counts[0],
            timed_invocations=observer_counts[1],
        )
        observer_qualified = bool(observer_calibration["summary"]["within_observer_overhead_limit"])
        del sizing_settling_destination, sizing_destination, sizing_source
        torch.cuda.empty_cache()

        windows = []
        for window_index in range(args.windows):
            generator = torch.Generator(device=device)
            window_seed = args.seed + window_index * 1_000_003
            generator.manual_seed(window_seed)
            source = torch.randint(
                0,
                256,
                (args.payload_bytes,),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            destination = torch.empty_like(source)
            settling_destination = torch.empty_like(source)
            measured = _measure_copy_window(
                torch,
                device=device,
                logical_device_index=args.device,
                expected_device_uuid=str(properties.uuid),
                source=source,
                destination=destination,
                settling_destination=settling_destination,
                settling_invocations=primary_counts[0],
                settling_minimum_gpu_ms=args.settling_minimum_gpu_ms,
                repeats=args.repeats,
                invocations_per_sample=primary_counts[1],
                minimum_gpu_active_ms=args.minimum_window_gpu_ms,
                range_prefix=f"ai_b200_copy_refresh_{args.run_id}_w{window_index}",
                observe=True,
                telemetry_interval_s=args.telemetry_interval_ms / 1000.0,
                maximum_telemetry_gap_s=args.maximum_telemetry_gap_ms / 1000.0,
                minimum_telemetry_samples=args.minimum_telemetry_samples,
            )
            windows.append(
                {
                    "window": window_index,
                    "seed": window_seed,
                    "measurement": measured,
                }
            )
            del settling_destination, destination, source
            torch.cuda.empty_cache()
        telemetry_after = query_gpu_telemetry(args.device, force_refresh=True)

    steady_gate = _copy_steady_state_qualification(
        windows,
        expected_window_count=args.windows,
        expected_gpu_index=args.device,
    )
    measurement_qualified = bool(observer_qualified and steady_gate["sustained_copy_kpi_qualified"])
    summary = None
    if measurement_qualified:
        window_medians = [float(window["measurement"]["throughput"]["p50"]) for window in windows]
        all_samples = [
            float(sample)
            for window in windows
            for sample in window["measurement"]["throughput"]["samples"]
        ]
        summary = {
            "statistic": "median of five fresh-window medians",
            "all_samples": _sample_summary(all_samples, "effective_requested_TB_per_s"),
            "window_medians": _sample_summary(
                window_medians,
                "effective_requested_TB_per_s",
            ),
            "window_median_bootstrap_95_ci": bootstrap_median_ci(
                window_medians,
                seed=args.seed + 101,
            ),
            "fraction_of_book_8_TB_per_s_nominal": statistics.median(window_medians) / 8.0,
        }

    virtualized = bool(environment["details"].get("virtualized"))
    return {
        "schema_version": 1,
        "kind": "b200_torch_cuda_tensor_copy_requested_byte_refresh",
        "status": "copy_refresh_complete" if measurement_qualified else "copy_refresh_disqualified",
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
            "repository_context_used": True,
            "reported_theoretical_fp16_tflops": float(lock_peaks[0]),
            "reported_theoretical_bandwidth_gbps": float(lock_peaks[1]),
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
            "manual_nvidia_smi_clock_or_power_change_performed": False,
            "power_limit_change_requested": False,
        },
        "workload": {
            "implementation_id": "torch_cuda_tensor_copy",
            "implementation_label": "torch CUDA tensor copy",
            "payload_bytes": args.payload_bytes,
            "input_distribution": "fresh seeded random uint8 per window",
            "requested_read_bytes_per_operation": args.payload_bytes,
            "requested_write_bytes_per_operation": args.payload_bytes,
            "requested_total_bytes_per_operation": requested_bytes,
            "byte_accounting": "requested source read plus requested destination write",
            "is_raw_physical_hbm_bandwidth": False,
            "copy_engine_identity_proved": False,
            "copy_engine_identity_requirement": "new matching Nsys trace",
        },
        "configuration": {
            "windows": args.windows,
            "repeats": args.repeats,
            "minimum_window_gpu_ms": args.minimum_window_gpu_ms,
            "settling_minimum_gpu_ms": args.settling_minimum_gpu_ms,
            "primary_invocation_counts": {
                "settling": primary_counts[0],
                "per_sample": primary_counts[1],
            },
            "observer_invocation_counts": {
                "settling": observer_counts[0],
                "per_sample": observer_counts[1],
            },
            "calibration_rounds": args.calibration_rounds,
            "calibration_repeats": args.calibration_repeats,
            "minimum_telemetry_samples": args.minimum_telemetry_samples,
            "telemetry_interval_ms": args.telemetry_interval_ms,
            "maximum_telemetry_gap_ms": args.maximum_telemetry_gap_ms,
        },
        "short_sizing_diagnostic": {
            "classification": "excluded from sustained requested-byte claims",
            "timing": short,
            "full_output_correctness_passed": True,
        },
        "observer_calibration": observer_calibration,
        "observer_qualified": observer_qualified,
        "windows": windows,
        "steady_state_qualification": steady_gate,
        "summary": summary,
        "unqualified_cross_window_aggregates_suppressed": not measurement_qualified,
        "claim_boundaries": {
            "requested_bytes_are_raw_physical_hbm_bytes": False,
            "copy_engine_identity_claimed": False,
            "profiled_timing_is_primary": False,
            "attained_is_absolute_hardware_max": False,
            "fresh_source_configuration_match_only": True,
            "graphics_clock_is_a_copy_acceptance_gate": False,
            "matched_roofline_input_available": measurement_qualified,
        },
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20_261_014)


def _add_settled_timing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--minimum-window-gpu-ms", type=float, default=1000.0)
    parser.add_argument("--settling-minimum-gpu-ms", type=float, default=3000.0)


def _add_observer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-rounds", type=int, default=8)
    parser.add_argument("--calibration-repeats", type=int, default=5)
    parser.add_argument("--telemetry-interval-ms", type=float, default=100.0)
    parser.add_argument("--maximum-telemetry-gap-ms", type=float, default=250.0)
    parser.add_argument("--minimum-telemetry-samples", type=int, default=8)
    parser.add_argument("--observer-overhead-limit-pct", type=float, default=0.5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    _add_common_arguments(search)
    _add_settled_timing_arguments(search)
    search.set_defaults(repeats=5)
    search.add_argument("--workspace-mib", dest="workspace_bytes", type=_parse_workspace_mib)
    search.set_defaults(workspace_bytes=WORKSPACE_CEILINGS_BYTES)
    search.add_argument("--max-heuristics", type=int, default=MAX_HEURISTICS_PER_WORKSPACE)
    search.add_argument("--maximum-candidates", type=int, default=DEFAULT_SEARCH_CANDIDATE_CAP)
    search.add_argument("--pair-rounds", type=int, default=2)
    search.add_argument("--short-repeats", type=int, default=3)

    confirm = subparsers.add_parser("confirm")
    _add_common_arguments(confirm)
    _add_settled_timing_arguments(confirm)
    _add_observer_arguments(confirm)
    confirm.add_argument("--search", type=Path, required=True)
    confirm.add_argument("--expected-search-lab-tree-sha256", required=True)
    confirm.add_argument("--candidate-signature", required=True)
    confirm.add_argument("--windows", type=int, default=5)

    peak = subparsers.add_parser("calibrate-peak")
    _add_common_arguments(peak)
    _add_settled_timing_arguments(peak)
    _add_observer_arguments(peak)
    peak.add_argument("--confirmation", type=Path, required=True)
    peak.add_argument("--expected-confirmation-lab-tree-sha256", required=True)
    peak.add_argument("--windows", type=int, default=5)
    peak.add_argument("--buffer-sets", type=int, default=2)
    peak.add_argument(
        "--fill-modes",
        type=_parse_fill_modes,
        default=_parse_fill_modes(
            "scaled_normal,uniform_rms_matched_to_scaled_normal,uniform_minus1_plus1,"
            "k_order_random_balanced,k_order_positive_then_negative,"
            "zero_power_switching_calibration"
        ),
    )

    profile = subparsers.add_parser("profile-replay")
    _add_common_arguments(profile)
    profile.add_argument("--peak-calibration", type=Path, required=True)
    profile.add_argument("--expected-peak-lab-tree-sha256", required=True)
    profile.add_argument(
        "--fill-mode",
        choices=(
            "scaled_normal",
            "uniform_rms_matched_to_scaled_normal",
            "uniform_minus1_plus1",
            "k_order_random_balanced",
            "k_order_positive_then_negative",
            "zero_power_switching_calibration",
        ),
        required=True,
    )
    profile.add_argument("--settling-minimum-gpu-ms", type=float, default=3000.0)

    copy = subparsers.add_parser("calibrate-copy")
    _add_common_arguments(copy)
    _add_settled_timing_arguments(copy)
    _add_observer_arguments(copy)
    copy.add_argument("--windows", type=int, default=5)
    copy.add_argument(
        "--payload-mib",
        dest="payload_bytes",
        type=lambda value: int(value) << 20,
        default=COPY_PAYLOAD_BYTES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runners = {
        "search": run_search,
        "confirm": run_confirm,
        "calibrate-peak": run_peak_calibration,
        "profile-replay": run_profile_replay,
        "calibrate-copy": run_copy_calibration,
    }
    try:
        _preflight_output(args.output)
        import torch

        with _bf16_fp32_accumulation_policy(torch):
            report = runners[args.command](args)
    except (DiagnosticGateError, RuntimeError, ValueError, FileExistsError) as error:
        failure = {
            "schema_version": 1,
            "kind": "b200_sustained_optimization_failure",
            "status": "rejected",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "error_type": type(error).__name__,
            "error": str(error),
            "lab_tree_sha256": _package_sha256(),
        }
        failure_output = None if isinstance(error, FileExistsError) else args.output
        try:
            _write_report(failure_output, failure)
        except FileExistsError:
            _write_report(None, failure)
        return 2
    try:
        _write_report(args.output, report)
    except FileExistsError as error:
        _write_report(
            None,
            {
                "schema_version": 1,
                "kind": "b200_sustained_optimization_failure",
                "status": "rejected",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": args.command,
                "error_type": type(error).__name__,
                "error": str(error),
                "lab_tree_sha256": _package_sha256(),
            },
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
