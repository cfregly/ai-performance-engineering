#!/usr/bin/env python3
"""Validate and analyze the sealed 24-run nested-graph-break benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


INPUT_SCHEMA = "nested-graph-breaks-b200-benchmark/v1"
OUTPUT_SCHEMA = "nested-graph-breaks-b200-analysis/v1"
BENCHMARK_ID = "torch-dynamo-nested-graph-breaks-b200-20261020"
SCOPE = (
    "controlled compiler microbenchmark; no serving, model-throughput, or "
    "model-quality claim"
)
PRIMARY_CASE = "nested_break_depth8"
CASES: Dict[str, Dict[str, int]] = {
    "no_break_depth4": {"depth": 4, "explicit_graph_breaks": 0},
    "nested_break_depth4": {"depth": 4, "explicit_graph_breaks": 1},
    "nested_break_depth8": {"depth": 8, "explicit_graph_breaks": 1},
}
ARMS = (0, 1)
SEEDS = (0, 1, 2, 3)
INPUT_ELEMENTS = 1 << 20
INPUT_BYTES = INPUT_ELEMENTS * 4
COLD_CALLS = 10
WARMUP_CALLS = 25
LATENCY_TRIALS = 100
THROUGHPUT_BLOCKS = 10
THROUGHPUT_CALLS_PER_BLOCK = 50
MIN_OBSERVED_SEED_SPEEDUP = 1.10
MAX_STARTUP_STEADY_P95_REGRESSION = 1.03
CONTROL_RATIO_MIN = 0.95
CONTROL_RATIO_MAX = 1.05
EXPECTED_HASH_LENGTH = 64


class ValidationError(Exception):
    """Raised when raw benchmark evidence does not satisfy the sealed protocol."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_seed_ratios(ratios: Sequence[float]) -> Dict[str, Any]:
    """Summarize the four observed paired-seed ratios without interval claims."""

    values = [float(value) for value in ratios]
    if len(values) != len(SEEDS):
        raise ValueError(f"expected exactly {len(SEEDS)} paired-seed ratios")
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("paired-seed ratios must be finite and positive")
    return {
        "geometric_mean_ratio": _geometric_mean(values),
        "observed_min_seed_ratio": min(values),
        "observed_max_seed_ratio": max(values),
        "observed_seed_ratios": values,
        "coverage_claim": None,
        "summary_role": (
            "descriptive four-seed summary; no confidence-interval or population-"
            "coverage claim"
        ),
    }


def _read_record(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError([f"cannot read {path}: {exc}"]) from exc
    if not isinstance(value, dict):
        raise ValidationError([f"{path}: result JSON must be an object"])
    value = dict(value)
    value["_source_path"] = str(path.resolve())
    return value


def discover_records(inputs: Sequence[Path]) -> List[Dict[str, Any]]:
    if not inputs:
        raise ValidationError(
            ["at least one result directory or JSON file is required"]
        )
    found: Dict[str, Path] = {}
    errors: List[str] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            candidates = [path]
        elif path.is_dir() and (path / "result.json").is_file():
            candidates = [path / "result.json"]
        elif path.is_dir():
            candidates = sorted(path.rglob("result.json"))
        else:
            errors.append(f"input does not exist: {path}")
            continue
        if not candidates:
            errors.append(f"no result.json files found under {path}")
        for candidate in candidates:
            found[str(candidate.resolve())] = candidate.resolve()
    if errors:
        raise ValidationError(errors)
    return [_read_record(found[key]) for key in sorted(found)]


def _path(record: Mapping[str, Any]) -> str:
    return str(record.get("_source_path", "<in-memory-record>"))


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _require_equal(
    errors: List[str], record: Mapping[str, Any], keys: Sequence[str], expected: Any
) -> None:
    actual = _nested(record, *keys)
    if actual != expected:
        errors.append(
            f"{_path(record)}: {'.'.join(keys)}={actual!r}, expected {expected!r}"
        )


def _require_hash(
    errors: List[str], record: Mapping[str, Any], keys: Sequence[str]
) -> None:
    value = _nested(record, *keys)
    if not isinstance(value, str) or len(value) != EXPECTED_HASH_LENGTH:
        errors.append(f"{_path(record)}: {'.'.join(keys)} is not a SHA-256 hex string")
        return
    try:
        int(value, 16)
    except ValueError:
        errors.append(f"{_path(record)}: {'.'.join(keys)} is not hexadecimal")


def _validate_raw_vector(
    errors: List[str], record: Mapping[str, Any], keys: Sequence[str], length: int
) -> Optional[List[float]]:
    value = _nested(record, *keys)
    if not isinstance(value, list) or len(value) != length:
        errors.append(
            f"{_path(record)}: {'.'.join(keys)} must contain exactly {length} values"
        )
        return None
    if not all(_is_finite_positive(item) for item in value):
        errors.append(
            f"{_path(record)}: {'.'.join(keys)} contains non-positive/non-finite data"
        )
        return None
    return [float(item) for item in value]


def _validate_sample_hash(
    errors: List[str], record: Mapping[str, Any], phase: str
) -> None:
    sample = _nested(record, "correctness", phase, "sample_output")
    if not isinstance(sample, Mapping):
        errors.append(f"{_path(record)}: correctness.{phase}.sample_output is missing")
        return
    indices = sample.get("indices")
    values = sample.get("values")
    digest = sample.get("sha256")
    expected_indices = [
        0,
        1,
        17,
        1024,
        INPUT_ELEMENTS // 3,
        INPUT_ELEMENTS // 2,
        INPUT_ELEMENTS - 1,
    ]
    if indices != expected_indices:
        errors.append(f"{_path(record)}: selected sample indices changed")
    if not isinstance(values, list) or len(values) != len(expected_indices):
        errors.append(f"{_path(record)}: selected sample values are incomplete")
        return
    if not all(
        isinstance(item, (int, float)) and math.isfinite(float(item)) for item in values
    ):
        errors.append(f"{_path(record)}: selected sample values are non-finite")
        return
    packed = b"".join(struct.pack("!d", float(value)) for value in values)
    recomputed = hashlib.sha256(packed).hexdigest()
    if digest != recomputed:
        errors.append(
            f"{_path(record)}: sample-output hash does not match selected values"
        )


def validate_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, int, int], Mapping[str, Any]]:
    errors: List[str] = []
    expected_keys = {
        (case_name, enabled, seed)
        for case_name in CASES
        for enabled in ARMS
        for seed in SEEDS
    }
    indexed: Dict[Tuple[str, int, int], Mapping[str, Any]] = {}
    cache_paths: set[str] = set()

    if len(records) != len(expected_keys):
        errors.append(
            f"found {len(records)} records, expected exactly {len(expected_keys)}"
        )

    for record in records:
        source = _path(record)
        case_name = record.get("case")
        enabled = record.get("enabled")
        seed = record.get("seed")
        if case_name not in CASES or enabled not in ARMS or seed not in SEEDS:
            errors.append(
                f"{source}: unexpected identity case={case_name!r} enabled={enabled!r} seed={seed!r}"
            )
            continue
        key = (str(case_name), int(enabled), int(seed))
        if key in indexed:
            errors.append(f"{source}: duplicate result identity {key}")
            continue
        indexed[key] = record

        expected_profile = bool(case_name == PRIMARY_CASE and seed == 0)
        _require_equal(errors, record, ("profile_requested",), expected_profile)
        _require_equal(errors, record, ("profile", "requested"), expected_profile)
        profile_status = _nested(record, "profile", "status")
        if expected_profile:
            if profile_status == "captured":
                _require_equal(errors, record, ("status",), "pass")
                _require_equal(errors, record, ("profile", "evidence_complete"), True)
                for marker_name in (
                    "cuda_graph_launch_event_count",
                    "cuda_kernel_event_count",
                ):
                    marker_count = _nested(record, "profile", marker_name)
                    if (
                        not isinstance(marker_count, int)
                        or isinstance(marker_count, bool)
                        or marker_count <= 0
                    ):
                        errors.append(
                            f"{source}: successful profile lacks {marker_name}"
                        )
                _require_equal(
                    errors,
                    record,
                    ("profile", "cudagraph_step_marker_used"),
                    False,
                )
                _require_equal(
                    errors,
                    record,
                    ("profile", "output_lifetime"),
                    "released before each subsequent invocation",
                )
                _require_hash(errors, record, ("profile", "trace_sha256"))
            elif profile_status not in (
                "failed",
                "failed_missing_required_markers",
            ):
                errors.append(f"{source}: requested profile has invalid status")
            else:
                _require_equal(
                    errors, record, ("status",), "benchmark_pass_profile_failed"
                )
                profile_error = _nested(record, "profile", "error")
                if not isinstance(profile_error, str) or not profile_error:
                    errors.append(f"{source}: failed profile did not retain its error")
        elif profile_status != "not_requested":
            errors.append(f"{source}: profile was captured outside the sealed scope")
        else:
            _require_equal(errors, record, ("status",), "pass")

        _require_equal(errors, record, ("schema",), INPUT_SCHEMA)
        _require_equal(errors, record, ("benchmark_id",), BENCHMARK_ID)
        _require_equal(errors, record, ("scope",), SCOPE)
        if record.get("status") not in ("pass", "benchmark_pass_profile_failed"):
            errors.append(
                f"{source}: benchmark status is not a retained benchmark pass"
            )

        _require_equal(
            errors,
            record,
            ("treatment", "name"),
            "torch._dynamo.config.nested_graph_breaks",
        )
        _require_equal(errors, record, ("treatment", "requested"), bool(enabled))
        _require_equal(errors, record, ("treatment", "actual"), bool(enabled))
        _require_equal(
            errors, record, ("treatment", "only_intended_arm_difference"), True
        )

        spec = CASES[str(case_name)]
        workload_expectations = {
            "depth": spec["depth"],
            "explicit_graph_breaks": spec["explicit_graph_breaks"],
            "pre_pointwise_ops_per_frame": 4,
            "post_pointwise_ops_per_frame": 4,
            "pointwise_math": ["sin", "multiply", "residual_add", "cos"],
            "input_elements": INPUT_ELEMENTS,
            "input_bytes": INPUT_BYTES,
            "input_dtype": "torch.float32",
            "input_distribution": "torch.randn with per-process CUDA generator",
        }
        for name, expected in workload_expectations.items():
            _require_equal(errors, record, ("workload", name), expected)
        _require_hash(errors, record, ("workload", "generated_source_sha256"))
        _require_hash(errors, record, ("workload", "math_signature_sha256"))
        generated_source = _nested(record, "workload", "generated_source")
        generated_source_hash = _nested(record, "workload", "generated_source_sha256")
        if not isinstance(generated_source, str):
            errors.append(f"{source}: generated workload source is missing")
        else:
            recomputed_source_hash = hashlib.sha256(
                generated_source.encode("utf-8")
            ).hexdigest()
            if recomputed_source_hash != generated_source_hash:
                errors.append(f"{source}: generated workload source hash mismatch")
            if (
                generated_source.count("torch._dynamo.graph_break()")
                != spec["explicit_graph_breaks"]
            ):
                errors.append(f"{source}: generated workload graph-break count changed")
            if ".item(" in generated_source:
                errors.append(f"{source}: generated workload contains tensor.item sync")
        expected_math_signature = {
            "depth": spec["depth"],
            "pre": ["sin", "multiply:0.5", "residual_add", "cos"],
            "post": ["sin", "multiply:0.25", "residual_add", "cos"],
        }
        _require_equal(
            errors, record, ("workload", "math_signature"), expected_math_signature
        )
        recomputed_math_hash = hashlib.sha256(
            json.dumps(
                expected_math_signature, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if _nested(record, "workload", "math_signature_sha256") != recomputed_math_hash:
            errors.append(f"{source}: mathematical workload hash mismatch")

        compile_expectations = {
            "api": "torch.compile",
            "backend": "inductor",
            "mode": "reduce-overhead",
            "fullgraph": False,
        }
        for name, expected in compile_expectations.items():
            _require_equal(errors, record, ("compile", name), expected)
        _require_equal(
            errors,
            record,
            ("compile", "mode_options", "triton.cudagraphs"),
            True,
        )

        _require_equal(errors, record, ("device", "type"), "cuda")
        _require_equal(errors, record, ("device", "logical_index"), 0)
        gpu_name = _nested(record, "device", "torch_name")
        if not isinstance(gpu_name, str) or "B200" not in gpu_name.upper():
            errors.append(f"{source}: device.torch_name is not a B200")
        expected_uuid = _nested(record, "device", "expected_gpu_uuid")
        visible_uuid = _nested(record, "device", "cuda_visible_devices")
        observed_uuid = _nested(record, "device", "observed_uuid")
        if (
            not isinstance(expected_uuid, str)
            or not expected_uuid.upper().startswith("GPU-")
            or visible_uuid != expected_uuid
            or observed_uuid != expected_uuid
        ):
            errors.append(
                f"{source}: UUID mask/observation does not match expected GPU UUID"
            )

        _require_equal(errors, record, ("correctness", "pass"), True)
        for correctness_phase in (
            "first_compiled_call",
            "post_steady_compiled_call",
        ):
            for field in (
                "pass",
                "assert_close_pass",
                "finite",
                "shape_match",
                "dtype_match",
            ):
                _require_equal(
                    errors,
                    record,
                    ("correctness", correctness_phase, field),
                    True,
                )
            _require_equal(
                errors, record, ("correctness", correctness_phase, "rtol"), 1e-4
            )
            _require_equal(
                errors, record, ("correctness", correctness_phase, "atol"), 1e-5
            )
            _require_equal(
                errors,
                record,
                ("correctness", correctness_phase, "reference_shape"),
                [INPUT_ELEMENTS],
            )
            _require_equal(
                errors,
                record,
                ("correctness", correctness_phase, "compiled_shape"),
                [INPUT_ELEMENTS],
            )
            _require_equal(
                errors,
                record,
                ("correctness", correctness_phase, "reference_dtype"),
                "torch.float32",
            )
            _require_equal(
                errors,
                record,
                ("correctness", correctness_phase, "compiled_dtype"),
                "torch.float32",
            )
            _validate_sample_hash(errors, record, correctness_phase)
        first_sample_hash = _nested(
            record,
            "correctness",
            "first_compiled_call",
            "sample_output",
            "sha256",
        )
        final_sample_hash = _nested(
            record,
            "correctness",
            "post_steady_compiled_call",
            "sample_output",
            "sha256",
        )
        if first_sample_hash != final_sample_hash:
            errors.append(f"{source}: first and post-steady sampled outputs differ")

        if not _is_finite_positive(
            _nested(record, "timing", "first_compiled_call_host_ms")
        ):
            errors.append(f"{source}: first compiled-call timing is invalid")
        cold = _validate_raw_vector(
            errors, record, ("timing", "cold_first_10", "raw_host_ms"), COLD_CALLS
        )
        latency = _validate_raw_vector(
            errors,
            record,
            ("timing", "steady_invocation_latency", "raw_host_ms"),
            LATENCY_TRIALS,
        )
        _require_equal(errors, record, ("timing", "warmup_calls"), WARMUP_CALLS)
        _require_equal(
            errors, record, ("timing", "compiler_counter_stability", "stable"), True
        )
        _require_equal(
            errors,
            record,
            ("timing", "compiler_counter_stability", "steady_latency_delta"),
            {},
        )
        _require_equal(
            errors,
            record,
            ("timing", "compiler_counter_stability", "batched_throughput_delta"),
            {},
        )
        blocks = _nested(record, "timing", "batched_throughput", "blocks")
        _require_equal(
            errors,
            record,
            ("timing", "batched_throughput", "calls_per_block"),
            THROUGHPUT_CALLS_PER_BLOCK,
        )
        if not isinstance(blocks, list) or len(blocks) != THROUGHPUT_BLOCKS:
            errors.append(
                f"{source}: throughput blocks must contain {THROUGHPUT_BLOCKS} blocks"
            )
        else:
            for block_index, block in enumerate(blocks):
                if (
                    not isinstance(block, Mapping)
                    or not _is_finite_positive(block.get("total_ms"))
                    or not _is_finite_positive(block.get("per_call_ms"))
                ):
                    errors.append(
                        f"{source}: throughput block {block_index} is invalid"
                    )
                    continue
                expected_per_call = (
                    float(block["total_ms"]) / THROUGHPUT_CALLS_PER_BLOCK
                )
                if not math.isclose(
                    float(block["per_call_ms"]),
                    expected_per_call,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{source}: throughput block {block_index} per-call math is wrong"
                    )
        if cold is not None:
            recorded_total = _nested(record, "timing", "cold_first_10", "total_host_ms")
            if not isinstance(recorded_total, (int, float)) or not math.isclose(
                float(recorded_total), sum(cold), rel_tol=1e-12, abs_tol=1e-9
            ):
                errors.append(
                    f"{source}: cold_first_10.total_host_ms does not match raw data"
                )
        if latency is not None:
            recorded_median = _nested(
                record, "timing", "steady_invocation_latency", "median_host_ms"
            )
            recorded_p95 = _nested(
                record, "timing", "steady_invocation_latency", "p95_host_ms"
            )
            if not isinstance(recorded_median, (int, float)) or not math.isclose(
                float(recorded_median),
                statistics.median(latency),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(f"{source}: steady median does not match raw data")
            if not isinstance(recorded_p95, (int, float)) or not math.isclose(
                float(recorded_p95),
                _percentile(latency, 95.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                errors.append(f"{source}: steady p95 does not match raw data")

        runtime_expectations = {
            "num_threads": 1,
            "num_interop_threads": 1,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        for name, expected in runtime_expectations.items():
            _require_equal(errors, record, ("runtime", name), expected)
        inductor_expectations = {
            "compile_threads": 1,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "autotune_remote_cache": False,
            "bundled_autotune_remote_cache": False,
        }
        for name, expected in inductor_expectations.items():
            _require_equal(
                errors, record, ("runtime", "inductor_options", name), expected
            )
        functorch_expectations = {
            "enable_autograd_cache": False,
            "enable_remote_autograd_cache": False,
            "bundled_autograd_cache": False,
        }
        for name, expected in functorch_expectations.items():
            _require_equal(
                errors, record, ("runtime", "functorch_options", name), expected
            )
        fixed_environment_expectations = {
            "CUDA_CACHE_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
            "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
            "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
        }
        for name, expected in fixed_environment_expectations.items():
            _require_equal(
                errors,
                record,
                ("runtime", "fixed_environment", name),
                expected,
            )
        for name in (
            "torch_version",
            "torch_git_version",
            "torch_cuda_version",
            "python",
        ):
            value = _nested(record, "runtime", name)
            if not isinstance(value, str) or not value:
                errors.append(f"{source}: runtime.{name} is missing")

        contract = _nested(record, "controller_contract")
        if not isinstance(contract, Mapping):
            errors.append(f"{source}: controller_contract is missing")
        else:
            if (
                contract.get("expected_gpu_uuid") != expected_uuid
                or contract.get("cuda_visible_devices") != expected_uuid
            ):
                errors.append(
                    f"{source}: controller contract UUID does not match device UUID"
                )
            cache_dirs = contract.get("cache_dirs")
            if not isinstance(cache_dirs, Mapping):
                errors.append(
                    f"{source}: controller cache directory evidence is missing"
                )
            else:
                per_record_paths: List[str] = []
                for env_name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                    entry = cache_dirs.get(env_name)
                    if (
                        not isinstance(entry, Mapping)
                        or entry.get("initially_empty") is not True
                        or entry.get("initial_entries") != []
                    ):
                        errors.append(f"{source}: {env_name} was not recorded cold")
                        continue
                    cache_path = entry.get("path")
                    if not isinstance(cache_path, str) or not cache_path:
                        errors.append(f"{source}: {env_name} path is missing")
                        continue
                    per_record_paths.append(cache_path)
                    if cache_path in cache_paths:
                        errors.append(
                            f"{source}: cache path reused across fresh processes: {cache_path}"
                        )
                    cache_paths.add(cache_path)
                if (
                    len(per_record_paths) == 2
                    and per_record_paths[0] == per_record_paths[1]
                ):
                    errors.append(
                        f"{source}: Inductor and Triton cache paths are identical"
                    )

        _require_hash(errors, record, ("provenance", "worker_source_sha256"))
        _require_hash(errors, record, ("provenance", "flag_source_snapshot_sha256"))
        _require_equal(errors, record, ("provenance", "flag_source_snapshot_line"), 587)
        _require_equal(
            errors, record, ("provenance", "stderr_capture_owner"), "controller"
        )
        dynamo_stats = _nested(record, "dynamo", "stats")
        if not isinstance(dynamo_stats, Mapping):
            errors.append(f"{source}: Dynamo graph-count stats are missing")
        else:
            unique_graphs = dynamo_stats.get("unique_graphs")
            if (
                not isinstance(unique_graphs, int)
                or isinstance(unique_graphs, bool)
                or unique_graphs <= 0
            ):
                errors.append(
                    f"{source}: Dynamo stats.unique_graphs must be a positive integer"
                )
        graph_break_reasons = _nested(record, "dynamo", "graph_break_reason_counts")
        if not isinstance(graph_break_reasons, Mapping):
            errors.append(f"{source}: Dynamo graph-break reasons are missing")
            graph_break_reasons = {}
        for counter_name in (
            "graph_break_event_sum",
            "graph_break_distinct_reason_count",
        ):
            value = _nested(record, "dynamo", counter_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{source}: Dynamo {counter_name} is missing/invalid")
        graph_break_event_sum = _nested(record, "dynamo", "graph_break_event_sum")
        graph_break_distinct_count = _nested(
            record, "dynamo", "graph_break_distinct_reason_count"
        )
        numeric_reason_counts = [
            count
            for count in graph_break_reasons.values()
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        ]
        if len(numeric_reason_counts) != len(graph_break_reasons):
            errors.append(f"{source}: Dynamo graph-break reason counts are invalid")
        elif graph_break_event_sum != sum(numeric_reason_counts):
            errors.append(
                f"{source}: Dynamo graph-break event sum does not match reasons"
            )
        if graph_break_distinct_count != len(graph_break_reasons):
            errors.append(
                f"{source}: Dynamo distinct graph-break count does not match reasons"
            )
        if case_name == "no_break_depth4":
            if graph_break_event_sum != 0:
                errors.append(f"{source}: no-break control recorded graph-break events")
            if graph_break_distinct_count != 0:
                errors.append(
                    f"{source}: no-break control recorded distinct graph-break reasons"
                )
            if graph_break_reasons:
                errors.append(
                    f"{source}: no-break control recorded graph-break reasons"
                )
        else:
            if (
                not isinstance(graph_break_event_sum, int)
                or isinstance(graph_break_event_sum, bool)
                or graph_break_event_sum <= 0
            ):
                errors.append(
                    f"{source}: nested-break case lacks positive graph-break events"
                )
            if (
                not isinstance(graph_break_distinct_count, int)
                or isinstance(graph_break_distinct_count, bool)
                or graph_break_distinct_count <= 0
            ):
                errors.append(
                    f"{source}: nested-break case lacks distinct graph-break reasons"
                )
            if not graph_break_reasons:
                errors.append(
                    f"{source}: nested-break case lacks a graph-break reason histogram"
                )

    actual_keys = set(indexed)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"missing result identities: {missing}")
    if extra:
        errors.append(f"unexpected result identities: {extra}")

    if actual_keys == expected_keys:
        first = indexed[sorted(indexed)[0]]
        fixed_runtime_signature = {
            "torch_version": _nested(first, "runtime", "torch_version"),
            "torch_git_version": _nested(first, "runtime", "torch_git_version"),
            "torch_cuda_version": _nested(first, "runtime", "torch_cuda_version"),
            "cudnn_version": _nested(first, "runtime", "cudnn_version"),
            "python": _nested(first, "runtime", "python"),
            "platform": _nested(first, "runtime", "platform"),
            "hostname": _nested(first, "runtime", "hostname"),
            "inductor_options": _nested(first, "runtime", "inductor_options"),
            "functorch_options": _nested(first, "runtime", "functorch_options"),
            "fixed_environment": _nested(first, "runtime", "fixed_environment"),
        }
        worker_hash = _nested(first, "provenance", "worker_source_sha256")
        flag_hash = _nested(first, "provenance", "flag_source_snapshot_sha256")
        for key, record in sorted(indexed.items()):
            current_signature = {
                "torch_version": _nested(record, "runtime", "torch_version"),
                "torch_git_version": _nested(record, "runtime", "torch_git_version"),
                "torch_cuda_version": _nested(record, "runtime", "torch_cuda_version"),
                "cudnn_version": _nested(record, "runtime", "cudnn_version"),
                "python": _nested(record, "runtime", "python"),
                "platform": _nested(record, "runtime", "platform"),
                "hostname": _nested(record, "runtime", "hostname"),
                "inductor_options": _nested(record, "runtime", "inductor_options"),
                "functorch_options": _nested(record, "runtime", "functorch_options"),
                "fixed_environment": _nested(record, "runtime", "fixed_environment"),
            }
            if current_signature != fixed_runtime_signature:
                errors.append(
                    f"{_path(record)}: runtime/config provenance differs across arms"
                )
            if _nested(record, "provenance", "worker_source_sha256") != worker_hash:
                errors.append(
                    f"{_path(record)}: worker source hash differs across records"
                )
            if (
                _nested(record, "provenance", "flag_source_snapshot_sha256")
                != flag_hash
            ):
                errors.append(
                    f"{_path(record)}: flag source snapshot hash differs across records"
                )

        for case_name in CASES:
            generated_hashes = {
                _nested(
                    indexed[(case_name, enabled, seed)],
                    "workload",
                    "generated_source_sha256",
                )
                for enabled in ARMS
                for seed in SEEDS
            }
            math_hashes = {
                _nested(
                    indexed[(case_name, enabled, seed)],
                    "workload",
                    "math_signature_sha256",
                )
                for enabled in ARMS
                for seed in SEEDS
            }
            if len(generated_hashes) != 1:
                errors.append(
                    f"{case_name}: generated workload source differs across arms/seeds"
                )
            if len(math_hashes) != 1:
                errors.append(
                    f"{case_name}: mathematical workload differs across arms/seeds"
                )
            for enabled in ARMS:
                seed_hashes = {
                    _nested(
                        indexed[(case_name, enabled, seed)],
                        "correctness",
                        "post_steady_compiled_call",
                        "sample_output",
                        "sha256",
                    )
                    for seed in SEEDS
                }
                if len(seed_hashes) != len(SEEDS):
                    errors.append(
                        f"{case_name} enabled={enabled}: seeded inputs/outputs are not distinct"
                    )

        control_math = _nested(
            indexed[("no_break_depth4", 0, 0)], "workload", "math_signature_sha256"
        )
        break_math = _nested(
            indexed[("nested_break_depth4", 0, 0)], "workload", "math_signature_sha256"
        )
        if control_math != break_math:
            errors.append(
                "depth-4 no-break control and break case do not share identical math"
            )

    if errors:
        raise ValidationError(errors)
    return indexed


def _run_metrics(record: Mapping[str, Any]) -> Dict[str, float]:
    cold = [
        float(value)
        for value in _nested(record, "timing", "cold_first_10", "raw_host_ms")
    ]
    latency = [
        float(value)
        for value in _nested(
            record, "timing", "steady_invocation_latency", "raw_host_ms"
        )
    ]
    blocks = _nested(record, "timing", "batched_throughput", "blocks")
    throughput = [float(block["per_call_ms"]) for block in blocks]
    return {
        "first_compiled_call_host_ms": float(
            _nested(record, "timing", "first_compiled_call_host_ms")
        ),
        "cold_first_10_total_host_ms": sum(cold),
        "cold_first_10_mean_host_ms": statistics.fmean(cold),
        "steady_median_invocation_host_ms": statistics.median(latency),
        "steady_p95_invocation_host_ms": _percentile(latency, 95.0),
        "throughput_median_per_call_host_ms": statistics.median(throughput),
    }


def _comparison_for_case(
    case_name: str, indexed: Mapping[Tuple[str, int, int], Mapping[str, Any]]
) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    cold_ratios: List[float] = []
    steady_ratios: List[float] = []
    throughput_ratios: List[float] = []
    for seed in SEEDS:
        baseline = _run_metrics(indexed[(case_name, 0, seed)])
        candidate = _run_metrics(indexed[(case_name, 1, seed)])
        cold_ratio = (
            baseline["cold_first_10_total_host_ms"]
            / candidate["cold_first_10_total_host_ms"]
        )
        steady_ratio = (
            baseline["steady_median_invocation_host_ms"]
            / candidate["steady_median_invocation_host_ms"]
        )
        throughput_ratio = (
            baseline["throughput_median_per_call_host_ms"]
            / candidate["throughput_median_per_call_host_ms"]
        )
        p95_ratio = (
            candidate["steady_p95_invocation_host_ms"]
            / baseline["steady_p95_invocation_host_ms"]
        )
        cold_ratios.append(cold_ratio)
        steady_ratios.append(steady_ratio)
        throughput_ratios.append(throughput_ratio)
        per_seed.append(
            {
                "seed": seed,
                "baseline_enabled_0": baseline,
                "candidate_enabled_1": candidate,
                "cold_first_10_speedup_ratio": cold_ratio,
                "steady_median_invocation_speedup_ratio": steady_ratio,
                "steady_p95_candidate_over_baseline": p95_ratio,
                "throughput_speedup_ratio": throughput_ratio,
            }
        )

    cold_summary = summarize_seed_ratios(cold_ratios)
    steady_summary = summarize_seed_ratios(steady_ratios)
    throughput_summary = summarize_seed_ratios(throughput_ratios)
    steady_p95_no_regression = all(
        item["steady_p95_candidate_over_baseline"] <= 1.0 for item in per_seed
    )
    steady_p95_within_three_percent = all(
        item["steady_p95_candidate_over_baseline"] <= MAX_STARTUP_STEADY_P95_REGRESSION
        for item in per_seed
    )
    steady_win = bool(
        steady_summary["observed_min_seed_ratio"] >= MIN_OBSERVED_SEED_SPEEDUP
        and steady_p95_no_regression
    )
    startup_win = bool(
        cold_summary["observed_min_seed_ratio"] >= MIN_OBSERVED_SEED_SPEEDUP
        and steady_p95_within_three_percent
    )
    return {
        "case": case_name,
        "role": (
            "predeclared primary"
            if case_name == PRIMARY_CASE
            else "predeclared no-break drift-control gate"
            if case_name == "no_break_depth4"
            else "secondary descriptive case"
        ),
        "per_seed": per_seed,
        "cold_first_10": cold_summary,
        "steady_median_invocation": steady_summary,
        "steady_p95_no_regression_each_seed": steady_p95_no_regression,
        "steady_p95_within_3_percent_each_seed": steady_p95_within_three_percent,
        "batched_throughput": throughput_summary,
        "steady_win_criteria_pass": steady_win,
        "startup_win_criteria_pass": startup_win,
    }


def _control_metric_validity(
    control: Mapping[str, Any], ratio_field: str, metric: str
) -> Dict[str, Any]:
    checks = [
        {
            "seed": int(item["seed"]),
            "baseline_over_candidate_ratio": float(item[ratio_field]),
            "within_predeclared_range": bool(
                CONTROL_RATIO_MIN <= float(item[ratio_field]) <= CONTROL_RATIO_MAX
            ),
        }
        for item in control["per_seed"]
    ]
    valid = all(item["within_predeclared_range"] for item in checks)
    return {
        "status": "PASS" if valid else "FAIL",
        "metric": metric,
        "allowed_inclusive_range": [CONTROL_RATIO_MIN, CONTROL_RATIO_MAX],
        "every_seed_required": True,
        "observed_seed_ratios": [
            item["baseline_over_candidate_ratio"] for item in checks
        ],
        "per_seed": checks,
        "gate_pass": valid,
    }


def analyze_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = validate_records(records)
    cases = {case_name: _comparison_for_case(case_name, indexed) for case_name in CASES}
    primary = cases[PRIMARY_CASE]
    control = cases["no_break_depth4"]
    steady_control = _control_metric_validity(
        control,
        "steady_median_invocation_speedup_ratio",
        "steady median synchronized invocation latency",
    )
    cold_control = _control_metric_validity(
        control,
        "cold_first_10_speedup_ratio",
        "cold first-10 synchronized compiled-call total",
    )
    steady_control_valid = bool(steady_control["gate_pass"])
    startup_control_valid = bool(
        steady_control["gate_pass"] and cold_control["gate_pass"]
    )
    control_validity = {
        "case": "no_break_depth4",
        "ratio_definition": "enabled=0 latency divided by enabled=1 latency",
        "steady": steady_control,
        "cold_first_10": cold_control,
        "steady_qualification_gate_pass": steady_control_valid,
        "startup_qualification_gate_pass": startup_control_valid,
        "failure_semantics": (
            "steady control drift blocks both routes; cold first-10 control drift "
            "blocks the startup route while leaving an otherwise qualified steady "
            "route eligible"
        ),
    }
    if primary["steady_win_criteria_pass"]:
        measurement_gate_pass = True
        measurement_win_kind = "steady_invocation"
    elif primary["startup_win_criteria_pass"]:
        measurement_gate_pass = True
        measurement_win_kind = "startup_only"
    else:
        measurement_gate_pass = False
        measurement_win_kind = None

    profile_evidence = []
    counter_evidence = []
    for key, record in sorted(indexed.items()):
        if bool(record.get("profile_requested")):
            profile_evidence.append(
                {
                    "case": key[0],
                    "enabled": key[1],
                    "seed": key[2],
                    "profile": record.get("profile"),
                }
            )
        counter_evidence.append(
            {
                "case": key[0],
                "enabled": key[1],
                "seed": key[2],
                "dynamo_stats": _nested(record, "dynamo", "stats"),
                "graph_break_event_sum": _nested(
                    record, "dynamo", "graph_break_event_sum"
                ),
                "graph_break_reason_counts": _nested(
                    record, "dynamo", "graph_break_reason_counts"
                ),
                "graph_break_histogram_nonempty": bool(
                    _nested(record, "dynamo", "graph_break_reason_counts")
                ),
                "inductor_metrics": record.get("inductor_metrics"),
            }
        )

    qualified_profile_arms = [
        item["enabled"]
        for item in profile_evidence
        if isinstance(item.get("profile"), Mapping)
        and item["profile"].get("status") == "captured"
        and item["profile"].get("evidence_complete") is True
        and int(item["profile"].get("cuda_graph_launch_event_count", 0)) > 0
        and int(item["profile"].get("cuda_kernel_event_count", 0)) > 0
    ]
    capture_qualified = sorted(qualified_profile_arms) == list(ARMS)
    capture_qualification = {
        "status": "PASS" if capture_qualified else "FAIL",
        "required_identities": [
            {"case": PRIMARY_CASE, "enabled": enabled, "seed": 0} for enabled in ARMS
        ],
        "qualified_arms": sorted(qualified_profile_arms),
        "cuda_graph_claim_permitted": capture_qualified,
        "failure_semantics": (
            "failed or incomplete profiler evidence does not invalidate retained "
            "timings, but it forbids a CUDA Graph capture claim"
        ),
    }
    qualified_steady_win = bool(
        primary["steady_win_criteria_pass"]
        and steady_control_valid
        and capture_qualified
    )
    qualified_startup_win = bool(
        primary["startup_win_criteria_pass"]
        and startup_control_valid
        and capture_qualified
    )
    if qualified_steady_win:
        verdict = "PASS"
        win_kind = "steady_invocation"
    elif qualified_startup_win:
        verdict = "PASS"
        win_kind = "startup_only"
    else:
        verdict = "INCONCLUSIVE"
        win_kind = None
    selected_route_control_valid = bool(
        steady_control_valid
        if measurement_win_kind == "steady_invocation"
        else startup_control_valid
        if measurement_win_kind == "startup_only"
        else False
    )

    return {
        "schema": OUTPUT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "scope": SCOPE,
        "generated_at": _utc_now(),
        "validation": {
            "status": "PASS",
            "records": len(records),
            "expected_matrix": {
                "cases": list(CASES),
                "arms": list(ARMS),
                "seeds": list(SEEDS),
            },
            "all_correctness_passed": True,
            "fresh_process_cache_paths_unique": True,
            "graph_break_mechanism_validated": True,
        },
        "predeclared_primary_case": PRIMARY_CASE,
        "decision": {
            "verdict": verdict,
            "verdict_scope": (
                "predeclared performance thresholds, no-break control validity, "
                "and two-arm CUDA Graph capture qualification"
            ),
            "win_kind": win_kind,
            "measurement_gate_pass": measurement_gate_pass,
            "measurement_win_kind": measurement_win_kind,
            "control_validity_pass": selected_route_control_valid,
            "steady_control_validity_pass": steady_control_valid,
            "startup_control_validity_pass": startup_control_valid,
            "capture_qualification_pass": capture_qualified,
            "qualified_steady_win": qualified_steady_win,
            "qualified_startup_win": qualified_startup_win,
            "primary_case": PRIMARY_CASE,
            "steady_rule": (
                "observed minimum of four paired-seed median speedup ratios >= 1.10; "
                "candidate invocation p95 <= baseline for every seed"
            ),
            "startup_rule": (
                "observed minimum of four paired-seed cold-first-10 speedup ratios "
                ">= 1.10; candidate steady invocation p95 "
                "no more than 3% above baseline for every seed"
            ),
            "control_rule": (
                "no_break_depth4 steady median baseline/candidate ratio must be "
                "within [0.95, 1.05] for every seed on both routes; startup also "
                "requires its cold first-10 ratio within [0.95, 1.05] for every seed"
            ),
            "failure_semantics": (
                "INCONCLUSIVE retains the full raw matrix; the measurement gate, "
                "control-validity gate, capture qualification, or more than one did "
                "not pass, so no qualified optimization-win claim is supported"
            ),
            "cuda_graph_claim_permitted": capture_qualified,
        },
        "cases": cases,
        "counter_evidence": counter_evidence,
        "profile_evidence": profile_evidence,
        "control_validity": control_validity,
        "capture_qualification": capture_qualification,
        "metric_definition": {
            "p95": "type-7 percentile over 100 synchronized host-timed invocations",
            "speedup_ratio": "enabled=0 latency divided by enabled=1 latency",
            "paired_seed_summary": {
                "paired_unit": "seed",
                "seeds": list(SEEDS),
                "acceptance_statistic": "observed minimum paired-seed ratio",
                "descriptive_statistic": "geometric mean of four seed ratios",
                "confidence_interval": None,
                "coverage_claim": None,
            },
        },
        "claim_limits": [
            "This is a controlled compiler microbenchmark.",
            "Invocation p95 is not a serving latency percentile.",
            "The eager execution is a correctness reference, not a performance competitor.",
            "The no-break case gates drift validity and does not establish nested-break benefit.",
            "No serving, model throughput, or model quality conclusion follows.",
        ],
        "retention": {
            "raw_results_retained": True,
            "raw_result_paths": [_path(record) for record in records],
        },
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    decision = analysis["decision"]
    lines = [
        "# Nested graph-break B200 microbenchmark",
        "",
        f"Verdict: **{decision['verdict']}**"
        + (f" ({decision['win_kind']})" if decision.get("win_kind") else ""),
        "",
        "CUDA Graph capture qualification: **{}**".format(
            analysis["capture_qualification"]["status"]
        ),
        "",
        "No-break steady control: **{}**".format(
            analysis["control_validity"]["steady"]["status"]
        ),
        "",
        "No-break cold first-10 control: **{}**".format(
            analysis["control_validity"]["cold_first_10"]["status"]
        ),
        "",
        "This is a controlled compiler microbenchmark. Invocation p95 is not a serving latency percentile.",
        "",
        "| Case | Seed | Cold first-10 speedup | Steady median speedup | Candidate/baseline p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for case_name in CASES:
        case = analysis["cases"][case_name]
        for item in case["per_seed"]:
            lines.append(
                "| {case} | {seed} | {cold:.4f}x | {steady:.4f}x | {p95:.4f}x |".format(
                    case=case_name,
                    seed=item["seed"],
                    cold=item["cold_first_10_speedup_ratio"],
                    steady=item["steady_median_invocation_speedup_ratio"],
                    p95=item["steady_p95_candidate_over_baseline"],
                )
            )
    lines.extend(
        [
            "",
            "The performance decision uses only the predeclared `nested_break_depth8` case. The `no_break_depth4` case is the predeclared drift-control gate; the other depth-4 result is descriptive.",
            "",
            "Raw result paths and compiler counters remain in the JSON analysis and source result directories.",
            "",
        ]
    )
    if not analysis["capture_qualification"]["cuda_graph_claim_permitted"]:
        lines.extend(
            [
                "CUDA Graph profiler evidence was incomplete, so this report makes no CUDA Graph capture claim.",
                "",
            ]
        )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="24 result directories/files, or a parent directory containing them.",
    )
    parser.add_argument("--out", type=Path, help="Write the full JSON analysis here.")
    parser.add_argument(
        "--markdown", type=Path, help="Write a short Markdown report here."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        records = discover_records(args.inputs)
        analysis = analyze_records(records)
        exit_code = 0
    except ValidationError as exc:
        analysis = {
            "schema": OUTPUT_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "scope": SCOPE,
            "generated_at": _utc_now(),
            "validation": {"status": "FAIL", "errors": exc.errors},
            "decision": {
                "verdict": "INVALID",
                "win_kind": None,
                "failure_semantics": "raw evidence retained; no performance claim",
            },
            "retention": {
                "raw_results_retained": True,
                "raw_input_paths": [
                    str(path.expanduser().resolve()) for path in args.inputs
                ],
            },
        }
        exit_code = 2

    serialized = json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        _atomic_write(args.out.expanduser().resolve(), serialized)
    else:
        sys.stdout.write(serialized)
    if args.markdown:
        if analysis["validation"]["status"] == "PASS":
            markdown = render_markdown(analysis)
        else:
            markdown = (
                "# Invalid benchmark evidence\n\n"
                + "\n".join(
                    f"- {error}" for error in analysis["validation"].get("errors", [])
                )
                + "\n"
            )
        _atomic_write(args.markdown.expanduser().resolve(), markdown)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
