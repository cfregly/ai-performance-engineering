#!/usr/bin/env python3
"""Deterministic qualification for the fixed two-B200 TE training study.

This module validates retained bytes and recomputes the fixed scientific
decision. It never launches PIQ, allocates an accelerator, or creates a
reader-facing package. GPU receipt v1, CPU receipt v2, and PIQ envelope v1
are unsupported. Receipt v3 plus PIQ envelope v2 is the only active measured
contract.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

if __package__:
    from .piq_gpu_training_evidence import validate_bundle as validate_gpu_bundle
else:
    from piq_gpu_training_evidence import validate_bundle as validate_gpu_bundle


CONTRACT_ID = "te_mxfp8_bf16_two_b200_v1"
DESIGN_KIND = "qualified_gpu_training_design"
REPORT_KIND = "qualified_gpu_training_report"
PAIR_IDS = ("pair-01", "pair-02", "pair-03", "pair-04")
SCHEDULE = (
    {"id": "admission-01", "kind": "admission", "order": ["baseline", "candidate"]},
    {"id": "pair-01", "kind": "pair", "order": ["baseline", "candidate"]},
    {"id": "pair-02", "kind": "pair", "order": ["candidate", "baseline"]},
    {"id": "pair-03", "kind": "pair", "order": ["candidate", "baseline"]},
    {"id": "pair-04", "kind": "pair", "order": ["baseline", "candidate"]},
)
REQUIRED_PHASES = (
    "data_open",
    "nccl_init",
    "periodic_save",
    "checkpoint_load",
    "restart_to_first_update",
    "final_save",
    "validation",
    "evidence_flush",
    "cleanup",
)
ADMISSION_PHASES = ("data_open", "nccl_init", "evidence_flush", "cleanup")
REQUIRED_LAYERS = (
    "host",
    "accelerator",
    "fabric",
    "storage",
    "orchestration",
    "runtime",
    "workload",
)
REQUIRED_IMPORT_ROLES = (
    "runner",
    "adapter",
    "verifier",
    "evaluator",
    "checkpoint",
    "telemetry",
    "helper",
)
ELIGIBLE_UPDATE_INDICES = tuple((*range(13, 96), *range(98, 193)))

# These five definitions close the machine-semantics gaps identified before
# constants were admitted.  Every design must embed this object byte-for-value.
MEASUREMENT_SEMANTICS = {
    "version": "gpu-training-machine-semantics-v1",
    "p95": {
        "estimator": "nearest_rank_ceiling",
        "eligible_update_indices": list(ELIGIBLE_UPDATE_INDICES),
        "rank_formula": "ceil(95*n/100), one_based",
        "excluded_update_indices": [*range(1, 13), 96, 97],
    },
    "coefficient_of_variation": {
        "standard_deviation": "population",
        "rounding": "none_for_gate",
        "gate_formula": "400*sum((n*x_i-sum_x)^2)<=n*sum_x^2",
        "display_decimal_places": 6,
        "display_rounding": "half_even",
    },
    "intervals": {
        "model": "half_open_monotonic_nanoseconds",
        "phase_duration": "union_across_ranks_no_clipping",
        "touching_intervals_merge": True,
        "coverage_rule": "each_rank_union_equals_primary_clock",
        "unexplained_gap": "any_positive_uncovered_nanosecond",
        "cross_phase_overlap": "allowed_and_not_summed",
    },
    "quality": {
        "nll_representation": "integer_total_nanonats_and_integer_target_bytes",
        "nll_gate_rounding": "none_rational_cross_multiplication",
        "nll_display_decimal_places": 9,
        "accuracy_representation": "integer_correct_bytes_and_integer_target_bytes",
        "accuracy_gate_rounding": "none_rational_cross_multiplication",
        "accuracy_display_decimal_places": 6,
        "display_rounding": "half_even",
    },
    "power": {
        "source": "nvidia_smi_power_limit_milliwatts",
        "observations_per_device_per_attempt": 2,
        "observation_labels": ["start", "end"],
        "absolute_tolerance_mw": 1000,
        "mutation_allowed": False,
    },
}

CRITERIA = {
    "criteria_version": "te-mxfp8-bf16-two-b200-v1",
    "accepted_pair_count": 4,
    "attempt_count": 10,
    "minimum_each_pair_gain": "1.05",
    "minimum_median_gain": "1.10",
    "maximum_gain_spread": "0.10",
    "maximum_within_arm_population_cv": "0.05",
    "maximum_median_candidate_to_baseline_p95": "0.95",
    "maximum_each_candidate_to_baseline_p95": "1.02",
    "minimum_absolute_nll_improvement_nats_per_byte": "0.10",
    "maximum_each_candidate_nll_gap_nats_per_byte": "0.03",
    "maximum_median_candidate_nll_gap_nats_per_byte": "0.02",
    "maximum_each_candidate_accuracy_gap_percentage_points": "0.75",
    "maximum_median_candidate_accuracy_gap_percentage_points": "0.50",
    "phase_candidate_multiplier": "1.10",
    "phase_candidate_slack_nanoseconds": 2_000_000_000,
    "updates_per_full_attempt": 192,
    "accumulation_rounds_per_full_attempt": 768,
    "rank_local_microbatches_per_full_attempt": 1536,
    "target_bytes_per_full_attempt": 12_582_912,
    "oracle_target_bytes": 1_048_576,
    "checkpoint_update": 96,
    "first_resumed_update": 97,
    "dry_forward_tolerance_nanonats": 1000,
    "no_failed_missing_retried_or_unaccepted_attempts": True,
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
RECEIPT_FIELDS = {
    "schema_version", "state", "execution_class", "workload_family",
    "contract_id", "workload_scope", "runtime_identifier",
    "hardware_identifier", "design", "registration",
    "source_import_closure", "baseline_review", "telemetry_review",
    "raw_census", "correctness_quality", "checkpoint_restart",
    "telemetry_coverage", "custody_readback_cleanup",
    "qualification_report", "scientific_review", "source_artifacts",
    "piq_envelope", "accepted_repeats", "failed_attempts",
    "profile_evidence", "claims",
}


class EvidenceError(ValueError):
    """Malformed or semantically inconsistent retained evidence."""


def canonical_json(value: Any) -> bytes:
    """Return the only accepted serialized form for deterministic reports."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise EvidenceError(f"nonfinite JSON constant: {value}")


def strict_json_bytes(raw: bytes, label: str = "JSON") -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_duplicate_keys,
                          parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError, EvidenceError) as exc:
        raise EvidenceError(f"invalid {label}: {exc}") from exc


def strict_load_json(path: Path, label: str = "JSON") -> Any:
    try:
        return strict_json_bytes(path.read_bytes(), label)
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _filled_text(value: Any) -> bool:
    return _text(value) and not value.startswith("UNFILLED_")


def _digest_text(value: Any) -> bool:
    return (isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
            and value != "0" * 64)


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and all(part not in ("", ".", "..")
                                           for part in path.parts))


def _ref(package: Path, value: Any, label: str, errors: list[str],
         *, prefix: str = "evidence/") -> Path | None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        errors.append(f"{label} must contain exactly path and sha256")
        return None
    relative = value.get("path")
    if not _safe_relative(relative) or (prefix and not str(relative).startswith(prefix)):
        errors.append(f"{label}.path must be a safe {prefix or 'package-relative '}path")
        return None
    digest = value.get("sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        errors.append(f"{label}.sha256 must be a lowercase SHA-256 digest")
        return None
    target = package
    try:
        for part in PurePosixPath(str(relative)).parts:
            target /= part
            if target.is_symlink():
                errors.append(f"{label} must not use symlinks: {relative}")
                return None
        target.resolve(strict=True).relative_to(package.resolve(strict=True))
    except (OSError, ValueError, RuntimeError):
        errors.append(f"{label} is missing or escapes the package: {relative}")
        return None
    try:
        if not target.is_file() or target.stat().st_size == 0:
            errors.append(f"{label} must name a nonempty regular file: {relative}")
            return None
        if _sha256(target) != digest:
            errors.append(f"{label}.sha256 does not match {relative}")
    except OSError as exc:
        errors.append(f"{label} is unreadable: {relative}: {exc}")
        return None
    return target


def _require_fields(value: Any, expected: set[str], label: str,
                    errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        errors.append(f"{label} fields differ (missing={missing}, extra={extra})")
        return False
    return True


def _canonical_absolute(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        return False
    path = PurePosixPath(value)
    return str(path) == value and all(part not in ("", ".", "..") for part in path.parts[1:])


def validate_gpu_design(design: Any) -> list[str]:
    """Validate the immutable, result-free design and the five machine semantics."""
    errors: list[str] = []
    fields = {
        "schema_version", "kind", "contract_id", "study_id", "workload_scope",
        "created_at", "execution_owner", "execution_class", "accelerator_count",
        "synthetic_fixture", "model", "data", "arms", "runtime", "topology",
        "determinism", "schedule", "criteria", "measurement_semantics",
        "required_phases", "budgets", "retention", "power_policy",
        "source_imports", "primary_clock_boundary",
    }
    if not _require_fields(design, fields, "GPU design", errors):
        return errors
    if (design.get("schema_version"), design.get("kind"), design.get("contract_id"),
            design.get("execution_class"), design.get("accelerator_count")) != (
                1, DESIGN_KIND, CONTRACT_ID, "gpu", 2):
        errors.append("GPU design version/kind/contract/execution class must match v3")
    for field in ("study_id", "workload_scope", "execution_owner"):
        if not _filled_text(design.get(field)):
            errors.append(f"GPU design {field} must be a nonempty string")
    if _timestamp(design.get("created_at")) is None:
        errors.append("GPU design created_at must be a UTC timestamp")
    if type(design.get("synthetic_fixture")) is not bool:
        errors.append("GPU design synthetic_fixture must be an explicit boolean")
    if design.get("schedule") != list(SCHEDULE):
        errors.append("GPU design schedule must be admission then the fixed four balanced pairs")
    if design.get("criteria") != CRITERIA:
        errors.append("GPU design criteria must equal exported CRITERIA")
    if design.get("measurement_semantics") != MEASUREMENT_SEMANTICS:
        errors.append("GPU design must embed the ratified machine semantics exactly")
    if design.get("required_phases") != list(REQUIRED_PHASES):
        errors.append("GPU design required phases must equal exported REQUIRED_PHASES")
    if design.get("primary_clock_boundary") != (
        "before_initial_checkpoint_open_through_post_cleanup_evidence_readback"
    ):
        errors.append("GPU design primary clock must use the complete-job boundary")

    model = design.get("model")
    required_model = {
        "blocks": 24, "hidden_width": 2048, "mlp_width": 8192,
        "attention_heads": 16, "head_dimension": 128, "sequence_length": 2048,
        "rank_local_microbatch": 4, "gradient_accumulation": 4,
        "global_batch": 32, "optimizer_updates": 192,
        "attention_backend": "te_fused_attention", "activation": "gelu",
        "normalization": "rmsnorm", "parameter_dtype": "bf16",
        "dropout": "0", "bias": False, "activation_recomputation": False,
        "compile": False, "cuda_graphs": False,
    }
    model_fields = set(required_model) | {"optimizer", "parameter_manifest_sha256"}
    if (not isinstance(model, dict) or set(model) != model_fields
            or any(model.get(k) != v for k, v in required_model.items())):
        errors.append("GPU design model does not freeze the selected TE workload")
    elif not isinstance(model.get("optimizer"), dict) or not _digest_text(model.get("parameter_manifest_sha256")):
        errors.append("GPU design model must bind optimizer/master state and parameter manifest")
    else:
        expected_optimizer = {
            "name": "Transformer Engine FusedAdam", "lr_peak": "0.0002",
            "betas": ["0.9", "0.95"], "eps": "0.00000001",
            "weight_decay": "0.1", "master_weights": "float32",
            "moments": "float32", "gradient_clip_norm": "1.0",
            "schedule": "16-update linear warmup then cosine to 0.00002",
        }
        if model["optimizer"] != expected_optimizer:
            errors.append("GPU design optimizer/master/moment/schedule policy drifted")

    data = design.get("data")
    data_fields = {
        "dataset_revision", "remote_files", "converter_sha256", "train_sha256",
        "validation_sha256", "sampler_sha256", "validation_oracle_windows",
        "seed", "cache_policy",
    }
    if not _require_fields(data, data_fields, "GPU design data", errors):
        pass
    elif (data.get("dataset_revision") != "b08601e04326c79dfdd32d625aee71d232d685c3"
          or data.get("validation_oracle_windows") != 512 or data.get("seed") != 1701
          or data.get("cache_policy") != "shared_staged_preread_once"):
        errors.append("GPU design data identity/sampler/oracle/cache policy drifted")
    if isinstance(data, dict):
        for field in ("converter_sha256", "train_sha256", "validation_sha256", "sampler_sha256"):
            if not _digest_text(data.get(field)):
                errors.append(f"GPU design data {field} must be a SHA-256 digest")
        expected_remote = [
            {"path": "train-00000-of-00002.parquet", "sha256": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"},
            {"path": "train-00001-of-00002.parquet", "sha256": "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"},
            {"path": "validation-00000-of-00001.parquet", "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"},
        ]
        if data.get("remote_files") != expected_remote:
            errors.append("GPU design must bind all three pinned remote dataset files")

    arms = design.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"baseline", "candidate", "sole_changed_variable"}:
        errors.append("GPU design arms must contain baseline, candidate and the sole change")
    else:
        expected_settings = {
            "baseline": "te.autocast(enabled=False)",
            "candidate": "te.autocast(enabled=True, recipe=MXFP8BlockScaling(fp8_format=Format.E4M3))",
        }
        for role, setting in expected_settings.items():
            arm = arms.get(role)
            if (not isinstance(arm, dict) or set(arm) != {"id", "configuration", "configuration_sha256"}
                    or arm.get("id") != ("bf16" if role == "baseline" else "mxfp8")
                    or arm.get("configuration") != setting
                    or arm.get("configuration_sha256") != _hash_value({"configuration": setting})):
                errors.append(f"GPU design {role} arm identity/configuration drifted")
        if arms.get("sole_changed_variable") != "transformer_engine_forward_autocast_recipe":
            errors.append("GPU design must name only the TE forward autocast recipe as changed")

    runtime = design.get("runtime")
    pins = {
        "pytorch": "2.15.0", "transformer_engine": "2.20.2",
        "transformer_engine_commit": "6ea2a74a9e98c99e6d7b164a33775cc457520027",
        "cuda_toolkit": "13.4.3", "cublas": "13.8.2",
    }
    runtime_fields = {
        "pytorch", "transformer_engine", "transformer_engine_commit",
        "cuda_toolkit", "cublas", "cudnn", "nccl", "driver",
        "image_digest", "python", "compiler_extensions",
        "attention_backend", "runtime_identifier",
    }
    if (not isinstance(runtime, dict) or set(runtime) != runtime_fields
            or any(runtime.get(k) != v for k, v in pins.items())):
        errors.append("GPU design runtime lacks the admitted TE/PyTorch/CUDA/cuBLAS pins")
    elif (any(not _filled_text(runtime.get(field)) for field in (
            "cudnn", "nccl", "driver", "image_digest", "python",
            "compiler_extensions", "runtime_identifier"))
          or runtime.get("attention_backend") != "te_fused_attention"
          or not str(runtime.get("image_digest", "")).startswith("sha256:")
          or not _digest_text(str(runtime.get("image_digest", ""))[7:])):
        errors.append("GPU design runtime must bind every loaded-library/environment identity")

    topology = design.get("topology")
    if not isinstance(topology, dict):
        errors.append("GPU design topology must be an object")
    else:
        topology_fields = {
            "node_count", "accelerator_count", "process_count", "ranks",
            "local_ranks", "compute_capability", "launcher", "backend",
            "parallelism", "device_uuids", "topology_sha256",
        }
        expected = {
            "node_count": 1, "accelerator_count": 2, "process_count": 2,
            "ranks": [0, 1], "local_ranks": [0, 1], "compute_capability": [10, 0],
            "launcher": "torchrun --standalone --nnodes=1 --nproc-per-node=2",
            "backend": "nccl", "parallelism": "ddp_only",
        }
        if set(topology) != topology_fields or any(topology.get(k) != v for k, v in expected.items()):
            errors.append("GPU design topology must be one host, two B200 ranks, DDP/NCCL")
        uuids = topology.get("device_uuids")
        if (not isinstance(uuids, list) or len(uuids) != 2
                or len(set(uuids)) != 2 or not all(_filled_text(v) for v in uuids)):
            errors.append("GPU design topology must bind two distinct device UUIDs")
        if not _digest_text(topology.get("topology_sha256")):
            errors.append("GPU design topology_sha256 must be a SHA-256 digest")

    determinism = design.get("determinism")
    if not isinstance(determinism, dict) or set(determinism) != {
        "seed", "nvte_allow_nondeterministic_algo", "torch_deterministic_errors",
        "cudnn_benchmark",
    } or (
        determinism.get("seed"), determinism.get("nvte_allow_nondeterministic_algo"),
        determinism.get("torch_deterministic_errors"), determinism.get("cudnn_benchmark")
    ) != (1701, "0", True, False):
        errors.append("GPU design determinism policy drifted")

    power = design.get("power_policy")
    uuids = topology.get("device_uuids", []) if isinstance(topology, dict) else []
    if not isinstance(power, dict) or set(power) != {
        "source", "expected_limit_mw_by_device", "tolerance_mw",
        "observations_per_attempt_per_device", "observation_labels", "mutation_allowed",
    }:
        errors.append("GPU design power policy fields must match the ratified rule")
    elif (power.get("source") != MEASUREMENT_SEMANTICS["power"]["source"]
          or power.get("tolerance_mw") != 1000
          or power.get("observations_per_attempt_per_device") != 2
          or power.get("observation_labels") != ["start", "end"]
          or power.get("mutation_allowed") is not False
          or set(power.get("expected_limit_mw_by_device", {})) != set(uuids)
          or any(not _positive_int(v) for v in power.get("expected_limit_mw_by_device", {}).values())):
        errors.append("GPU design power policy must require two read-only per-device samples within 1000 mW")

    imports = design.get("source_imports")
    if not isinstance(imports, list) or [item.get("role") for item in imports if isinstance(item, dict)] != list(REQUIRED_IMPORT_ROLES):
        errors.append("GPU design must bind the exact resolved first-party import-role closure")
    else:
        seen_paths: set[str] = set()
        for item in imports:
            if not isinstance(item, dict) or set(item) != {
                "role", "expected_absolute_path", "expected_sha256",
                "resolved_absolute_path", "resolved_sha256",
            }:
                errors.append("GPU design source import entry fields are invalid")
                continue
            expected = item.get("expected_absolute_path")
            resolved = item.get("resolved_absolute_path")
            if (not _canonical_absolute(expected) or "UNFILLED_" in expected
                    or resolved != expected or expected in seen_paths):
                errors.append("GPU design resolved imports must equal distinct admitted absolute paths")
            else:
                seen_paths.add(expected)
            digest = item.get("expected_sha256")
            if not _digest_text(digest) or item.get("resolved_sha256") != digest:
                errors.append("GPU design resolved import hashes must equal admitted SHA-256 values")

    budgets = design.get("budgets")
    if (not isinstance(budgets, dict) or set(budgets) != {
            "gpu_count", "gpu_hours", "additional_spend_usd",
            "hard_spend_cap_usd", "hard_wall_minutes",
        } or budgets != {
            "gpu_count": 2, "gpu_hours": 2, "additional_spend_usd": 0,
            "hard_spend_cap_usd": 50, "hard_wall_minutes": 60,
        }):
        errors.append("GPU design must freeze the two-GPU, two-GPU-hour, zero-spend budget")
    retention = design.get("retention")
    if (not isinstance(retention, dict) or set(retention) != {
            "initial_checkpoint_sha256", "keep_raw_evidence",
            "keep_restartable_final_checkpoint_per_arm", "deletions_recorded",
        } or not _digest_text(retention.get("initial_checkpoint_sha256"))
          or retention.get("keep_raw_evidence") is not True
          or retention.get("keep_restartable_final_checkpoint_per_arm") is not True
          or retention.get("deletions_recorded") is not True):
        errors.append("GPU design retention/cleanup policy is incomplete")
    return errors


def _merge_intervals(intervals: Any, start: int, end: int,
                     label: str, errors: list[str]) -> list[tuple[int, int]]:
    if not isinstance(intervals, list) or not intervals:
        errors.append(f"{label} must contain intervals")
        return []
    parsed: list[tuple[int, int]] = []
    for index, interval in enumerate(intervals):
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(value) is not int for value in interval)
                or not start <= interval[0] < interval[1] <= end):
            errors.append(f"{label}[{index}] must be a nonempty half-open interval inside the primary clock")
            continue
        parsed.append((interval[0], interval[1]))
    parsed.sort()
    merged: list[tuple[int, int]] = []
    for left, right in parsed:
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def _union_duration(intervals: Any, start: int, end: int,
                    label: str, errors: list[str]) -> int:
    return sum(right - left for left, right in _merge_intervals(intervals, start, end, label, errors))


def validate_gpu_checkpoint(checkpoint: Any,
                            detached_objects: dict[str, dict] | None = None) -> list[str]:
    """Validate checkpoint/restart state without materializing detached tensors."""
    errors: list[str] = []
    fields = {
        "periodic_files", "final_files", "pre_exit_state", "post_load_state",
        "state_components", "saved_next_batch_sha256", "resumed_next_batch_sha256",
        "pre_exit_dry_forward_loss_nanonats", "reloaded_dry_forward_loss_nanonats",
        "step_before_exit", "step_after_load", "first_resumed_step_before",
        "first_resumed_step_after", "update_97_occurrences", "final_clean_process_load",
        "te_external_state_present", "durable_protocol",
    }
    if not _require_fields(checkpoint, fields, "GPU checkpoint", errors):
        return errors
    components = ["model", "fp32_master", "adam_exp_avg", "adam_exp_avg_sq",
                  "scheduler", "global_step", "sampler", "rng_rank_0", "rng_rank_1"]
    if checkpoint.get("state_components") != components:
        errors.append("GPU checkpoint state component census is incomplete")
    before = checkpoint.get("pre_exit_state")
    after = checkpoint.get("post_load_state")
    if (not isinstance(before, dict) or set(before) != set(components)
            or not isinstance(after, dict) or after != before
            or any(not isinstance(v, str) or SHA256_RE.fullmatch(v) is None for v in before.values())):
        errors.append("GPU checkpoint model/master/moment/scheduler/step/sampler/RNG state did not round-trip")
    for field in ("periodic_files", "final_files"):
        files = checkpoint.get(field)
        if not isinstance(files, list) or not files:
            errors.append(f"GPU checkpoint {field} must contain detached object records")
            continue
        for item in files:
            if not isinstance(item, dict) or set(item) != {"object_id", "sha256", "size_bytes"}:
                errors.append(f"GPU checkpoint {field} entry is malformed")
                continue
            if not _text(item.get("object_id")) or not _positive_int(item.get("size_bytes")) or not isinstance(item.get("sha256"), str) or SHA256_RE.fullmatch(item["sha256"]) is None:
                errors.append(f"GPU checkpoint {field} entry requires nonempty retained bytes and hash")
            if detached_objects is not None:
                retained = detached_objects.get(item.get("object_id"))
                if not isinstance(retained, dict) or retained.get("sha256") != item.get("sha256") or retained.get("size_bytes") != item.get("size_bytes") or retained.get("availability") != "retained":
                    errors.append(f"GPU checkpoint {field} detached object lacks matching retained custody")
    if checkpoint.get("saved_next_batch_sha256") != checkpoint.get("resumed_next_batch_sha256") or not isinstance(checkpoint.get("saved_next_batch_sha256"), str) or SHA256_RE.fullmatch(checkpoint["saved_next_batch_sha256"]) is None:
        errors.append("GPU checkpoint resumed next-batch digest does not match the saved cursor")
    dry_before = checkpoint.get("pre_exit_dry_forward_loss_nanonats")
    dry_after = checkpoint.get("reloaded_dry_forward_loss_nanonats")
    if (type(dry_before) is not int or type(dry_after) is not int
            or abs(dry_before - dry_after) > CRITERIA["dry_forward_tolerance_nanonats"]):
        errors.append("GPU checkpoint reload dry-forward delta exceeds 1e-6")
    if (checkpoint.get("step_before_exit"), checkpoint.get("step_after_load"),
            checkpoint.get("first_resumed_step_before"), checkpoint.get("first_resumed_step_after"),
            checkpoint.get("update_97_occurrences")) != (96, 96, 96, 97, 1):
        errors.append("GPU checkpoint must resume update 97 exactly once from step 96")
    if checkpoint.get("final_clean_process_load") is not True:
        errors.append("GPU final checkpoint must load in a clean process")
    if checkpoint.get("te_external_state_present") is not False:
        errors.append("GPU checkpoint must prove no arm-specific external TE state")
    if checkpoint.get("durable_protocol") != "flush_fsync_atomic_rename_directory_fsync_manifest_last":
        errors.append("GPU checkpoint durable-write protocol drifted")
    return errors


def validate_gpu_telemetry(telemetry: Any, primary_clock: dict[str, int] | None = None,
                           ranks: list[dict] | None = None,
                           required_phases: tuple[str, ...] = REQUIRED_PHASES) -> list[str]:
    """Validate required full-stack coverage and interval semantics."""
    errors: list[str] = []
    fields = {
        "layers", "parser_accepted", "raw_parser_accepted", "dropped_records",
        "overflow", "truncated", "coverage_by_rank", "phase_intervals",
        "observer_sources", "rank_device_bindings", "summary_ref",
        "raw_custody_ref", "heavy_profiles",
    }
    if not _require_fields(telemetry, fields, "GPU telemetry", errors):
        return errors
    if primary_clock is None or set(primary_clock) != {"start_ns", "end_ns"}:
        errors.append("GPU telemetry requires the exact primary monotonic clock")
        return errors
    start, end = primary_clock.get("start_ns"), primary_clock.get("end_ns")
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        errors.append("GPU telemetry primary clock is invalid")
        return errors
    layers = telemetry.get("layers")
    if layers != [{"name": name, "state": "observed"} for name in REQUIRED_LAYERS]:
        errors.append("GPU telemetry must observe every required full-stack layer")
    if (telemetry.get("parser_accepted") is not True
            or telemetry.get("raw_parser_accepted") is not True
            or telemetry.get("dropped_records") != 0
            or telemetry.get("overflow") is not False
            or telemetry.get("truncated") is not False):
        errors.append("GPU telemetry requires parser acceptance and no dropped/overflow/truncated data")
    rank_ids = [rank.get("rank") for rank in ranks or []]
    coverage = telemetry.get("coverage_by_rank")
    if (not isinstance(coverage, list)
            or any(not isinstance(item, dict) or set(item) != {"rank", "intervals"}
                   for item in coverage)
            or [item.get("rank") for item in coverage if isinstance(item, dict)] != rank_ids):
        errors.append("GPU telemetry must retain primary-clock coverage for every rank")
    else:
        for item in coverage:
            merged = _merge_intervals(item.get("intervals"), start, end,
                                      f"GPU telemetry rank {item.get('rank')} coverage", errors)
            if merged != [(start, end)]:
                errors.append("GPU telemetry has an unexplained positive coverage gap")
    phases = telemetry.get("phase_intervals")
    if not isinstance(phases, dict) or set(phases) != set(required_phases):
        errors.append("GPU telemetry phase interval census is incomplete")
    else:
        for phase in required_phases:
            if _union_duration(phases[phase], start, end, f"GPU telemetry {phase}", errors) <= 0:
                errors.append(f"GPU telemetry phase {phase} must have positive observed duration")
    sources = telemetry.get("observer_sources")
    if (not isinstance(sources, list)
            or [item.get("layer") for item in sources if isinstance(item, dict)] != list(REQUIRED_LAYERS)
            or any(not isinstance(item, dict)
                   or set(item) != {"layer", "producer", "sha256"}
                   or not _text(item.get("producer"))
                   or not isinstance(item.get("sha256"), str)
                   or SHA256_RE.fullmatch(item["sha256"]) is None
                   for item in sources)):
        errors.append("GPU telemetry observer sources must exactly bind every layer in order")
    bindings = telemetry.get("rank_device_bindings")
    expected_bindings = [{"rank": item.get("rank"), "device_uuid": item.get("device_uuid")} for item in ranks or []]
    if bindings != expected_bindings:
        errors.append("GPU telemetry rank/device bindings differ from the observed rank census")
    for field in ("summary_ref", "raw_custody_ref"):
        ref = telemetry.get(field)
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not _safe_relative(ref.get("path")) or not isinstance(ref.get("sha256"), str) or SHA256_RE.fullmatch(ref["sha256"]) is None:
            errors.append(f"GPU telemetry {field} must bind retained indexed bytes")
    profiles = telemetry.get("heavy_profiles")
    if not isinstance(profiles, list):
        errors.append("GPU telemetry heavy_profiles must be an explicit list. Empty is permitted")
    else:
        for profile in profiles:
            if (not isinstance(profile, dict) or set(profile) != {
                    "kind", "capture_performed", "parser_accepted", "study_id",
                    "design_sha256", "artifact",
                } or profile.get("kind") != "gpu_profile"
                    or profile.get("capture_performed") is not True
                    or profile.get("parser_accepted") is not True
                    or not _text(profile.get("study_id"))
                    or not _digest_text(profile.get("design_sha256"))):
                errors.append("GPU telemetry heavy profile must be a real parser-accepted capture")
                continue
            artifact = profile.get("artifact")
            if (not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}
                    or not _safe_relative(artifact.get("path"))
                    or not _digest_text(artifact.get("sha256"))):
                errors.append("GPU telemetry heavy profile artifact reference is invalid")
    return errors


def _power_errors(design: dict, observations: Any, ranks: list[dict]) -> list[str]:
    errors: list[str] = []
    uuids = [rank.get("device_uuid") for rank in ranks]
    expected = design["power_policy"]["expected_limit_mw_by_device"]
    wanted = [(uuid, label) for uuid in uuids for label in ("start", "end")]
    seen: list[tuple[Any, Any]] = []
    if not isinstance(observations, list) or len(observations) != 4:
        return ["GPU attempt must retain exactly two power-limit observations per device"]
    for item in observations:
        if not isinstance(item, dict) or set(item) != {
            "device_uuid", "label", "source", "observed_limit_mw", "mutation_attempted",
        }:
            errors.append("GPU power observation fields are invalid")
            continue
        key = (item.get("device_uuid"), item.get("label"))
        seen.append(key)
        observed = item.get("observed_limit_mw")
        target = expected.get(item.get("device_uuid"))
        if item.get("source") != MEASUREMENT_SEMANTICS["power"]["source"]:
            errors.append("GPU power observation source drifted")
        if item.get("mutation_attempted") is not False:
            errors.append("GPU study forbids clock/power/fan mutation")
        if type(observed) is not int or type(target) is not int or abs(observed - target) > 1000:
            errors.append("GPU observed power limit exceeds the admitted 1000 mW tolerance")
    if seen != wanted:
        errors.append("GPU power observations must be in device/start/end order without omissions")
    return errors


def _quality_record(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "nll_sum_nanonats", "target_bytes", "correct_bytes",
    }:
        errors.append(f"{label} must use exact NLL/count operands")
        return
    if not _positive_int(value.get("nll_sum_nanonats")):
        errors.append(f"{label}.nll_sum_nanonats must be a positive integer")
    if value.get("target_bytes") != CRITERIA["oracle_target_bytes"]:
        errors.append(f"{label}.target_bytes must equal the frozen 512-window oracle")
    correct = value.get("correct_bytes")
    if type(correct) is not int or not 0 <= correct <= value.get("target_bytes", -1):
        errors.append(f"{label}.correct_bytes must be an exact bounded count")


def normalize_gpu_attempt(design: dict, group: dict, envelope: dict,
                          raw: Any) -> dict:
    """Normalize one canonical raw attempt or raise :class:`EvidenceError`."""
    errors: list[str] = []
    fields = {
        "schema", "study_id", "design_sha256", "group_id", "run_id",
        "attempt_id", "role", "arm_configuration_sha256", "status", "accepted",
        "failure_reason", "started_at", "ended_at", "primary_clock", "elapsed_ns",
        "identity", "ranks", "power_observations", "useful_work", "safety",
        "updates", "quality", "checkpoint", "telemetry", "cleanup",
    }
    if not _require_fields(raw, fields, "GPU raw attempt", errors):
        raise EvidenceError(". ".join(errors))
    if raw.get("schema") != "piq.gpu-training-attempt.v1":
        errors.append("GPU raw attempt schema is unsupported")
    if raw.get("study_id") != design.get("study_id") or raw.get("group_id") != group.get("id"):
        errors.append("GPU raw attempt study/group identity drifted")
    if raw.get("role") not in ("baseline", "candidate") or raw.get("role") not in group.get("order", []):
        errors.append("GPU raw attempt role is not admitted by its group")
    arm = design.get("arms", {}).get(raw.get("role"), {})
    if raw.get("arm_configuration_sha256") != arm.get("configuration_sha256"):
        errors.append("GPU raw attempt arm configuration hash drifted")
    if raw.get("status") != "succeeded" or raw.get("accepted") is not True or raw.get("failure_reason") is not None:
        errors.append("GPU qualification requires every attempt to succeed and be accepted")
    if _timestamp(raw.get("started_at")) is None or _timestamp(raw.get("ended_at")) is None or _timestamp(raw.get("ended_at")) <= _timestamp(raw.get("started_at")):
        errors.append("GPU raw attempt must retain an ordered UTC window")
    clock = raw.get("primary_clock")
    if (not isinstance(clock, dict) or set(clock) != {"start_ns", "end_ns"}
            or type(clock.get("start_ns")) is not int or type(clock.get("end_ns")) is not int
            or not 0 <= clock["start_ns"] < clock["end_ns"]
            or raw.get("elapsed_ns") != clock["end_ns"] - clock["start_ns"]):
        errors.append("GPU raw attempt elapsed_ns must derive from the complete primary clock")

    imports = design.get("source_imports", [])
    identity = raw.get("identity")
    expected_identity = {
        "runtime_identifier": design.get("runtime", {}).get("runtime_identifier"),
        "hardware_identifier": design.get("topology", {}).get("topology_sha256"),
        "source_sha256s": [item.get("resolved_sha256") for item in imports],
        "resolved_imports": imports,
        "data_sha256s": {
            key: design.get("data", {}).get(key)
            for key in ("train_sha256", "validation_sha256", "sampler_sha256")
        },
        "initial_checkpoint_sha256": design.get("retention", {}).get("initial_checkpoint_sha256"),
        "topology_sha256": design.get("topology", {}).get("topology_sha256"),
        "environment_image_digest": design.get("runtime", {}).get("image_digest"),
    }
    if identity != expected_identity:
        errors.append("GPU attempt source/import/runtime/hardware/data/checkpoint identity drifted")

    ranks = raw.get("ranks")
    expected_uuids = design.get("topology", {}).get("device_uuids", [])
    if not isinstance(ranks, list) or len(ranks) != 2:
        errors.append("GPU attempt requires two complete CUDA ranks")
        ranks = []
    else:
        expected_rank_fields = {
            "rank", "local_rank", "pid", "host", "device_type", "device_uuid",
            "output_ref", "arm_configuration_sha256",
        }
        for index, rank in enumerate(ranks):
            if not isinstance(rank, dict) or set(rank) != expected_rank_fields:
                errors.append("GPU rank census fields are invalid")
                continue
            if (rank.get("rank"), rank.get("local_rank"), rank.get("device_type"),
                    rank.get("device_uuid"), rank.get("arm_configuration_sha256")) != (
                        index, index, "cuda", expected_uuids[index], arm.get("configuration_sha256")):
                errors.append("GPU rank/local-rank/device/arm mapping drifted")
            if not _positive_int(rank.get("pid")) or not _text(rank.get("host")):
                errors.append("GPU rank census requires a positive PID and one observed host")
            ref = rank.get("output_ref")
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or not _safe_relative(ref.get("path")) or not isinstance(ref.get("sha256"), str) or SHA256_RE.fullmatch(ref["sha256"]) is None:
                errors.append("GPU rank output reference is invalid")
        if ([rank.get("rank") for rank in ranks] != [0, 1]
                or [rank.get("local_rank") for rank in ranks] != [0, 1]
                or len({rank.get("pid") for rank in ranks}) != 2
                or len({rank.get("host") for rank in ranks}) != 1
                or len({rank.get("device_uuid") for rank in ranks}) != 2):
            errors.append("GPU attempt rank census must be two distinct ranks/PIDs/devices on one host")
    errors.extend(_power_errors(design, raw.get("power_observations"), ranks))

    admission = group.get("kind") == "admission"
    useful = raw.get("useful_work")
    expected_useful = ({
        "optimizer_updates": 12, "accumulation_rounds": 48,
        "rank_local_microbatches": 96, "target_bytes": 786_432,
        "periodic_checkpoints": 0, "clean_restarts": 0,
        "first_resumed_updates": 0, "final_checkpoints": 0,
        "validation_windows": 0, "retries": 0, "skipped_updates": 0,
        "duplicate_updates": 0,
    } if admission else {
        "optimizer_updates": 192, "accumulation_rounds": 768,
        "rank_local_microbatches": 1536, "target_bytes": 12_582_912,
        "periodic_checkpoints": 1, "clean_restarts": 1,
        "first_resumed_updates": 1, "final_checkpoints": 1,
        "validation_windows": 512, "retries": 0, "skipped_updates": 0,
        "duplicate_updates": 0,
    })
    if (not isinstance(useful, dict)
            or set(useful) != set(expected_useful) | {"update_indices"}
            or any(useful.get(k) != v for k, v in expected_useful.items())):
        errors.append("GPU attempt useful-work census differs from the fixed complete job")
    expected_update_indices = list(range(1, 13 if admission else 193))
    if not isinstance(useful, dict) or useful.get("update_indices") != expected_update_indices:
        errors.append("GPU attempt update census differs from its admitted 12- or 192-update population")

    safety = raw.get("safety")
    expected_safety = {
        "finite_losses": True, "finite_gradient_norms": True,
        "overflow_count": 0, "skipped_optimizer_updates": 0,
        "nccl_errors": 0, "cuda_errors": 0, "xid_errors": 0,
        "ecc_errors": 0, "fallback_warnings": 0,
    }
    if not isinstance(safety, dict) or safety != expected_safety:
        errors.append("GPU attempt numerical/runtime safety census failed")

    updates = raw.get("updates")
    if (not isinstance(updates, list) or len(updates) != len(expected_update_indices)
            or [item.get("index") for item in updates if isinstance(item, dict)] != expected_update_indices
            or any(not isinstance(item, dict) or set(item) != {"index", "latency_ns"}
                   or not _positive_int(item.get("latency_ns")) for item in updates)):
        errors.append("GPU attempt must retain one positive integer latency for each update")

    quality = raw.get("quality")
    if not isinstance(quality, dict) or set(quality) != {"initial_bf16", "final_bf16"}:
        errors.append("GPU attempt quality must retain shared-initial and final BF16 operands")
    else:
        _quality_record(quality["initial_bf16"], "GPU initial quality", errors)
        if admission:
            if quality["final_bf16"] is not None:
                errors.append("GPU admission probe must not fabricate final BF16 evaluation")
        else:
            _quality_record(quality["final_bf16"], "GPU final quality", errors)
    if admission:
        if raw.get("checkpoint") is not None:
            errors.append("GPU admission probe must not fabricate checkpoint/restart evidence")
    else:
        errors.extend(validate_gpu_checkpoint(raw.get("checkpoint")))
    if isinstance(clock, dict):
        errors.extend(validate_gpu_telemetry(
            raw.get("telemetry"), clock, ranks,
            ADMISSION_PHASES if admission else REQUIRED_PHASES,
        ))
    telemetry = raw.get("telemetry")
    if isinstance(telemetry, dict) and "artifacts" in envelope:
        indexed_refs = {
            (item.get("path"), item.get("sha256"))
            for item in envelope.get("artifacts", []) if isinstance(item, dict)
        }
        for field in ("summary_ref", "raw_custody_ref"):
            ref = telemetry.get(field)
            if (not isinstance(ref, dict)
                    or (ref.get("path"), ref.get("sha256")) not in indexed_refs):
                errors.append(f"GPU telemetry {field} is not in the exact PIQ artifact index")
        for profile in telemetry.get("heavy_profiles", []) if isinstance(
                telemetry.get("heavy_profiles"), list) else []:
            ref = profile.get("artifact") if isinstance(profile, dict) else None
            if (not isinstance(ref, dict)
                    or (ref.get("path"), ref.get("sha256")) not in indexed_refs):
                errors.append("GPU telemetry heavy profile is not in the exact PIQ artifact index")

    cleanup = raw.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup != {
        "launcher_exited": True, "workers_exited": True, "process_group_closed": True,
        "observers_closed": True, "owned_gpu_processes": 0,
        "owned_allocations": 0, "scratch_removed": True,
        "retained_bytes_readable": True,
    }:
        errors.append("GPU attempt terminal cleanup is incomplete")
    if errors:
        raise EvidenceError(". ".join(dict.fromkeys(errors)))
    assert isinstance(updates, list) and isinstance(clock, dict)
    normalized = dict(raw)
    normalized["eligible_update_latencies_ns"] = ([] if admission else [
        updates[index - 1]["latency_ns"] for index in ELIGIBLE_UPDATE_INDICES
    ])
    normalized["phase_durations_ns"] = {
        phase: _union_duration(raw["telemetry"]["phase_intervals"][phase],
                               clock["start_ns"], clock["end_ns"], phase, [])
        for phase in (ADMISSION_PHASES if admission else REQUIRED_PHASES)
    }
    return normalized


def _nearest_rank_p95(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceError("p95 population is empty")
    rank = (95 * len(ordered) + 99) // 100
    return ordered[rank - 1]


def _fraction_text(value: Fraction, places: int = 9) -> str:
    # Deterministic decimal rendering is explanatory only. Gates use Fraction.
    scale = 10 ** places
    numerator = value.numerator * scale
    quotient, remainder = divmod(numerator, value.denominator)
    twice = remainder * 2
    if twice > value.denominator or (twice == value.denominator and quotient % 2):
        quotient += 1
    whole, fraction = divmod(quotient, scale)
    return f"{whole}.{fraction:0{places}d}".rstrip("0").rstrip(".")


def _median(values: list[Fraction]) -> Fraction:
    if not values:
        raise EvidenceError("median population is empty")
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _cv_pass(values: list[int]) -> tuple[bool, Fraction]:
    n = len(values)
    total = sum(values)
    squared = sum((n * value - total) ** 2 for value in values)
    # cv^2 = squared / (n * total^2).  Preserve the exact squared ratio.
    cv_squared = Fraction(squared, n * total * total)
    return 400 * squared <= n * total * total, cv_squared


def evaluate_gpu_study(design: dict, attempts: list[dict],
                       *, design_sha256: str) -> dict:
    """Recompute every fixed gate using integers and rational arithmetic."""
    errors = validate_gpu_design(design)
    expected_order = [(group["id"], role) for group in SCHEDULE for role in group["order"]]
    actual_order = [(item.get("group_id"), item.get("role")) for item in attempts]
    if actual_order != expected_order or len(attempts) != 10:
        errors.append("GPU study must retain all ten attempts in the fixed five-group order")
    if len({item.get("attempt_id") for item in attempts}) != len(attempts):
        errors.append("GPU study attempt IDs must be distinct. Retries and replacements are forbidden")
    if len({item.get("run_id") for item in attempts}) != 5:
        errors.append("GPU study must retain exactly one real run ID per group")
    if any(item.get("status") != "succeeded" or item.get("accepted") is not True
           or item.get("failure_reason") is not None for item in attempts):
        errors.append("GPU study has a failed, missing, or unaccepted attempt")
    observer_sources_consistent = True
    if attempts:
        observer_sources = attempts[0].get("telemetry", {}).get("observer_sources")
        if any(item.get("telemetry", {}).get("observer_sources") != observer_sources
               for item in attempts[1:]):
            errors.append("GPU telemetry observer source identity differs across attempts")
            observer_sources_consistent = False

    pairs: list[dict[str, Any]] = []
    gain_values: list[Fraction] = []
    p95_ratios: list[Fraction] = []
    nll_gaps: list[Fraction] = []
    accuracy_gaps: list[Fraction] = []
    arm_times = {"baseline": [], "candidate": []}
    shared_initial = (attempts[0].get("quality", {}).get("initial_bf16")
                      if attempts else None)
    quality_inputs_passed = len(attempts) == 10 and shared_initial is not None and all(
        item.get("quality", {}).get("initial_bf16") == shared_initial
        for item in attempts
    )
    if not quality_inputs_passed:
        errors.append("GPU study must share one initial BF16 quality oracle across all ten attempts")
    for pair_id in PAIR_IDS:
        members = {item.get("role"): item for item in attempts if item.get("group_id") == pair_id}
        if set(members) != {"baseline", "candidate"}:
            errors.append(f"GPU study {pair_id} lacks both arms")
            continue
        baseline, candidate = members["baseline"], members["candidate"]
        tb, tm = baseline.get("elapsed_ns"), candidate.get("elapsed_ns")
        if not _positive_int(tb) or not _positive_int(tm):
            errors.append(f"GPU study {pair_id} elapsed operands are invalid")
            continue
        gain = Fraction(tb, tm)
        gain_values.append(gain)
        arm_times["baseline"].append(tb)
        arm_times["candidate"].append(tm)
        if gain < Fraction(105, 100):
            errors.append(f"GPU study {pair_id} complete-job gain is below 1.05")

        base_p95 = _nearest_rank_p95(baseline.get("eligible_update_latencies_ns", []))
        candidate_p95 = _nearest_rank_p95(candidate.get("eligible_update_latencies_ns", []))
        p95_ratio = Fraction(candidate_p95, base_p95)
        p95_ratios.append(p95_ratio)
        if p95_ratio > Fraction(102, 100):
            errors.append(f"GPU study {pair_id} candidate p95 ratio exceeds 1.02")

        for role, attempt in members.items():
            initial = attempt["quality"]["initial_bf16"]
            final = attempt["quality"]["final_bf16"]
            if (initial["nll_sum_nanonats"] - final["nll_sum_nanonats"]
                    < initial["target_bytes"] * 100_000_000):
                errors.append(f"GPU study {pair_id} {role} absolute learning is below 0.10 nat/byte")
                quality_inputs_passed = False
        base_final = baseline["quality"]["final_bf16"]
        candidate_final = candidate["quality"]["final_bf16"]
        count = base_final["target_bytes"]
        if candidate_final["target_bytes"] != count:
            errors.append(f"GPU study {pair_id} quality byte counts differ")
            quality_inputs_passed = False
        nll_gap = Fraction(candidate_final["nll_sum_nanonats"] - base_final["nll_sum_nanonats"], count * 1_000_000_000)
        accuracy_gap = Fraction(base_final["correct_bytes"] - candidate_final["correct_bytes"], count)
        nll_gaps.append(nll_gap)
        accuracy_gaps.append(accuracy_gap)
        if nll_gap > Fraction(3, 100):
            errors.append(f"GPU study {pair_id} candidate NLL gap exceeds 0.03 nat/byte")
            quality_inputs_passed = False
        if accuracy_gap > Fraction(75, 10_000):
            errors.append(f"GPU study {pair_id} candidate byte-accuracy gap exceeds 0.75 point")
            quality_inputs_passed = False

        phase_results: dict[str, dict[str, Any]] = {}
        for phase in REQUIRED_PHASES:
            base_duration = baseline["phase_durations_ns"][phase]
            candidate_duration = candidate["phase_durations_ns"][phase]
            passed = 10 * candidate_duration <= 11 * base_duration + 20_000_000_000
            if not passed:
                errors.append(f"GPU study {pair_id} {phase} exceeds 1.10*baseline+2s")
            phase_results[phase] = {
                "baseline_ns": base_duration, "candidate_ns": candidate_duration,
                "passed": passed,
            }
        pairs.append({
            "id": pair_id,
            "baseline_elapsed_ns": tb,
            "candidate_elapsed_ns": tm,
            "gain_ratio": _fraction_text(gain),
            "baseline_p95_update_ns": base_p95,
            "candidate_p95_update_ns": candidate_p95,
            "candidate_to_baseline_p95_ratio": _fraction_text(p95_ratio),
            "candidate_nll_gap_nats_per_byte": _fraction_text(nll_gap),
            "candidate_accuracy_gap_percentage_points": _fraction_text(accuracy_gap * 100),
            "phases": phase_results,
        })

    median_gain = _median(gain_values) if gain_values else Fraction(0)
    spread = max(gain_values) - min(gain_values) if gain_values else Fraction(0)
    median_p95 = _median(p95_ratios) if p95_ratios else Fraction(0)
    median_nll_gap = _median(nll_gaps) if nll_gaps else Fraction(0)
    median_accuracy_gap = _median(accuracy_gaps) if accuracy_gaps else Fraction(0)
    if median_gain < Fraction(110, 100):
        errors.append("GPU study median complete-job gain is below 1.10")
    if spread > Fraction(10, 100):
        errors.append("GPU study paired gain spread exceeds 0.10")
    if median_p95 > Fraction(95, 100):
        errors.append("GPU study median candidate/baseline p95 ratio exceeds 0.95")
    if median_nll_gap > Fraction(2, 100):
        errors.append("GPU study median candidate NLL gap exceeds 0.02 nat/byte")
    if median_accuracy_gap > Fraction(50, 10_000):
        errors.append("GPU study median candidate byte-accuracy gap exceeds 0.50 point")

    cv_results: dict[str, dict[str, Any]] = {}
    for role in ("baseline", "candidate"):
        values = arm_times[role]
        if len(values) != 4:
            errors.append(f"GPU study {role} CV requires four complete times")
            passed, squared = False, Fraction(0)
        else:
            passed, squared = _cv_pass(values)
            if not passed:
                errors.append(f"GPU study {role} population CV exceeds 0.05")
        cv_results[role] = {"cv_squared": _fraction_text(squared, 12), "passed": passed}

    full_attempts = [item for item in attempts if item.get("group_id") in PAIR_IDS]
    checkpoint_passed = len(full_attempts) == 8 and all(not validate_gpu_checkpoint(item.get("checkpoint")) for item in full_attempts)
    telemetry_passed = observer_sources_consistent and all(not validate_gpu_telemetry(
        item.get("telemetry"), item.get("primary_clock"), item.get("ranks"),
        ADMISSION_PHASES if item.get("group_id") == "admission-01" else REQUIRED_PHASES,
    ) for item in attempts)
    cleanup_passed = all(item.get("cleanup", {}).get("owned_gpu_processes") == 0
                         and item.get("cleanup", {}).get("owned_allocations") == 0
                         for item in attempts)
    errors = list(dict.fromkeys(errors))
    verdict = "QUALIFIED" if not errors else "INCONCLUSIVE"
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "contract_id": CONTRACT_ID,
        "study_id": design.get("study_id"),
        "design_sha256": design_sha256,
        "verdict": verdict,
        "accepted_pair_ids": list(PAIR_IDS) if verdict == "QUALIFIED" else [],
        "attempt_census": {
            "group_role_order": [[group, role] for group, role in actual_order],
            "attempt_count": len(attempts),
            "run_count": len({item.get("run_id") for item in attempts}),
        },
        "gates": {
            "complete_census": actual_order == expected_order and len(attempts) == 10,
            "all_attempts_accepted": all(item.get("status") == "succeeded" and item.get("accepted") is True for item in attempts),
            "useful_work_and_safety": all(item.get("useful_work", {}).get("optimizer_updates") == (12 if item.get("group_id") == "admission-01" else 192) and item.get("safety") == {
                "finite_losses": True, "finite_gradient_norms": True,
                "overflow_count": 0, "skipped_optimizer_updates": 0,
                "nccl_errors": 0, "cuda_errors": 0, "xid_errors": 0,
                "ecc_errors": 0, "fallback_warnings": 0,
            } for item in attempts),
            "checkpoint_restart": checkpoint_passed,
            "telemetry": telemetry_passed,
            "cleanup": cleanup_passed,
            "gain": len(gain_values) == 4 and median_gain >= Fraction(110, 100) and all(value >= Fraction(105, 100) for value in gain_values),
            "repeatability": len(gain_values) == 4 and spread <= Fraction(10, 100) and all(item["passed"] for item in cv_results.values()),
            "update_latency": len(p95_ratios) == 4 and median_p95 <= Fraction(95, 100) and all(value <= Fraction(102, 100) for value in p95_ratios),
            "quality": (quality_inputs_passed and len(nll_gaps) == 4
                        and median_nll_gap <= Fraction(2, 100)
                        and median_accuracy_gap <= Fraction(50, 10_000)),
            "phase_regressions": all(all(value["passed"] for value in pair["phases"].values()) for pair in pairs),
        },
        "metrics": {
            "pairs": pairs,
            "median_gain_ratio": _fraction_text(median_gain),
            "gain_spread": _fraction_text(spread),
            "within_arm_population_cv": cv_results,
            "median_candidate_to_baseline_p95_ratio": _fraction_text(median_p95),
            "median_candidate_nll_gap_nats_per_byte": _fraction_text(median_nll_gap),
            "median_candidate_accuracy_gap_percentage_points": _fraction_text(median_accuracy_gap * 100),
        },
        "errors": errors,
    }

def _load_ref_json(package: Path, ref: Any, label: str, errors: list[str]) -> tuple[Path | None, Any]:
    path = _ref(package, ref, label, errors)
    if path is None:
        return None, None
    try:
        return path, strict_load_json(path, label)
    except EvidenceError as exc:
        errors.append(str(exc))
        return path, None


def _review_errors(review: Any, label: str, design: dict, design_sha: str,
                   execution_owner: str) -> list[str]:
    errors: list[str] = []
    common = {"study_id", "design_sha256", "reviewer", "independent", "verdict", "reviewed_at", "reasoning"}
    extras = ({"baseline_id", "optimized_standard", "primary_sources"}
              if label == "baseline review" else
              {"required_layers", "required_phases", "machine_semantics_sha256"}
              if label == "telemetry review" else
              {"qualification_report_sha256", "assessments"})
    if not _require_fields(review, common | extras, f"GPU {label}", errors):
        return errors
    if review.get("study_id") != design.get("study_id") or review.get("design_sha256") != design_sha:
        errors.append(f"GPU {label} design identity mismatch")
    if not _text(review.get("reviewer")) or review.get("reviewer") == execution_owner or review.get("independent") is not True or review.get("verdict") != "PASS" or not _text(review.get("reasoning")) or _timestamp(review.get("reviewed_at")) is None:
        errors.append(f"GPU {label} must be a substantive independent PASS")
    if label == "baseline review":
        if review.get("baseline_id") != "bf16" or review.get("optimized_standard") is not True or not isinstance(review.get("primary_sources"), list) or not review["primary_sources"] or any(not isinstance(v, str) or not v.startswith("https://") for v in review["primary_sources"]):
            errors.append("GPU baseline review must defend the optimized BF16 standard from primary sources")
    elif label == "telemetry review":
        if review.get("required_layers") != list(REQUIRED_LAYERS) or review.get("required_phases") != list(REQUIRED_PHASES) or review.get("machine_semantics_sha256") != _hash_value(MEASUREMENT_SEMANTICS):
            errors.append("GPU telemetry review must ratify every layer, phase and machine semantic")
    else:
        assessments = review.get("assessments")
        if not isinstance(assessments, list) or sorted(assessments) != sorted([
            "source_identity", "controls", "correctness_quality", "checkpoint_restart",
            "timing_boundary", "telemetry", "custody_cleanup", "scope_limitations",
        ]):
            errors.append("GPU scientific review assessment census is incomplete")
    return errors


def _detached_index(envelope: dict) -> dict[str, dict]:
    values = envelope.get("detached_objects")
    if not isinstance(values, list):
        return {}
    return {item.get("id"): item for item in values if isinstance(item, dict) and _text(item.get("id"))}


def _validate_gpu_receipt(package: Path, receipt: dict) -> list[str]:
    errors: list[str] = []
    if not _require_fields(receipt, RECEIPT_FIELDS, "GPU receipt v3", errors):
        return errors
    if (receipt.get("schema_version"), receipt.get("state"), receipt.get("execution_class"),
            receipt.get("workload_family"), receipt.get("contract_id")) != (
                3, "measured", "gpu", "gpu_training", CONTRACT_ID):
        errors.append("GPU receipt must declare measurement v3 for the fixed gpu_training contract")
    for field in ("workload_scope", "runtime_identifier", "hardware_identifier"):
        if not _text(receipt.get(field)):
            errors.append(f"GPU receipt {field} must be a nonempty string")

    document_names = (
        "design", "registration", "source_import_closure", "baseline_review",
        "telemetry_review", "raw_census", "correctness_quality",
        "checkpoint_restart", "telemetry_coverage", "custody_readback_cleanup",
        "qualification_report", "scientific_review",
    )
    documents: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name in document_names:
        path, value = _load_ref_json(package, receipt.get(name), f"GPU receipt {name}", errors)
        if path is not None:
            paths[name] = path
        documents[name] = value
    design = documents.get("design")
    if not isinstance(design, dict):
        errors.append("GPU receipt design must be a JSON object")
        return errors
    errors.extend(validate_gpu_design(design))
    design_sha = receipt.get("design", {}).get("sha256") if isinstance(receipt.get("design"), dict) else None
    if (receipt.get("workload_scope") != design.get("workload_scope")
            or receipt.get("runtime_identifier") != design.get("runtime", {}).get("runtime_identifier")
            or receipt.get("hardware_identifier") != design.get("topology", {}).get("topology_sha256")):
        errors.append("GPU receipt workload/runtime/hardware identity must equal the design")

    source_refs = receipt.get("source_artifacts")
    expected_source_hashes = [item.get("resolved_sha256") for item in design.get("source_imports", [])]
    if not isinstance(source_refs, list) or len(source_refs) != len(REQUIRED_IMPORT_ROLES):
        errors.append("GPU receipt must bind every first-party source artifact")
        source_refs = []
    actual_source_hashes: list[str] = []
    for index, ref in enumerate(source_refs):
        _ref(package, ref, f"GPU receipt source_artifacts[{index}]", errors)
        if isinstance(ref, dict):
            actual_source_hashes.append(ref.get("sha256"))
    if actual_source_hashes != expected_source_hashes:
        errors.append("GPU receipt source artifacts differ from the resolved import closure")

    envelope_path, envelope = _load_ref_json(package, receipt.get("piq_envelope"), "GPU receipt PIQ v2 envelope", errors)
    if envelope_path is None or not isinstance(envelope, dict):
        return errors
    bundle_errors = validate_gpu_bundle(envelope_path.parent)
    errors.extend(bundle_errors)
    if (envelope.get("schema_version"), envelope.get("kind"), envelope.get("scope"),
            envelope.get("experiment_kind"), envelope.get("workload_family"),
            envelope.get("scientific_disposition")) != (
                2, "piq_experiment_evidence", "diagnostic",
                "te_mxfp8_gpu_training", "gpu_training", "DIAGNOSTIC_ONLY"):
        errors.append(
            "GPU receipt must bind the diagnostic-only PIQ v2 te_mxfp8_gpu_training variant"
        )
    if envelope.get("study_id") != design.get("study_id"):
        errors.append("GPU receipt design and PIQ envelope study identities differ")

    closure = documents.get("source_import_closure")
    if not isinstance(closure, dict) or closure != {
        "schema_version": 1, "kind": "gpu_training_source_import_closure",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "imports": design.get("source_imports"), "accepted": True,
    }:
        errors.append("GPU source/import closure must equal the admitted resolved paths and hashes")

    raw_attempts: list[dict] = []
    raw_rows: list[dict] = []
    group_map = {group["id"]: group for group in SCHEDULE}
    for index, attempt in enumerate(envelope.get("attempts", [])):
        if not isinstance(attempt, dict):
            errors.append(f"GPU PIQ attempt[{index}] must be an object")
            continue
        raw_ref = attempt.get("raw_result")
        raw_path = None
        if isinstance(raw_ref, dict) and _safe_relative(raw_ref.get("path")):
            raw_path = envelope_path.parent / raw_ref["path"]
        try:
            raw = strict_load_json(raw_path, f"GPU raw attempt {index}") if raw_path is not None else None
        except EvidenceError as exc:
            errors.append(str(exc))
            raw = None
        if not isinstance(raw, dict):
            continue
        try:
            normalized = normalize_gpu_attempt(design, group_map.get(attempt.get("group_id"), {}), envelope, raw)
        except (EvidenceError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"GPU canonical attempt normalization failed: {exc}")
            continue
        if any(normalized.get(key) != attempt.get(other) for key, other in (
                ("attempt_id", "id"), ("run_id", "run_id"),
                ("group_id", "group_id"), ("role", "role"),
                ("status", "status"), ("accepted", "accepted"),
                ("failure_reason", "failure_reason"))):
            errors.append("GPU PIQ envelope attempt identity/disposition differs from canonical raw evidence")
        raw_attempts.append(normalized)
        raw_rows.append({
            "group_id": attempt.get("group_id"), "role": attempt.get("role"),
            "run_id": attempt.get("run_id"), "attempt_id": attempt.get("id"),
            "raw_result": raw_ref,
        })

    raw_census = documents.get("raw_census")
    expected_census = {
        "schema_version": 1, "kind": "gpu_training_raw_census",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "attempts": raw_rows,
    }
    if raw_census != expected_census:
        errors.append("GPU raw census must conserve all ten PIQ attempts in exact order")

    detached = _detached_index(envelope)
    for attempt in raw_attempts:
        if attempt.get("group_id") in PAIR_IDS:
            errors.extend(validate_gpu_checkpoint(attempt.get("checkpoint"), detached))
    computed = evaluate_gpu_study(design, raw_attempts, design_sha256=str(design_sha))
    report = documents.get("qualification_report")
    if report != computed:
        errors.append("GPU qualification report does not match deterministic recomputation")
    report_path = paths.get("qualification_report")
    if report_path is not None and report_path.read_bytes() != canonical_json(computed):
        errors.append("GPU qualification report bytes are not canonical deterministic JSON")
    if computed.get("verdict") != "QUALIFIED":
        errors.extend(computed.get("errors", []))

    expected_quality = {
        "schema_version": 1, "kind": "gpu_training_correctness_quality",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "accepted": computed.get("gates", {}).get("quality") is True,
        "metrics": computed.get("metrics", {}),
    }
    if documents.get("correctness_quality") != expected_quality:
        errors.append("GPU correctness/quality record must equal recomputed exact operands and verdicts")
    expected_checkpoint = {
        "schema_version": 1, "kind": "gpu_training_checkpoint_restart",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "accepted": computed.get("gates", {}).get("checkpoint_restart") is True,
        "attempt_ids": [item.get("attempt_id") for item in raw_attempts if item.get("group_id") in PAIR_IDS],
        "detached_object_ids": sorted(detached),
    }
    if documents.get("checkpoint_restart") != expected_checkpoint:
        errors.append("GPU checkpoint/restart aggregate does not match the full attempt/object census")
    expected_telemetry = {
        "schema_version": 1, "kind": "gpu_training_telemetry_coverage",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "accepted": computed.get("gates", {}).get("telemetry") is True,
        "required_layers": list(REQUIRED_LAYERS),
        "required_phases": list(REQUIRED_PHASES),
        "attempt_ids": [item.get("attempt_id") for item in raw_attempts],
    }
    if documents.get("telemetry_coverage") != expected_telemetry:
        errors.append("GPU telemetry aggregate does not match required layers/phases/attempt census")

    registration = documents.get("registration")
    expected_plan_hashes = {group["id"]: _hash_value(group) for group in SCHEDULE}
    if not isinstance(registration, dict) or set(registration) != {
        "schema_version", "kind", "study_id", "design_sha256", "registered_at",
        "authority", "record_id", "group_plan_sha256s",
    } or (registration.get("schema_version"), registration.get("kind"),
          registration.get("study_id"), registration.get("design_sha256"),
          registration.get("group_plan_sha256s")) != (
              1, "gpu_training_registration", design.get("study_id"), design_sha,
              expected_plan_hashes):
        errors.append("GPU registration must bind the immutable design and all five native group plans")
        registered_at = None
    else:
        registered_at = _timestamp(registration.get("registered_at"))
        if registered_at is None or not _text(registration.get("authority")) or not _text(registration.get("record_id")):
            errors.append("GPU registration requires a UTC time and authoritative record identity")

    baseline = documents.get("baseline_review")
    telemetry_review = documents.get("telemetry_review")
    scientific = documents.get("scientific_review")
    errors.extend(_review_errors(baseline, "baseline review", design, str(design_sha), design.get("execution_owner")))
    errors.extend(_review_errors(telemetry_review, "telemetry review", design, str(design_sha), design.get("execution_owner")))
    errors.extend(_review_errors(scientific, "scientific review", design, str(design_sha), design.get("execution_owner")))
    starts = [_timestamp(item.get("started_at")) for item in raw_attempts]
    ends = [_timestamp(item.get("ended_at")) for item in raw_attempts]
    baseline_at = _timestamp(baseline.get("reviewed_at")) if isinstance(baseline, dict) else None
    telemetry_at = _timestamp(telemetry_review.get("reviewed_at")) if isinstance(telemetry_review, dict) else None
    scientific_at = _timestamp(scientific.get("reviewed_at")) if isinstance(scientific, dict) else None
    design_at = _timestamp(design.get("created_at"))
    if (design_at is None or registered_at is None or baseline_at is None
            or telemetry_at is None or any(
                design_at >= value for value in (registered_at, baseline_at, telemetry_at)
            )):
        errors.append("GPU immutable design must precede registration and pre-execution reviews")
    if any(value is None or design_at is None or registered_at is None
           or baseline_at is None or telemetry_at is None or value <= design_at
           or value <= registered_at or value <= baseline_at or value <= telemetry_at
           for value in starts):
        errors.append("GPU design/reviews/registration must precede every attempt")
    if any(value is None or scientific_at is None or scientific_at <= value for value in ends):
        errors.append("GPU scientific review must follow every attempt")
    if (not isinstance(scientific, dict)
            or scientific.get("qualification_report_sha256") != receipt.get("qualification_report", {}).get("sha256")):
        errors.append("GPU scientific review must bind the exact deterministic qualification report")

    custody = documents.get("custody_readback_cleanup")
    expected_custody = {
        "schema_version": 1, "kind": "gpu_training_custody_readback_cleanup",
        "study_id": design.get("study_id"), "design_sha256": design_sha,
        "detached_object_ids": sorted(detached),
        "local_readback": True, "piq_readback": True, "export_readback": True,
        "all_retained_bytes_readable": True, "all_owned_processes_released": True,
        "all_owned_allocations_released": True, "failure_census_complete": True,
        "cleanup_attempt_ids": [item.get("attempt_id") for item in raw_attempts],
    }
    if custody != expected_custody:
        errors.append("GPU custody/readback/cleanup aggregate is incomplete or stale")

    repeats = receipt.get("accepted_repeats")
    if not isinstance(repeats, list) or [item.get("id") for item in repeats if isinstance(item, dict)] != list(PAIR_IDS):
        errors.append("GPU receipt accepted_repeats must contain pair-01 through pair-04")
        repeats = []
    for index, repeat in enumerate(repeats):
        if not isinstance(repeat, dict) or set(repeat) != {"id", "accepted", "raw_result"} or repeat.get("accepted") is not True:
            errors.append("GPU accepted repeat record is malformed")
            continue
        _, pair = _load_ref_json(package, repeat.get("raw_result"), f"GPU accepted pair[{index}]", errors)
        expected_pair = next((item for item in computed.get("metrics", {}).get("pairs", []) if item.get("id") == repeat.get("id")), None)
        if pair != {"schema_version": 1, "kind": "gpu_training_pair_result", "pair": expected_pair}:
            errors.append(f"GPU accepted repeat {repeat.get('id')} does not bind its recomputed pair")
    failed = receipt.get("failed_attempts")
    if failed != []:
        errors.append("GPU qualified receipt failed_attempts must be empty. Failures stay in incomplete evidence")

    profiles = receipt.get("profile_evidence")
    if not isinstance(profiles, list):
        errors.append("GPU profile_evidence must be a list. Empty is valid")
        profiles = []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or set(profile) != {
            "kind", "capture_performed", "parser_accepted", "study_id",
            "design_sha256", "artifact",
        } or (profile.get("kind"), profile.get("capture_performed"),
              profile.get("parser_accepted"), profile.get("study_id"),
              profile.get("design_sha256")) != (
                  "gpu_profile", True, True, design.get("study_id"), design_sha):
            errors.append("GPU profile evidence must be a real accepted parser-bound capture")
        else:
            _ref(package, profile.get("artifact"), f"GPU profile_evidence[{index}]", errors)

    claims = receipt.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("GPU receipt claims must be a nonempty list")
        claims = []
    claim_ids: list[Any] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict) or set(claim) != {
            "id", "source", "accepted_repeat_ids", "profile_claim",
        }:
            errors.append(f"GPU claim[{index}] fields are invalid")
            continue
        claim_ids.append(claim.get("id"))
        if not _text(claim.get("id")) or claim.get("accepted_repeat_ids") != list(PAIR_IDS):
            errors.append("GPU claim must bind all four accepted pair IDs")
        if claim.get("source") != receipt.get("qualification_report"):
            errors.append("GPU claim source must be the exact deterministic qualification report")
        _ref(package, claim.get("source"), f"GPU claim[{index}].source", errors)
        if type(claim.get("profile_claim")) is not bool or (claim.get("profile_claim") and not profiles):
            errors.append("GPU profile claim requires a real nonempty accepted capture")
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("GPU claim IDs must be distinct")
    return list(dict.fromkeys(errors))


def validate_gpu_receipt(package: Path, receipt: dict) -> list[str]:
    """Fail closed on malformed containers. Valid means qualified retained bytes."""
    try:
        return _validate_gpu_receipt(package.resolve(), receipt)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError,
            ZeroDivisionError) as exc:
        return [f"malformed GPU receipt evidence ({type(exc).__name__}): {exc}"]


def gpu_receipt_binds_envelope(package: Path, receipt: dict,
                               envelope_path: Path) -> bool:
    """Return true only for a valid v3 receipt's exact PIQ-v2 envelope bytes."""
    if validate_gpu_receipt(package, receipt):
        return False
    ref = receipt.get("piq_envelope")
    if not isinstance(ref, dict) or not _safe_relative(ref.get("path")):
        return False
    try:
        expected = (package / ref["path"]).resolve(strict=True)
        actual = envelope_path.resolve(strict=True)
        return expected == actual and _sha256(actual) == ref.get("sha256")
    except (OSError, ValueError):
        return False


def build_gpu_public_summary(package: Path, receipt: dict) -> dict:
    """Build the deterministic public allowlist from an already qualified receipt."""
    errors = validate_gpu_receipt(package, receipt)
    if errors:
        raise EvidenceError("cannot project a non-qualified GPU receipt: " + ". ".join(errors))
    design_path = package / receipt["design"]["path"]
    report_path = package / receipt["qualification_report"]["path"]
    design = strict_load_json(design_path, "GPU design")
    report = strict_load_json(report_path, "GPU qualification report")
    if design.get("synthetic_fixture") is not False:
        raise EvidenceError("cannot project a synthetic GPU fixture")
    public_pairs = []
    for ordinal, pair in enumerate(report["metrics"]["pairs"], 1):
        public_pairs.append({
            "ordinal": ordinal,
            "gain_ratio": pair["gain_ratio"],
            "candidate_to_baseline_p95_ratio": pair["candidate_to_baseline_p95_ratio"],
            "candidate_nll_gap_nats_per_byte": pair["candidate_nll_gap_nats_per_byte"],
            "candidate_accuracy_gap_percentage_points": pair["candidate_accuracy_gap_percentage_points"],
            "phase_verdicts": {name: value["passed"] for name, value in pair["phases"].items()},
        })
    return {
        "schema_version": 1,
        "kind": "gpu_training_public_summary",
        "contract_id": CONTRACT_ID,
        "qualification": "QUALIFIED",
        "method": {
            "workload_family": "gpu_training",
            "workload_scope": design["workload_scope"],
            "baseline": "Transformer Engine BF16",
            "candidate": "Transformer Engine MXFP8 E4M3 forward autocast",
            "sole_changed_variable": design["arms"]["sole_changed_variable"],
            "complete_boundary": design["primary_clock_boundary"],
            "balanced_pair_count": 4,
        },
        "results": {
            "median_complete_job_gain_ratio": report["metrics"]["median_gain_ratio"],
            "gain_spread": report["metrics"]["gain_spread"],
            "within_arm_population_cv": report["metrics"]["within_arm_population_cv"],
            "median_candidate_to_baseline_p95_ratio": report["metrics"]["median_candidate_to_baseline_p95_ratio"],
            "median_candidate_nll_gap_nats_per_byte": report["metrics"]["median_candidate_nll_gap_nats_per_byte"],
            "median_candidate_accuracy_gap_percentage_points": report["metrics"]["median_candidate_accuracy_gap_percentage_points"],
            "correctness_quality_passed": report["gates"]["quality"],
            "checkpoint_restart_passed": report["gates"]["checkpoint_restart"],
            "telemetry_passed": report["gates"]["telemetry"],
            "repeats": public_pairs,
        },
        "evidence": {
            "private_qualification_report_sha256": receipt["qualification_report"]["sha256"],
            "private_design_sha256": receipt["design"]["sha256"],
        },
        "scope_and_limits": {
            "scope": "one staged byte-level model, exact pinned environment, one two-B200 host, 192 updates",
            "exclusions": [
                "long-run pretraining convergence", "other models or datasets",
                "multi-node behavior", "serving performance", "cold object-store input",
            ],
            "tradeoff": "MXFP8 adds quantization work and leaves BF16 communication, FP32 optimizer work, validation, checkpoint I/O, restart, evidence custody, and cleanup in the complete path.",
            "simpler_option": "Use BF16 when the complete gain, BF16-evaluated quality, restart evidence, or exact low-precision stack does not pass the fixed gates.",
            "unmeasured": ["portability beyond the exact environment", "long-horizon downstream quality"],
        },
    }
