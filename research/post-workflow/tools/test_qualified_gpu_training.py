from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.qualified_gpu_training import (
    ADMISSION_PHASES,
    CONTRACT_ID,
    CRITERIA,
    ELIGIBLE_UPDATE_INDICES,
    MEASUREMENT_SEMANTICS,
    PAIR_IDS,
    REQUIRED_IMPORT_ROLES,
    REQUIRED_LAYERS,
    REQUIRED_PHASES,
    SCHEDULE,
    EvidenceError,
    _cv_pass,
    _nearest_rank_p95,
    build_gpu_public_summary,
    canonical_json,
    evaluate_gpu_study,
    normalize_gpu_attempt,
    strict_load_json,
    validate_gpu_checkpoint,
    validate_gpu_design,
    validate_gpu_receipt,
    validate_gpu_telemetry,
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_value(value: object) -> str:
    return _hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode())


def _json(path: Path, value: object, *, canonical: bool = False) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) if canonical else (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(payload)
    return {"path": path.as_posix(), "sha256": _hash_bytes(payload)}


def _bundle_ref(bundle: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(bundle).as_posix(), "sha256": _hash_bytes(path.read_bytes())}


def _package_ref(package: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(package).as_posix(), "sha256": _hash_bytes(path.read_bytes())}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_design(source_hashes: list[str], *, synthetic: bool) -> dict:
    uuids = ["GPU-SYNTHETIC-B200-0", "GPU-SYNTHETIC-B200-1"]
    initial_sha = _hash_bytes(b"synthetic-initial-checkpoint")
    baseline = "te.autocast(enabled=False)"
    candidate = "te.autocast(enabled=True, recipe=MXFP8BlockScaling(fp8_format=Format.E4M3))"
    imports = []
    for role, digest in zip(REQUIRED_IMPORT_ROLES, source_hashes, strict=True):
        path = f"/opt/piq-study/{role}.py"
        imports.append({
            "role": role,
            "expected_absolute_path": path,
            "expected_sha256": digest,
            "resolved_absolute_path": path,
            "resolved_sha256": digest,
        })
    return {
        "schema_version": 1,
        "kind": "qualified_gpu_training_design",
        "contract_id": CONTRACT_ID,
        "study_id": "synthetic-te-study-v1",
        "workload_scope": "synthetic fixed two-B200 TE MXFP8-versus-BF16 training fixture",
        "created_at": "2026-10-31T10:00:00Z",
        "execution_owner": "synthetic-execution-owner",
        "execution_class": "gpu",
        "accelerator_count": 2,
        "synthetic_fixture": synthetic,
        "model": {
            "blocks": 24, "hidden_width": 2048, "mlp_width": 8192,
            "attention_heads": 16, "head_dimension": 128, "sequence_length": 2048,
            "rank_local_microbatch": 4, "gradient_accumulation": 4,
            "global_batch": 32, "optimizer_updates": 192,
            "attention_backend": "te_fused_attention", "activation": "gelu",
            "normalization": "rmsnorm", "parameter_dtype": "bf16",
            "dropout": "0", "bias": False, "activation_recomputation": False,
            "compile": False, "cuda_graphs": False,
            "parameter_manifest_sha256": _hash_bytes(b"synthetic-parameter-manifest"),
            "optimizer": {
                "name": "Transformer Engine FusedAdam", "lr_peak": "0.0002",
                "betas": ["0.9", "0.95"], "eps": "0.00000001",
                "weight_decay": "0.1", "master_weights": "float32",
                "moments": "float32", "gradient_clip_norm": "1.0",
                "schedule": "16-update linear warmup then cosine to 0.00002",
            },
        },
        "data": {
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "remote_files": [
                {"path": "train-00000-of-00002.parquet", "sha256": "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"},
                {"path": "train-00001-of-00002.parquet", "sha256": "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"},
                {"path": "validation-00000-of-00001.parquet", "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"},
            ],
            "converter_sha256": _hash_bytes(b"synthetic-converter"),
            "train_sha256": _hash_bytes(b"synthetic-train-u8"),
            "validation_sha256": _hash_bytes(b"synthetic-validation-u8"),
            "sampler_sha256": _hash_bytes(b"synthetic-sampler"),
            "validation_oracle_windows": 512,
            "seed": 1701,
            "cache_policy": "shared_staged_preread_once",
        },
        "arms": {
            "baseline": {"id": "bf16", "configuration": baseline,
                         "configuration_sha256": _hash_value({"configuration": baseline})},
            "candidate": {"id": "mxfp8", "configuration": candidate,
                          "configuration_sha256": _hash_value({"configuration": candidate})},
            "sole_changed_variable": "transformer_engine_forward_autocast_recipe",
        },
        "runtime": {
            "pytorch": "2.15.0", "transformer_engine": "2.20.2",
            "transformer_engine_commit": "6ea2a74a9e98c99e6d7b164a33775cc457520027",
            "cuda_toolkit": "13.4.3", "cublas": "13.8.2", "cudnn": "9.12.0",
            "nccl": "2.28.0", "driver": "synthetic-driver",
            "image_digest": "sha256:" + _hash_bytes(b"synthetic-image"),
            "python": "3.13.9", "compiler_extensions": "synthetic-extension-lock",
            "attention_backend": "te_fused_attention",
            "runtime_identifier": "synthetic-pytorch-2.15-te-2.20.2-cuda-13.4.3",
        },
        "topology": {
            "node_count": 1, "accelerator_count": 2, "process_count": 2,
            "ranks": [0, 1], "local_ranks": [0, 1],
            "compute_capability": [10, 0],
            "launcher": "torchrun --standalone --nnodes=1 --nproc-per-node=2",
            "backend": "nccl", "parallelism": "ddp_only", "device_uuids": uuids,
            "topology_sha256": _hash_bytes(b"synthetic-one-host-two-b200-topology"),
        },
        "determinism": {
            "seed": 1701, "nvte_allow_nondeterministic_algo": "0",
            "torch_deterministic_errors": True, "cudnn_benchmark": False,
        },
        "schedule": copy.deepcopy(list(SCHEDULE)),
        "criteria": copy.deepcopy(CRITERIA),
        "measurement_semantics": copy.deepcopy(MEASUREMENT_SEMANTICS),
        "required_phases": list(REQUIRED_PHASES),
        "budgets": {"gpu_count": 2, "gpu_hours": 2, "additional_spend_usd": 0,
                    "hard_spend_cap_usd": 50, "hard_wall_minutes": 60},
        "retention": {
            "initial_checkpoint_sha256": initial_sha,
            "keep_raw_evidence": True,
            "keep_restartable_final_checkpoint_per_arm": True,
            "deletions_recorded": True,
        },
        "power_policy": {
            "source": "nvidia_smi_power_limit_milliwatts",
            "expected_limit_mw_by_device": {uuid: 1_000_000 for uuid in uuids},
            "tolerance_mw": 1000,
            "observations_per_attempt_per_device": 2,
            "observation_labels": ["start", "end"],
            "mutation_allowed": False,
        },
        "source_imports": imports,
        "primary_clock_boundary": "before_initial_checkpoint_open_through_post_cleanup_evidence_readback",
    }


def build_gpu_fixture(package: Path, *, synthetic: bool = True) -> dict:
    """Create small invented contract evidence. It describes no real execution."""
    bundle = package / "evidence/piq-study"
    qualification = package / "evidence/qualification"
    bundle.mkdir(parents=True, exist_ok=True)
    qualification.mkdir(parents=True, exist_ok=True)

    source_paths: list[Path] = []
    for role in REQUIRED_IMPORT_ROLES:
        path = bundle / f"source/{role}.py"
        _write_text(path, f"# synthetic {role}. Never executed\nROLE = {role!r}\n")
        source_paths.append(path)
    source_hashes = [_hash_bytes(path.read_bytes()) for path in source_paths]
    design = _fixture_design(source_hashes, synthetic=synthetic)
    design_path = bundle / "design.json"
    _json(design_path, design)
    design_sha = _hash_bytes(design_path.read_bytes())

    data_identity_path = bundle / "data-identity.json"
    _json(data_identity_path, {
        "schema": "synthetic-data-identity.v1",
        "dataset_revision": design["data"]["dataset_revision"],
        "train_sha256": design["data"]["train_sha256"],
        "validation_sha256": design["data"]["validation_sha256"],
        "sampler_sha256": design["data"]["sampler_sha256"],
        "synthetic_fixture": True,
    })

    attempt_specs: list[tuple[dict, str, str, int]] = []
    pair_elapsed = {
        "pair-01": {"baseline": 120_000_000_000, "candidate": 108_000_000_000},
        "pair-02": {"baseline": 121_000_000_000, "candidate": 109_000_000_000},
        "pair-03": {"baseline": 119_000_000_000, "candidate": 107_000_000_000},
        "pair-04": {"baseline": 122_000_000_000, "candidate": 110_000_000_000},
    }
    for group in SCHEDULE:
        run_id = f"synthetic-run-{group['id']}"
        for role in group["order"]:
            elapsed = ({"baseline": 12_000_000_000, "candidate": 11_000_000_000}[role]
                       if group["kind"] == "admission" else pair_elapsed[group["id"]][role])
            attempt_specs.append((group, role, run_id, elapsed))

    detached_values = [{
        "id": "initial-checkpoint", "role": "initial_checkpoint",
        "sha256": design["retention"]["initial_checkpoint_sha256"],
        "size_bytes": 8_000_000_000,
    }]
    for group, role, _, _ in attempt_specs:
        if group["kind"] == "admission":
            continue
        attempt_id = f"{group['id']}-{role}-attempt"
        for checkpoint_role in ("periodic", "final"):
            object_id = f"{attempt_id}-{checkpoint_role}-checkpoint"
            detached_values.append({
                "id": object_id, "role": f"{checkpoint_role}_checkpoint",
                "sha256": _hash_bytes(object_id.encode()),
                "size_bytes": 8_100_000_000 if checkpoint_role == "periodic" else 8_200_000_000,
            })

    readback_entries = [
        {"id": item["id"], "sha256": item["sha256"],
         "size_bytes": item["size_bytes"], "available": True}
        for item in detached_values
    ]
    custody_paths: dict[str, Path] = {}
    custody_contracts = {
        "custody-record": ("piq.gpu-training-custody-record.v1", "custody"),
        "local-readback": ("piq.gpu-training-readback.v1", "local_durable"),
        "piq-readback": ("piq.gpu-training-readback.v1", "piq"),
        "export-readback": ("piq.gpu-training-readback.v1", "export"),
    }
    for name in ("custody-record", "local-readback", "piq-readback", "export-readback"):
        path = bundle / f"custody/{name}.json"
        schema, scope = custody_contracts[name]
        _json(path, {
            "schema": schema, "scope": scope,
            "verified_at": "2026-10-31T12:55:00Z", "immutable": True,
            "objects": readback_entries, "synthetic_fixture": True,
        })
        custody_paths[name] = path
    detached_objects = []
    for item in detached_values:
        detached_objects.append({
            **item, "media_type": "application/x-pytorch-checkpoint",
            "custody_record": _bundle_ref(bundle, custody_paths["custody-record"]),
            "local_readback": _bundle_ref(bundle, custody_paths["local-readback"]),
            "piq_readback": _bundle_ref(bundle, custody_paths["piq-readback"]),
            "export_readback": _bundle_ref(bundle, custody_paths["export-readback"]),
            "availability": "retained",
        })
    detached_map = {item["id"]: item for item in detached_objects}

    telemetry_summary_path = bundle / "telemetry/summary-index.json"
    telemetry_raw_path = bundle / "telemetry/raw-custody-index.json"
    _json(telemetry_summary_path, {"schema": "synthetic-gpu-telemetry-summary.v1",
                                   "complete": True, "synthetic_fixture": True})
    _json(telemetry_raw_path, {"schema": "synthetic-gpu-telemetry-raw-index.v1",
                               "complete": True, "synthetic_fixture": True})
    telemetry_summary_ref = _bundle_ref(bundle, telemetry_summary_path)
    telemetry_raw_ref = _bundle_ref(bundle, telemetry_raw_path)

    raw_attempts: list[dict] = []
    raw_paths: dict[str, Path] = {}
    rank_paths: dict[tuple[str, int], Path] = {}
    attempt_started = datetime(2026, 10, 31, 11, 0, tzinfo=timezone.utc)
    components = ["model", "fp32_master", "adam_exp_avg", "adam_exp_avg_sq",
                  "scheduler", "global_step", "sampler", "rng_rank_0", "rng_rank_1"]
    for attempt_index, (group, role, run_id, elapsed) in enumerate(attempt_specs):
        attempt_id = f"{group['id']}-{role}-attempt"
        started = attempt_started + timedelta(minutes=attempt_index * 10)
        ended = started + timedelta(seconds=max(1, elapsed // 1_000_000_000))
        arm_hash = design["arms"][role]["configuration_sha256"]
        ranks = []
        for rank in (0, 1):
            rank_path = bundle / f"rank-outputs/{attempt_id}-rank-{rank}.json"
            _json(rank_path, {"schema": "synthetic-rank-output.v1", "attempt_id": attempt_id,
                              "rank": rank, "synthetic_fixture": True})
            rank_paths[(attempt_id, rank)] = rank_path
            ranks.append({
                "rank": rank, "local_rank": rank, "pid": 20_000 + attempt_index * 10 + rank,
                "host": "synthetic-one-host", "device_type": "cuda",
                "device_uuid": design["topology"]["device_uuids"][rank],
                "output_ref": _bundle_ref(bundle, rank_path),
                "arm_configuration_sha256": arm_hash,
            })
        start_ns = 1_000_000_000
        end_ns = start_ns + elapsed
        phases = ADMISSION_PHASES if group["kind"] == "admission" else REQUIRED_PHASES
        phase_intervals = {}
        for phase_index, phase in enumerate(phases):
            left = start_ns + (phase_index + 1) * 500_000_000
            phase_intervals[phase] = [
                [left, left + 600_000_000],
                [left + 400_000_000, left + 1_000_000_000],
            ]
        update_count = 12 if group["kind"] == "admission" else 192
        updates = [{"index": index, "latency_ns": (500_000_000 if role == "baseline" else 450_000_000) + index * 1000}
                   for index in range(1, update_count + 1)]
        initial_quality = {"nll_sum_nanonats": 4_000_000_000 * 1_048_576,
                           "target_bytes": 1_048_576, "correct_bytes": 250_000}
        final_base = 3_800_000_000 * 1_048_576 - attempt_index * 1000
        final_quality = {
            "nll_sum_nanonats": final_base + (10_000_000 * 1_048_576 if role == "candidate" else 0),
            "target_bytes": 1_048_576,
            "correct_bytes": 320_000 - (5_000 if role == "candidate" else 0),
        }
        checkpoint = None
        if group["kind"] == "pair":
            state = {component: _hash_bytes(f"{attempt_id}-{component}".encode()) for component in components}
            periodic = detached_map[f"{attempt_id}-periodic-checkpoint"]
            final = detached_map[f"{attempt_id}-final-checkpoint"]
            checkpoint = {
                "periodic_files": [{"object_id": periodic["id"], "sha256": periodic["sha256"], "size_bytes": periodic["size_bytes"]}],
                "final_files": [{"object_id": final["id"], "sha256": final["sha256"], "size_bytes": final["size_bytes"]}],
                "pre_exit_state": state, "post_load_state": copy.deepcopy(state),
                "state_components": components,
                "saved_next_batch_sha256": _hash_bytes(f"{attempt_id}-next-batch".encode()),
                "resumed_next_batch_sha256": _hash_bytes(f"{attempt_id}-next-batch".encode()),
                "pre_exit_dry_forward_loss_nanonats": 3_900_000_000,
                "reloaded_dry_forward_loss_nanonats": 3_900_000_500,
                "step_before_exit": 96, "step_after_load": 96,
                "first_resumed_step_before": 96, "first_resumed_step_after": 97,
                "update_97_occurrences": 1, "final_clean_process_load": True,
                "te_external_state_present": False,
                "durable_protocol": "flush_fsync_atomic_rename_directory_fsync_manifest_last",
            }
        useful = ({
            "optimizer_updates": 12, "accumulation_rounds": 48,
            "rank_local_microbatches": 96, "target_bytes": 786_432,
            "periodic_checkpoints": 0, "clean_restarts": 0,
            "first_resumed_updates": 0, "final_checkpoints": 0,
            "validation_windows": 0,
        } if group["kind"] == "admission" else {
            "optimizer_updates": 192, "accumulation_rounds": 768,
            "rank_local_microbatches": 1536, "target_bytes": 12_582_912,
            "periodic_checkpoints": 1, "clean_restarts": 1,
            "first_resumed_updates": 1, "final_checkpoints": 1,
            "validation_windows": 512,
        })
        useful.update({"retries": 0, "skipped_updates": 0, "duplicate_updates": 0,
                       "update_indices": list(range(1, update_count + 1))})
        raw = {
            "schema": "piq.gpu-training-attempt.v1",
            "study_id": design["study_id"], "design_sha256": design_sha,
            "group_id": group["id"], "run_id": run_id, "attempt_id": attempt_id,
            "role": role, "arm_configuration_sha256": arm_hash,
            "status": "succeeded", "accepted": True, "failure_reason": None,
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "ended_at": ended.isoformat().replace("+00:00", "Z"),
            "primary_clock": {"start_ns": start_ns, "end_ns": end_ns},
            "elapsed_ns": elapsed,
            "identity": {
                "runtime_identifier": design["runtime"]["runtime_identifier"],
                "hardware_identifier": design["topology"]["topology_sha256"],
                "source_sha256s": source_hashes,
                "resolved_imports": design["source_imports"],
                "data_sha256s": {key: design["data"][key] for key in ("train_sha256", "validation_sha256", "sampler_sha256")},
                "initial_checkpoint_sha256": design["retention"]["initial_checkpoint_sha256"],
                "topology_sha256": design["topology"]["topology_sha256"],
                "environment_image_digest": design["runtime"]["image_digest"],
            },
            "ranks": ranks,
            "power_observations": [
                {"device_uuid": uuid, "label": label,
                 "source": "nvidia_smi_power_limit_milliwatts",
                 "observed_limit_mw": 1_000_000 + (-500 if label == "start" else 500),
                 "mutation_attempted": False}
                for uuid in design["topology"]["device_uuids"] for label in ("start", "end")
            ],
            "useful_work": useful,
            "safety": {
                "finite_losses": True, "finite_gradient_norms": True,
                "overflow_count": 0, "skipped_optimizer_updates": 0,
                "nccl_errors": 0, "cuda_errors": 0, "xid_errors": 0,
                "ecc_errors": 0, "fallback_warnings": 0,
            },
            "updates": updates,
            "quality": {"initial_bf16": initial_quality, "final_bf16": final_quality},
            "checkpoint": checkpoint,
            "telemetry": {
                "layers": [{"name": name, "state": "observed"} for name in REQUIRED_LAYERS],
                "parser_accepted": True, "raw_parser_accepted": True,
                "dropped_records": 0, "overflow": False, "truncated": False,
                "coverage_by_rank": [
                    {"rank": rank, "intervals": [[start_ns, start_ns + elapsed // 2],
                                                  [start_ns + elapsed // 2, end_ns]]}
                    for rank in (0, 1)
                ],
                "phase_intervals": phase_intervals,
                "observer_sources": [
                    {"layer": name, "producer": f"synthetic-{name}-observer",
                     "sha256": _hash_bytes(f"observer-{name}".encode())}
                    for name in REQUIRED_LAYERS
                ],
                "rank_device_bindings": [{"rank": rank, "device_uuid": design["topology"]["device_uuids"][rank]} for rank in (0, 1)],
                "summary_ref": telemetry_summary_ref,
                "raw_custody_ref": telemetry_raw_ref,
                "heavy_profiles": [],
            },
            "cleanup": {
                "launcher_exited": True, "workers_exited": True,
                "process_group_closed": True, "observers_closed": True,
                "owned_gpu_processes": 0, "owned_allocations": 0,
                "scratch_removed": True, "retained_bytes_readable": True,
            },
        }
        raw_path = bundle / f"attempts/{attempt_id}.json"
        _json(raw_path, raw)
        raw_paths[attempt_id] = raw_path
        raw_attempts.append(raw)

    normalized = [
        normalize_gpu_attempt(design, group, {}, raw)
        for (group, _, _, _), raw in zip(attempt_specs, raw_attempts, strict=True)
    ]
    report = evaluate_gpu_study(design, normalized, design_sha256=design_sha)
    if report["verdict"] != "QUALIFIED":
        raise AssertionError(report["errors"])

    source_closure_path = bundle / "source-import-closure.json"
    _json(source_closure_path, {
        "schema_version": 1, "kind": "gpu_training_source_import_closure",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "imports": design["source_imports"], "accepted": True,
    })
    raw_rows = [{
        "group_id": raw["group_id"], "role": raw["role"], "run_id": raw["run_id"],
        "attempt_id": raw["attempt_id"], "raw_result": _bundle_ref(bundle, raw_paths[raw["attempt_id"]]),
    } for raw in raw_attempts]
    raw_census_path = bundle / "raw-census.json"
    _json(raw_census_path, {
        "schema_version": 1, "kind": "gpu_training_raw_census",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "attempts": raw_rows,
    })
    quality_path = bundle / "quality.json"
    _json(quality_path, {
        "schema_version": 1, "kind": "gpu_training_correctness_quality",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "accepted": True, "metrics": report["metrics"],
    })
    full_attempt_ids = [raw["attempt_id"] for raw in raw_attempts if raw["group_id"] in PAIR_IDS]
    checkpoint_path = bundle / "checkpoint-restart.json"
    _json(checkpoint_path, {
        "schema_version": 1, "kind": "gpu_training_checkpoint_restart",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "accepted": True, "attempt_ids": full_attempt_ids,
        "detached_object_ids": sorted(item["id"] for item in detached_objects),
    })
    telemetry_path = bundle / "telemetry-coverage.json"
    _json(telemetry_path, {
        "schema_version": 1, "kind": "gpu_training_telemetry_coverage",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "accepted": True, "required_layers": list(REQUIRED_LAYERS),
        "required_phases": list(REQUIRED_PHASES),
        "attempt_ids": [raw["attempt_id"] for raw in raw_attempts],
    })
    custody_aggregate_path = bundle / "custody-readback-cleanup.json"
    _json(custody_aggregate_path, {
        "schema_version": 1, "kind": "gpu_training_custody_readback_cleanup",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "detached_object_ids": sorted(item["id"] for item in detached_objects),
        "local_readback": True, "piq_readback": True, "export_readback": True,
        "all_retained_bytes_readable": True, "all_owned_processes_released": True,
        "all_owned_allocations_released": True, "failure_census_complete": True,
        "cleanup_attempt_ids": [raw["attempt_id"] for raw in raw_attempts],
    })
    execution_path = bundle / "execution-index.json"
    _json(execution_path, {"schema": "synthetic-execution-index.v1",
                           "attempt_ids": [raw["attempt_id"] for raw in raw_attempts],
                           "synthetic_fixture": True})
    cleanup_path = bundle / "cleanup-index.json"
    _json(cleanup_path, {"schema": "synthetic-cleanup-index.v1",
                         "attempt_ids": [raw["attempt_id"] for raw in raw_attempts],
                         "complete": True, "synthetic_fixture": True})

    assignments: dict[str, Path] = {}
    resources_by_group: dict[str, tuple[dict, dict]] = {}
    for group in SCHEDULE:
        group_raw = [raw for raw in raw_attempts if raw["group_id"] == group["id"]]
        requested = {"executor": "assigned-gpu-host", "node_count": 1,
                     "accelerator_count": 2, "process_count": 2, "timeout_seconds": 420}
        observed = {"node_count": 1, "accelerator_count": 2, "process_count": 2,
                    "host": "synthetic-one-host", "device_uuids": design["topology"]["device_uuids"]}
        resources_by_group[group["id"]] = (requested, observed)
        assignment_path = bundle / f"assignments/{group['id']}.json"
        _json(assignment_path, {
            "schema": "piq.gpu-training-assignment.v1",
            "study_id": design["study_id"], "group_id": group["id"],
            "run_id": group_raw[0]["run_id"],
            "scheduler_job_id": f"synthetic-job-{group['id']}",
            "owner": "synthetic-execution-owner", "node": "synthetic-one-host",
            "session_id": f"synthetic-session-{group['id']}",
            "cgroup": f"/synthetic/{group['id']}",
            "device_uuids": design["topology"]["device_uuids"],
            "source_sha256s": source_hashes,
            "runtime_identifier": design["runtime"]["runtime_identifier"],
            "executable_sha256": _hash_bytes(b"synthetic-python-executable"),
            "plan_sha256": _hash_value(group),
            "window": {"started_at": group_raw[0]["started_at"], "ended_at": group_raw[-1]["ended_at"]},
            "requested_resources": requested, "observed_resources": observed,
            "attempt_processes": [{"attempt_id": raw["attempt_id"],
                                   "pids": [rank["pid"] for rank in raw["ranks"]]}
                                  for raw in group_raw],
        })
        assignments[group["id"]] = assignment_path

    executions = []
    envelope_attempts = []
    for group in SCHEDULE:
        group_raw = [raw for raw in raw_attempts if raw["group_id"] == group["id"]]
        assignment_ref = _bundle_ref(bundle, assignments[group["id"]])
        requested, observed = resources_by_group[group["id"]]
        executions.append({
            "group_id": group["id"], "run_id": group_raw[0]["run_id"],
            "order": group["order"], "attempt_ids": [raw["attempt_id"] for raw in group_raw],
            "requested_resources": requested, "observed_resources": observed,
            "assignment_evidence": assignment_ref,
        })
        for raw in group_raw:
            envelope_attempts.append({
                "id": raw["attempt_id"], "run_id": raw["run_id"],
                "group_id": raw["group_id"], "role": raw["role"],
                "status": raw["status"], "accepted": raw["accepted"],
                "failure_reason": raw["failure_reason"],
                "raw_result": _bundle_ref(bundle, raw_paths[raw["attempt_id"]]),
                "assignment_evidence": assignment_ref,
            })

    workload_evidence = {
        "execution": _bundle_ref(bundle, execution_path),
        "quality": _bundle_ref(bundle, quality_path),
        "checkpoint_restart": _bundle_ref(bundle, checkpoint_path),
        "telemetry_summary": _bundle_ref(bundle, telemetry_path),
        "telemetry_raw_custody_index": telemetry_raw_ref,
        "cleanup": _bundle_ref(bundle, cleanup_path),
        "readback": _bundle_ref(bundle, custody_paths["local-readback"]),
        "export": _bundle_ref(bundle, custody_paths["export-readback"]),
    }
    artifact_items = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "envelope.json":
            relative = path.relative_to(bundle).as_posix()
            artifact_items.append({
                "path": relative, "sha256": _hash_bytes(path.read_bytes()),
                "size_bytes": path.stat().st_size,
                "role": "synthetic_fixture_artifact",
            })
    source_refs = [_bundle_ref(bundle, path) for path in source_paths]
    envelope = {
        "schema_version": 2, "kind": "piq_experiment_evidence",
        "scope": "diagnostic", "scientific_disposition": "DIAGNOSTIC_ONLY",
        "experiment_kind": "te_mxfp8_gpu_training",
        "workload_family": "gpu_training", "study_id": design["study_id"],
        "experiment": {
            "id": design["study_id"],
            "question": "Synthetic fixture: does complete-job MXFP8 pass the frozen gates?",
            "hypothesis": "Synthetic fixture only. No execution claim.",
            "workload_scope": design["workload_scope"],
            "design": _bundle_ref(bundle, design_path),
            "source_artifacts": source_refs,
            "data": _bundle_ref(bundle, data_identity_path),
            "initial_checkpoint": {"object_id": "initial-checkpoint",
                                   "sha256": detached_map["initial-checkpoint"]["sha256"],
                                   "size_bytes": detached_map["initial-checkpoint"]["size_bytes"]},
        },
        "stack": {
            "piq_revision": "1" * 40,
            "runtime_identifier": design["runtime"]["runtime_identifier"],
            "hardware_identifier": design["topology"]["topology_sha256"],
            "layers": [{"name": name, "state": "observed"} for name in REQUIRED_LAYERS],
        },
        "objective": {"metric": "complete_job_elapsed", "unit": "nanoseconds",
                      "direction": "minimize", "started_at": raw_attempts[0]["started_at"],
                      "ended_at": raw_attempts[-1]["ended_at"]},
        "executions": executions,
        "attempts": envelope_attempts,
        "comparison": {
            "baseline_attempt_ids": [raw["attempt_id"] for raw in raw_attempts if raw["role"] == "baseline"],
            "candidate_attempt_ids": [raw["attempt_id"] for raw in raw_attempts if raw["role"] == "candidate"],
            "scientific_disposition": "DIAGNOSTIC_ONLY",
            "reason": "PIQ v2 retains diagnostics. Only measurement receipt v3 may qualify.",
        },
        "correctness": {"accepted": True, "artifact": _bundle_ref(bundle, quality_path)},
        "workload_evidence": workload_evidence,
        "detached_objects": detached_objects,
        "artifacts": artifact_items,
    }
    envelope_path = bundle / "envelope.json"
    _json(envelope_path, envelope)

    registration_path = qualification / "registration.json"
    _json(registration_path, {
        "schema_version": 1, "kind": "gpu_training_registration",
        "study_id": design["study_id"], "design_sha256": design_sha,
        "registered_at": "2026-10-31T10:30:00Z",
        "authority": "synthetic-advance-registry", "record_id": "synthetic-registration-1",
        "group_plan_sha256s": {group["id"]: _hash_value(group) for group in SCHEDULE},
    })
    baseline_review_path = qualification / "baseline-review.json"
    _json(baseline_review_path, {
        "study_id": design["study_id"], "design_sha256": design_sha,
        "reviewer": "synthetic-baseline-reviewer", "independent": True,
        "verdict": "PASS", "reviewed_at": "2026-10-31T10:10:00Z",
        "reasoning": "Synthetic review fixture defends the same optimized TE graph and FP32 master state.",
        "baseline_id": "bf16", "optimized_standard": True,
        "primary_sources": ["https://docs.nvidia.com/deeplearning/transformer-engine/"],
    })
    telemetry_review_path = qualification / "telemetry-review.json"
    _json(telemetry_review_path, {
        "study_id": design["study_id"], "design_sha256": design_sha,
        "reviewer": "synthetic-telemetry-reviewer", "independent": True,
        "verdict": "PASS", "reviewed_at": "2026-10-31T10:20:00Z",
        "reasoning": "Synthetic review fixture ratifies interval, quality, CV, p95 and power semantics.",
        "required_layers": list(REQUIRED_LAYERS), "required_phases": list(REQUIRED_PHASES),
        "machine_semantics_sha256": _hash_value(MEASUREMENT_SEMANTICS),
    })
    report_path = qualification / "qualification-report.json"
    _json(report_path, report, canonical=True)
    scientific_review_path = qualification / "scientific-review.json"
    _json(scientific_review_path, {
        "study_id": design["study_id"], "design_sha256": design_sha,
        "reviewer": "synthetic-scientific-reviewer", "independent": True,
        "verdict": "PASS", "reviewed_at": "2026-10-31T13:00:00Z",
        "reasoning": "Synthetic fixture review checks every retained contract dimension. It proves no execution.",
        "qualification_report_sha256": _hash_bytes(report_path.read_bytes()),
        "assessments": ["source_identity", "controls", "correctness_quality",
                        "checkpoint_restart", "timing_boundary", "telemetry",
                        "custody_cleanup", "scope_limitations"],
    })
    pair_refs = []
    for pair in report["metrics"]["pairs"]:
        pair_path = qualification / f"pairs/{pair['id']}.json"
        _json(pair_path, {"schema_version": 1, "kind": "gpu_training_pair_result",
                          "pair": pair})
        pair_refs.append({"id": pair["id"], "accepted": True,
                          "raw_result": _package_ref(package, pair_path)})

    receipt = {
        "schema_version": 3, "state": "measured", "execution_class": "gpu",
        "workload_family": "gpu_training", "contract_id": CONTRACT_ID,
        "workload_scope": design["workload_scope"],
        "runtime_identifier": design["runtime"]["runtime_identifier"],
        "hardware_identifier": design["topology"]["topology_sha256"],
        "design": _package_ref(package, design_path),
        "registration": _package_ref(package, registration_path),
        "source_import_closure": _package_ref(package, source_closure_path),
        "baseline_review": _package_ref(package, baseline_review_path),
        "telemetry_review": _package_ref(package, telemetry_review_path),
        "raw_census": _package_ref(package, raw_census_path),
        "correctness_quality": _package_ref(package, quality_path),
        "checkpoint_restart": _package_ref(package, checkpoint_path),
        "telemetry_coverage": _package_ref(package, telemetry_path),
        "custody_readback_cleanup": _package_ref(package, custody_aggregate_path),
        "qualification_report": _package_ref(package, report_path),
        "scientific_review": _package_ref(package, scientific_review_path),
        "source_artifacts": [_package_ref(package, path) for path in source_paths],
        "piq_envelope": _package_ref(package, envelope_path),
        "accepted_repeats": pair_refs,
        "failed_attempts": [], "profile_evidence": [],
        "claims": [{"id": "complete-job-qualified-gain",
                    "source": _package_ref(package, report_path),
                    "accepted_repeat_ids": list(PAIR_IDS), "profile_claim": False}],
    }
    receipt_path = package / "evidence/measurement-receipt.json"
    _json(receipt_path, receipt)
    return receipt


def _normalized_fixture(package: Path, receipt: dict) -> tuple[dict, dict, list[dict]]:
    design = strict_load_json(package / receipt["design"]["path"])
    envelope_path = package / receipt["piq_envelope"]["path"]
    envelope = strict_load_json(envelope_path)
    groups = {item["id"]: item for item in SCHEDULE}
    attempts = []
    for item in envelope["attempts"]:
        raw = strict_load_json(envelope_path.parent / item["raw_result"]["path"])
        attempts.append(normalize_gpu_attempt(
            design, groups[item["group_id"]], envelope, raw,
        ))
    return design, envelope, attempts


class QualifiedGpuTrainingTests(unittest.TestCase):
    def test_complete_synthetic_fixture_validates_but_cannot_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            self.assertEqual([], validate_gpu_receipt(package, receipt))
            with self.assertRaisesRegex(EvidenceError, "synthetic"):
                build_gpu_public_summary(package, receipt)

    def test_design_closes_all_five_machine_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design = strict_load_json(package / receipt["design"]["path"])
            self.assertEqual([], validate_gpu_design(design))
            self.assertEqual("nearest_rank_ceiling", design["measurement_semantics"]["p95"]["estimator"])
            self.assertEqual("population", design["measurement_semantics"]["coefficient_of_variation"]["standard_deviation"])
            self.assertEqual("half_open_monotonic_nanoseconds", design["measurement_semantics"]["intervals"]["model"])
            self.assertIn("integer_total_nanonats", design["measurement_semantics"]["quality"]["nll_representation"])
            self.assertEqual(1000, design["measurement_semantics"]["power"]["absolute_tolerance_mw"])

    def test_report_byte_edit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            path = package / receipt["qualification_report"]["path"]
            path.write_text(path.read_text().replace("  \"accepted_pair_ids\"", " \"accepted_pair_ids\""))
            receipt["qualification_report"]["sha256"] = _hash_bytes(path.read_bytes())
            self.assertTrue(any("canonical" in error or "scientific review" in error
                                for error in validate_gpu_receipt(package, receipt)))

    def test_admission_cannot_enter_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            receipt["claims"][0]["accepted_repeat_ids"][0] = "admission-01"
            errors = validate_gpu_receipt(package, receipt)
            self.assertTrue(any("all four" in error for error in errors))

    def test_gain_boundary_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design = strict_load_json(package / receipt["design"]["path"])
            envelope = strict_load_json(package / receipt["piq_envelope"]["path"])
            bundle = (package / receipt["piq_envelope"]["path"]).parent
            normalized = []
            groups = {item["id"]: item for item in SCHEDULE}
            for item in envelope["attempts"]:
                raw = strict_load_json(bundle / item["raw_result"]["path"])
                normalized.append(normalize_gpu_attempt(design, groups[item["group_id"]], envelope, raw))
            pair = [item for item in normalized if item["group_id"] == "pair-01"]
            baseline = next(item for item in pair if item["role"] == "baseline")
            candidate = next(item for item in pair if item["role"] == "candidate")
            baseline["elapsed_ns"] = 105
            candidate["elapsed_ns"] = 100
            # Other pairs keep the median gate passing. The exact per-pair boundary passes.
            report = evaluate_gpu_study(design, normalized, design_sha256=receipt["design"]["sha256"])
            self.assertFalse(any("pair-01 complete-job gain" in error for error in report["errors"]))
            baseline["elapsed_ns"] = 104
            report = evaluate_gpu_study(design, normalized, design_sha256=receipt["design"]["sha256"])
            self.assertTrue(any("pair-01 complete-job gain" in error for error in report["errors"]))

    def test_p95_population_and_population_cv_boundaries_are_exact(self) -> None:
        self.assertEqual(178, len(ELIGIBLE_UPDATE_INDICES))
        self.assertEqual((13, 95, 98, 192), (
            ELIGIBLE_UPDATE_INDICES[0], ELIGIBLE_UPDATE_INDICES[82],
            ELIGIBLE_UPDATE_INDICES[83], ELIGIBLE_UPDATE_INDICES[-1],
        ))
        self.assertEqual(170, _nearest_rank_p95(range(1, 179)))
        passed, squared = _cv_pass([95, 105])
        self.assertTrue(passed)
        self.assertEqual("1/400", str(squared))
        self.assertFalse(_cv_pass([94, 106])[0])

    def test_power_limit_boundary_and_source_import_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design = strict_load_json(package / receipt["design"]["path"])
            envelope_path = package / receipt["piq_envelope"]["path"]
            envelope = strict_load_json(envelope_path)
            attempt = next(item for item in envelope["attempts"]
                           if item["group_id"] == "pair-01")
            raw = strict_load_json(envelope_path.parent / attempt["raw_result"]["path"])
            group = next(item for item in SCHEDULE if item["id"] == "pair-01")
            expected = design["power_policy"]["expected_limit_mw_by_device"]

            at_boundary = copy.deepcopy(raw)
            observation = at_boundary["power_observations"][0]
            observation["observed_limit_mw"] = expected[observation["device_uuid"]] + 1000
            normalize_gpu_attempt(design, group, envelope, at_boundary)

            over_boundary = copy.deepcopy(at_boundary)
            over_boundary["power_observations"][0]["observed_limit_mw"] += 1
            with self.assertRaisesRegex(EvidenceError, "1000 mW tolerance"):
                normalize_gpu_attempt(design, group, envelope, over_boundary)

            wrong_source = copy.deepcopy(raw)
            wrong_source["power_observations"][0]["source"] = "synthetic-setter"
            with self.assertRaisesRegex(EvidenceError, "source drifted"):
                normalize_gpu_attempt(design, group, envelope, wrong_source)

            import_drift = copy.deepcopy(raw)
            import_drift["identity"]["resolved_imports"][0]["resolved_sha256"] = "f" * 64
            with self.assertRaisesRegex(EvidenceError, "identity drifted"):
                normalize_gpu_attempt(design, group, envelope, import_drift)

    def test_checkpoint_mutations_cover_required_restart_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            _, _, attempts = _normalized_fixture(package, receipt)
            checkpoint = next(item["checkpoint"] for item in attempts
                              if item["group_id"] == "pair-01")
            self.assertEqual([], validate_gpu_checkpoint(checkpoint))
            dry_boundary = copy.deepcopy(checkpoint)
            dry_boundary["reloaded_dry_forward_loss_nanonats"] = (
                checkpoint["pre_exit_dry_forward_loss_nanonats"] + 1000
            )
            self.assertEqual([], validate_gpu_checkpoint(dry_boundary))
            mutations = [
                ("periodic_files", [], "periodic_files"),
                ("post_load_state", {}, "round-trip"),
                ("reloaded_dry_forward_loss_nanonats",
                 checkpoint["pre_exit_dry_forward_loss_nanonats"] + 1001,
                 "dry-forward"),
                ("update_97_occurrences", 2, "update 97"),
                ("final_clean_process_load", False, "clean process"),
            ]
            for field, value, expected in mutations:
                with self.subTest(field=field):
                    changed = copy.deepcopy(checkpoint)
                    changed[field] = value
                    self.assertTrue(any(expected in error
                                        for error in validate_gpu_checkpoint(changed)))

    def test_gain_latency_and_phase_thresholds_pass_on_boundary_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design, _, original = _normalized_fixture(package, receipt)

            median_exact = copy.deepcopy(original)
            for pair_id in PAIR_IDS:
                for item in median_exact:
                    if item["group_id"] == pair_id:
                        item["elapsed_ns"] = 110 if item["role"] == "baseline" else 100
            report = evaluate_gpu_study(
                design, median_exact, design_sha256=receipt["design"]["sha256"],
            )
            self.assertNotIn(
                "GPU study median complete-job gain is below 1.10", report["errors"],
            )
            for item in median_exact:
                if item["group_id"] in PAIR_IDS and item["role"] == "baseline":
                    item["elapsed_ns"] = 109
            report = evaluate_gpu_study(
                design, median_exact, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU study median complete-job gain is below 1.10", report["errors"],
            )

            spread_exact = copy.deepcopy(original)
            baseline_times = dict(zip(PAIR_IDS, (105, 105, 115, 115), strict=True))
            for item in spread_exact:
                if item["group_id"] in PAIR_IDS:
                    item["elapsed_ns"] = (
                        baseline_times[item["group_id"]]
                        if item["role"] == "baseline" else 100
                    )
            report = evaluate_gpu_study(
                design, spread_exact, design_sha256=receipt["design"]["sha256"],
            )
            self.assertNotIn(
                "GPU study paired gain spread exceeds 0.10", report["errors"],
            )
            next(item for item in spread_exact
                 if item["group_id"] == "pair-04" and item["role"] == "baseline")[
                     "elapsed_ns"] = 116
            report = evaluate_gpu_study(
                design, spread_exact, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU study paired gain spread exceeds 0.10", report["errors"],
            )

            latency = copy.deepcopy(original)
            for item in latency:
                if item["group_id"] in PAIR_IDS:
                    value = 90 if item["role"] == "candidate" else 100
                    item["eligible_update_latencies_ns"] = [value] * 178
            pair_candidate = next(item for item in latency
                                  if item["group_id"] == "pair-01"
                                  and item["role"] == "candidate")
            pair_candidate["eligible_update_latencies_ns"] = [102] * 178
            report = evaluate_gpu_study(
                design, latency, design_sha256=receipt["design"]["sha256"],
            )
            self.assertFalse(any("pair-01 candidate p95 ratio" in error
                                 for error in report["errors"]))
            pair_candidate["eligible_update_latencies_ns"] = [103] * 178
            report = evaluate_gpu_study(
                design, latency, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("pair-01 candidate p95 ratio" in error
                                for error in report["errors"]))

            for item in latency:
                if item["group_id"] in PAIR_IDS:
                    item["eligible_update_latencies_ns"] = [100] * 178
            for pair_id, value in zip(PAIR_IDS, (90, 90, 100, 100), strict=True):
                next(item for item in latency if item["group_id"] == pair_id
                     and item["role"] == "candidate")["eligible_update_latencies_ns"] = [value] * 178
            report = evaluate_gpu_study(
                design, latency, design_sha256=receipt["design"]["sha256"],
            )
            self.assertNotIn(
                "GPU study median candidate/baseline p95 ratio exceeds 0.95",
                report["errors"],
            )
            for pair_id, value in zip(PAIR_IDS, (90, 91, 100, 101), strict=True):
                next(item for item in latency if item["group_id"] == pair_id
                     and item["role"] == "candidate")["eligible_update_latencies_ns"] = [value] * 178
            report = evaluate_gpu_study(
                design, latency, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU study median candidate/baseline p95 ratio exceeds 0.95",
                report["errors"],
            )

            phase = copy.deepcopy(original)
            baseline = next(item for item in phase if item["group_id"] == "pair-01"
                            and item["role"] == "baseline")
            candidate = next(item for item in phase if item["group_id"] == "pair-01"
                             and item["role"] == "candidate")
            baseline["phase_durations_ns"]["data_open"] = 10_000_000_000
            candidate["phase_durations_ns"]["data_open"] = 13_000_000_000
            report = evaluate_gpu_study(
                design, phase, design_sha256=receipt["design"]["sha256"],
            )
            self.assertFalse(any("pair-01 data_open" in error for error in report["errors"]))
            candidate["phase_durations_ns"]["data_open"] += 1
            report = evaluate_gpu_study(
                design, phase, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("pair-01 data_open" in error for error in report["errors"]))

    def test_telemetry_gap_interval_and_observer_parity_mutations_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design, _, attempts = _normalized_fixture(package, receipt)
            attempt = next(item for item in attempts if item["group_id"] == "pair-01")
            telemetry = attempt["telemetry"]
            clock, ranks = attempt["primary_clock"], attempt["ranks"]
            self.assertEqual([], validate_gpu_telemetry(telemetry, clock, ranks))

            gap = copy.deepcopy(telemetry)
            start, end = clock["start_ns"], clock["end_ns"]
            middle = start + (end - start) // 2
            gap["coverage_by_rank"][0]["intervals"] = [
                [start, middle], [middle + 1, end],
            ]
            self.assertTrue(any("gap" in error
                                for error in validate_gpu_telemetry(gap, clock, ranks)))

            outside = copy.deepcopy(telemetry)
            outside["phase_intervals"]["data_open"] = [[start, end + 1]]
            self.assertTrue(any("inside the primary clock" in error
                                for error in validate_gpu_telemetry(outside, clock, ranks)))

            changed_sources = copy.deepcopy(attempts)
            changed_sources[3]["telemetry"]["observer_sources"][0]["producer"] = "other-observer"
            report = evaluate_gpu_study(
                design, changed_sources, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU telemetry observer source identity differs across attempts",
                report["errors"],
            )
            self.assertFalse(report["gates"]["telemetry"])

    def test_quality_rational_boundaries_use_unrounded_integer_operands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design, _, attempts = _normalized_fixture(package, receipt)
            baseline = next(item for item in attempts
                            if item["group_id"] == "pair-01" and item["role"] == "baseline")
            candidate = next(item for item in attempts
                             if item["group_id"] == "pair-01" and item["role"] == "candidate")
            count = baseline["quality"]["final_bf16"]["target_bytes"]

            candidate["quality"]["final_bf16"]["nll_sum_nanonats"] = (
                baseline["quality"]["final_bf16"]["nll_sum_nanonats"]
                + 30_000_000 * count
            )
            report = evaluate_gpu_study(
                design, attempts, design_sha256=receipt["design"]["sha256"],
            )
            self.assertFalse(any("pair-01 candidate NLL gap" in error
                                 for error in report["errors"]))
            candidate["quality"]["final_bf16"]["nll_sum_nanonats"] += 1
            report = evaluate_gpu_study(
                design, attempts, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("pair-01 candidate NLL gap" in error
                                for error in report["errors"]))

            baseline_correct = baseline["quality"]["final_bf16"]["correct_bytes"]
            candidate["quality"]["final_bf16"]["nll_sum_nanonats"] = (
                baseline["quality"]["final_bf16"]["nll_sum_nanonats"]
            )
            candidate["quality"]["final_bf16"]["correct_bytes"] = (
                baseline_correct - (3 * count // 400)
            )
            report = evaluate_gpu_study(
                design, attempts, design_sha256=receipt["design"]["sha256"],
            )
            self.assertFalse(any("pair-01 candidate byte-accuracy gap" in error
                                 for error in report["errors"]))
            candidate["quality"]["final_bf16"]["correct_bytes"] -= 1
            report = evaluate_gpu_study(
                design, attempts, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("pair-01 candidate byte-accuracy gap" in error
                                for error in report["errors"]))

    def test_absolute_learning_and_median_quality_boundaries_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design, _, original = _normalized_fixture(package, receipt)

            learning = copy.deepcopy(original)
            attempt = next(item for item in learning if item["group_id"] == "pair-01"
                           and item["role"] == "candidate")
            initial = attempt["quality"]["initial_bf16"]
            attempt["quality"]["final_bf16"]["nll_sum_nanonats"] = (
                initial["nll_sum_nanonats"]
                - initial["target_bytes"] * 100_000_000
            )
            report = evaluate_gpu_study(
                design, learning, design_sha256=receipt["design"]["sha256"],
            )
            self.assertFalse(any("pair-01 candidate absolute learning" in error
                                 for error in report["errors"]))
            attempt["quality"]["final_bf16"]["nll_sum_nanonats"] += 1
            report = evaluate_gpu_study(
                design, learning, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("pair-01 candidate absolute learning" in error
                                for error in report["errors"]))
            self.assertFalse(report["gates"]["quality"])

            nll = copy.deepcopy(original)
            for pair_id, gap in zip(PAIR_IDS, (10_000_000, 10_000_000,
                                               30_000_000, 30_000_000), strict=True):
                baseline = next(item for item in nll if item["group_id"] == pair_id
                                and item["role"] == "baseline")
                candidate = next(item for item in nll if item["group_id"] == pair_id
                                 and item["role"] == "candidate")
                count = baseline["quality"]["final_bf16"]["target_bytes"]
                candidate["quality"]["final_bf16"]["nll_sum_nanonats"] = (
                    baseline["quality"]["final_bf16"]["nll_sum_nanonats"]
                    + gap * count
                )
            report = evaluate_gpu_study(
                design, nll, design_sha256=receipt["design"]["sha256"],
            )
            self.assertNotIn(
                "GPU study median candidate NLL gap exceeds 0.02 nat/byte",
                report["errors"],
            )
            next(item for item in nll if item["group_id"] == "pair-02"
                 and item["role"] == "candidate")["quality"]["final_bf16"][
                     "nll_sum_nanonats"] += 1
            report = evaluate_gpu_study(
                design, nll, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU study median candidate NLL gap exceeds 0.02 nat/byte",
                report["errors"],
            )

            accuracy = copy.deepcopy(original)
            gap_counts = (5000, 5000, 5485, 5485)
            for pair_id, gap in zip(PAIR_IDS, gap_counts, strict=True):
                baseline = next(item for item in accuracy if item["group_id"] == pair_id
                                and item["role"] == "baseline")
                candidate = next(item for item in accuracy if item["group_id"] == pair_id
                                 and item["role"] == "candidate")
                candidate["quality"]["final_bf16"]["correct_bytes"] = (
                    baseline["quality"]["final_bf16"]["correct_bytes"] - gap
                )
            report = evaluate_gpu_study(
                design, accuracy, design_sha256=receipt["design"]["sha256"],
            )
            self.assertNotIn(
                "GPU study median candidate byte-accuracy gap exceeds 0.50 point",
                report["errors"],
            )
            next(item for item in accuracy if item["group_id"] == "pair-02"
                 and item["role"] == "candidate")["quality"]["final_bf16"][
                     "correct_bytes"] -= 1
            report = evaluate_gpu_study(
                design, accuracy, design_sha256=receipt["design"]["sha256"],
            )
            self.assertIn(
                "GPU study median candidate byte-accuracy gap exceeds 0.50 point",
                report["errors"],
            )

    def test_nested_unsupported_design_field_and_attempt_reorder_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            design, _, attempts = _normalized_fixture(package, receipt)
            changed_design = copy.deepcopy(design)
            changed_design["runtime"]["ignored_override"] = True
            self.assertTrue(any("runtime" in error
                                for error in validate_gpu_design(changed_design)))
            attempts[2], attempts[3] = attempts[3], attempts[2]
            report = evaluate_gpu_study(
                design, attempts, design_sha256=receipt["design"]["sha256"],
            )
            self.assertTrue(any("fixed five-group order" in error
                                for error in report["errors"]))

    def test_wrong_p95_population_fails_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            envelope = strict_load_json(package / receipt["piq_envelope"]["path"])
            design = strict_load_json(package / receipt["design"]["path"])
            attempt = next(item for item in envelope["attempts"] if item["group_id"] == "pair-01")
            raw = strict_load_json((package / receipt["piq_envelope"]["path"]).parent / attempt["raw_result"]["path"])
            raw["updates"].pop()
            with self.assertRaises(ValueError):
                normalize_gpu_attempt(design, next(item for item in SCHEDULE if item["id"] == "pair-01"), envelope, raw)

    def test_unsupported_receipt_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            receipt["ignored_quality_override"] = True
            self.assertTrue(any("fields differ" in error for error in validate_gpu_receipt(package, receipt)))

    def test_receipt_refuses_non_te_piq_v2_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            receipt = build_gpu_fixture(package)
            envelope_path = package / receipt["piq_envelope"]["path"]
            envelope = strict_load_json(envelope_path)
            envelope["experiment_kind"] = "checkpoint_write_batching_stage0"
            _json(envelope_path, envelope)
            receipt["piq_envelope"]["sha256"] = _hash_bytes(envelope_path.read_bytes())
            self.assertIn(
                "GPU receipt must bind the diagnostic-only PIQ v2 "
                "te_mxfp8_gpu_training variant",
                validate_gpu_receipt(package, receipt),
            )


if __name__ == "__main__":
    unittest.main()
