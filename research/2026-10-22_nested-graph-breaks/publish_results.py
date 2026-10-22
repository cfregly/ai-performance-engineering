#!/usr/bin/env python3
"""Create public projections of retained nested-graph-break benchmark results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_RECORDS = 24
PUBLIC_GPU_UUID = "GPU-<b200-device>"
PUBLIC_HOSTNAME = "<b200-host>"
MANIFEST_SCHEMA = "nested-graph-breaks-public-results/v1"
DIAGNOSIS_SCHEMA = "nested-graph-breaks-public-diagnosis/v1"
QUALIFICATION_SCHEMA = "nested-graph-breaks-public-qualification/v2"


class PublicationError(Exception):
    """Raised when a retained input cannot be projected without ambiguity."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"cannot read JSON from {path}: {exc}") from exc


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> bytes:
    data = _json_bytes(value)
    _write_bytes(path, data)
    return data


def _format_path(path: Sequence[Any]) -> str:
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + str(part)
    return rendered


def _lookup(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for part in path:
        current = current[part]
    return current


def _replace(
    value: MutableMapping[str, Any],
    path: Sequence[Any],
    replacement: Any,
    category: str,
    changes: list[dict[str, Any]],
) -> None:
    current: Any = value
    for part in path[:-1]:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise PublicationError(
                f"required redaction field is missing: {_format_path(path)}"
            ) from exc
    final = path[-1]
    try:
        original = current[final]
    except (KeyError, IndexError, TypeError) as exc:
        raise PublicationError(
            f"required redaction field is missing: {_format_path(path)}"
        ) from exc
    if original == replacement:
        raise PublicationError(
            f"input already contains the public placeholder at {_format_path(path)}"
        )
    current[final] = replacement
    changes.append(
        {
            "path": _format_path(path),
            "category": category,
            "original_type": type(original).__name__,
            "public_type": type(replacement).__name__,
            "replacement": replacement,
        }
    )


def _iter_values(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterable[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_values(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_values(item, path + (index,))
    else:
        yield path, value


def _diff_paths(left: Any, right: Any, path: tuple[Any, ...] = ()) -> list[str]:
    if type(left) is not type(right):
        return [_format_path(path)]
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return [_format_path(path)]
        differences: list[str] = []
        for key in sorted(left):
            differences.extend(_diff_paths(left[key], right[key], path + (key,)))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [_format_path(path)]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            differences.extend(_diff_paths(left_item, right_item, path + (index,)))
        return differences
    return [] if left == right else [_format_path(path)]


def _redact_object_ids(
    record: MutableMapping[str, Any],
    record_id: str,
    changes: list[dict[str, Any]],
) -> None:
    semantic_paths = {
        "top-level-counter-map": (
            "counter_lifecycle",
            "top_level_mapping",
            "object_id",
        ),
        "graph-break-counter": (
            "counter_lifecycle",
            "preparation",
            "graph_break",
            "object_id",
        ),
        "stats-counter": (
            "counter_lifecycle",
            "preparation",
            "stats",
            "object_id",
        ),
    }
    for semantic_name, path in semantic_paths.items():
        _replace(
            record,
            path,
            f"<object-id:{record_id}:{semantic_name}>",
            "process-local object identifier",
            changes,
        )
    for observation_index in range(2):
        for group_name, semantic_name in (
            ("graph_break", "graph-break-counter"),
            ("stats", "stats-counter"),
        ):
            _replace(
                record,
                (
                    "counter_lifecycle",
                    "observations",
                    observation_index,
                    "groups",
                    group_name,
                    "object_id",
                ),
                f"<object-id:{record_id}:{semantic_name}>",
                "process-local object identifier",
                changes,
            )


def _public_record(
    source: Mapping[str, Any], record_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = copy.deepcopy(source)
    changes: list[dict[str, Any]] = []

    for path in (
        ("controller_contract", "cuda_visible_devices"),
        ("controller_contract", "expected_gpu_uuid"),
        ("device", "cuda_visible_devices"),
        ("device", "expected_gpu_uuid"),
        ("device", "observed_uuid"),
    ):
        _replace(record, path, PUBLIC_GPU_UUID, "GPU UUID", changes)

    _replace(
        record,
        ("device", "optional_torch_device_uuid"),
        f"<torch-device-object:{record_id}>",
        "process-local device object address",
        changes,
    )
    _replace(
        record,
        ("runtime", "hostname"),
        PUBLIC_HOSTNAME,
        "hostname",
        changes,
    )
    _replace(
        record,
        ("runtime", "pid"),
        f"<process-id:{record_id}>",
        "process identifier",
        changes,
    )

    for cache_name, suffix in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        _replace(
            record,
            ("controller_contract", "cache_dirs", cache_name, "path"),
            f"<cache-root>/{record_id}/{suffix}",
            "filesystem path and username",
            changes,
        )

    _replace(
        record,
        ("provenance", "worker_source_path"),
        "<source-root>/benchmark.py",
        "filesystem path and username",
        changes,
    )
    _replace(
        record,
        ("provenance", "flag_source_snapshot_path"),
        "<source-root>/pytorch-main-dynamo-config.py",
        "filesystem path and username",
        changes,
    )

    argv = record.get("provenance", {}).get("argv")
    if not isinstance(argv, list):
        raise PublicationError(f"{record_id}: provenance.argv is missing")
    _replace(
        record,
        ("provenance", "argv", 0),
        "<source-root>/benchmark.py",
        "filesystem path and username",
        changes,
    )
    for flag, replacement, category in (
        (
            "--out",
            f"<result-root>/{record_id}",
            "filesystem path and username",
        ),
        ("--expected-gpu-uuid", PUBLIC_GPU_UUID, "GPU UUID"),
    ):
        try:
            index = argv.index(flag) + 1
        except ValueError as exc:
            raise PublicationError(f"{record_id}: {flag} is missing from argv") from exc
        if index >= len(argv):
            raise PublicationError(f"{record_id}: {flag} has no value in argv")
        _replace(record, ("provenance", "argv", index), replacement, category, changes)

    profile = record.get("profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("trace_path"), str):
        _replace(
            record,
            ("profile", "trace_path"),
            f"<result-root>/{record_id}/profile.chrome.json",
            "filesystem path and username",
            changes,
        )

    _redact_object_ids(record, record_id, changes)
    return record, changes


def _assert_public(record: Mapping[str, Any], private_values: Sequence[str]) -> None:
    gpu_uuid = re.compile(r"GPU-[0-9a-fA-F]{8,}(?:-[0-9a-fA-F]+)+")
    pointer = re.compile(r"0x[0-9a-fA-F]{6,}")
    for path, item in _iter_values(record):
        if not isinstance(item, str):
            continue
        if item.startswith("/"):
            raise PublicationError(
                f"public projection retains an absolute path at {_format_path(path)}"
            )
        if gpu_uuid.search(item):
            raise PublicationError(
                f"public projection retains a GPU UUID at {_format_path(path)}"
            )
        if pointer.search(item):
            raise PublicationError(
                f"public projection retains a process address at {_format_path(path)}"
            )
        for private_value in private_values:
            if private_value and private_value in item:
                raise PublicationError(
                    f"public projection retains private metadata at {_format_path(path)}"
                )


def _scientific_analysis_view(analysis: Mapping[str, Any]) -> dict[str, Any]:
    view = copy.deepcopy(analysis)
    view.pop("generated_at", None)
    view.pop("retention", None)
    for item in view.get("profile_evidence", []):
        profile = item.get("profile")
        if isinstance(profile, MutableMapping):
            profile.pop("trace_path", None)
    return view


def _run_analyzer(
    analyzer: Path,
    output: Path,
    reference_analysis_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="nested-graph-public-analysis-") as raw_dir:
        raw_root = Path(raw_dir)
        raw_json = raw_root / "analyzer-result.json"
        raw_markdown = raw_root / "analyzer-report.md"
        command = [
            sys.executable,
            str(analyzer.resolve()),
            str(output.resolve()),
            "--out",
            str(raw_json),
            "--markdown",
            str(raw_markdown),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise PublicationError(
                "analyzer rejected the public projections with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

        raw_analysis_bytes = raw_json.read_bytes()
        analysis = json.loads(raw_analysis_bytes)
        if analysis.get("validation", {}).get("status") != "PASS":
            raise PublicationError("analyzer did not return validation PASS")

        analysis_changes: list[dict[str, Any]] = []
        raw_paths = analysis.get("retention", {}).get("raw_result_paths")
        if not isinstance(raw_paths, list) or len(raw_paths) != EXPECTED_RECORDS:
            raise PublicationError("analyzer did not retain all public result paths")
        for index, raw_path in enumerate(raw_paths):
            record_id = Path(raw_path).parent.name
            _replace(
                analysis,
                ("retention", "raw_result_paths", index),
                f"{record_id}/result.json",
                "local analyzer path",
                analysis_changes,
            )

        reference_analysis = _read_json(reference_analysis_path)
        scientific_match = _scientific_analysis_view(
            analysis
        ) == _scientific_analysis_view(reference_analysis)
        if not scientific_match:
            raise PublicationError(
                "public projection analysis differs from the retained scientific analysis"
            )

        public_analysis_bytes = _write_json(output / "analyzer-result.json", analysis)
        report_bytes = raw_markdown.read_bytes()
        _write_bytes(output / "analyzer-report.md", report_bytes)

    analyzer_record = {
        "exit_code": 0,
        "validation_status": "PASS",
        "decision_verdict": analysis["decision"]["verdict"],
        "decision_win_kind": analysis["decision"]["win_kind"],
        "reference_analysis_sha256": _sha256(reference_analysis_path.read_bytes()),
        "scientific_output_matches_retained_analysis": scientific_match,
        "comparison_excludes_only": [
            "generated_at",
            "retention paths",
            "profile trace paths",
        ],
        "raw_analyzer_output_sha256": _sha256(raw_analysis_bytes),
        "public_analyzer_output_sha256": _sha256(public_analysis_bytes),
        "public_report_sha256": _sha256(report_bytes),
        "public_projection_changes": analysis_changes,
        "public_command": (
            "python analyze.py results --out analyzer-result.json --markdown analyzer-report.md"
        ),
    }
    return analysis, analyzer_record, report_bytes.decode("utf-8")


def _round(value: float) -> float:
    return round(float(value), 9)


def _phase_metrics(
    indexed: Mapping[tuple[str, int, int], Mapping[str, Any]],
    case_name: str,
    seed: int,
) -> dict[str, Any]:
    baseline = indexed[(case_name, 0, seed)]["timing"]["cold_first_10"]["raw_host_ms"]
    candidate = indexed[(case_name, 1, seed)]["timing"]["cold_first_10"]["raw_host_ms"]

    def comparison(baseline_value: float, candidate_value: float) -> dict[str, float]:
        return {
            "baseline_enabled_0_ms": _round(baseline_value),
            "candidate_enabled_1_ms": _round(candidate_value),
            "baseline_over_candidate_ratio": _round(baseline_value / candidate_value),
        }

    baseline_call_1 = float(baseline[0])
    candidate_call_1 = float(candidate[0])
    baseline_calls_2_to_10 = sum(float(value) for value in baseline[1:])
    candidate_calls_2_to_10 = sum(float(value) for value in candidate[1:])
    baseline_first_10 = sum(float(value) for value in baseline)
    candidate_first_10 = sum(float(value) for value in candidate)
    return {
        "seed": seed,
        "call_1": comparison(baseline_call_1, candidate_call_1),
        "calls_2_through_10_total": comparison(
            baseline_calls_2_to_10, candidate_calls_2_to_10
        ),
        "calls_1_through_10_total": comparison(baseline_first_10, candidate_first_10),
    }


def _build_diagnosis(
    indexed: Mapping[tuple[str, int, int], Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    phase_split = {
        case_name: [_phase_metrics(indexed, case_name, seed) for seed in range(4)]
        for case_name in (
            "no_break_depth4",
            "nested_break_depth4",
            "nested_break_depth8",
        )
    }
    primary = phase_split["nested_break_depth8"]
    control = phase_split["no_break_depth4"]
    primary_call_1_ratios = [
        item["call_1"]["baseline_over_candidate_ratio"] for item in primary
    ]
    primary_later_ratios = [
        item["calls_2_through_10_total"]["baseline_over_candidate_ratio"]
        for item in primary
    ]
    primary_total_ratios = [
        item["calls_1_through_10_total"]["baseline_over_candidate_ratio"]
        for item in primary
    ]
    control_total_ratios = [
        item["calls_1_through_10_total"]["baseline_over_candidate_ratio"]
        for item in control
    ]
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "ratio_definition": "enabled=0 host time divided by enabled=1 host time",
        "phase_split_by_case_and_seed": phase_split,
        "primary_startup_diagnosis": {
            "case": "nested_break_depth8",
            "call_1_ratio_range": [
                min(primary_call_1_ratios),
                max(primary_call_1_ratios),
            ],
            "calls_2_through_10_ratio_range": [
                min(primary_later_ratios),
                max(primary_later_ratios),
            ],
            "calls_1_through_10_ratio_range": [
                min(primary_total_ratios),
                max(primary_total_ratios),
            ],
            "fixed_minimum_ratio": 1.1,
            "status": "FAIL",
            "interpretation": (
                "Enabled=1 is faster across calls 2 through 10 in every seed. "
                "Its slower first compiled call dominates the predeclared first-10 "
                "total, so every seed misses the startup threshold."
            ),
        },
        "no_break_startup_control_diagnosis": {
            "case": "no_break_depth4",
            "allowed_inclusive_range": [0.95, 1.05],
            "first_10_ratio_by_seed": {
                str(seed): control_total_ratios[seed] for seed in range(4)
            },
            "failing_seeds": [
                seed
                for seed, ratio in enumerate(control_total_ratios)
                if not 0.95 <= ratio <= 1.05
            ],
            "seed_1_phase_split": control[1],
            "status": "FAIL",
            "interpretation": (
                "Seed 1 misses the lower control bound because its first compiled "
                "call ratio is 0.941899568. Calls 2 through 10 are neutral at "
                "1.001055019."
            ),
        },
        "analyzer_route_outcomes": {
            "steady": "PASS",
            "startup": "FAIL",
            "qualified_steady_win": analysis["decision"]["qualified_steady_win"],
            "qualified_startup_win": analysis["decision"]["qualified_startup_win"],
        },
    }


def _build_qualification(
    analysis: Mapping[str, Any], final_decision: Mapping[str, Any]
) -> dict[str, Any]:
    frozen_decision = final_decision["scientific_result"]["frozen_analyzer_verdict"]
    if frozen_decision != analysis["decision"]:
        raise PublicationError(
            "retained final decision does not contain the analyzer decision"
        )
    return {
        "schema": QUALIFICATION_SCHEMA,
        "protocol_scientific_verdict": analysis["decision"],
        "route_outcomes": {
            "steady": {
                "status": "PASS",
                "measurement_gate_pass": analysis["cases"]["nested_break_depth8"][
                    "steady_win_criteria_pass"
                ],
                "control_gate_pass": analysis["control_validity"]["steady"][
                    "gate_pass"
                ],
                "capture_gate_pass": analysis["capture_qualification"][
                    "cuda_graph_claim_permitted"
                ],
            },
            "startup": {
                "status": "FAIL",
                "measurement_gate_pass": analysis["cases"]["nested_break_depth8"][
                    "startup_win_criteria_pass"
                ],
                "control_gate_pass": analysis["control_validity"]["cold_first_10"][
                    "gate_pass"
                ],
                "capture_gate_pass": analysis["capture_qualification"][
                    "cuda_graph_claim_permitted"
                ],
            },
        },
        "historical_all_gates_content_decision": final_decision[
            "human_content_condition"
        ],
        "current_research_publication_decision": {
            "decision": "AUTHORIZED",
            "content_permitted": True,
            "scope": (
                "the exact disclosed research package under "
                "research/2026-10-22_nested-graph-breaks"
            ),
            "accepted_tradeoff": (
                "the steady route passes while the startup route and its seed-1 "
                "no-break startup control do not pass"
            ),
            "scientific_gate_effect": "none",
            "retroactive_startup_gate_pass": False,
        },
        "decision_boundary": (
            "The fixed protocol scientifically qualifies the steady invocation "
            "route and does not qualify the startup route. At experiment closure, "
            "the historical all-gates condition withheld the planned Whiteboard "
            "content. The current explicit authorization accepts the disclosed "
            "startup tradeoff and permits this exact research package. It does not "
            "change a scientific gate or retroactively pass the startup route."
        ),
        "historical_evidence_boundary": {
            "classification": "retained standalone compiler microbenchmark",
            "hardware": "one NVIDIA B200 executed each worker",
            "profiler": "retained PyTorch Chrome CPU and CUDA profile summaries",
            "not_established": [
                "a new canonical repository-harness qualification",
                "Nsight Compute qualification",
                "Nsight Systems qualification",
                "serving latency or throughput",
                "model quality",
                "multi-GPU scaling",
            ],
            "profile_trace_bytes_published": False,
        },
        "claim_limits": analysis["claim_limits"],
    }


def _build_results_readme() -> str:
    return """# Public result evidence

This directory contains public projections of all 24 retained B200 result
records. The projections preserve timing samples, counters, flags, generated
workload code and hashes, correctness evidence, profile summaries, statuses,
and verdict inputs. Operational identifiers and private paths use explicit
placeholders.

## Decision boundaries

- The fixed analyzer returns `PASS` for a qualified steady invocation win.
- The startup route remains `FAIL`. It fails in all four primary seeds, and
  the no-break startup control also fails at seed 1.
- The historical all-gates content condition withheld the planned Whiteboard
  content at experiment closure.
- The current explicit authorization accepts the disclosed startup tradeoff
  and permits this exact research package. It does not change any gate or turn
  the startup failure into a pass.

## Files

- [`analyzer-result.json`](analyzer-result.json) is the full analyzer output
  over the public projections. Only its local result paths are projected to
  repository-relative paths.
- [`analyzer-report.md`](analyzer-report.md) is the analyzer's rendered report.
- [`diagnosis.json`](diagnosis.json) separates call 1 from calls 2 through 10
  for every case and seed.
- [`qualification.json`](qualification.json) records the scientific verdict,
  historical content decision, current publication authorization, and claim
  limits.
- [`publication-manifest.json`](publication-manifest.json) records the original
  and public SHA-256 values plus every changed field path. It does not disclose
  the original private values.

Each `<case>-seed<seed>-flag<arm>/result.json` file is one projected retained
worker record.

From the archive root, rerun the standard-library analyzer with:

```bash
python analyze.py results --out /tmp/nested-graphs-analysis.json \\
  --markdown /tmp/nested-graphs-analysis.md
```

These are historical standalone compiler-microbenchmark measurements with
PyTorch Chrome profile evidence. They do not establish a new repository-harness,
NCU, Nsys, serving, model-quality, or multi-GPU qualification.
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--reference-analysis", required=True, type=Path)
    parser.add_argument("--final-decision", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_root = args.source_results.resolve()
    output = args.output.resolve()
    source_paths = sorted(source_root.glob("*/result.json"))
    if len(source_paths) != EXPECTED_RECORDS:
        raise PublicationError(
            f"found {len(source_paths)} result records, expected {EXPECTED_RECORDS}"
        )

    output.mkdir(parents=True, exist_ok=True)
    records_manifest: list[dict[str, Any]] = []
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for source_path in source_paths:
        record_id = source_path.parent.name
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        if _json_bytes(source) != source_bytes:
            raise PublicationError(
                f"{record_id}: source JSON formatting is not reproducible"
            )
        public, changes = _public_record(source, record_id)
        changed_paths = sorted(_diff_paths(source, public))
        logged_paths = sorted(change["path"] for change in changes)
        if changed_paths != logged_paths:
            raise PublicationError(
                f"{record_id}: change log does not match the public projection"
            )
        private_values = [
            str(_lookup(source, ("runtime", "hostname"))),
            str(_lookup(source, ("device", "expected_gpu_uuid"))),
        ]
        _assert_public(public, private_values)

        public_path = output / record_id / "result.json"
        public_bytes = _write_json(public_path, public)
        records_manifest.append(
            {
                "record_id": record_id,
                "source_relative_path": f"results/{record_id}/result.json",
                "public_relative_path": f"{record_id}/result.json",
                "original_sha256": _sha256(source_bytes),
                "public_sha256": _sha256(public_bytes),
                "byte_identical": source_bytes == public_bytes,
                "changes": changes,
            }
        )
        key = (str(public["case"]), int(public["enabled"]), int(public["seed"]))
        if key in indexed:
            raise PublicationError(f"duplicate record identity {key}")
        indexed[key] = public

    analysis, analyzer_record, _ = _run_analyzer(
        args.analyzer.resolve(),
        output,
        args.reference_analysis.resolve(),
    )
    final_decision = _read_json(args.final_decision.resolve())
    diagnosis_bytes = _write_json(
        output / "diagnosis.json", _build_diagnosis(indexed, analysis)
    )
    qualification_bytes = _write_json(
        output / "qualification.json",
        _build_qualification(analysis, final_decision),
    )
    readme_bytes = _build_results_readme().encode("utf-8")
    _write_bytes(output / "README.md", readme_bytes)

    supporting_files = {}
    for name in (
        "README.md",
        "analyzer-result.json",
        "analyzer-report.md",
        "diagnosis.json",
        "qualification.json",
    ):
        supporting_files[name] = {"sha256": _sha256((output / name).read_bytes())}
    assert supporting_files["diagnosis.json"]["sha256"] == _sha256(diagnosis_bytes)
    assert supporting_files["qualification.json"]["sha256"] == _sha256(
        qualification_bytes
    )
    assert supporting_files["README.md"]["sha256"] == _sha256(readme_bytes)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "benchmark_id": analysis["benchmark_id"],
        "projection_boundary": {
            "source": "24 private retained result records",
            "public": "24 field-redacted JSON projections",
            "public_files_are_exact_private_bytes": False,
            "hash_scope": (
                "original_sha256 covers each retained private input. public_sha256 "
                "covers the corresponding public projection."
            ),
            "preserved_without_value_changes": [
                "all timing samples and derived timing fields",
                "all counter measurements and mechanism flags",
                "all treatment and compile flags",
                "generated workload source and hashes",
                "correctness values and hashes",
                "profile counts, names, statuses and trace hashes",
                "result statuses and analyzer verdict inputs",
            ],
            "redacted": [
                "GPU UUIDs",
                "hostnames",
                "filesystem paths and embedded usernames",
                "process identifiers",
                "process-local object identifiers and addresses",
            ],
            "not_present_in_source_records": [
                "GPU PCI identifiers",
                "thread identifiers",
                "standalone username fields",
            ],
        },
        "records": records_manifest,
        "analyzer_replay": analyzer_record,
        "supporting_files": supporting_files,
        "excluded_private_evidence": [
            "control logs",
            "credential material",
            "raw telemetry XML",
            "Chrome trace bytes",
            "the full retained final-decision and custody documents",
        ],
    }
    _write_json(output / "publication-manifest.json", manifest)
    print(
        json.dumps(
            {
                "records": len(records_manifest),
                "validation": analysis["validation"]["status"],
                "verdict": analysis["decision"]["verdict"],
                "win_kind": analysis["decision"]["win_kind"],
                "startup": "FAIL",
                "steady": "PASS",
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
