"""Validate retained GPU-time flamegraph evidence and presentation metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def validate_profile_conventions(package: Path) -> list[str]:
    """Return actionable errors for a package's active Zymtrace views."""
    package = Path(package).resolve()
    views = package / "zymtrace/views"
    if not views.is_dir():
        return []
    errors: list[str] = []

    def read(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: cannot read JSON ({exc})")
            return {}

    def retained(spec, label):
        if not isinstance(spec, dict):
            errors.append(f"{label}: requires path and sha256")
            return None
        relative = spec.get("path", spec.get("source_path"))
        if not isinstance(relative, str) or not relative:
            errors.append(f"{label}: missing retained source path")
            return None
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or Path(relative).is_absolute():
            errors.append(f"{label}: source path must stay within the package")
            return None
        try:
            data = path.read_bytes()
        except OSError:
            errors.append(f"{label}: missing source {relative}")
            return None
        digest = hashlib.sha256(data).hexdigest()
        if not data or digest != spec.get("sha256"):
            errors.append(f"{label}: source sha256 mismatch or empty file: {relative}")
        return path

    provenance_path = views / "provenance.json"
    provenance = read(provenance_path) if provenance_path.exists() else {}
    semantics_path = views / "profile-semantics.json"
    width_metadata = " ".join(str(provenance.get(key, "")) for key in ("width_unit", "method")) if isinstance(provenance, dict) else ""
    if any(term in width_metadata.lower() for term in ("record count", "event row", "count weight")):
        errors.append("active flamegraphs use count weights: public performance flamegraphs require GPU time fractions")
    if not semantics_path.exists():
        errors.append("zymtrace/views/profile-semantics.json is required for active profile views")
        return errors
    semantics = read(semantics_path)
    if not isinstance(semantics, dict) or not isinstance(provenance, dict):
        return [*errors, "profile semantics and provenance must be objects"]
    if semantics.get("schema_version") != 1:
        errors.append("profile semantics schema_version must be 1")
    weights = semantics.get("weights", {})
    if not isinstance(weights, dict) or weights.get("quantity") != "gpu_time_share" or weights.get("unit") != "fraction" or weights.get("normalization") != "per_worker_sum":
        errors.append("weights must declare quantity=gpu_time_share, unit=fraction and normalization=per_worker_sum, never count weights or guessed absolute units")
        weights = weights if isinstance(weights, dict) else {}
    weight_field = weights.get("field")
    if weight_field != "c":
        errors.append("weights.field must be the documented source time field c")
        weight_field = "c"
    authority = semantics.get("unit_authority", {})
    if not isinstance(authority, dict):
        authority = {}
    if authority.get("verified") is not True or authority.get("schema_field") != "JsonEvent.c" or authority.get("schema_value") != "Time spent in this event.":
        errors.append("unit authority requires verified=true and JsonEvent.c='Time spent in this event.'")
    if authority.get("absolute_export_unit") != "undocumented":
        errors.append("unit authority must disclose absolute_export_unit=undocumented")
    url = authority.get("source_url", "")
    if not isinstance(url, str) or urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
        errors.append("unit authority requires a source_url")
    authority_path = retained(authority, "unit authority")
    if authority_path is not None:
        retained_authority = read(authority_path)
        expected_authority = {
            "schema_field": "JsonEvent.c",
            "schema_value": "Time spent in this event.",
            "quantity": "gpu_time",
            "normalization": "per_worker_sum",
            "unit": "fraction",
            "absolute_export_unit": "undocumented",
        }
        if not isinstance(retained_authority, dict) or any(retained_authority.get(key) != value for key, value in expected_authority.items()):
            errors.append("retained unit authority must verify the time field, CUDA GPU-time quantity and fraction normalization without an absolute-unit assumption")
    scope = semantics.get("scope", {})
    if not isinstance(scope, dict):
        scope = {}
    phases = scope.get("phases")
    if not isinstance(phases, list) or not phases or any(not isinstance(p, str) or not p.strip() for p in phases):
        errors.append("scope.phases must name the captured workload phases")
    if scope.get("normalized_width") is not True or not isinstance(scope.get("disclosure"), str) or not scope.get("disclosure", "").strip():
        errors.append("scope requires normalized_width=true and an explicit per-worker percentage width disclosure")
    arms = semantics.get("arms", {})
    if not isinstance(arms, dict) or not arms:
        return [*errors, "profile semantics requires per-arm evidence"]
    seen_pids = set()
    for name, arm in arms.items():
        label = f"profile arm {name}"
        if not isinstance(arm, dict):
            errors.append(f"{label}: must be an object")
            continue
        pid = arm.get("expected_pid")
        gpu = arm.get("gpu_uuid")
        if type(pid) is not int or pid <= 0:
            errors.append(f"{label}: expected_pid must be a positive integer")
        elif pid in seen_pids:
            errors.append(f"{label}: expected_pid must be distinct per arm")
        else:
            seen_pids.add(pid)
        if not isinstance(gpu, str) or not gpu:
            errors.append(f"{label}: requires a GPU UUID filter")
        path = retained(arm.get("events"), label)
        if path is None:
            continue
        raw = read(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
            errors.append(f"{label}: source must contain an events list")
            continue
        strings = raw.get("strings", [])
        def value(row, key):
            val = row.get(key)
            if key in ("k", "u") and type(val) is int and isinstance(strings, list) and 0 <= val < len(strings):
                return strings[val]
            return val
        accepted = []
        accepted_rows = []
        invalid = False
        for index, row in enumerate(raw["events"]):
            if not isinstance(row, dict):
                continue
            source_gpu = value(row, "u")
            if row.get("pid") != pid or value(row, "k") != "cuda" or not isinstance(source_gpu, str) or source_gpu.removeprefix("GPU-") != str(gpu).removeprefix("GPU-") or row.get("sid") is None:
                continue
            weight = row.get(weight_field)
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                invalid = True
                continue
            accepted.append(weight)
            accepted_rows.append((index, row))
        if invalid:
            errors.append(f"{label}: scoped GPU weights must be positive and finite")
        try:
            total = math.fsum(accepted)
        except OverflowError:
            total = float("inf")
        count, excluded = len(accepted), len(raw["events"]) - len(accepted)
        if not math.isfinite(total) or total <= 0:
            errors.append(f"{label}: summed GPU weights must be positive and finite")
        fractions = math.fsum(weight / total for weight in accepted) if math.isfinite(total) and total > 0 else 0.0
        actual = {"accepted_records": count, "excluded_records": excluded, "sum_exported_time_weights": total, "sum_gpu_time_fractions": fractions}
        prov = provenance.get("inputs", {}).get(name, {}) if isinstance(provenance.get("inputs"), dict) else {}
        if not isinstance(prov, dict):
            prov = {}
        if prov.get("pid") != pid:
            errors.append(f"{label}: provenance PID does not match expected_pid")
        for key, expected in actual.items():
            for source, metadata in (("semantics", arm), ("provenance", prov)):
                recorded = metadata.get(key)
                matches = (type(recorded) in (int, float) and math.isfinite(recorded) and math.isclose(recorded, expected, rel_tol=1e-12, abs_tol=1e-15)) if key in ("sum_exported_time_weights", "sum_gpu_time_fractions") else type(recorded) is int and recorded == expected
                if not matches:
                    errors.append(f"{label}: {source} {key} does not match filtered source ({expected})")
        outputs = provenance.get("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}

        def bound_output(relative):
            target = (package / relative).resolve()
            if Path(relative).is_absolute() or not target.is_relative_to(package) or not target.is_file():
                errors.append(f"{label}: missing or invalid active image or sidecar {relative}")
                return None
            spec = outputs.get(relative)
            if not isinstance(spec, dict):
                errors.append(f"{label}: active image or sidecar missing from provenance.outputs: {relative}")
                return None
            data = target.read_bytes()
            if not data or hashlib.sha256(data).hexdigest() != spec.get("sha256") or type(spec.get("bytes")) is not int or len(data) != spec.get("bytes"):
                errors.append(f"{label}: active image or sidecar hash/bytes mismatch: {relative}")
            return target

        images = arm.get("images")
        if not isinstance(images, list) or not images:
            errors.append(f"{label}: requires active images")
        else:
            for image in images:
                if not isinstance(image, str):
                    errors.append(f"{label}: image path must be a string")
                    continue
                bound_output(image)
                sidecar_path = bound_output(str(Path(image).with_suffix(".stacks.json")))
                if sidecar_path is None:
                    continue
                sidecar = read(sidecar_path)
                if not isinstance(sidecar, dict):
                    errors.append(f"{label}: sidecar must be an object")
                    continue
                if sidecar.get("pid") != pid or sidecar.get("gpu_uuid") != gpu or sidecar.get("unit") != "fraction of captured GPU time":
                    errors.append(f"{label}: sidecar PID/GPU/unit must match filtered GPU-time fractions")
                records = sidecar.get("records")
                if not isinstance(records, list) or len(records) != len(accepted_rows):
                    errors.append(f"{label}: sidecar records must cover exactly the filtered source rows")
                    continue
                by_index = {}
                for record in records:
                    if not isinstance(record, dict) or type(record.get("raw_event_index")) is not int or record["raw_event_index"] in by_index:
                        errors.append(f"{label}: sidecar raw_event_index must be a unique source index")
                        continue
                    by_index[record["raw_event_index"]] = record
                for source_index, row in accepted_rows:
                    record = by_index.get(source_index, {})
                    expected_fraction = row[weight_field] / total if math.isfinite(total) and total > 0 else None
                    if record.get("exported_time_weight") != row[weight_field] or record.get("gpu_time_fraction") != expected_fraction:
                        errors.append(f"{label}: sidecar gpu_time_fraction or exported_time_weight differs from filtered source at raw_event_index={source_index}")
                    source_frames = row.get("f")
                    if not isinstance(source_frames, list) or not source_frames or record.get("frame_ids") != source_frames:
                        errors.append(f"{label}: sidecar frame_ids differ from full source stack at raw_event_index={source_index}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    errors = validate_profile_conventions(args.package)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
