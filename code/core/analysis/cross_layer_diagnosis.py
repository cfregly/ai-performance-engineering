"""Correlate timed diagnostic signals without treating overlap as a root cause."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from core.diagnostics.evidence import finite_number, write_bundle

SCHEMA = "aisp.diagnostic-signals.v1"


def validate_signal(row):
    required = ("metric", "value", "unit", "start_unix_s", "end_unix_s", "clock_domain", "scope", "role")
    if not isinstance(row, dict) or any(key not in row for key in required):
        raise ValueError(f"Signal requires {', '.join(required)}")
    if row["role"] not in {"symptom", "counter", "measurement"}:
        raise ValueError("Signal role must be symptom, counter or measurement")
    for key in ("metric", "unit", "clock_domain"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{key} must be a nonempty string")
    start = finite_number(row["start_unix_s"], "start_unix_s")
    end = finite_number(row["end_unix_s"], "end_unix_s")
    finite_number(row["value"], "signal value")
    if end <= start:
        raise ValueError("Signal interval must have positive duration")
    if not isinstance(row["scope"], dict) or not row["scope"]:
        raise ValueError("Signal scope must identify a host, path or workload")
    if not any(row["scope"].get(k) for k in ("host", "interface", "path", "workload_id", "endpoint_alias_hash")):
        raise ValueError("Signal scope lacks a host, interface, path, workload_id or endpoint_alias_hash")
    if any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in row["scope"].items()):
        raise ValueError("Scope keys and values must be nonempty strings")
    if row["role"] == "counter" and row.get("semantics") != "delta":
        raise ValueError("Counter correlation requires interval deltas, not cumulative snapshots")
    return dict(row)


def correlate(signals, *, max_clock_skew_s=None):
    rows = [validate_signal(row) for row in signals]
    if max_clock_skew_s is not None:
        finite_number(max_clock_skew_s, "max_clock_skew_s")
    matches, context, excluded = [], [], []
    symptoms = [r for r in rows if r["role"] == "symptom" and r["value"] > 0]
    evidence = [r for r in rows if r["role"] == "measurement" or (r["role"] == "counter" and r["value"] > 0)]
    for symptom in symptoms:
        for counter in evidence:
            a, b = symptom["scope"], counter["scope"]
            common = a.keys() & b.keys()
            identity_keys = {"host", "path", "workload_id", "endpoint_alias_hash"}
            if not (common & identity_keys) or any(a[k] != b[k] for k in common):
                excluded.append({"symptom": symptom["metric"], "counter": counter["metric"], "reason": "resource_scope_mismatch"})
                continue
            same_clock = symptom["clock_domain"] == counter["clock_domain"]
            if not same_clock and max_clock_skew_s is None:
                excluded.append({"symptom": symptom["metric"], "counter": counter["metric"], "reason": "clock_alignment_unverified"})
                continue
            uncertainty = 0 if same_clock else max_clock_skew_s
            overlap = min(symptom["end_unix_s"], counter["end_unix_s"]) - max(symptom["start_unix_s"], counter["start_unix_s"]) - uncertainty
            if overlap <= 0:
                continue
            if counter["role"] == "measurement":
                context.append({"symptom": symptom, "measurement": counter,
                                "overlap_lower_bound_s": overlap, "clock_skew_bound_s": uncertainty,
                                "interpretation": "A measurement from an overlapping interval on a matching scope. No baseline or causal conclusion is implied."})
            else:
                matches.append({"symptom": symptom, "counter": counter,
                                "overlap_lower_bound_s": overlap, "clock_skew_bound_s": uncertainty,
                                "interpretation": "Temporal association on a matching scope. This does not establish cause."})
    findings = [{"summary": f"{m['symptom']['metric']} overlaps a positive {m['counter']['metric']} delta.",
                 "next_measurement": "Repeat the same workload while changing one suspected mechanism and retain both endpoint traces."}
                for m in matches]
    if symptoms and not matches:
        findings.append({"summary": "The supplied evidence does not associate the symptom with a measured fabric counter change.",
                         "next_measurement": "Collect matching workload/path intervals and verify clock alignment. Missing evidence does not clear the fabric."})
    return {"schema": "aisp.cross-layer-diagnosis.v1", "status": "ok" if matches else "partial" if rows else "unavailable",
            "signals": rows, "correlations": matches, "contextual_measurements": context,
            "excluded_pairs": excluded, "findings": findings,
            "valid_for_performance_claim": False,
            "inference_limits": ["A counter sampled over a long interval cannot locate an event within that interval.",
                                 "Unmeasured paths and missing switch access remain unknown.",
                                 "Application latency includes queueing and rank readiness, not only byte transfer."]}


def artifact_signals(payload):
    """Accept normalized exports from collectors without guessing their units."""
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic artifact must be a JSON object")
    if payload.get("schema") == SCHEMA or "signals" in payload:
        signals = payload.get("signals")
        if not isinstance(signals, list):
            raise ValueError("Diagnostic signals must be a list")
        if signals and str(payload.get("status", "")).lower() in {
            "error", "invalid", "failed", "rejected", "unsupported", "unavailable", "skipped"
        }:
            raise ValueError("Failed or unavailable diagnostic artifacts cannot contribute signals")
        return [validate_signal(row) for row in signals]
    raise ValueError("Artifact has no timed signals. Export aisp.diagnostic-signals.v1 with units, scope and clock identity.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Timed signal export or diagnostic artifact containing signals. Repeat for each source.")
    parser.add_argument("--max-clock-skew-ms", type=float,
                        help="Measured bound between different clock domains. Omit when alignment is unverified.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        sources, signals = [], []
        for path in args.input:
            content = path.read_bytes()
            payload = json.loads(content)
            digest = hashlib.sha256(content).hexdigest()
            sources.append({"path": str(path.resolve()), "sha256": digest, "payload": payload})
            for signal in artifact_signals(payload):
                signals.append({**signal, "source_sha256": digest})
        skew = args.max_clock_skew_ms / 1000 if args.max_clock_skew_ms is not None else None
        report = correlate(signals, max_clock_skew_s=skew)
        output = write_bundle(args.run_dir, "cross-layer-diagnose", sources, report)
        print(json.dumps({"status": report["status"], "correlations": len(report["correlations"]), "report": str(output)}))
        return 2 if report["status"] == "unavailable" else 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
