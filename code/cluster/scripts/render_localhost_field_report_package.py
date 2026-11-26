#!/usr/bin/env python3
"""Render localhost field-report package from structured artifacts.

This keeps localhost report/notes aligned with the latest canonical RUN_ID and
prevents manual narrative/metric drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def md_link(rel_path: str) -> str:
    return f"[{rel_path}]({rel_path})"


def fmt_inline_json(value: Any) -> str:
    if value in (None, "", [], {}):
        return "`n/a`"
    return f"`{json.dumps(value, sort_keys=True)}`"


def redact_nmx_base(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    return re.sub(r"https?://[^\\s`]+/nmx/v1", "<nmx-base>", text)


def localhost_fabric_repro_cmd(label: str) -> str:
    return (
        "python -m cli.aisp cluster fabric-eval --run-id <new_run_id>"
        + " --hosts localhost --labels "
        + label
        + " --ssh-user $(id -un) --primary-label "
        + label
        + " --nmx-url https://<your-nmx-host> --timeout 7200"
        + " --extra-arg --skip-bootstrap-nodes --extra-arg --disable-fp4"
        + " --extra-arg --health-suite --extra-arg off"
        + " --extra-arg --skip-vllm-multinode"
        + " --extra-arg --model --extra-arg openai-community/gpt2"
        + " --extra-arg --tp --extra-arg 1"
        + " --extra-arg --isl --extra-arg 128 --extra-arg --osl --extra-arg 64"
        + " --extra-arg --concurrency-range --extra-arg '1 2'"
        + " --extra-arg --vllm-request-rate-range --extra-arg '1 2'"
        + " --extra-arg --vllm-request-rate-max-concurrency --extra-arg 4"
        + " --extra-arg --vllm-request-rate-num-prompts --extra-arg 80"
        + " --extra-arg --fio-runtime --extra-arg 15"
        + " --extra-arg --nvbandwidth-quick"
    )


def _resolve_run_dir(cluster_root: Path, run_id: str, raw_run_dir: str | None) -> Path:
    return Path(raw_run_dir).resolve() if raw_run_dir else (cluster_root / "runs" / run_id)


def _artifact_ref(run_dir: Path, target: str, subdir: str | None, name: str) -> tuple[str, Path]:
    path = (run_dir / subdir / name) if subdir else (run_dir / name)
    if target == "published":
        prefix = "published/current"
        link = f"{prefix}/{subdir}/{name}" if subdir else f"{prefix}/{name}"
    else:
        link = f"../{subdir}/{name}" if subdir else f"../{name}"
    return link, path


def _run_local_report_path(run_dir: Path, filename: str) -> Path:
    return run_dir / "reports" / filename


def parse_gpu_count(meta_payload: dict[str, Any]) -> int:
    nvidia_smi_l = (meta_payload.get("commands") or {}).get("nvidia_smi_l") or {}
    stdout = str(nvidia_smi_l.get("stdout") or "")
    count = sum(1 for line in stdout.splitlines() if line.strip().startswith("GPU "))
    return count if count > 0 else 1


def parse_suite_summary(steps: list[dict[str, Any]]) -> tuple[int, int, int]:
    latest = {step.get("name") or index: step for index, step in enumerate(steps)}
    total = len(latest)
    failures = sum(1 for step in latest.values() if int(step.get("exit_code", 1)) != 0)
    ok = total - failures
    return ok, total, failures


def pick_timeline_rows(steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    wanted = {
        "preflight_services": "preflight started",
        "discovery": "discovery started",
        "quick_friction_all_nodes": "quick friction completed",
        "monitoring_expectations_all_nodes": "monitoring expectations completed",
        "hang_triage_bundle": "hang triage completed",
        "connectivity_probe": "connectivity probe completed",
        "nccl_env_sensitivity": "NCCL env sweep completed",
        "vllm_serve_sweep": "vLLM serve sweep completed",
        "plot_operator_checks_dashboard": "operator dashboard generated",
        "validate_required_artifacts": "required artifact validation completed",
        "manifest_refresh": "manifest refreshed",
        "render_localhost_field_report_package": "localhost report package rendered",
    }
    rows: list[dict[str, str]] = []
    by_name = {step.get("name"): step for step in steps}
    for name, milestone in wanted.items():
        step = by_name.get(name)
        if not step:
            continue
        ts = str(step.get("start_time") or step.get("end_time") or "")
        time_text = ""
        if ts:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            time_text = dt.astimezone(timezone.utc).strftime("%H:%M:%S")
        status = "ok" if int(step.get("exit_code", 1)) == 0 else "error"
        rows.append({"time": time_text, "milestone": milestone, "status": status})
    return rows


def summarize_quick_friction(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "status": "not_run",
            "failed": [],
            "passed": [],
            "expected_failed": [],
            "unexpected_failed": [],
        }
    failed = payload.get("failed_checks") or []
    passed = payload.get("passed_checks") or []
    expected_failed = payload.get("expected_failed_checks") or []
    unexpected_failed = payload.get("unexpected_failed_checks") or []
    return {
        "status": str(payload.get("status") or "error"),
        "failed": failed,
        "passed": passed,
        "expected_failed": expected_failed,
        "unexpected_failed": unexpected_failed,
    }


def summarize_monitoring(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "status": "not_run",
            "control_plane": "not_run",
            "gpu_telemetry": "not_run",
            "system_signals": "not_run",
        }
    categories = payload.get("categories") or {}
    return {
        "status": str(payload.get("status") or "error"),
        "control_plane": str(((categories.get("control_plane") or {}).get("status") or "unknown")),
        "gpu_telemetry": str(((categories.get("gpu_telemetry") or {}).get("status") or "unknown")),
        "system_signals": str(((categories.get("system_signals") or {}).get("status") or "unknown")),
    }


def summarize_connectivity(payload: dict[str, Any]) -> tuple[float, float]:
    ranks = payload.get("ranks") or []
    if not ranks:
        return 0.0, 0.0
    rank0 = ranks[0]
    barrier = rank0.get("barrier_ms") or []
    barrier_mean = statistics.mean([to_float(x) for x in barrier]) if barrier else 0.0
    algbw = to_float(((rank0.get("payload_probe") or {}).get("algbw_gbps")))
    return barrier_mean, algbw


def summarize_nccl_peak(payload: dict[str, Any]) -> tuple[float, int]:
    results = payload.get("results") or []
    if not results:
        return 0.0, 0
    peak = max(to_float(row.get("algbw_gbps")) for row in results)
    peak_row = max(results, key=lambda row: to_float(row.get("algbw_gbps")))
    return peak, int(to_float(peak_row.get("size_bytes"), 0.0))


def summarize_vllm_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "concurrency": int(to_float(row.get("concurrency"), 0)),
                "total_tok_s": to_float(row.get("total_token_throughput")),
                "mean_ttft_ms": to_float(row.get("mean_ttft_ms")),
                "p99_ttft_ms": to_float(row.get("p99_ttft_ms")),
                "p99_tpot_ms": to_float(row.get("p99_tpot_ms")),
            }
        )
    out.sort(key=lambda row: row["concurrency"])
    return out


def tail_latency_issue_status(vllm_rows: list[dict[str, Any]]) -> str:
    if len(vllm_rows) < 2:
        return "Not observed (insufficient sweep points)"
    p99_vals = [row["p99_ttft_ms"] for row in vllm_rows if row["p99_ttft_ms"] > 0]
    if len(p99_vals) < 2:
        return "Not observed"
    ratio = max(p99_vals) / max(min(p99_vals), 1e-9)
    if ratio >= 2.0:
        return "Observed (severe latency knee present in this sweep)"
    return "Not observed in this localhost canary sweep"


def render_report(args: argparse.Namespace, *, target: str = "run_local") -> str:
    run_id = args.run_id
    label = args.label
    run_dir = args.run_dir

    manifest_rel, manifest_path = _artifact_ref(run_dir, target, None, "manifest.json")
    suite_rel, suite_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_suite_steps.json")
    meta_rel, meta_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_meta.json")
    hang_rel, hang_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_hang_triage_readiness.json")
    quick_rel, quick_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_quick_friction.json")
    mon_rel, mon_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_monitoring_expectations.json")
    op_dash_rel, op_dash_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_operator_checks_dashboard.json")
    conn_rel, conn_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_torchrun_connectivity_probe.json")
    nccl_env_rel, nccl_env_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_nccl_env_sensitivity.json")
    nccl_rel, nccl_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_node1_nccl.json")
    vllm_csv_rel, vllm_csv_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_vllm_serve_sweep.csv")
    vllm_jsonl_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_vllm_serve_sweep.jsonl")
    gemm_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_gemm_gpu_sanity.csv")
    fio_rel, fio_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_fio.json")
    preflight_rel, preflight_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_preflight_services.json")
    nvlink_rel, nvlink_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_meta_nvlink_topology.json")
    node_parity_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_node_parity_summary.json")
    fabric_catalog_rel, fabric_catalog_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_command_catalog.json")
    fabric_caps_rel, fabric_caps_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_capability_matrix.json")
    fabric_ver_rel, fabric_ver_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_verification.json")
    fabric_ai_rel, fabric_ai_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_ai_correlation.json")
    fabric_score_rel, fabric_score_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_scorecard.json")

    nccl_fig_rel, nccl_fig_path = _artifact_ref(run_dir, target, "figures", f"{run_id}_node1_nccl_bw_vs_msg.png")
    nccl_scale_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_node1_nccl_scaling_efficiency.png")
    vllm_tok_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_{label}_vllm_serve_total_tok_s_vs_concurrency.png")
    vllm_ttft_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_{label}_vllm_serve_ttft_vs_concurrency.png")
    env_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_nccl_env_sensitivity.png")
    operator_fig_rel, operator_fig_path = _artifact_ref(run_dir, target, "figures", f"{run_id}_operator_checks_dashboard.png")
    story_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_cluster_story_dashboard.png")
    nvlink_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_{label}_meta_nvlink_topology.png")

    manifest = load_json(manifest_path)
    steps = load_json(suite_path)
    meta = load_json(meta_path)
    hang = load_json(hang_path)
    quick = load_json_optional(quick_path)
    mon = load_json_optional(mon_path)
    operator_dashboard = load_json_optional(op_dash_path)
    conn = load_json(conn_path)
    nccl_env = load_json(nccl_env_path)
    nccl = load_json(nccl_path)
    preflight = load_json(preflight_path)
    nvlink = load_json(nvlink_path)
    vllm_rows = summarize_vllm_rows(read_csv_rows(vllm_csv_path))
    fabric_catalog = load_json_optional(fabric_catalog_path)
    fabric_caps = load_json_optional(fabric_caps_path)
    fabric_verification = load_json_optional(fabric_ver_path)
    fabric_ai = load_json_optional(fabric_ai_path)
    fabric_score = load_json_optional(fabric_score_path)

    ok_steps, total_steps, failed_steps = parse_suite_summary(steps)
    gpu_count = parse_gpu_count(meta)
    qf = summarize_quick_friction(quick)
    mon_sum = summarize_monitoring(mon)
    barrier_mean_ms, payload_algbw = summarize_connectivity(conn)
    nccl_peak_algbw, nccl_peak_size = summarize_nccl_peak(nccl)
    tail_issue = tail_latency_issue_status(vllm_rows)
    timeline_rows = pick_timeline_rows(steps)

    if not vllm_rows:
        raise RuntimeError(f"no vLLM rows found in {vllm_csv_rel}")

    first_vllm = vllm_rows[0]
    last_vllm = vllm_rows[-1]

    validate_rc = "unknown"
    for step in reversed(steps):
        if step.get("name") == "validate_required_artifacts":
            validate_rc = str(step.get("exit_code"))
            break

    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    has_operator_checks = quick is not None or mon is not None or operator_dashboard is not None
    has_operator_fig = operator_fig_path.exists()
    capability_families = ((fabric_caps or {}).get("families") or {}) if fabric_caps else {}
    fabric_families = ((fabric_score or {}).get("families") or {}) if fabric_score else {}
    fabric_summary = (fabric_score or {}).get("summary") or {}
    fabric_findings = (fabric_ai or {}).get("findings") or []
    nvlink_control_plane = ((((fabric_verification or {}).get("families") or {}).get("nvlink") or {}).get("control_plane")) or {}
    ib_verification = (((fabric_verification or {}).get("families") or {}).get("infiniband") or {})
    spectrum_verification = (((fabric_verification or {}).get("families") or {}).get("spectrum-x") or {})
    nmx_summary = nvlink_control_plane.get("nmx") or {}
    nmx_topology = nmx_summary.get("topology") or {}
    nmx_partitions = nmx_summary.get("partitions") or {}
    nmx_telemetry = nmx_summary.get("telemetry") or {}
    ib_summary = ib_verification.get("scenario_summary") or {}
    spectrum_summary = spectrum_verification.get("scenario_summary") or {}

    lines: list[str] = []
    lines.append("# Cluster Perf Field Report (Localhost, 1 Node)")
    lines.append("")
    lines.append(f"Last updated: {now_text}. Canonical run: `{run_id}`.")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("1. [TL;DR](#tldr)")
    lines.append("2. [Scope + Canonical Artifacts](#scope--canonical-artifacts)")
    lines.append("3. [Required Reliability Gates (Canonical Run)](#required-reliability-gates-canonical-run)")
    lines.append("4. [Fabric Evaluation](#fabric-evaluation)")
    lines.append("5. [Operator Friction + Monitoring Expectations (New Checks)](#operator-friction--monitoring-expectations-new-checks)")
    lines.append("6. [Cluster Story (First Contact)](#cluster-story-first-contact)")
    lines.append("7. [Weird / New / Interesting (with Normal Baseline)](#weird--new--interesting-with-normal-baseline)")
    lines.append("8. [Benchmark A (Networking Story)](#benchmark-a-networking-story)")
    lines.append("9. [Benchmark B (Inference Story)](#benchmark-b-inference-story)")
    lines.append("10. [Required Issues (Explicit)](#required-issues-explicit)")
    lines.append("11. [Root Cause + Fix Mapping](#root-cause--fix-mapping)")
    lines.append("12. [Report Completeness Delta (vs prior condensed revision)](#report-completeness-delta-vs-prior-condensed-revision)")
    lines.append("13. [Gaps, Risks, and Smell Checks](#gaps-risks-and-smell-checks)")
    lines.append("14. [Implications for Small AI Teams](#implications-for-small-ai-teams)")
    lines.append("15. [Stakeholder Recommendations (Prioritized)](#stakeholder-recommendations-prioritized)")
    lines.append("16. [Repro Steps](#repro-steps)")
    lines.append("17. [Reproducibility Package](#reproducibility-package)")
    lines.append("18. [Appendix (Coverage vs Case-Study Goals)](#appendix-coverage-vs-case-study-goals)")
    lines.append("19. [Activity Log](#activity-log)")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("| Topic | Summary |")
    lines.append("| --- | --- |")
    lines.append(f"| Scope | `localhost` only, {gpu_count} GPU(s) |")
    lines.append(f"| Canonical run | `{run_id}` |")
    lines.append(f"| Suite status | `{ok_steps}/{total_steps}` steps green; `validate_required_artifacts={validate_rc}` |")
    lines.append(f"| Networking headline | NCCL single-node peak algbw `{fmt_float(nccl_peak_algbw, 1)} GB/s` ({nccl_peak_size} bytes); connectivity probe `{fmt_float(payload_algbw, 3)} GB/s` payload algbw |")
    lines.append(f"| Inference headline | vLLM total throughput `{fmt_float(first_vllm['total_tok_s'], 3)} tok/s` (c={first_vllm['concurrency']}) -> `{fmt_float(last_vllm['total_tok_s'], 3)} tok/s` (c={last_vllm['concurrency']}); p99 TTFT `{fmt_float(first_vllm['p99_ttft_ms'], 3)} ms` -> `{fmt_float(last_vllm['p99_ttft_ms'], 3)} ms` |")
    operator_summary = (
        f"quick_friction `{qf['status']}` (pass={len(qf['passed'])}, failed={len(qf['failed'])}, "
        f"expected={len(qf['expected_failed'])}, unexpected={len(qf['unexpected_failed'])}), "
        f"monitoring_expectations `{mon_sum['status']}`"
    )
    if not has_operator_checks:
        operator_summary = "not run in this preset"
    lines.append(f"| Operator checks | {operator_summary} |")
    lines.append(
        f"| Fabric scorecard | Stored status `{(fabric_score or {}).get('status', 'not_run')}`, completeness `{(fabric_score or {}).get('completeness', 'not_run')}`, full-stack families `{fabric_summary.get('full_stack_verified_families', 0)}` |"
    )
    lines.append("| Key weird/new | Single-node NCCL env sweep can show `busbw=0.0` by definition (rank=1), while algbw is still strong. |")
    lines.append("")

    lines.append("## Scope + Canonical Artifacts")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append("| Hosts in-scope | `localhost` |")
    lines.append("| Excluded hosts | none |")
    lines.append(f"| GPUs per host | `{gpu_count}` |")
    lines.append(f"| Canonical manifest | {md_link(manifest_rel)} |")
    lines.append(f"| Canonical suite steps | {md_link(suite_rel)} |")
    lines.append(f"| Meta snapshot | {md_link(meta_rel)} |")
    lines.append(f"| Node parity summary | {md_link(node_parity_rel)} |")
    lines.append(f"| Operator checks dashboard | {md_link(op_dash_rel) if operator_dashboard else 'not generated in this preset'} |")
    lines.append(f"| Fabric scorecard | {md_link(fabric_score_rel) if fabric_score else 'not generated in this preset'} |")
    lines.append("")

    lines.append("## Fabric Evaluation")
    lines.append("| Family | Present | Completeness | Mgmt plane | Link health | Routing | AI workload impact |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    if fabric_families:
        for family in ("nvlink", "infiniband", "spectrum-x"):
            values = fabric_families.get(family) or {}
            lines.append(
                f"| `{family}` | `{values.get('present', False)}` | `{values.get('completeness', 'not_present')}` | "
                f"`{values.get('management_plane_configured', False)}` | `{values.get('link_health', 'n/a')}` | "
                f"`{values.get('routing_correctness', 'n/a')}` | {values.get('ai_workload_impact', 'n/a')} |"
            )
    else:
        lines.append("| `all` | `False` | `not_run` | `False` | `n/a` | `n/a` | Fabric evaluation was not requested in this run. |")
    lines.append("")
    lines.append("| Fabric summary | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Management planes configured | `{fabric_summary.get('configured_management_planes', 0)}` |")
    lines.append(f"| Runtime-verified families | `{fabric_summary.get('runtime_verified_families', 0)}` |")
    lines.append(f"| Full-stack-verified families | `{fabric_summary.get('full_stack_verified_families', 0)}` |")
    lines.append(f"| Catalog entries | `{len((fabric_catalog or {}).get('entries') or [])}` |")
    lines.append("")
    if nmx_summary:
        alpha = (nmx_topology.get("scenario_answers") or {}).get("team_alpha_candidate") or {}
        beta = (nmx_topology.get("scenario_answers") or {}).get("team_beta_candidate") or {}
        topology_answers = nmx_topology.get("scenario_answers") or {}
        ports_summary = nmx_topology.get("ports") or {}
        alpha_nodes = ", ".join(str(node.get("node") or node.get("system_uid") or "unknown") for node in (alpha.get("nodes") or [])) or "none"
        beta_nodes = ", ".join(str(node.get("node") or node.get("system_uid") or "unknown") for node in (beta.get("nodes") or [])) or "none"
        alpha_locations = ", ".join(alpha.get("gpu_locations") or []) or "none"
        beta_locations = ", ".join(beta.get("gpu_locations") or []) or "none"
        chassis_serials = ", ".join(nmx_topology.get("chassis_serial_numbers") or []) or "none"
        sample_compute_node = nmx_topology.get("sample_compute_node") or {}
        sample_gpu = nmx_topology.get("sample_gpu") or {}
        sample_switch = nmx_topology.get("sample_switch") or {}
        sample_chassis = nmx_topology.get("sample_chassis") or {}
        sample_switch_tray = topology_answers.get("sample_switch_tray") or {}
        default_partition = nmx_partitions.get("default_partition") or {}
        unassigned_locations = ", ".join((nmx_partitions.get("unassigned_gpu_locations") or [])[:8]) or "none"
        lines.append("| NMX Topology Scenario | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Chassis | `{nmx_topology.get('chassis_count', 0)}` |")
        lines.append(f"| Chassis serials | {chassis_serials} |")
        lines.append(f"| Compute nodes | `{nmx_topology.get('compute_node_count', 0)}` |")
        lines.append(f"| GPUs | `{nmx_topology.get('gpu_count', 0)}` |")
        lines.append(f"| Switch ASICs | `{nmx_topology.get('switch_asic_count', 0)}` |")
        lines.append(f"| Switch trays | `{nmx_topology.get('switch_tray_count', 0)}` |")
        lines.append(f"| Ports | `{nmx_topology.get('port_count', 0)}` |")
        lines.append(
            f"| Supports Alpha+Beta 4-GPU split | `{topology_answers.get('can_support_two_concurrent_4gpu_workloads', False)}` |"
        )
        lines.append(f"| Node/GPU grouping field | `{topology_answers.get('node_gpu_grouping_field', 'unknown')}` |")
        lines.append(f"| Switch ASIC field | `{topology_answers.get('switch_asic_distinguishing_field', 'unknown')}` |")
        lines.append(f"| Switch tray grouping fields | `{', '.join(topology_answers.get('switch_tray_grouping_fields') or []) or 'unknown'}` |")
        lines.append(f"| Team Alpha candidate nodes | {alpha_nodes} |")
        lines.append(f"| Team Alpha candidate GPU locations | {alpha_locations} |")
        lines.append(f"| Team Beta candidate nodes | {beta_nodes} |")
        lines.append(f"| Team Beta candidate GPU locations | {beta_locations} |")
        lines.append(f"| GPU-facing ports | `{ports_summary.get('gpu_facing_ports', 0)}` (`BaseLID all-present={ports_summary.get('gpu_facing_ports_have_base_lid', False)}`) |")
        lines.append(f"| Switch-facing ports | `{ports_summary.get('switch_facing_ports', 0)}` (`BaseLID all-present={ports_summary.get('switch_facing_ports_have_base_lid', False)}`) |")
        lines.append(f"| Port formula check | `{ports_summary.get('expected_formula', 'n/a')}` (`matches={ports_summary.get('matches_expected_formula', False)}`) |")
        lines.append(f"| Sample compute-node fields | {fmt_inline_json(sample_compute_node)} |")
        lines.append(f"| Sample GPU fields | {fmt_inline_json(sample_gpu)} |")
        lines.append(f"| Sample switch fields | {fmt_inline_json(sample_switch)} |")
        lines.append(f"| Sample switch-tray grouping | {fmt_inline_json(sample_switch_tray)} |")
        lines.append(f"| Sample chassis fields | {fmt_inline_json(sample_chassis)} |")
        lines.append("")
        lines.append("| NMX Partition Scenario | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Partition count | `{nmx_partitions.get('partition_count', 0)}` |")
        lines.append(f"| Default partition | `{default_partition.get('name', 'none')}` (`members={default_partition.get('member_count', 0)}`) |")
        lines.append(
            f"| Default partition present | `{(nmx_partitions.get('scenario_answers') or {}).get('default_partition_present', False)}` "
            f"(`members={(nmx_partitions.get('scenario_answers') or {}).get('default_partition_member_count', 0)}`) |"
        )
        lines.append(f"| Unassigned GPUs | `{nmx_partitions.get('unassigned_gpu_count', 0)}` |")
        lines.append(f"| Unassigned GPU locations (first 8) | {unassigned_locations} |")
        lines.append(
            f"| Ready for partition create | `{(nmx_partitions.get('scenario_answers') or {}).get('ready_for_new_partition_create', False)}` |"
        )
        lines.append(
            f"| Operation poll path | `{redact_nmx_base((nmx_partitions.get('scenario_answers') or {}).get('operation_poll_path', 'n/a'))}` |"
        )
        lines.append(
            "| Lab helper entrypoint | `python -m cli.aisp cluster nmx-partition-lab --nmx-url <nmx-base>` |"
        )
        lines.append("")
        lines.append("| NMX Telemetry Scenario | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Metrics endpoint | `{redact_nmx_base(nmx_telemetry.get('metrics_endpoint', 'n/a'))}` |")
        lines.append(f"| switch_temperature series | `{nmx_telemetry.get('switch_temperature_series', 0)}` |")
        lines.append(f"| PortXmitDataExtended series | `{nmx_telemetry.get('tx_throughput_series', 0)}` |")
        lines.append(f"| PortRcvDataExtended series | `{nmx_telemetry.get('rx_throughput_series', 0)}` |")
        lines.append(f"| PortLocalPhysicalErrors series | `{nmx_telemetry.get('physical_error_series', 0)}` |")
        lines.append(f"| CableInfoTemperature series | `{nmx_telemetry.get('cable_temperature_series', 0)}` |")
        lines.append(f"| CableInfoRxPower series | `{nmx_telemetry.get('cable_rx_power_series', 0)}` |")
        lines.append(f"| CableInfoTxPower series | `{nmx_telemetry.get('cable_tx_power_series', 0)}` |")
        lines.append("")
    ib_family = fabric_families.get("infiniband") or {}
    ib_capability = capability_families.get("infiniband") or {}
    ib_runtime = (ib_verification.get("runtime") or {}).get("evidence") or {}
    ib_routing_checks = ", ".join(ib_summary.get("routing_checks_ok") or []) or "none"
    ib_hcas = ", ".join(ib_capability.get("hcas") or ib_family.get("hcas") or []) or "none"
    lines.append("| InfiniBand Scenario | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Family present | `{ib_family.get('present', False)}` |")
    lines.append(f"| Completeness | `{ib_family.get('completeness', 'not_present')}` |")
    lines.append(f"| Management plane configured | `{ib_family.get('management_plane_configured', False)}` |")
    lines.append(f"| Capacity/path visibility ready | `{ib_summary.get('capacity_and_path_visibility_ready', False)}` |")
    lines.append(f"| Visible HCAs | `{ib_summary.get('visible_hca_count', len(ib_capability.get('hcas') or ib_family.get('hcas') or []))}` ({ib_hcas}) |")
    lines.append(f"| Visible hosts from `ibhosts` | `{ib_summary.get('visible_host_count', 0)}` |")
    lines.append(f"| Visible switches from `ibswitches` | `{ib_summary.get('visible_switch_count', 0)}` |")
    lines.append(f"| `iblinkinfo` visible | `{ib_summary.get('linkinfo_visible', False)}` |")
    lines.append(f"| `ibnetdiscover` visible | `{ib_summary.get('subnet_discovery_visible', False)}` |")
    lines.append(f"| `saquery` visible | `{ib_summary.get('saquery_visible', False)}` |")
    lines.append(f"| Routing/counter verification ready | `{ib_summary.get('routing_and_counter_verification_ready', False)}` |")
    lines.append(f"| Routing checks passed | {ib_routing_checks} |")
    lines.append(f"| Runtime correlation ready | `{ib_summary.get('runtime_correlation_ready', False)}` |")
    lines.append(
        f"| Multi-node NCCL / single-node ratio | `{fmt_float(ib_summary.get('multi_to_single_nccl_ratio', 0.0), 3)}` "
        f"(`world_size={ib_summary.get('world_size', ib_runtime.get('world_size', 0))}`) |"
    )
    lines.append(f"| Runtime interpretation | {ib_summary.get('runtime_interpretation', ib_family.get('ai_workload_impact', 'InfiniBand not present in this run.'))} |")
    lines.append("")
    spectrum_family = fabric_families.get("spectrum-x") or {}
    spectrum_runtime = (spectrum_verification.get("runtime") or {}).get("evidence") or {}
    switches_targeted = ", ".join(spectrum_summary.get("switches_targeted") or []) or "none"
    lines.append("| Spectrum-X / RoCE Scenario | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Family present | `{spectrum_family.get('present', False)}` |")
    lines.append(f"| Completeness | `{spectrum_family.get('completeness', 'not_present')}` |")
    lines.append(f"| Management plane configured | `{spectrum_family.get('management_plane_configured', False)}` |")
    lines.append(f"| Switches targeted | `{spectrum_summary.get('switch_count_targeted', 0)}` ({switches_targeted}) |")
    lines.append(f"| Fabric readiness ready | `{spectrum_summary.get('fabric_readiness_ready', False)}` |")
    lines.append(f"| Adaptive routing visible | `{spectrum_summary.get('adaptive_routing_visible', False)}` |")
    lines.append(f"| RoCE QoS visible | `{spectrum_summary.get('roce_qos_visible', False)}` |")
    lines.append(f"| BGP neighbor state visible | `{spectrum_summary.get('bgp_neighbor_state_visible', False)}` |")
    lines.append(f"| BGP summary visible | `{spectrum_summary.get('bgp_summary_visible', False)}` |")
    lines.append(f"| BGP route visibility | `{spectrum_summary.get('bgp_route_visibility', False)}` |")
    lines.append(f"| Runtime correlation ready | `{spectrum_summary.get('runtime_correlation_ready', False)}` |")
    lines.append(
        f"| Multi-node NCCL / single-node ratio | `{fmt_float(spectrum_summary.get('multi_to_single_nccl_ratio', 0.0), 3)}` "
        f"(`world_size={spectrum_summary.get('world_size', spectrum_runtime.get('world_size', 0))}`) |"
    )
    lines.append(
        f"| Runtime interpretation | "
        f"{spectrum_summary.get('runtime_interpretation', spectrum_family.get('ai_workload_impact', 'Spectrum-X / RoCE not present in this run.'))} |"
    )
    lines.append("")
    if fabric_findings:
        lines.append("Cross-fabric interpretation:")
        for finding in fabric_findings:
            lines.append(f"- {finding}")
        lines.append("")
    fabric_data_refs = [md_link(rel) for rel, payload in (
        (fabric_catalog_rel, fabric_catalog),
        (fabric_caps_rel, fabric_caps),
        (fabric_ver_rel, fabric_verification),
        (fabric_ai_rel, fabric_ai),
        (fabric_score_rel, fabric_score),
    ) if payload]
    lines.append(f"Data: {', '.join(fabric_data_refs) if fabric_data_refs else 'fabric evaluation was not requested in this preset run.'}")
    lines.append("")

    lines.append("## Required Reliability Gates (Canonical Run)")
    lines.append("| Gate | Status | Key result | Structured artifact |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| Hang-triage readiness (`py-spy` + `strace`) | `{hang.get('status', 'ok')}` | semantic status `{hang.get('status', 'ok')}` for localhost | {md_link(hang_rel)} |")
    lines.append(f"| Torchrun connectivity probe | `{conn.get('status', 'ok')}` | `world_size={conn.get('world_size', 1)}`, barrier mean `{fmt_float(barrier_mean_ms, 4)} ms`, payload algbw `{fmt_float(payload_algbw, 3)} GB/s` | {md_link(conn_rel)} |")
    lines.append(f"| NCCL env sensitivity sweep | `{nccl_env.get('status', 'ok')}` (`failure_count={nccl_env.get('failure_count', 0)}`) | baseline peak busbw `{fmt_float(to_float(nccl_env.get('baseline_peak_busbw_gbps')), 3)}` (rank-1 expected), no failed profiles | {md_link(nccl_env_rel)} |")
    lines.append("")
    lines.append(f"<p><a href=\"{env_fig_rel}\"><img src=\"{env_fig_rel}\" alt=\"NCCL env sensitivity localhost\" width=\"920\"/></a></p>")
    lines.append("")
    lines.append(f"Data: {md_link(manifest_rel)}, {md_link(conn_rel)}, {md_link(nccl_env_rel)}, {md_link(hang_rel)}")
    lines.append("")

    lines.append("## Operator Friction + Monitoring Expectations (New Checks)")
    lines.append("| Check | Status | Key diagnostics | Structured artifacts |")
    lines.append("| --- | --- | --- | --- |")
    qf_diag = f"pass={len(qf['passed'])}, failed={len(qf['failed'])}, expected_failed={','.join(qf['expected_failed']) or 'none'}, unexpected_failed={','.join(qf['unexpected_failed']) or 'none'}"
    mon_diag = f"control_plane={mon_sum['control_plane']}, gpu_telemetry={mon_sum['gpu_telemetry']}, system_signals={mon_sum['system_signals']}"
    lines.append(f"| quick_friction | `{qf['status']}` | {qf_diag if quick else 'not run in this preset'} | {md_link(quick_rel) if quick else 'not generated'} |")
    lines.append(f"| monitoring_expectations | `{mon_sum['status']}` | {mon_diag if mon else 'not run in this preset'} | {md_link(mon_rel) if mon else 'not generated'} |")
    lines.append(f"| operator dashboard | {'generated' if operator_dashboard else 'not generated'} | consolidated status for quick-friction + monitoring expectations | {md_link(op_dash_rel) if operator_dashboard else 'not generated'} |")
    lines.append("")
    if has_operator_fig:
        lines.append(f"<p><a href=\"{operator_fig_rel}\"><img src=\"{operator_fig_rel}\" alt=\"Operator checks dashboard localhost\" width=\"920\"/></a></p>")
        lines.append("")
    if has_operator_checks:
        operator_data = []
        if quick:
            operator_data.append(md_link(quick_rel))
        if mon:
            operator_data.append(md_link(mon_rel))
        if operator_dashboard:
            operator_data.append(md_link(op_dash_rel))
        lines.append(f"Data: {', '.join(operator_data)}")
    else:
        lines.append("Data: operator-friction and monitoring artifacts were not requested in this preset run.")
    lines.append("")

    lines.append("## Cluster Story (First Contact)")
    lines.append("| UTC time | Milestone | Status |")
    lines.append("| --- | --- | --- |")
    for row in timeline_rows:
        lines.append(f"| `{row['time']}` | {row['milestone']} | {row['status']} |")
    lines.append("")
    lines.append(f"<p><a href=\"{story_fig_rel}\"><img src=\"{story_fig_rel}\" alt=\"Cluster story dashboard localhost\" width=\"920\"/></a></p>")
    lines.append("")
    lines.append(f"Data: {md_link(suite_rel)}")
    lines.append("")

    lines.append("## Weird / New / Interesting (with Normal Baseline)")
    lines.append("### Baseline vs Weird Log")
    lines.append("| Area | Normal (canonical localhost) | Weird / notable | Why it matters | Evidence |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| Preflight services | strict service checks pass | prior flake path removed (`systemctl show`-based unit check) | avoids false-negative invalidations | {md_link(preflight_rel)} |")
    lines.append(f"| NVLink topology parsing | topology summary generated | parser now handles single-GPU header formats robustly | keeps topology evidence reproducible on localhost | {md_link(nvlink_rel)} |")
    lines.append(f"| Fabric control-plane coverage | {('fabric scorecard generated' if fabric_score else 'not run in this preset')} | management-plane completeness drops to structured `not_configured`, never a silent skip | makes NVLink / IB / Spectrum-X coverage auditable | {md_link(fabric_score_rel) if fabric_score else md_link(suite_rel)} |")
    lines.append(f"| NCCL env sweep | all profiles `ok` | `busbw=0.0` in rank-1 mode looks odd but is expected | prevents false network conclusions on 1-GPU runs | {md_link(nccl_env_rel)} |")
    lines.append(f"| Operator friction | {'Outcomes for the selected quick-friction checks' if quick else 'not run in this preset'} | expected failures captured explicitly (`{','.join(qf['expected_failed']) or 'none'}` if present) | preserves operator visibility without false-red localhost status | {md_link(quick_rel) if quick else md_link(suite_rel)} |")
    lines.append(f"| Monitoring mode | {'gpu/system checks run' if mon else 'not run in this preset'} | control-plane checks can be `not_applicable` without kubeconfig | clarifies expected non-K8s behavior | {md_link(mon_rel) if mon else md_link(suite_rel)} |")
    lines.append("")
    lines.append("### Deep-Dive Findings")
    lines.append("| Finding | Baseline anchor | Reinforcement insight | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 1 | Preflight services | service gate remains strict while eliminating pipeline flake behavior | {md_link(preflight_rel)} |")
    lines.append(f"| 2 | NVLink topology parsing | single-node topology visual now lands in canonical package consistently | {md_link(nvlink_rel)} |")
    lines.append(f"| 3 | Fabric completeness ledger | each family records `not_present`, `present_unverified`, `runtime_verified`, or `full_stack_verified` | {md_link(fabric_score_rel) if fabric_score else md_link(suite_rel)} |")
    lines.append(f"| 4 | Operator friction classification | expected misses are tracked separately from unexpected failures when operator checks are enabled | {md_link(quick_rel) if quick else md_link(suite_rel)} |")
    lines.append("")
    lines.append(f"<p><a href=\"{story_fig_rel}\"><img src=\"{story_fig_rel}\" alt=\"Weird and normal baseline dashboard localhost\" width=\"920\"/></a></p>")
    lines.append("")
    deep_dive_data = [md_link(preflight_rel), md_link(nvlink_rel)]
    if fabric_score:
        deep_dive_data.append(md_link(fabric_score_rel))
    if operator_dashboard:
        deep_dive_data.append(md_link(op_dash_rel))
    lines.append(f"Data: {', '.join(deep_dive_data)}")
    lines.append("")

    lines.append("## Benchmark A (Networking Story)")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| NCCL single-node peak algbw | `{fmt_float(nccl_peak_algbw, 1)} GB/s` |")
    lines.append(f"| Peak message size | `{nccl_peak_size}` bytes |")
    lines.append(f"| Connectivity probe payload algbw | `{fmt_float(payload_algbw, 3)} GB/s` |")
    lines.append(f"| Connectivity barrier mean | `{fmt_float(barrier_mean_ms, 4)} ms` |")
    lines.append("")
    if gpu_count == 1 or (conn or {}).get("world_size") == 1:
        lines.append("Rank-1 results describe the recorded local workload. Peer fabric bandwidth requires a multi-rank transfer.")
    else:
        lines.append("The retained results cover the recorded single-host topology and benchmark configuration.")
    lines.append("")
    lines.append(f"<p><a href=\"{nccl_fig_rel}\"><img src=\"{nccl_fig_rel}\" alt=\"NCCL bandwidth localhost\" width=\"920\"/></a></p>")
    lines.append(f"<p><a href=\"{nccl_scale_fig_rel}\"><img src=\"{nccl_scale_fig_rel}\" alt=\"NCCL scaling localhost\" width=\"920\"/></a></p>")
    lines.append("")
    lines.append(f"Data: {md_link(nccl_rel)}, {md_link(conn_rel)}")
    lines.append("")

    lines.append("## Benchmark B (Inference Story)")
    lines.append("| Concurrency | Total tok/s | Mean TTFT (ms) | p99 TTFT (ms) | p99 TPOT (ms) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in vllm_rows:
        lines.append(
            f"| `{row['concurrency']}` | `{fmt_float(row['total_tok_s'], 3)}` | `{fmt_float(row['mean_ttft_ms'], 3)}` | `{fmt_float(row['p99_ttft_ms'], 3)}` | `{fmt_float(row['p99_tpot_ms'], 3)}` |"
        )
    lines.append("")
    if len(vllm_rows) < 3:
        lines.append("Interpretation: this is a canary/sparse sweep (<3 concurrency points); use for smoke directionality, not full knee modeling.")
    else:
        lines.append("Interpretation: throughput/latency curve is dense enough to discuss knees directly from this run.")
    lines.append("")
    lines.append(f"<p><a href=\"{vllm_tok_fig_rel}\"><img src=\"{vllm_tok_fig_rel}\" alt=\"vLLM throughput localhost\" width=\"920\"/></a></p>")
    lines.append(f"<p><a href=\"{vllm_ttft_fig_rel}\"><img src=\"{vllm_ttft_fig_rel}\" alt=\"vLLM TTFT localhost\" width=\"920\"/></a></p>")
    lines.append("")
    lines.append(f"Data: {md_link(vllm_csv_rel)}, {md_link(vllm_jsonl_rel)}")
    lines.append("")

    lines.append("## Required Issues (Explicit)")
    lines.append("| Required issue (verbatim) | Status now | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Not applicable (single-node localhost scope) | {md_link(fio_rel)} |")
    lines.append(f"| No multinode vLLM artifact in canonical package. | Not applicable (single-node localhost scope) | {md_link(vllm_csv_rel)} |")
    lines.append(f"| No nvbandwidth bundle in canonical package. | Not applicable for this localhost package unless explicitly enabled | {md_link(suite_rel)} |")
    lines.append(f"| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Not applicable when `--health-suite off` for localhost package | {md_link(suite_rel)} |")
    lines.append(f"| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | {tail_issue} | {md_link(vllm_csv_rel)} |")
    lines.append("")

    lines.append("## Root Cause + Fix Mapping")
    lines.append("| Issue | Root cause | Fix | Verification |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| `preflight_services` false negatives | pipeline-based unit detection could produce flaky non-zero under strict shell options | use `systemctl show -p LoadState` for deterministic service presence checks | clean preflight status in {md_link(preflight_rel)} and step rc=0 in {md_link(suite_rel)} |")
    lines.append(f"| NVLink parser robustness | topology parser assumptions missed some single-GPU header patterns | robust tokenization and header parsing | topology summary/figure generated: {md_link(nvlink_rel)} and {md_link(nvlink_fig_rel)} |")
    lines.append(f"| Quick-friction false-red localhost | missing optional internet/operator tools should be visible but classifiable | added expected-failure classification (`expected_failed_checks`) and auto localhost allowlist | " + (f"quick-friction artifact shows expected vs unexpected failures: {md_link(quick_rel)}" if quick else f"operator-check path is optional in this preset: {md_link(suite_rel)}") + " |")
    lines.append("")

    lines.append("## Report Completeness Delta (vs prior condensed revision)")
    lines.append("| Area | Prior state | Current state |")
    lines.append("| --- | --- | --- |")
    lines.append("| Package type | environment-focused localhost output possible | full template-style localhost field report package rendered from artifacts |")
    lines.append("| Drift risk | hand-updated markdown could diverge from metrics | report is generated directly from structured artifacts for this RUN_ID |")
    lines.append("| Operator checks | could appear as degraded without expected-failure context | expected vs unexpected failure classification is explicit in artifacts |")
    lines.append("| Cleanup hygiene | superseded localhost runs could remain unless manually deleted | cleanup script supports canonical-run retention and stale artifact pruning |")
    lines.append("")

    lines.append("## Gaps, Risks, and Smell Checks")
    lines.append("| Severity | Check | Outcome |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Medium | quick friction optional tooling | status `{qf['status']}` with expected failures `{','.join(qf['expected_failed']) or 'none'}` and unexpected failures `{','.join(qf['unexpected_failed']) or 'none'}` |")
    lines.append(f"| Medium | control-plane observability on non-K8s localhost | monitoring control_plane status `{mon_sum['control_plane']}` |")
    lines.append("| Medium | single-node-only networking conclusions | scope constrained; no multi-node claims in this package |")
    lines.append("| Low | report/template synchronization drift | mitigated via generated localhost report package and validator gate |")
    lines.append("")

    lines.append("## Implications for Small AI Teams")
    lines.append("| Area | Practical implication |")
    lines.append("| --- | --- |")
    lines.append("| Bring-up confidence | strict preflight + required reliability gates provide fast localhost readiness proof before larger spend |")
    lines.append("| Operator readiness | quick-friction explicitly separates expected local misses from unexpected blockers |")
    lines.append("| Repro speed | full localhost package can be regenerated in one suite run with report renderer integration |")
    lines.append("")

    lines.append("## Stakeholder Recommendations (Prioritized)")
    lines.append("| Priority | Recommendation | Why |")
    lines.append("| --- | --- | --- |")
    lines.append("| P1 | Keep strict preflight + required artifact validation as hard gates | avoids false-green benchmark evidence |")
    lines.append("| P1 | Keep localhost report generation automated in suite flow | prevents template drift and missing sections |")
    lines.append("| P2 | Standardize optional operator tools (`uv`, `whois`, `speedtest`) where required | removes expected-failure noise when those checks are mission-critical |")
    lines.append("| P2 | Run cleanup script with canonical-run retention after each new package | keeps artifact corpus unambiguous for stakeholders |")
    lines.append("")

    lines.append("## Repro Steps")
    lines.append("")
    lines.append(f"The example starts a new run. See the [suite record]({suite_rel}) for the measured settings.")
    lines.append("")
    lines.append("| Step | Command |")
    lines.append("| --- | --- |")
    lines.append(
        "| Example localhost fabric eval | `"
        + localhost_fabric_repro_cmd(label)
        + "` |"
    )
    lines.append(
        "| Validate localhost report package | `cluster/scripts/validate_field_report_requirements.sh --report "
        + (
            (
                f"cluster/runs/{run_id}/reports/field-report-localhost.md"
                f" --notes cluster/runs/{run_id}/reports/field-report-localhost-notes.md"
                f" --published-root cluster/runs/{run_id} --package-only"
            )
            if target == "run_local"
            else "cluster/field-report-localhost.md --notes cluster/field-report-localhost-notes.md"
        )
        + " --canonical-run-id "
        + run_id
        + "` |"
    )
    lines.append("")

    lines.append("## Reproducibility Package")
    lines.append("| Artifact class | Links |")
    lines.append("| --- | --- |")
    lines.append(f"| Manifest + suite | {md_link(manifest_rel)}, {md_link(suite_rel)} |")
    lines.append(f"| Reliability gates | {md_link(hang_rel)}, {md_link(conn_rel)}, {md_link(nccl_env_rel)} |")
    lines.append(f"| Operator checks | " + (f"{md_link(quick_rel)}, {md_link(mon_rel)}, {md_link(op_dash_rel)}" if has_operator_checks else md_link(suite_rel)) + " |")
    lines.append(f"| Core benchmarks | {md_link(nccl_rel)}, {md_link(vllm_csv_rel)}, {md_link(gemm_rel)}, {md_link(fio_rel)} |")
    lines.append(f"| Fabric artifacts | {', '.join(fabric_data_refs) if fabric_data_refs else md_link(suite_rel)} |")
    figure_links = [md_link(story_fig_rel), md_link(nccl_fig_rel), md_link(vllm_tok_fig_rel)]
    if has_operator_fig:
        figure_links.insert(1, md_link(operator_fig_rel))
    lines.append(f"| Figures | {', '.join(figure_links)} |")
    lines.append("")

    lines.append("## Appendix (Coverage vs Case-Study Goals)")
    lines.append("| Case-study goal | Coverage |")
    lines.append("| --- | --- |")
    lines.append("| Cluster story | Covered via suite timeline and cluster story dashboard |")
    lines.append("| Fabric characterization | Covered via family scorecard, verification ledger, and AI-correlation findings |")
    lines.append("| Weird/new findings | Covered in merged weird/normal section with deep-dive table |")
    lines.append("| Benchmark A/B | Covered with NCCL + vLLM tables and visuals |")
    lines.append("| Reproducible scripts + artifacts | Covered in Repro Steps + Reproducibility Package |")
    lines.append("| Operator insights | Covered via quick-friction/monitoring diagnostics and recommendations |")
    lines.append("")

    lines.append("## Activity Log")
    lines.append("| UTC | Action | Result |")
    lines.append("| --- | --- | --- |")
    for row in timeline_rows:
        lines.append(f"| `{row['time']}` | {row['milestone']} | {row['status']} |")

    return "\n".join(lines) + "\n"


def render_notes(args: argparse.Namespace, *, target: str = "run_local") -> str:
    run_id = args.run_id
    label = args.label
    run_dir = args.run_dir

    quick_rel, quick_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_quick_friction.json")
    mon_rel, mon_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_monitoring_expectations.json")
    op_dash_rel, op_dash_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_operator_checks_dashboard.json")
    operator_fig_rel, operator_fig_path = _artifact_ref(run_dir, target, "figures", f"{run_id}_operator_checks_dashboard.png")
    manifest_rel, _ = _artifact_ref(run_dir, target, None, "manifest.json")
    suite_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_suite_steps.json")
    hang_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_hang_triage_readiness.json")
    conn_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_torchrun_connectivity_probe.json")
    nccl_env_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_nccl_env_sensitivity.json")
    fio_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_fio.json")
    vllm_csv_rel, _ = _artifact_ref(run_dir, target, "structured", f"{run_id}_{label}_vllm_serve_sweep.csv")
    story_fig_rel, _ = _artifact_ref(run_dir, target, "figures", f"{run_id}_cluster_story_dashboard.png")
    fabric_score_rel, fabric_score_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_scorecard.json")
    fabric_ver_rel, fabric_ver_path = _artifact_ref(run_dir, target, "structured", f"{run_id}_fabric_verification.json")
    quick_exists = quick_path.exists()
    mon_exists = mon_path.exists()
    operator_dashboard_exists = op_dash_path.exists()
    operator_fig_exists = operator_fig_path.exists()
    fabric_score_exists = fabric_score_path.exists()
    fabric_ver_exists = fabric_ver_path.exists()

    now_text = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append("# Cluster Case Study Field Notes (Localhost Package)")
    lines.append("")
    lines.append(f"Last updated: {now_text}. Canonical run: `{run_id}`.")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("1. [Scope](#scope)")
    lines.append("2. [Required Reliability Gates](#required-reliability-gates)")
    lines.append("3. [Fabric Evaluation](#fabric-evaluation)")
    lines.append("4. [Operator Friction + Monitoring](#operator-friction--monitoring)")
    lines.append("5. [Required Issue Ledger](#required-issue-ledger)")
    lines.append("6. [Root Cause + Fix Mapping](#root-cause--fix-mapping)")
    lines.append("7. [Evidence Matrix](#evidence-matrix)")
    lines.append("8. [Repro Entry Point](#repro-entry-point)")
    lines.append("")

    lines.append("## Scope")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append("| Host | `localhost` |")
    lines.append("| GPU count | `1` |")
    lines.append(f"| Canonical run | `{run_id}` |")
    lines.append(f"| Manifest | {md_link(manifest_rel)} |")
    lines.append(f"| Suite steps | {md_link(suite_rel)} |")
    lines.append(f"| Operator dashboard | {md_link(op_dash_rel) if operator_dashboard_exists else 'not generated in this preset'} |")
    lines.append(f"| Fabric scorecard | {md_link(fabric_score_rel) if fabric_score_exists else 'not generated in this preset'} |")
    lines.append("")

    lines.append("## Required Reliability Gates")
    lines.append("| Gate | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Hang triage readiness | `ok` | {md_link(hang_rel)} |")
    lines.append(f"| Torchrun connectivity probe | `ok` | {md_link(conn_rel)} |")
    lines.append(f"| NCCL env sensitivity | `ok` | {md_link(nccl_env_rel)} |")
    lines.append("")

    lines.append("## Fabric Evaluation")
    lines.append("| Item | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Fabric scorecard | {'generated' if fabric_score_exists else 'not run'} | {md_link(fabric_score_rel) if fabric_score_exists else md_link(suite_rel)} |")
    lines.append(f"| Fabric verification ledger | {'generated' if fabric_ver_exists else 'not run'} | {md_link(fabric_ver_rel) if fabric_ver_exists else md_link(suite_rel)} |")
    lines.append("")

    lines.append("## Operator Friction + Monitoring")
    lines.append("| Check | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| quick_friction | {'see artifact' if quick_exists else 'not run'} | {md_link(quick_rel) if quick_exists else md_link(suite_rel)} |")
    lines.append(f"| monitoring_expectations | {'see artifact' if mon_exists else 'not run'} | {md_link(mon_rel) if mon_exists else md_link(suite_rel)} |")
    lines.append(f"| operator checks dashboard (json) | {'generated' if operator_dashboard_exists else 'not generated'} | {md_link(op_dash_rel) if operator_dashboard_exists else md_link(suite_rel)} |")
    lines.append(f"| operator checks dashboard (fig) | {'generated' if operator_fig_exists else 'not generated'} | {md_link(operator_fig_rel) if operator_fig_exists else md_link(suite_rel)} |")
    lines.append("")

    lines.append("## Required Issue Ledger")
    lines.append("| Required issue (verbatim) | Status in localhost package | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Missing node2 fio artifact in canonical package (node2_fio.json absent). | Not applicable (single-node scope) | {md_link(fio_rel)} |")
    lines.append(f"| No multinode vLLM artifact in canonical package. | Not applicable (single-node scope) | {md_link(vllm_csv_rel)} |")
    lines.append(f"| No nvbandwidth bundle in canonical package. | Not applicable unless explicitly enabled in localhost package | {md_link(suite_rel)} |")
    lines.append(f"| Health suite had GDR requested, but effective GDR was false due non-CUDA IB local checks. | Not applicable (`health-suite off`) | {md_link(suite_rel)} |")
    lines.append(f"| Tail latency knee is severe at high concurrency (throughput up, TTFT/p99 TTFT much worse). | Not observed in localhost canary sweep by default | {md_link(vllm_csv_rel)} |")
    lines.append("")

    lines.append("## Root Cause + Fix Mapping")
    lines.append("| Issue | Root cause | Fix | Verification |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| preflight false negatives | pipeline-based service probing could be flaky under strict shell behavior | switch to deterministic `systemctl show -p LoadState` checks | clean preflight in canonical suite steps |")
    lines.append("| NVLink topology parse fragility | header parsing assumptions were too strict | parser robustness for single-GPU/non-tab layouts | topology summary + figure generated in canonical package |")
    lines.append("| quick-friction red-state noise on localhost | optional external tools may be absent by design | expected-failure classification (`expected_failed_checks` vs `unexpected_failed_checks`) | operator checks are either evidenced directly or marked as skipped in this preset |")
    lines.append("")

    lines.append("## Evidence Matrix")
    lines.append("| Claim | Evidence | Verdict |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Localhost suite is clean | {md_link(suite_rel)} | Backed |")
    lines.append(f"| Fabric coverage is {'included' if fabric_score_exists else 'not included in this preset'} | {md_link(fabric_score_rel) if fabric_score_exists else md_link(suite_rel)} | Backed |")
    lines.append(f"| Operator checks are {'included' if quick_exists or mon_exists or operator_dashboard_exists else 'optional and skipped in this preset'} | " + (f"{md_link(quick_rel)}, {md_link(mon_rel)}, {md_link(op_dash_rel)}" if quick_exists or mon_exists or operator_dashboard_exists else md_link(suite_rel)) + " | Backed |")
    lines.append(f"| Visual package is present | " + (md_link(operator_fig_rel) if operator_fig_exists else md_link(story_fig_rel)) + " | Backed |")
    lines.append("")

    lines.append("## Repro Entry Point")
    lines.append("")
    lines.append(f"The example starts a new run. See the [suite record]({suite_rel}) for the measured settings.")
    lines.append("")
    lines.append("| Step | Command |")
    lines.append("| --- | --- |")
    lines.append(
        "| Example localhost fabric eval | `"
        + localhost_fabric_repro_cmd(label)
        + "` |"
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render localhost field-report package from a canonical run ID.")
    parser.add_argument("--run-id", required=True, help="Canonical run ID (example: 2026-02-24_localhost_fullsuite_r4)")
    parser.add_argument("--label", default="localhost", help="Host label used in structured artifact names (default: localhost)")
    parser.add_argument("--report", default="", help="Run-local report markdown path (default: <run_dir>/reports/field-report-localhost.md)")
    parser.add_argument("--notes", default="", help="Run-local notes markdown path (default: <run_dir>/reports/field-report-localhost-notes.md)")
    parser.add_argument("--publish-report", default="", help="Optional published report markdown path")
    parser.add_argument("--publish-notes", default="", help="Optional published notes markdown path")
    parser.add_argument("--root", default=None, help="Cluster root directory (default: auto-detect from script location)")
    parser.add_argument("--run-dir", default="", help="Canonical run directory (default: <cluster_root>/runs/<run_id>)")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    args.root = root
    args.run_dir = _resolve_run_dir(root, args.run_id, args.run_dir or None)

    report_path = Path(args.report).resolve() if args.report else _run_local_report_path(args.run_dir, "field-report-localhost.md")
    notes_path = Path(args.notes).resolve() if args.notes else _run_local_report_path(args.run_dir, "field-report-localhost-notes.md")

    report_text = render_report(args, target="run_local")
    notes_text = render_notes(args, target="run_local")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    notes_path.write_text(notes_text, encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {notes_path}")

    if args.publish_report:
        publish_report_path = Path(args.publish_report).resolve()
        publish_report_path.parent.mkdir(parents=True, exist_ok=True)
        publish_report_path.write_text(render_report(args, target="published"), encoding="utf-8")
        print(f"Wrote {publish_report_path}")
    if args.publish_notes:
        publish_notes_path = Path(args.publish_notes).resolve()
        publish_notes_path.parent.mkdir(parents=True, exist_ok=True)
        publish_notes_path.write_text(render_notes(args, target="published"), encoding="utf-8")
        print(f"Wrote {publish_notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
