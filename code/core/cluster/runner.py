from __future__ import annotations

import socket
import subprocess
import threading
import time
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.harness.progress import ProgressEvent, ProgressRecorder

_CLUSTER_PROGRESS_POLL_SECONDS = 2.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cluster_root() -> Path:
    return _repo_root() / "cluster"


def _promote_run_script() -> Path:
    """Path to `cluster/scripts/promote_run.py` in this checkout (not derived from `--repo-root`)."""
    return Path(__file__).resolve().parents[2] / "cluster" / "scripts" / "promote_run.py"


def _watch_run_for_promotion_script() -> Path:
    """Path to `cluster/scripts/watch_run_for_promotion.py` in this checkout."""
    return Path(__file__).resolve().parents[2] / "cluster" / "scripts" / "watch_run_for_promotion.py"


def _cluster_run_dir(run_id: str) -> Path:
    return _cluster_root() / "runs" / run_id


def _cluster_run_layout(run_id: str, *, repo_root: Optional[Path] = None) -> Dict[str, str]:
    cluster_root = (repo_root / "cluster") if repo_root is not None else _cluster_root()
    run_dir = cluster_root / "runs" / run_id
    return {
        "run_dir": str(run_dir),
        "structured_dir": str(run_dir / "structured"),
        "raw_dir": str(run_dir / "raw"),
        "figures_dir": str(run_dir / "figures"),
        "reports_dir": str(run_dir / "reports"),
        "progress_dir": str(run_dir / "progress"),
        "progress_path": str(run_dir / "progress" / "run_progress.json"),
        "suite_steps_path": str(run_dir / "structured" / f"{run_id}_suite_steps.json"),
        "manifest_path": str(run_dir / "manifest.json"),
    }


def _sanitize_label(raw: str) -> str:
    return str(raw).replace(".", "_").replace(":", "_")


def _cluster_host_labels(hosts: List[str], labels: Optional[List[str]]) -> List[str]:
    resolved: List[str] = []
    label_values = list(labels or [])
    for index, host in enumerate(hosts):
        label = label_values[index].strip() if index < len(label_values) and str(label_values[index]).strip() else ""
        resolved.append(label or _sanitize_label(host))
    return resolved


def _is_local_host_name(host: str) -> bool:
    host_value = str(host or "").strip()
    if not host_value:
        return False
    fqdn = socket.getfqdn() or socket.gethostname()
    candidates = {
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.gethostname().split(".", 1)[0],
        fqdn,
        fqdn.split(".", 1)[0],
    }
    return host_value in candidates


def _split_cli_values(raw: str) -> List[str]:
    values = str(raw or "").replace(",", " ").split()
    return [value for value in values if value]


def _cluster_suite_progress_state(
    *,
    hosts: List[str],
    labels: Optional[List[str]],
    primary_label: Optional[str],
    extra_args: Optional[List[str]],
    coverage_baseline_run_id: Optional[str],
    oob_if: Optional[str],
    socket_ifname: Optional[str],
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "bootstrap_nodes": True,
        "run_quick_friction": True,
        "run_monitoring_expectations": True,
        "run_nccl_env_sensitivity": True,
        "health_suite_mode": "collectives",
        "run_vllm_request_rate_sweep": False,
        "run_vllm_multinode_mode": "auto",
        "vllm_multinode_concurrency_values": ["64"],
        "enable_fp4": True,
        "enable_mamf": False,
        "enable_allreduce_stability": False,
        "enable_allreduce_latency_comp": False,
        "enable_allgather_control_plane": False,
        "enable_nccl_alltoall": False,
        "enable_nccl_algo_comparison": False,
        "run_c2c": False,
        "run_numa_mem_bw": False,
        "run_train_step": False,
        "run_train_step_explicit": False,
        "train_step_single_node": True,
        "train_step_multi_node": True,
        "run_checkpoint_io": False,
        "run_nvbandwidth_mode": "auto",
        "run_gpu_stream_mode": "on",
        "run_fabric_eval": False,
        "render_localhost_report_mode": "auto",
        "check_ib_sharp": False,
        "modern_llm_profile": False,
        "multinode_readiness_check_only": False,
        "coverage_baseline": bool(str(coverage_baseline_run_id or "").strip()),
    }
    explicit_vllm_override = False
    args = [str(arg) for arg in (extra_args or [])]
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--skip-quick-friction":
            state["run_quick_friction"] = False
        elif arg == "--run-quick-friction":
            state["run_quick_friction"] = True
        elif arg == "--skip-monitoring-expectations":
            state["run_monitoring_expectations"] = False
        elif arg == "--run-monitoring-expectations":
            state["run_monitoring_expectations"] = True
        elif arg == "--skip-nccl-env-sensitivity":
            state["run_nccl_env_sensitivity"] = False
        elif arg == "--run-nccl-env-sensitivity":
            state["run_nccl_env_sensitivity"] = True
        elif arg == "--health-suite" and index + 1 < len(args):
            index += 1
            state["health_suite_mode"] = args[index]
        elif arg == "--run-vllm-request-rate-sweep":
            state["run_vllm_request_rate_sweep"] = True
            explicit_vllm_override = True
        elif arg == "--skip-vllm-request-rate-sweep":
            state["run_vllm_request_rate_sweep"] = False
            explicit_vllm_override = True
        elif arg == "--run-vllm-multinode":
            state["run_vllm_multinode_mode"] = "on"
        elif arg == "--skip-vllm-multinode":
            state["run_vllm_multinode_mode"] = "off"
        elif arg == "--vllm-multinode-concurrency-range" and index + 1 < len(args):
            index += 1
            values = _split_cli_values(args[index])
            if values:
                state["vllm_multinode_concurrency_values"] = values
        elif arg == "--vllm-multinode-concurrency" and index + 1 < len(args):
            index += 1
            state["vllm_multinode_concurrency_values"] = [args[index]]
        elif arg == "--enable-fp4":
            state["enable_fp4"] = True
        elif arg == "--disable-fp4":
            state["enable_fp4"] = False
        elif arg == "--enable-mamf":
            state["enable_mamf"] = True
        elif arg == "--enable-allreduce-stability":
            state["enable_allreduce_stability"] = True
        elif arg == "--disable-allreduce-stability":
            state["enable_allreduce_stability"] = False
        elif arg == "--enable-allreduce-latency-comp":
            state["enable_allreduce_latency_comp"] = True
        elif arg == "--enable-allgather-control-plane":
            state["enable_allgather_control_plane"] = True
        elif arg == "--enable-nccl-alltoall":
            state["enable_nccl_alltoall"] = True
        elif arg == "--disable-nccl-alltoall":
            state["enable_nccl_alltoall"] = False
        elif arg == "--enable-nccl-algo-comparison":
            state["enable_nccl_algo_comparison"] = True
        elif arg == "--disable-nccl-algo-comparison":
            state["enable_nccl_algo_comparison"] = False
        elif arg == "--run-c2c":
            state["run_c2c"] = True
        elif arg == "--run-numa-mem-bw":
            state["run_numa_mem_bw"] = True
        elif arg == "--run-train-step":
            state["run_train_step"] = True
            state["run_train_step_explicit"] = True
        elif arg == "--train-step-single-node":
            state["train_step_single_node"] = True
        elif arg == "--train-step-multi-node":
            state["train_step_multi_node"] = True
        elif arg == "--run-checkpoint-io":
            state["run_checkpoint_io"] = True
        elif arg == "--run-nvbandwidth":
            state["run_nvbandwidth_mode"] = "on"
        elif arg == "--skip-nvbandwidth":
            state["run_nvbandwidth_mode"] = "off"
        elif arg == "--run-gpu-stream":
            state["run_gpu_stream_mode"] = "on"
        elif arg == "--skip-gpu-stream":
            state["run_gpu_stream_mode"] = "off"
        elif arg == "--run-fabric-eval":
            state["run_fabric_eval"] = True
        elif arg == "--skip-fabric-eval":
            state["run_fabric_eval"] = False
        elif arg == "--render-localhost-report":
            state["render_localhost_report_mode"] = "on"
        elif arg == "--skip-render-localhost-report":
            state["render_localhost_report_mode"] = "off"
        elif arg == "--check-ib-sharp":
            state["check_ib_sharp"] = True
        elif arg == "--modern-llm-profile":
            state["modern_llm_profile"] = True
        elif arg == "--multinode-readiness-check-only":
            state["multinode_readiness_check_only"] = True
        elif arg in {
            "--model",
            "--tp",
            "--isl",
            "--osl",
            "--concurrency-range",
            "--vllm-repeats",
            "--vllm-request-rate-range",
            "--vllm-request-rate-repeats",
            "--vllm-request-rate-max-concurrency",
            "--vllm-request-rate-num-prompts",
        } and index + 1 < len(args):
            explicit_vllm_override = True
            index += 1
        elif arg == "--coverage-baseline-run-id" and index + 1 < len(args):
            index += 1
            state["coverage_baseline"] = bool(str(args[index]).strip())
        index += 1

    host_count = len(hosts)
    has_socket_bootstrap = bool(str(socket_ifname or oob_if or "").strip())
    if state["modern_llm_profile"]:
        state["run_vllm_request_rate_sweep"] = True
        state["run_nvbandwidth_mode"] = "on"
        state["enable_allreduce_stability"] = True
        state["enable_allreduce_latency_comp"] = True
        state["enable_allgather_control_plane"] = True
        state["enable_nccl_alltoall"] = True
        state["enable_nccl_algo_comparison"] = True
        if not state["run_train_step_explicit"]:
            state["run_train_step"] = True
    if state["run_vllm_multinode_mode"] == "on":
        state["run_vllm_multinode"] = True
    elif state["run_vllm_multinode_mode"] == "off":
        state["run_vllm_multinode"] = False
    else:
        state["run_vllm_multinode"] = host_count > 1
    if state["run_nvbandwidth_mode"] == "on":
        state["run_nvbandwidth"] = True
    elif state["run_nvbandwidth_mode"] == "off":
        state["run_nvbandwidth"] = False
    else:
        state["run_nvbandwidth"] = host_count > 1
    state["run_gpu_stream"] = state["run_gpu_stream_mode"] == "on"
    label_values = _cluster_host_labels(hosts, labels)
    resolved_primary_label = str(primary_label or "").strip() or (label_values[0] if label_values else _default_primary_label())
    is_localhost_package = host_count == 1 and bool(hosts) and _is_local_host_name(hosts[0])
    if state["run_fabric_eval"] and is_localhost_package and not explicit_vllm_override:
        state["run_vllm_request_rate_sweep"] = False
    state.update(
        {
            "host_count": host_count,
            "has_socket_bootstrap": has_socket_bootstrap,
            "labels": label_values,
            "primary_label": resolved_primary_label,
            "is_localhost_package": is_localhost_package,
            "render_localhost_report": state["render_localhost_report_mode"] == "on"
            or (state["render_localhost_report_mode"] == "auto" and is_localhost_package),
        }
    )
    return state


def _cluster_suite_planned_steps(
    *,
    run_id: str,
    hosts: List[str],
    labels: Optional[List[str]],
    primary_label: Optional[str],
    extra_args: Optional[List[str]],
    coverage_baseline_run_id: Optional[str],
    oob_if: Optional[str],
    socket_ifname: Optional[str],
) -> List[str]:
    state = _cluster_suite_progress_state(
        hosts=hosts,
        labels=labels,
        primary_label=primary_label,
        extra_args=extra_args,
        coverage_baseline_run_id=coverage_baseline_run_id,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
    )
    if state["multinode_readiness_check_only"]:
        return ["multinode_readiness", "manifest_refresh"]

    steps: List[str] = []
    host_count = int(state["host_count"])
    labels_resolved = list(state["labels"])
    run_vllm_multinode = bool(state["run_vllm_multinode"])

    if state["bootstrap_nodes"]:
        steps.append("bootstrap_nodes")
    steps.extend(
        [
            "preflight_services",
            "discovery",
        ]
    )
    if state["run_quick_friction"]:
        steps.append("quick_friction_all_nodes")
    if state["run_monitoring_expectations"]:
        steps.append("monitoring_expectations_all_nodes")
    steps.extend(
        [
            "hang_triage_bundle",
            "connectivity_probe",
            "nccl_single_node",
        ]
    )
    if host_count > 1 and bool(str(oob_if or "").strip()):
        steps.append("nccl_multi_node")
    if state["run_nccl_env_sensitivity"]:
        steps.append("nccl_env_sensitivity")
    health_suite_mode = str(state["health_suite_mode"])
    if host_count > 1 and health_suite_mode != "off":
        steps.append(f"health_suite_{health_suite_mode}")
    if host_count > 1 and state["check_ib_sharp"] and bool(str(oob_if or "").strip()):
        steps.append("ib_sharp_check")
    steps.append("vllm_serve_sweep")
    if state["run_vllm_request_rate_sweep"]:
        steps.append("vllm_request_rate_sweep")
    if run_vllm_multinode and host_count > 1:
        for value in state["vllm_multinode_concurrency_values"]:
            steps.append(f"vllm_serve_multinode_c{value}")
    steps.append("gemm_sanity")
    if state["run_gpu_stream"]:
        steps.append("gpu_stream_all_nodes")
    if state["enable_fp4"]:
        steps.append("fp4_checks")
    if state["enable_mamf"]:
        steps.append("mamf_finder")
    if state["enable_allreduce_stability"]:
        steps.append("allreduce_stability")
    if state["enable_allreduce_latency_comp"]:
        steps.append("allreduce_latency_comp")
    if state["enable_allgather_control_plane"]:
        steps.append("allgather_control_plane")
    if state["enable_nccl_alltoall"]:
        steps.append("nccl_alltoall_single_node")
        if host_count > 1 and bool(str(oob_if or "").strip()):
            steps.append("nccl_alltoall_multi_node")
    if state["enable_nccl_algo_comparison"]:
        steps.append("nccl_algo_comparison")
    if state["run_c2c"]:
        steps.append("c2c_memcpy")
    if state["run_numa_mem_bw"]:
        steps.append("numa_mem_bw")
    if state["run_train_step"]:
        if state["train_step_single_node"]:
            steps.append("train_step_single_node")
        if state["train_step_multi_node"] and host_count > 1:
            steps.append("train_step_multi_node")
    if state["run_checkpoint_io"]:
        steps.append("checkpoint_io")
    steps.append("fio_all_nodes")
    if state["run_nvbandwidth"]:
        steps.append("nvbandwidth_all_nodes")

    steps.append("plot_nccl_single_node")
    if host_count > 1 and bool(str(oob_if or "").strip()):
        steps.append("plot_nccl_multi_node")
    if state["enable_nccl_alltoall"]:
        steps.append("plot_nccl_alltoall_single_node")
        if host_count > 1 and bool(str(oob_if or "").strip()):
            steps.append("plot_nccl_alltoall_multi_node")
    if state["run_nccl_env_sensitivity"]:
        steps.append("plot_nccl_env_sensitivity")
    steps.extend(
        [
            "plot_vllm_serve",
            "analyze_vllm_slo_goodput",
            "plot_vllm_slo_goodput",
        ]
    )
    if state["run_vllm_request_rate_sweep"]:
        steps.extend(
            [
                "analyze_vllm_request_rate_slo_goodput",
                "plot_vllm_request_rate_sweep",
                "plot_vllm_request_rate_slo_goodput",
            ]
        )
    if run_vllm_multinode and host_count > 1:
        steps.extend(
            [
                "plot_vllm_serve_multinode",
                "analyze_vllm_multinode_slo_goodput",
                "plot_vllm_multinode_slo_goodput",
            ]
        )
    if host_count > 1 and health_suite_mode != "off":
        steps.append("plot_iperf3_oob")
    steps.append("plot_gemm_sanity")
    if state["enable_mamf"]:
        steps.append("plot_mamf")
    if state["enable_allreduce_stability"]:
        steps.append("plot_allreduce_stability")
    if state["enable_allreduce_latency_comp"]:
        steps.append("plot_allreduce_latency_comp")
    if state["enable_allgather_control_plane"]:
        steps.append("plot_allgather_control_plane")
    if state["enable_nccl_algo_comparison"]:
        steps.append("plot_nccl_algo_comparison")
    for label in labels_resolved:
        steps.append(f"plot_fio_{run_id}_{label}_fio")
    if state["run_nvbandwidth"]:
        for label in labels_resolved:
            steps.append(f"plot_nvbandwidth_{run_id}_{label}_nvbandwidth")
    if state["run_gpu_stream"]:
        for label in labels_resolved:
            steps.append(f"plot_gpu_stream_{run_id}_{label}_gpu_stream")
    if state["run_c2c"]:
        steps.append("plot_c2c_memcpy")
    if state["run_numa_mem_bw"]:
        for label in labels_resolved:
            steps.append(f"plot_numa_mem_bw_{run_id}_{label}_numa_mem_bw")
    if state["run_train_step"] and state["train_step_single_node"]:
        steps.append("plot_train_step_single")
    if state["run_train_step"] and state["train_step_multi_node"] and host_count > 1:
        steps.append("plot_train_step_multi")
    for label in labels_resolved:
        steps.append(f"plot_nvlink_topology_{run_id}_{label}_meta")
    steps.append("plot_cluster_story_dashboard")
    if state["run_quick_friction"] or state["run_monitoring_expectations"]:
        steps.append("plot_operator_checks_dashboard")
    if state["run_fabric_eval"]:
        steps.append("build_fabric_eval")
    steps.extend(
        [
            "build_cluster_scorecard",
            "build_mlperf_alignment",
            "analyze_benchmark_coverage",
        ]
    )
    if state["coverage_baseline"]:
        steps.append("build_coverage_delta")
    steps.extend(
        [
            "validate_required_artifacts",
            "manifest_refresh",
        ]
    )
    if state["render_localhost_report"]:
        steps.append("render_localhost_field_report_package")
    if not steps:
        return ["cluster_eval_suite"]
    return steps


def _read_cluster_suite_steps(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    latest: Dict[str, Dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        latest[name] = row
    return latest


def _cluster_suite_current_step(suite_log_dir: Path, *, completed_steps: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not suite_log_dir.exists():
        return None
    latest_candidate: Optional[Path] = None
    latest_mtime = -1.0
    for log_path in suite_log_dir.glob("*.log"):
        step_name = log_path.stem
        recorded = completed_steps.get(step_name)
        if recorded is not None:
            try:
                exit_code = int(recorded.get("exit_code"))
            except Exception:
                exit_code = 0
            if exit_code != 75:
                continue
        try:
            mtime = log_path.stat().st_mtime
        except OSError:
            continue
        if mtime >= latest_mtime:
            latest_candidate = log_path
            latest_mtime = mtime
    return latest_candidate.stem if latest_candidate is not None else None


def _cluster_suite_progress_current(
    *,
    run_id: str,
    planned_steps: List[str],
    suite_steps_path: Path,
    suite_log_dir: Path,
) -> Dict[str, Any]:
    latest_by_name = _read_cluster_suite_steps(suite_steps_path)
    completed_names = set()
    for name, row in latest_by_name.items():
        try:
            exit_code = int(row.get("exit_code", 0) or 0)
        except Exception:
            exit_code = 0
        if exit_code != 75:
            completed_names.add(name)
    current_step = _cluster_suite_current_step(suite_log_dir, completed_steps=latest_by_name)
    known_steps = set(planned_steps)
    extra_steps = {name for name in completed_names if name not in known_steps}
    if current_step and current_step not in known_steps:
        extra_steps.add(current_step)
    total_steps = max(1, len(planned_steps) + len(extra_steps))
    completed_count = len(completed_names)
    percent_complete = min(100.0, (float(completed_count) / float(total_steps)) * 100.0)
    active_step = current_step or (planned_steps[completed_count] if completed_count < len(planned_steps) else None)
    phase_index = completed_count + 1 if active_step else completed_count
    phase_index = max(1, min(total_steps, phase_index))
    return {
        "phase": "cluster_eval_suite",
        "phase_index": phase_index,
        "total_phases": total_steps,
        "step": active_step or ("complete" if completed_count >= total_steps else "starting"),
        "step_detail": f"completed {completed_count}/{total_steps} suite steps",
        "percent_complete": percent_complete,
        "metrics": {
            "completed_steps": completed_count,
            "total_steps": total_steps,
            "current_step": active_step,
            "planned_steps": planned_steps,
            "suite_steps_path": str(suite_steps_path),
        },
    }


def _emit_cluster_suite_progress(
    progress_recorder: ProgressRecorder,
    *,
    run_id: str,
    planned_steps: List[str],
    suite_steps_path: Path,
    suite_log_dir: Path,
    status: str,
) -> None:
    current = _cluster_suite_progress_current(
        run_id=run_id,
        planned_steps=planned_steps,
        suite_steps_path=suite_steps_path,
        suite_log_dir=suite_log_dir,
    )
    percent_complete = 100.0 if status == "completed" else current["percent_complete"]
    progress_recorder.emit(
        ProgressEvent(
            phase=str(current.get("phase") or "cluster_eval_suite"),
            phase_index=int(current.get("phase_index") or 1),
            total_phases=int(current.get("total_phases") or max(1, len(planned_steps))),
            step=str(current.get("step") or status),
            step_detail=str(current.get("step_detail") or f"status={status}"),
            percent_complete=percent_complete,
            artifacts=[str(suite_steps_path)],
            metrics={
                **dict(current.get("metrics") or {}),
                "status": status,
            },
        )
    )


def _run_cluster_suite_with_progress(
    cmd: List[str],
    *,
    cwd: Path,
    timeout_seconds: Optional[int],
    run_id: str,
    planned_steps: List[str],
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    layout = _cluster_run_layout(run_id, repo_root=repo_root)
    progress_recorder = ProgressRecorder(run_id=run_id, progress_path=Path(layout["progress_path"]))
    suite_steps_path = Path(layout["suite_steps_path"])
    suite_log_dir = Path(layout["raw_dir"]) / f"{run_id}_suite"
    _emit_cluster_suite_progress(
        progress_recorder,
        run_id=run_id,
        planned_steps=planned_steps,
        suite_steps_path=suite_steps_path,
        suite_log_dir=suite_log_dir,
        status="running",
    )

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stop_event = threading.Event()

    def _poll_progress() -> None:
        while not stop_event.wait(_CLUSTER_PROGRESS_POLL_SECONDS):
            _emit_cluster_suite_progress(
                progress_recorder,
                run_id=run_id,
                planned_steps=planned_steps,
                suite_steps_path=suite_steps_path,
                suite_log_dir=suite_log_dir,
                status="running",
            )

    thread = threading.Thread(
        target=_poll_progress,
        name=f"cluster-progress-{run_id}",
        daemon=True,
    )
    thread.start()
    try:
        stdout, stderr = proc.communicate(
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        returncode = proc.returncode
        return {
            "command": cmd,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "error": f"timeout after {timeout_seconds}s",
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "error": str(exc),
            "duration_ms": duration_ms,
        }
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        status = "completed" if proc.poll() == 0 else "failed"
        _emit_cluster_suite_progress(
            progress_recorder,
            run_id=run_id,
            planned_steps=planned_steps,
            suite_steps_path=suite_steps_path,
            suite_log_dir=suite_log_dir,
            status=status,
        )


def _run_cmd(cmd: List[str], *, cwd: Optional[Path] = None, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": (exc.stdout or ""),
            "stderr": (exc.stderr or ""),
            "error": f"timeout after {timeout_seconds}s",
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "duration_ms": duration_ms,
        }


def _default_run_id_prefix() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _default_primary_label() -> str:
    host = socket.gethostname() or "local"
    return host.split(".")[0] or "local"


_COMMON_EVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "common-answer-fast": {
        "description": (
            "Agent-friendly fast answer bundle for the benchmark questions people ask first: "
            "NCCL all-reduce, compact vLLM concurrency and request-rate sweeps, GEMM sanity, "
            "GPU STREAM, fio, quick nvbandwidth, scorecard, coverage, and MLPerf-alignment. "
            "Uses a smaller modern LLM profile and skips non-benchmark friction/monitoring extras."
        ),
        "extra_args": [
            "--skip-quick-friction",
            "--skip-monitoring-expectations",
            "--disable-fp4",
            "--health-suite",
            "off",
            "--skip-vllm-multinode",
            "--model",
            "openai/gpt-oss-20b",
            "--tp",
            "1",
            "--isl",
            "512",
            "--osl",
            "128",
            "--concurrency-range",
            "8 16 32",
            "--run-vllm-request-rate-sweep",
            "--vllm-request-rate-range",
            "1 2 4",
            "--vllm-request-rate-max-concurrency",
            "32",
            "--vllm-request-rate-num-prompts",
            "128",
            "--fio-runtime",
            "15",
            "--run-nvbandwidth",
            "--nvbandwidth-quick",
        ],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "core-system": {
        "description": (
            "Common single-node or cluster evaluation bundle: discovery, NCCL all-reduce, "
            "vLLM concurrency sweep, vLLM request-rate sweep, GEMM sanity, GPU STREAM, fio, "
            "nvbandwidth, scorecard, coverage, and MLPerf-alignment summary."
        ),
        "extra_args": [
            "--run-vllm-request-rate-sweep",
            "--run-nvbandwidth",
        ],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "modern-llm": {
        "description": (
            "High-signal modern LLM system evaluation: core-system plus request-rate repeats, "
            "allreduce stability, allreduce latency composition, allgather control-plane, "
            "NCCL all-to-all, NCCL algorithm comparison, train-step, and strict completeness gates."
        ),
        "extra_args": ["--modern-llm-profile"],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "allreduce_stability",
            "allreduce_latency_comp",
            "allgather_control_plane",
            "nccl_alltoall",
            "nccl_algo_comparison",
            "train_step",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "fabric-systems": {
        "description": (
            "Canonical AI fabric evaluation bundle: modern-llm runtime coverage plus "
            "fabric inventory, management-plane verification, runtime link checks, "
            "and AI-workload correlation across NVLink, InfiniBand, and Spectrum-X / RoCE."
        ),
        "extra_args": [
            "--modern-llm-profile",
            "--no-strict-canonical-completeness",
            "--run-fabric-eval",
        ],
        "artifact_roles": [
            "meta",
            "manifest",
            "suite_steps",
            "nccl_allreduce",
            "vllm_concurrency_sweep",
            "vllm_request_rate_sweep",
            "gemm_sanity",
            "gpu_stream",
            "fio",
            "nvbandwidth",
            "allreduce_stability",
            "allreduce_latency_comp",
            "allgather_control_plane",
            "nccl_alltoall",
            "nccl_algo_comparison",
            "train_step",
            "fabric_command_catalog",
            "fabric_capability_matrix",
            "fabric_verification",
            "fabric_ai_correlation",
            "fabric_scorecard",
            "cluster_scorecard",
            "benchmark_coverage_analysis",
            "mlperf_alignment",
        ],
    },
    "multinode-readiness": {
        "description": (
            "Fail-fast multi-node contract validation only. Produces readiness evidence without running workloads."
        ),
        "extra_args": [
            "--modern-llm-profile",
            "--multinode-readiness-check-only",
        ],
        "artifact_roles": [
            "multinode_readiness",
            "manifest",
        ],
    },
}


def _smoke_eval(run_id: str, *, primary_label: Optional[str] = None, timeout_seconds: int = 120) -> Dict[str, Any]:
    cluster_root = _cluster_root()
    scripts_dir = cluster_root / "scripts"
    run_dir = _cluster_run_dir(run_id)
    struct_dir = run_dir / "structured"
    raw_dir = run_dir / "raw"
    reports_dir = run_dir / "reports"
    figures_dir = run_dir / "figures"
    for path in (struct_dir, raw_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    label = primary_label or _default_primary_label()
    meta_path = struct_dir / f"{run_id}_{label}_meta.json"

    collect_script = scripts_dir / "collect_system_info.sh"
    manifest_script = scripts_dir / "write_manifest.py"

    if not collect_script.exists():
        return {"success": False, "error": f"Missing script: {collect_script}"}
    if not manifest_script.exists():
        return {"success": False, "error": f"Missing script: {manifest_script}"}

    collect_cmd = ["bash", str(collect_script), "--output", str(meta_path), "--label", str(label)]
    collect = _run_cmd(collect_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    if collect.get("returncode") != 0:
        return {
            "success": False,
            "mode": "smoke",
            "run_id": run_id,
            "primary_label": label,
            "meta_path": str(meta_path),
            "collect": collect,
            "error": "collect_system_info failed",
        }

    manifest_cmd = [
        "python3",
        str(manifest_script),
        "--root",
        str(cluster_root),
        "--run-id",
        str(run_id),
        "--run-dir",
        str(run_dir),
    ]
    manifest = _run_cmd(manifest_cmd, cwd=cluster_root, timeout_seconds=timeout_seconds)
    manifest_path = (manifest.get("stdout") or "").strip().splitlines()[-1] if manifest.get("stdout") else ""

    return {
        "success": bool(manifest.get("returncode") == 0),
        "mode": "smoke",
        "run_id": run_id,
        "primary_label": label,
        "meta_path": str(meta_path),
        "manifest_path": manifest_path or None,
        "collect": collect,
        "manifest": manifest,
        **_cluster_run_layout(run_id),
    }


def run_cluster_eval_suite(
    *,
    mode: str = "smoke",
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    primary_label: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the cluster eval suite (full) or a fast local smoke run.

    - mode=smoke:
      Writes under `cluster/runs/<run_id>/...` via collect_system_info.sh and write_manifest.py.

    - mode=full:
      Invokes `cluster/scripts/run_cluster_eval_suite.sh` (requires --hosts).
    """
    run_id_value = run_id or _default_run_id_prefix()
    mode_value = (mode or "smoke").strip().lower()
    timeout = int(timeout_seconds) if timeout_seconds is not None else None

    if mode_value in {"smoke", "local", "mini"}:
        return _smoke_eval(run_id_value, primary_label=primary_label, timeout_seconds=timeout or 120)

    if mode_value not in {"full", "eval", "suite"}:
        return {"success": False, "error": f"Unknown mode: {mode!r} (expected smoke|full)"}

    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "run_cluster_eval_suite.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    hosts_list = hosts or []
    hosts_list = [h.strip() for h in hosts_list if isinstance(h, str) and h.strip()]
    if not hosts_list:
        return {"success": False, "error": "--hosts is required in full mode", "mode": "full", "run_id": run_id_value}

    cmd: List[str] = ["bash", str(script), "--hosts", ",".join(hosts_list), "--run-id", str(run_id_value)]
    if labels:
        labels_list = [l.strip() for l in labels if isinstance(l, str) and l.strip()]
        if labels_list:
            cmd.extend(["--labels", ",".join(labels_list)])
    if ssh_user:
        cmd.extend(["--ssh-user", str(ssh_user)])
    if ssh_key:
        cmd.extend(["--ssh-key", str(ssh_key)])
    if oob_if:
        cmd.extend(["--oob-if", str(oob_if)])
    if socket_ifname:
        cmd.extend(["--socket-ifname", str(socket_ifname)])
    if nccl_ib_hca:
        cmd.extend(["--nccl-ib-hca", str(nccl_ib_hca)])
    if primary_label:
        cmd.extend(["--primary-label", str(primary_label)])
    if extra_args:
        cmd.extend([str(x) for x in extra_args if str(x).strip()])

    planned_steps = _cluster_suite_planned_steps(
        run_id=run_id_value,
        hosts=hosts_list,
        labels=labels,
        primary_label=primary_label,
        extra_args=extra_args,
        coverage_baseline_run_id=None,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
    )
    result = _run_cluster_suite_with_progress(
        cmd,
        cwd=cluster_root,
        timeout_seconds=timeout,
        run_id=run_id_value,
        planned_steps=planned_steps,
    )
    success = result.get("returncode") == 0
    return {
        "success": bool(success),
        "mode": "full",
        "run_id": run_id_value,
        "command": cmd,
        **_cluster_run_layout(run_id_value),
        **result,
    }


def run_cluster_common_eval(
    *,
    preset: str = "core-system",
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    nmx_url: Optional[str] = None,
    nmx_token: Optional[str] = None,
    ib_mgmt_host: Optional[str] = None,
    ib_mgmt_user: Optional[str] = None,
    ib_mgmt_ssh_key: Optional[str] = None,
    cumulus_hosts: Optional[List[str]] = None,
    cumulus_user: Optional[str] = None,
    cumulus_ssh_key: Optional[str] = None,
    primary_label: Optional[str] = None,
    coverage_baseline_run_id: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    preset_name = (preset or "core-system").strip().lower()
    preset_payload = _COMMON_EVAL_PRESETS.get(preset_name)
    if preset_payload is None:
        return {
            "success": False,
            "error": f"Unknown preset: {preset!r} (expected one of {', '.join(sorted(_COMMON_EVAL_PRESETS))})",
        }

    merged_extra_args = [str(arg) for arg in preset_payload["extra_args"]]
    if nmx_url:
        merged_extra_args.extend(["--nmx-url", str(nmx_url)])
    if nmx_token:
        merged_extra_args.extend(["--nmx-token", str(nmx_token)])
    if ib_mgmt_host:
        merged_extra_args.extend(["--ib-mgmt-host", str(ib_mgmt_host)])
    if ib_mgmt_user:
        merged_extra_args.extend(["--ib-mgmt-user", str(ib_mgmt_user)])
    if ib_mgmt_ssh_key:
        merged_extra_args.extend(["--ib-mgmt-ssh-key", str(ib_mgmt_ssh_key)])
    if cumulus_hosts:
        normalized_cumulus_hosts = [str(host).strip() for host in cumulus_hosts if str(host).strip()]
        if normalized_cumulus_hosts:
            merged_extra_args.extend(["--cumulus-hosts", ",".join(normalized_cumulus_hosts)])
    if cumulus_user:
        merged_extra_args.extend(["--cumulus-user", str(cumulus_user)])
    if cumulus_ssh_key:
        merged_extra_args.extend(["--cumulus-ssh-key", str(cumulus_ssh_key)])
    if coverage_baseline_run_id:
        merged_extra_args.extend(["--coverage-baseline-run-id", str(coverage_baseline_run_id)])
    if extra_args:
        merged_extra_args.extend(str(arg) for arg in extra_args if str(arg).strip())

    artifact_roles = list(preset_payload["artifact_roles"])
    if "--multinode-readiness-check-only" in merged_extra_args:
        artifact_roles = ["multinode_readiness", "manifest"]

    result = run_cluster_eval_suite(
        mode="full",
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        primary_label=primary_label,
        extra_args=merged_extra_args,
        timeout_seconds=timeout_seconds,
    )
    return {
        "preset": preset_name,
        "preset_description": preset_payload["description"],
        "artifact_roles": artifact_roles,
        "coverage_baseline_run_id": coverage_baseline_run_id,
        **result,
    }


def run_cluster_fabric_eval(
    *,
    run_id: Optional[str] = None,
    hosts: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[str] = None,
    oob_if: Optional[str] = None,
    socket_ifname: Optional[str] = None,
    nccl_ib_hca: Optional[str] = None,
    nmx_url: Optional[str] = None,
    nmx_token: Optional[str] = None,
    ib_mgmt_host: Optional[str] = None,
    ib_mgmt_user: Optional[str] = None,
    ib_mgmt_ssh_key: Optional[str] = None,
    cumulus_hosts: Optional[List[str]] = None,
    cumulus_user: Optional[str] = None,
    cumulus_ssh_key: Optional[str] = None,
    primary_label: Optional[str] = None,
    coverage_baseline_run_id: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
    require_management_plane: bool = False,
) -> Dict[str, Any]:
    merged_extra_args = [str(arg) for arg in (extra_args or []) if str(arg).strip()]
    if require_management_plane:
        merged_extra_args.append("--require-management-plane")

    result = run_cluster_common_eval(
        preset="fabric-systems",
        run_id=run_id,
        hosts=hosts,
        labels=labels,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        nmx_url=nmx_url,
        nmx_token=nmx_token,
        ib_mgmt_host=ib_mgmt_host,
        ib_mgmt_user=ib_mgmt_user,
        ib_mgmt_ssh_key=ib_mgmt_ssh_key,
        cumulus_hosts=cumulus_hosts,
        cumulus_user=cumulus_user,
        cumulus_ssh_key=cumulus_ssh_key,
        primary_label=primary_label,
        coverage_baseline_run_id=coverage_baseline_run_id,
        extra_args=merged_extra_args or None,
        timeout_seconds=timeout_seconds,
    )
    return {
        "entrypoint": "cluster.fabric-eval",
        "require_management_plane": bool(require_management_plane),
        **result,
    }


def build_cluster_nmx_partition_lab(
    *,
    nmx_url: Optional[str] = None,
    nmx_token: Optional[str] = None,
    alpha_name: str = "AlphaPartition",
    beta_name: str = "BetaPartition",
    alpha_size: int = 4,
    beta_size: int = 4,
) -> Dict[str, Any]:
    from cluster.fabric import build_nmx_partition_lab_payload

    url = (nmx_url or "").strip()
    if not url:
        return {
            "success": False,
            "error": "NMX URL is required. Pass --nmx-url to target the fabric management plane.",
        }

    payload = build_nmx_partition_lab_payload(
        nmx_url=url,
        nmx_token=nmx_token,
        alpha_name=alpha_name,
        beta_name=beta_name,
        alpha_size=int(alpha_size),
        beta_size=int(beta_size),
    )
    return {
        "success": payload.get("status") != "error",
        "entrypoint": "cluster.nmx-partition-lab",
        **payload,
    }


def validate_field_report_requirements(
    *,
    report: Optional[str] = None,
    notes: Optional[str] = None,
    template: Optional[str] = None,
    runbook: Optional[str] = None,
    canonical_run_id: Optional[str] = None,
    allow_run_id: Optional[List[str]] = None,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Run the field-report validator script and return stdout/stderr."""
    root = _repo_root()
    script = root / "cluster" / "scripts" / "validate_field_report_requirements.sh"
    if not script.exists():
        return {"success": False, "error": f"Missing validator script: {script}"}

    cmd: List[str] = ["bash", str(script)]
    if report:
        cmd.extend(["--report", str(report)])
    if notes:
        cmd.extend(["--notes", str(notes)])
    if template:
        cmd.extend(["--template", str(template)])
    if runbook:
        cmd.extend(["--runbook", str(runbook)])
    if canonical_run_id:
        cmd.extend(["--canonical-run-id", str(canonical_run_id)])
    if allow_run_id:
        for rid in allow_run_id:
            if rid:
                cmd.extend(["--allow-run-id", str(rid)])

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    payload: Dict[str, Any] = {"success": bool(success), **result}
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"validator failed (returncode={rc})"
    return payload


def build_canonical_package(
    *,
    canonical_run_id: str,
    comparison_run_ids: Optional[List[str]] = None,
    historical_run_ids: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Materialize a clean, non-destructive cluster package for one canonical run family."""
    canonical_run = (canonical_run_id or "").strip()
    if not canonical_run:
        return {"success": False, "error": "canonical_run_id is required"}

    root = _repo_root()
    cluster_root = _cluster_root()
    script = cluster_root / "scripts" / "build_canonical_package.py"
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    comparison = [rid.strip() for rid in (comparison_run_ids or []) if isinstance(rid, str) and rid.strip()]
    historical = [rid.strip() for rid in (historical_run_ids or []) if isinstance(rid, str) and rid.strip()]

    output_dir_value = (output_dir or "cluster/canonical_package").strip()
    cmd: List[str] = [
        "python3",
        str(script),
        "--canonical-run-id",
        canonical_run,
        "--output-dir",
        output_dir_value,
    ]
    for rid in comparison:
        cmd.extend(["--comparison-run-id", rid])
    for rid in historical:
        cmd.extend(["--historical-run-id", rid])

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    package_root = (root / output_dir_value).resolve()

    payload: Dict[str, Any] = {
        "success": bool(success),
        "canonical_run_id": canonical_run,
        "comparison_run_ids": comparison,
        "historical_run_ids": historical,
        "output_dir": str(package_root),
        "package_readme_path": str(package_root / "README.md"),
        "package_manifest_path": str(package_root / "package_manifest.json"),
        "cleanup_keep_run_ids_path": str(package_root / "cleanup_keep_run_ids.txt"),
        "historical_reference_path": str(package_root / "historical-multinode-reference.md"),
        **result,
    }
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"canonical package build failed (returncode={rc})"
    return payload


def promote_cluster_run(
    *,
    run_id: str,
    label: str = "localhost",
    allow_run_ids: Optional[List[str]] = None,
    publish_report_path: Optional[str] = None,
    publish_notes_path: Optional[str] = None,
    skip_render_localhost_report: bool = False,
    skip_validate_localhost_report: bool = False,
    cleanup: bool = False,
    timeout_seconds: int = 300,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote one run-local result tree into the published cluster package.

    When ``repo_root`` is set (tests, alternate checkouts), run metadata is read from
    ``<repo_root>/cluster/runs/...`` while ``promote_run.py`` is always invoked from this
    repository checkout so temp trees do not need a full ``cluster/scripts`` copy.
    """
    run_id_value = (run_id or "").strip()
    if not run_id_value:
        return {"success": False, "error": "run_id is required"}

    root = Path(repo_root).resolve() if repo_root else _repo_root()
    cluster_root = root / "cluster"
    script = _promote_run_script()
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    publish_report_value = (publish_report_path or "cluster/field-report-localhost.md").strip()
    publish_notes_value = (publish_notes_path or "cluster/field-report-localhost-notes.md").strip()
    allow = [rid.strip() for rid in (allow_run_ids or []) if isinstance(rid, str) and rid.strip()]

    cmd: List[str] = [
        "python3",
        str(script),
        "--run-id",
        run_id_value,
        "--label",
        str(label or "localhost"),
        "--repo-root",
        str(root),
        "--publish-report-path",
        publish_report_value,
        "--publish-notes-path",
        publish_notes_value,
    ]
    for rid in allow:
        cmd.extend(["--allow-run-id", rid])
    if skip_render_localhost_report:
        cmd.append("--skip-render-localhost-report")
    if skip_validate_localhost_report:
        cmd.append("--skip-validate-localhost-report")
    if cleanup:
        cmd.append("--cleanup")

    result = _run_cmd(cmd, cwd=root, timeout_seconds=int(timeout_seconds))
    success = result.get("returncode") == 0
    payload: Dict[str, Any] = {
        "success": bool(success),
        "run_id": run_id_value,
        "label": label or "localhost",
        "repo_root": str(root),
        **_cluster_run_layout(run_id_value, repo_root=root),
        "published_root": str(cluster_root / "published" / "current"),
        "published_structured_dir": str(cluster_root / "published" / "current" / "structured"),
        "published_raw_dir": str(cluster_root / "published" / "current" / "raw"),
        "published_figures_dir": str(cluster_root / "published" / "current" / "figures"),
        "published_reports_dir": str(cluster_root / "published" / "current" / "reports"),
        "published_manifest_path": str(cluster_root / "published" / "current" / "manifest.json"),
        "published_localhost_report_path": str((root / publish_report_value).resolve()),
        "published_localhost_notes_path": str((root / publish_notes_value).resolve()),
        "allow_run_ids": allow,
        **result,
    }
    stdout = result.get("stdout") or ""
    try:
        parsed = json.loads(stdout)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        payload.update(parsed)
    if not success and not payload.get("error"):
        rc = payload.get("returncode")
        payload["error"] = f"promote run failed (returncode={rc})"
    return payload


def watch_cluster_run_for_promotion(
    *,
    run_id: str,
    pid: int,
    label: str = "localhost",
    allow_run_ids: Optional[List[str]] = None,
    publish_report_path: Optional[str] = None,
    publish_notes_path: Optional[str] = None,
    skip_render_localhost_report: bool = False,
    skip_validate_localhost_report: bool = False,
    cleanup: bool = False,
    poll_interval_seconds: float = 30.0,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach a detached watcher that promotes a completed run once required artifacts are present."""
    run_id_value = (run_id or "").strip()
    if not run_id_value:
        return {"success": False, "error": "run_id is required"}
    if not isinstance(pid, int) or pid <= 0:
        return {"success": False, "error": "pid must be a positive integer"}

    root = Path(repo_root).resolve() if repo_root else _repo_root()
    cluster_root = root / "cluster"
    script = _watch_run_for_promotion_script()
    run_dir = cluster_root / "runs" / run_id_value
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}
    if not run_dir.exists():
        return {"success": False, "error": f"Missing run dir: {run_dir}"}

    publish_report_value = (publish_report_path or "cluster/field-report-localhost.md").strip()
    publish_notes_value = (publish_notes_path or "cluster/field-report-localhost-notes.md").strip()
    allow = [rid.strip() for rid in (allow_run_ids or []) if isinstance(rid, str) and rid.strip()]

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    launch_log_path = raw_dir / f"{run_id_value}_postrun_promote_watch.launch.log"
    watch_status_path = raw_dir / f"{run_id_value}_postrun_promote_watch_status.json"

    cmd: List[str] = [
        sys.executable,
        str(script),
        "--repo-root",
        str(root),
        "--run-id",
        run_id_value,
        "--pid",
        str(pid),
        "--label",
        label or "localhost",
        "--publish-report-path",
        publish_report_value,
        "--publish-notes-path",
        publish_notes_value,
        "--poll-interval-seconds",
        str(poll_interval_seconds),
    ]
    if skip_render_localhost_report:
        cmd.append("--skip-render-localhost-report")
    if skip_validate_localhost_report:
        cmd.append("--skip-validate-localhost-report")
    if cleanup:
        cmd.append("--cleanup")
    for rid in allow:
        cmd.extend(["--allow-run-id", rid])

    with launch_log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return {
        "success": True,
        "run_id": run_id_value,
        "repo_root": str(root),
        "watcher_pid": proc.pid,
        "watch_command": cmd,
        "watch_status_path": str(watch_status_path),
        "launch_log_path": str(launch_log_path),
        **_cluster_run_layout(run_id_value, repo_root=root),
    }
