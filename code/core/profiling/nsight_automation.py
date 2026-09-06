#!/usr/bin/env python3
"""Automated Nsight Systems and Nsight Compute profiling for Blackwell.

Provides automated profiling workflows with:
- Metric selection for different workload types
- Batch profiling across multiple configurations
- Report generation with hotspot detection
- Integration with benchmark harness
"""

import argparse
import os
import re
import subprocess
import json
import signal
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence

from core.utils.logger import get_logger
from core.profiling.profiler_config import (
    MINIMAL_METRICS,
    validate_ncu_app_range_capture,
    validate_ncu_replay_mode,
)

logger = get_logger(__name__)


class NsightAutomation:
    """Automated Nsight profiling."""
    
    # Metric sets for different workload types
    METRIC_SETS = {
        'memory_bound': [
            'dram__bytes_read.sum',
            'dram__bytes_write.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
            'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
            'lts__t_sectors_op_read.sum',
            'lts__t_sectors_op_write.sum',
            'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
        ],
        'compute_bound': [
            'sm__cycles_active.avg',
            'sm__cycles_active.sum',
            'sm__pipe_tensor_cycles_active.avg',
            'smsp__inst_executed.avg',
            'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_fp32_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_fp64_pred_on.sum',
        ],
        'tensor_core': [
            'sm__pipe_tensor_cycles_active.avg',
            'sm__pipe_tensor_op_hmma_cycles_active.avg',
            'smsp__inst_executed_pipe_tensor.avg',
            'smsp__sass_thread_inst_executed_op_fp16_pred_on.sum',
            'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum',
        ],
        'communication': [
            'nvlink__bytes_read.sum',
            'nvlink__bytes_write.sum',
            'pcie__bytes_read.sum',
            'pcie__bytes_write.sum',
        ],
        'occupancy': [
            'sm__warps_active.avg.pct_of_peak_sustained_active',
            'sm__maximum_warps_per_active_cycle_pct',
            'achieved_occupancy',
        ],
    }
    
    def __init__(self, output_dir: Path = Path("artifacts/nsight")):
        """Initialize Nsight automation.
        
        Args:
            output_dir: Directory for profiling outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(__file__).resolve().parents[2]
        self.run_cwd = Path(
            os.environ.get("AISP_PROFILE_WORKDIR", str(self.repo_root))
        ).resolve()
        self.last_error: Optional[str] = None
        self.last_run: Dict[str, Any] = {}
        self._ncu_sets_cache: Optional[set[str]] = None
        self._last_resolved_ncu_set: Optional[str] = None
        self._python_startup_stub_dir: Optional[Path] = None
        self._pending_warnings: List[str] = []
        
        # Check availability
        self.nsys_available = self._check_command("nsys")
        self.ncu_available = self._check_command("ncu")
        
        logger.info(f"Nsight Systems: {'✓' if self.nsys_available else '✗'}")
        logger.info(f"Nsight Compute: {'✓' if self.ncu_available else '✗'}")
    
    def _check_command(self, cmd: str) -> bool:
        """Check if command is available."""
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _wait_for_output_artifact(
        self,
        output_path: Path,
        settle_seconds: float = 5.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """Wait briefly for a late-finalizing profiler artifact to appear."""
        deadline = time.monotonic() + max(float(settle_seconds), 0.0)
        warned_os_error = False
        while time.monotonic() <= deadline:
            try:
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
            except OSError as exc:
                if not warned_os_error:
                    self._record_runtime_warning(
                        f"Profiler artifact poll failed for {output_path}; artifact readiness checks may be incomplete",
                        exc=exc,
                    )
                    warned_os_error = True
            time.sleep(max(float(poll_interval), 0.05))
        return False

    def _record_runtime_warning(
        self,
        message: str,
        *,
        exc: Optional[BaseException] = None,
    ) -> str:
        """Persist a profiler warning in logs and structured run metadata."""
        detail = f"{message}: {exc}" if exc is not None else message
        logger.warning(detail)
        if self.last_run:
            self.last_run.setdefault("warnings", []).append(detail)
        else:
            self._pending_warnings.append(detail)
        return detail

    def _begin_run(self, metadata: Dict[str, Any]) -> None:
        """Start a new structured run record and attach any early warnings."""
        self.last_run = dict(metadata)
        if self._pending_warnings:
            self.last_run["warnings"] = list(self._pending_warnings)
            self._pending_warnings.clear()

    def _available_ncu_sets(self) -> set[str]:
        """Return available Nsight Compute set identifiers (best effort)."""
        if self._ncu_sets_cache is not None:
            return set(self._ncu_sets_cache)
        if not self.ncu_available:
            self._ncu_sets_cache = set()
            return set()
        try:
            proc = subprocess.run(
                ["ncu", "--list-sets"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            sets: set[str] = set()
            for raw_line in (proc.stdout or "").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("identifier") or line.startswith("-"):
                    continue
                match = re.match(r"^([A-Za-z0-9._-]+)\s+", line)
                if match:
                    sets.add(match.group(1).lower())
            self._ncu_sets_cache = sets
            return set(sets)
        except Exception as exc:
            # Keep this best-effort; validation falls back to alias defaults.
            self._record_runtime_warning(
                "Failed to enumerate available Nsight Compute metric sets; falling back to alias defaults",
                exc=exc,
            )
            self._ncu_sets_cache = set()
            return set()

    def _resolve_ncu_set(self, metric_set: str) -> str:
        """Resolve user-facing metric-set aliases to an installed NCU --set value."""
        metric_set_norm = str(metric_set or "").strip().lower()
        alias_candidates = {
            "full": ["full"],
            "roofline": ["roofline"],
            # Nsight versions vary: some expose speed-of-light, others expose basic.
            "speed-of-light": ["speed-of-light", "basic"],
            # Prefer `basic` first for minimal runs; it is substantially lower
            # overhead while still providing SpeedOfLight-derived signals.
            "minimal": ["basic", "speed-of-light"],
            "basic": ["basic", "speed-of-light"],
        }
        if metric_set_norm not in alias_candidates:
            allowed = ", ".join(sorted(alias_candidates.keys()))
            raise ValueError(f"Unsupported metric_set={metric_set!r}; expected one of: {allowed}")

        available = self._available_ncu_sets()
        candidates = alias_candidates[metric_set_norm]
        if not available:
            # If discovery fails, use the first candidate so command construction still works.
            resolved = candidates[0]
            self._last_resolved_ncu_set = resolved
            return resolved

        for candidate in candidates:
            if candidate in available:
                self._last_resolved_ncu_set = candidate
                return candidate

        available_display = ", ".join(sorted(available)) if available else "<unknown>"
        raise ValueError(
            f"Requested metric_set={metric_set!r} is not available on this Nsight Compute install. "
            f"Available sets: {available_display}"
        )

    def _normalize_nvtx_includes(self, nvtx_includes: Optional[Sequence[str]]) -> List[str]:
        """Normalize NVTX include filters for better push/pop range compatibility."""
        normalized: List[str] = []
        for raw in nvtx_includes or []:
            tag = str(raw).strip()
            if not tag:
                continue
            candidates = [tag]
            # ncu plain include expressions often target start/end ranges only.
            # Add push/pop-style variant when caller passed a simple range label.
            if (
                "/" not in tag
                and "::" not in tag
                and "@" not in tag
                and not tag.endswith("/")
            ):
                candidates.append(f"{tag}/")
            for candidate in candidates:
                if candidate not in normalized:
                    normalized.append(candidate)
        return normalized

    def _ensure_python_startup_stub(self) -> Path:
        """Materialize a startup-safe `sitecustomize.py` shim for profiling subprocesses."""
        if self._python_startup_stub_dir is not None:
            return self._python_startup_stub_dir
        stub_dir = Path(tempfile.gettempdir()) / "aisp_profile_python_startup"
        stub_dir.mkdir(parents=True, exist_ok=True)
        startup_hooks = {
            "sitecustomize.py": (
                '"""AISP profiling startup shim.\n'
                "\n"
                "Overrides host-level sitecustomize side effects during profiler launches.\n"
                '"""\n'
            ),
            "usercustomize.py": (
                '"""AISP profiling startup shim.\n'
                "\n"
                "Shadows incompatible host-level usercustomize hooks during profiler launches.\n"
                '"""\n'
            ),
        }
        for filename, contents in startup_hooks.items():
            stub_path = stub_dir / filename
            if not stub_path.exists() or stub_path.read_text() != contents:
                stub_path.write_text(contents)
        self._python_startup_stub_dir = stub_dir
        return stub_dir

    def _build_env(
        self,
        force_lineinfo: bool = False,
        extra_env: Optional[Dict[str, str]] = None,
        sanitize_python_startup: bool = True,
    ) -> Dict[str, str]:
        """Build environment with repo root on PYTHONPATH for child commands."""
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        pythonpath_entries: List[str] = [str(self.repo_root)]
        if sanitize_python_startup:
            stub_dir = self._ensure_python_startup_stub()
            pythonpath_entries.insert(0, str(stub_dir))
            # Keep user-site packages out of profiler child processes.
            env.setdefault("PYTHONNOUSERSITE", "1")
        # Do not force NVTE_PROJECT_BUILDING by default. On pinned serving-stack
        # environments this can trigger Transformer Engine ABI/symbol mismatches
        # during vLLM imports in profiler subprocesses.
        force_nvte_project_building = str(
            env.get("AISP_PROFILE_FORCE_NVTE_PROJECT_BUILDING", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if force_nvte_project_building:
            env["NVTE_PROJECT_BUILDING"] = "1"
        else:
            env.pop("NVTE_PROJECT_BUILDING", None)
        existing = env.get("PYTHONPATH", "")
        if existing:
            pythonpath_entries.append(existing)
        deduped_entries: List[str] = []
        seen = set()
        for entry in pythonpath_entries:
            if not entry or entry in seen:
                continue
            seen.add(entry)
            deduped_entries.append(entry)
        env["PYTHONPATH"] = os.pathsep.join(deduped_entries)
        if force_lineinfo:
            def _append_flag(key: str, flag: str) -> None:
                current = env.get(key, "").strip()
                if flag not in current.split():
                    env[key] = f"{flag} {current}".strip()
            _append_flag("NVCC_PREPEND_FLAGS", "-lineinfo")
            _append_flag("TORCH_NVCC_FLAGS", "-lineinfo")
        return env

    @staticmethod
    def _is_startup_sanitizer_issue(error_text: str) -> bool:
        """Detect profiler child failures caused by startup-sanitized Python env."""
        text = str(error_text or "").lower()
        return (
            "error in sitecustomize" in text
            or "error in usercustomize" in text
            or "no module named 'asyncio'" in text
            or "no module named '_posixsubprocess'" in text
            or ("libth_common.so" in text and "pyobjectslotd1ev" in text)
            or "could not find `transformer-engine` pypi package" in text
            or "could not find `transformer-engine` pypi package.".lower() in text
        )
    
    def profile_nsys(
        self,
        command: List[str],
        output_name: str,
        trace_cuda: bool = True,
        trace_nvtx: bool = True,
        trace_osrt: bool = True,
        full_timeline: bool = False,
        trace_forks: bool = False,
        preset: str = "light",
        force_lineinfo: bool = True,
        timeout_seconds: Optional[float] = None,
        wait_mode: str = "primary",
        finalize_grace_seconds: float = 20.0,
        extra_env: Optional[Dict[str, str]] = None,
        sanitize_python_startup: bool = True,
        capture_range_cuda_profiler_api: bool = False,
        cuda_graph_trace: Optional[str] = None,
    ) -> Optional[Path]:
        """Run Nsight Systems profiling.
        
        Args:
            command: Command to profile
            output_name: Base name for output file
            trace_cuda: Trace CUDA API calls
            trace_nvtx: Trace NVTX markers
            trace_osrt: Trace OS runtime
            full_timeline: If True, include driver/cu/pti traces and richer capture flags
            trace_forks: If True, trace child processes before exec
            capture_range_cuda_profiler_api: Capture only between the target's
                cudaProfilerStart()/cudaProfilerStop() calls and flush on stop
            cuda_graph_trace: Optional CUDA Graph trace granularity ("graph" or
                "node"). Leave unset to retain the installed Nsight default.
            wait_mode: NSYS wait mode ('primary' or 'all')
            finalize_grace_seconds: Grace period after SIGINT on timeout
            extra_env: Optional environment overrides for profiled process
            sanitize_python_startup: Prefix a safe `sitecustomize.py` shim on PYTHONPATH
        
        Presets:
            - light (default): cuda,nvtx,osrt, no sampling/ctx switch.
            - full: adds cuda-hw, cublas, cusolver, cusparse, cudnn, fork tracing.

        Returns:
            output_path: Path to .nsys-rep file, or None if failed
        """
        if not self.nsys_available:
            logger.error("Nsight Systems not available")
            return None
        self.last_error = None
        self.last_run = {}
        self._pending_warnings = []
        
        output_path = self.output_dir / f"{output_name}.nsys-rep"
        
        # Build nsys command
        nsys_cmd = [
            'nsys', 'profile',
            '--output', str(output_path),
            '--force-overwrite', 'true',
        ]
        
        trace_categories = []

        # Apply preset overrides first
        preset_normalized = (preset or "light").strip().lower()
        if preset_normalized == "full":
            full_timeline = True
            trace_forks = True
        elif preset_normalized == "light":
            full_timeline = False
        if trace_cuda:
            trace_categories.append('cuda')
        if trace_nvtx:
            trace_categories.append('nvtx')
        if trace_osrt:
            trace_categories.append('osrt')
        if full_timeline:
            trace_categories.extend(['cuda-hw', 'cublas', 'cusolver', 'cusparse', 'cudnn'])
        if trace_categories:
            # dedupe while preserving order
            seen = set()
            deduped = []
            for cat in trace_categories:
                if cat not in seen:
                    seen.add(cat)
                    deduped.append(cat)
            nsys_cmd.extend(['--trace', ",".join(deduped)])
            if full_timeline or preset_normalized == "full":
                logger.warning("NSYS full timeline enabled: traces will be larger and runs slower. Ensure TMPDIR has ample space.")

        # Prefer no sampling/ctx-switch overhead when hunting source attribution
        nsys_cmd.extend([
            '--sample', 'none',
            '--cpuctxsw', 'none',
            '--cuda-memory-usage', 'true',
            '--cuda-um-gpu-page-faults', 'true',
            '--cuda-um-cpu-page-faults', 'true',
        ])
        if trace_forks:
            nsys_cmd.extend(['--trace-fork-before-exec', 'true'])
        if capture_range_cuda_profiler_api:
            nsys_cmd.extend([
                '--capture-range=cudaProfilerApi',
                '--capture-range-end=stop',
            ])
        cuda_graph_trace_norm: Optional[str] = None
        if cuda_graph_trace is not None:
            cuda_graph_trace_norm = str(cuda_graph_trace).strip().lower()
            if cuda_graph_trace_norm not in {"graph", "node"}:
                raise ValueError("cuda_graph_trace must be 'graph', 'node', or None")
            if not trace_cuda:
                raise ValueError("cuda_graph_trace requires trace_cuda=True")
            nsys_cmd.append(f"--cuda-graph-trace={cuda_graph_trace_norm}")
        wait_mode_norm = str(wait_mode or "primary").strip().lower()
        if wait_mode_norm not in {"primary", "all"}:
            raise ValueError("wait_mode must be 'primary' or 'all'")
        nsys_cmd.extend(["--wait", wait_mode_norm])
        
        nsys_cmd.extend(command)
        
        logger.info(f"Running: {' '.join(nsys_cmd)}")
        self._begin_run({
            "tool": "nsys",
            "cmd": nsys_cmd,
            "cwd": str(self.run_cwd),
            "timeout_seconds": timeout_seconds,
            "preset": preset_normalized,
            "capture_range_cuda_profiler_api": bool(capture_range_cuda_profiler_api),
            "cuda_graph_trace": cuda_graph_trace_norm,
            "wait_mode": wait_mode_norm,
            "finalize_grace_seconds": finalize_grace_seconds,
            "sanitize_python_startup": bool(sanitize_python_startup),
        })
        
        try:
            process = subprocess.Popen(
                nsys_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._build_env(
                    force_lineinfo=force_lineinfo,
                    extra_env=extra_env,
                    sanitize_python_startup=sanitize_python_startup,
                ),
                cwd=str(self.run_cwd),
                start_new_session=True,
            )
            result_stdout = ""
            result_stderr = ""
            try:
                result_stdout, result_stderr = process.communicate(
                    timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
                )
            except subprocess.TimeoutExpired as exc:
                partial_stdout = str(exc.stdout or "")
                partial_stderr = str(exc.stderr or "")
                finalize = self._finalize_timed_out_nsys(
                    process,
                    grace_seconds=finalize_grace_seconds,
                )
                result_stdout = partial_stdout + str(finalize.get("stdout", ""))
                result_stderr = partial_stderr + str(finalize.get("stderr", ""))
                self.last_run.update(
                    {
                        "timeout_hit": True,
                        "stdout": result_stdout,
                        "stderr": result_stderr,
                        "graceful_finalize_attempted": True,
                        "graceful_finalize_completed": bool(finalize.get("completed")),
                        "finalize_signals": finalize.get("signals", []),
                        "defunct_launcher_detected": bool(finalize.get("defunct_launcher_detected")),
                        "returncode": process.returncode,
                    }
                )
                report_ready = False
                if bool(finalize.get("completed")) and output_path.exists():
                    report_ready = True
                elif self._wait_for_output_artifact(
                    output_path,
                    settle_seconds=min(max(float(finalize_grace_seconds), 5.0), 30.0),
                ):
                    report_ready = True
                if report_ready:
                    logger.warning(
                        "Nsight Systems timed out but produced a usable report during finalization."
                    )
                    self.last_run["output"] = str(output_path)
                    self.last_error = None
                    return output_path
                self.last_error = f"Nsight Systems timed out after {timeout_seconds}s"
                if finalize.get("defunct_launcher_detected"):
                    self.last_error += " (defunct nsys-launcher detected during finalization)"
                logger.error(self.last_error)
                return None
            result = subprocess.CompletedProcess(
                args=nsys_cmd,
                returncode=process.returncode,
                stdout=result_stdout,
                stderr=result_stderr,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    nsys_cmd,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            if not output_path.exists() and not self._wait_for_output_artifact(
                output_path,
                settle_seconds=10.0,
            ):
                self.last_run.update(
                    {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                        "timeout_hit": False,
                        "graceful_finalize_attempted": False,
                    }
                )
                self.last_error = (
                    "Nsight Systems exited successfully but no report artifact was "
                    f"produced at {output_path}"
                )
                logger.error(self.last_error)
                return None
            logger.info(f"Nsight Systems trace saved to {output_path}")
            self.last_run.update(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "timeout_hit": False,
                    "output": str(output_path),
                    "graceful_finalize_attempted": False,
                }
            )
            return output_path
        except subprocess.CalledProcessError as e:
            # Automatic fallback: drop full_timeline categories and retry once
            self.last_error = e.stderr or e.stdout or str(e)
            logger.error(f"Nsight Systems failed: {self.last_error}")
            if self._wait_for_output_artifact(output_path, settle_seconds=10.0):
                failure_detail = self.last_error or f"returncode={e.returncode}"
                self.last_error = (
                    "Nsight Systems returned non-zero status but produced a report artifact "
                    f"at {output_path}; treating profiler capture as failed. Detail: "
                    f"{failure_detail}"
                )
                logger.warning(self.last_error)
                self.last_run.update(
                    {
                        "stdout": e.output,
                        "stderr": e.stderr,
                        "returncode": e.returncode,
                        "timeout_hit": False,
                        "output": str(output_path),
                        "artifact_on_failure": str(output_path),
                        "graceful_finalize_attempted": False,
                    }
                )
                return None
            if sanitize_python_startup and self._is_startup_sanitizer_issue(self.last_error):
                logger.warning(
                    "Retrying NSYS capture with sanitize_python_startup=False due to "
                    "startup environment incompatibility."
                )
                return self.profile_nsys(
                    command,
                    output_name,
                    trace_cuda=trace_cuda,
                    trace_nvtx=trace_nvtx,
                    trace_osrt=trace_osrt,
                    full_timeline=full_timeline,
                    trace_forks=trace_forks,
                    capture_range_cuda_profiler_api=capture_range_cuda_profiler_api,
                    cuda_graph_trace=cuda_graph_trace_norm,
                    preset=preset_normalized,
                    force_lineinfo=force_lineinfo,
                    timeout_seconds=timeout_seconds,
                    wait_mode=wait_mode_norm,
                    finalize_grace_seconds=finalize_grace_seconds,
                    extra_env=extra_env,
                    sanitize_python_startup=False,
                )
            if full_timeline or preset_normalized == "full":
                logger.warning("Retrying NSYS capture with preset=light (reduced trace categories)")
                return self.profile_nsys(
                    command,
                    output_name,
                    trace_cuda=trace_cuda,
                    trace_nvtx=trace_nvtx,
                    trace_osrt=trace_osrt,
                    full_timeline=False,
                    trace_forks=False,
                    capture_range_cuda_profiler_api=capture_range_cuda_profiler_api,
                    cuda_graph_trace=cuda_graph_trace_norm,
                    preset="light",
                    force_lineinfo=force_lineinfo,
                    timeout_seconds=timeout_seconds,
                    wait_mode=wait_mode_norm,
                    finalize_grace_seconds=finalize_grace_seconds,
                    extra_env=extra_env,
                    sanitize_python_startup=sanitize_python_startup,
                )
            return None

    def _detect_nsys_defunct_launcher(self, parent_pid: int) -> bool:
        """Best-effort detection of defunct nsys-launcher child processes."""
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid,ppid,stat,cmd"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return False
            return self._parse_ps_for_defunct_launcher(result.stdout, parent_pid)
        except Exception as exc:
            self._record_runtime_warning(
                "Failed to inspect process table for defunct nsys-launcher children; timeout diagnostics are incomplete",
                exc=exc,
            )
            return False

    @staticmethod
    def _parse_ps_for_defunct_launcher(ps_output: str, parent_pid: int) -> bool:
        """Parse `ps -eo pid,ppid,stat,cmd` output for defunct nsys-launcher rows."""
        for line in str(ps_output or "").splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            _, ppid, stat, cmd = parts
            if ppid != str(parent_pid):
                continue
            if "nsys-launcher" in cmd and ("<defunct>" in cmd or "Z" in stat):
                return True
        return False

    def _finalize_timed_out_nsys(self, process: subprocess.Popen, grace_seconds: float) -> Dict[str, Any]:
        """Attempt to gracefully finalize NSYS output after timeout."""
        signals_sent: List[str] = []
        stdout_accum = ""
        stderr_accum = ""
        completed = False
        cleanup_warnings: List[str] = []

        def _record_finalize_warning(message: str, exc: BaseException) -> None:
            cleanup_warnings.append(self._record_runtime_warning(message, exc=exc))

        try:
            os.killpg(process.pid, signal.SIGINT)
            signals_sent.append("SIGINT")
        except Exception as exc:
            _record_finalize_warning(
                "Nsight Systems finalization failed while sending SIGINT to the profiler process group",
                exc,
            )

        try:
            stdout_chunk, stderr_chunk = process.communicate(timeout=max(1.0, grace_seconds))
            stdout_accum += stdout_chunk or ""
            stderr_accum += stderr_chunk or ""
            completed = True
        except subprocess.TimeoutExpired as exc:
            stdout_accum += str(exc.stdout or "")
            stderr_accum += str(exc.stderr or "")

        if not completed:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                signals_sent.append("SIGTERM")
            except Exception as exc:
                _record_finalize_warning(
                    "Nsight Systems finalization failed while sending SIGTERM to the profiler process group",
                    exc,
                )
            try:
                stdout_chunk, stderr_chunk = process.communicate(timeout=5)
                stdout_accum += stdout_chunk or ""
                stderr_accum += stderr_chunk or ""
                completed = True
            except subprocess.TimeoutExpired as exc:
                stdout_accum += str(exc.stdout or "")
                stderr_accum += str(exc.stderr or "")

        if not completed:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                signals_sent.append("SIGKILL")
            except Exception as exc:
                _record_finalize_warning(
                    "Nsight Systems finalization failed while sending SIGKILL to the profiler process group",
                    exc,
                )
            try:
                stdout_chunk, stderr_chunk = process.communicate(timeout=2)
                stdout_accum += stdout_chunk or ""
                stderr_accum += stderr_chunk or ""
            except Exception as exc:
                _record_finalize_warning(
                    "Nsight Systems finalization failed while collecting trailing profiler output after SIGKILL",
                    exc,
                )

        defunct_detected = self._detect_nsys_defunct_launcher(process.pid)
        return {
            "completed": completed,
            "stdout": stdout_accum,
            "stderr": stderr_accum,
            "signals": signals_sent,
            "defunct_launcher_detected": defunct_detected,
            "warnings": cleanup_warnings,
        }
    
    def build_ncu_command(
        self,
        *,
        command: List[str],
        output_path: Path,
        workload_type: str = 'memory_bound',
        kernel_filter: Optional[str] = None,
        kernel_name_base: Optional[str] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
        profile_from_start: Optional[str] = None,
        sampling_interval: Optional[int] = None,
        metric_set: str = 'full',
        launch_skip: Optional[int] = None,
        launch_count: Optional[int] = None,
        replay_mode: str = 'application',
    ) -> List[str]:
        """Build the Nsight Compute command without executing it."""
        if workload_type not in self.METRIC_SETS:
            raise ValueError(f"Unsupported workload_type: {workload_type}")
        metrics = self.METRIC_SETS[workload_type]
        ncu_set = self._resolve_ncu_set(metric_set)
        replay_mode = validate_ncu_replay_mode(replay_mode)
        nvtx_filters = (
            list(dict.fromkeys(str(tag).strip() for tag in nvtx_includes or [] if str(tag).strip()))
            if replay_mode == 'app-range'
            else self._normalize_nvtx_includes(nvtx_includes)
        )
        if replay_mode == 'app-range':
            if ncu_set != 'basic':
                raise ValueError("app-range requires metric_set='minimal' or 'basic'.")
            metrics = MINIMAL_METRICS
            validate_ncu_app_range_capture(
                nvtx_includes=nvtx_filters,
                metrics=metrics,
                kernel_filter=kernel_filter,
                launch_skip=launch_skip,
                launch_count=launch_count,
                sampling_interval=sampling_interval,
                profile_from_start=profile_from_start,
            )
        ncu_cmd = [
            'ncu',
            '--set', ncu_set,
            '--target-processes', 'all',
            '--export', str(output_path),
            '--force-overwrite',
        ]
        if replay_mode:
            ncu_cmd.extend(['--replay-mode', replay_mode])
        # Only add custom metrics when using the full set; other sets bring their own.
        if metrics and (ncu_set == 'full' or replay_mode == 'app-range'):
            ncu_cmd.extend(['--metrics', ",".join(metrics)])
        if kernel_filter:
            if kernel_name_base:
                ncu_cmd.extend(['--kernel-name-base', str(kernel_name_base)])
            ncu_cmd.extend(['--kernel-name', kernel_filter])
        if nvtx_filters:
            ncu_cmd.append('--nvtx')
            for tag in nvtx_filters:
                ncu_cmd.extend(['--nvtx-include', tag])
        if profile_from_start:
            normalized = str(profile_from_start).strip().lower()
            if normalized not in {'on', 'off'}:
                raise ValueError("profile_from_start must be 'on' or 'off'")
            ncu_cmd.extend(['--profile-from-start', normalized])
        if launch_skip is not None:
            ncu_cmd.extend(['--launch-skip', str(launch_skip)])
        if launch_count is not None:
            ncu_cmd.extend(['--launch-count', str(launch_count)])
        if sampling_interval:
            ncu_cmd.extend(['--pm-sampling-interval', str(sampling_interval)])
        ncu_cmd.extend(command)
        return ncu_cmd

    def profile_ncu(
        self,
        command: List[str],
        output_name: str,
        workload_type: str = 'memory_bound',
        kernel_filter: Optional[str] = None,
        kernel_name_base: Optional[str] = None,
        nvtx_includes: Optional[Sequence[str]] = None,
        profile_from_start: Optional[str] = None,
        force_lineinfo: bool = True,
        timeout_seconds: Optional[float] = None,
        sampling_interval: Optional[int] = None,
        metric_set: str = 'full',
        launch_skip: Optional[int] = None,
        launch_count: Optional[int] = None,
        replay_mode: str = 'application',
        sanitize_python_startup: bool = True,
    ) -> Optional[Path]:
        """Run Nsight Compute profiling.
        
        Args:
            command: Command to profile
            output_name: Base name for output file
            workload_type: Type of workload for metric selection
            kernel_filter: Optional kernel name filter (auto-limits launches when set)
            kernel_name_base: Optional kernel-name base mode for filter matching
            nvtx_includes: Optional NVTX range include filters (requires NVTX ranges in target code)
            profile_from_start: Optional NCU profiling gate ('on' or 'off')
            sampling_interval: pm-sampling-interval value (cycles between samples)
            metric_set: Metric set to use ('full', 'roofline', 'minimal', 'speed-of-light', 'basic')
            launch_skip: Number of kernel launches to skip before profiling
            launch_count: Number of kernel launches to profile (None = all remaining)
            replay_mode: 'application', 'kernel', or 'app-range' (one full NVTX range, minimal metrics)
        
        Returns:
            output_path: Path to .ncu-rep file, or None if failed
        """
        replay_mode = validate_ncu_replay_mode(replay_mode)
        if not self.ncu_available:
            logger.error("Nsight Compute not available")
            return None
        self.last_error = None
        self.last_run = {}
        self._pending_warnings = []
        self._last_resolved_ncu_set = None

        output_path = self.output_dir / f"{output_name}.ncu-rep"
        if kernel_filter:
            # Auto-limit when a kernel filter is specified to avoid timeouts
            # on workloads with many launches; caller can override explicitly.
            if launch_count is None:
                launch_count = 1
        ncu_cmd = self.build_ncu_command(
            command=command,
            output_path=output_path,
            workload_type=workload_type,
            kernel_filter=kernel_filter,
            kernel_name_base=kernel_name_base,
            nvtx_includes=nvtx_includes,
            profile_from_start=profile_from_start,
            sampling_interval=sampling_interval,
            metric_set=metric_set,
            launch_skip=launch_skip,
            launch_count=launch_count,
            replay_mode=replay_mode,
        )
        
        logger.info(f"Running: {' '.join(ncu_cmd[:6])} ...")
        self._begin_run({
            "tool": "ncu",
            "cmd": ncu_cmd,
            "cwd": str(self.run_cwd),
            "timeout_seconds": timeout_seconds,
            "workload_type": workload_type,
            "sampling_interval": sampling_interval,
            "metric_set": metric_set,
            "metric_set_resolved": self._last_resolved_ncu_set,
            "launch_skip": launch_skip,
            "launch_count": launch_count,
            "replay_mode": replay_mode,
            "kernel_filter": kernel_filter,
            "kernel_name_base": kernel_name_base,
            "nvtx_includes": list(nvtx_includes or []),
            "profile_from_start": profile_from_start,
        })
        capture_started_ns = time.time_ns()
        
        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
                env=self._build_env(
                    force_lineinfo=force_lineinfo,
                    sanitize_python_startup=sanitize_python_startup,
                ),
                cwd=str(self.run_cwd),
            )
            logger.info(f"Nsight Compute report saved to {output_path}")
            self.last_run.update(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "timeout_hit": False,
                    "output": str(output_path),
                }
            )
            output_text = f"{result.stdout}\n{result.stderr}"
            capture_errors: List[str] = []
            if "No kernels were profiled." in output_text:
                capture_errors.append("Nsight Compute captured zero kernels.")
            if "No metrics to collect found in sections." in output_text:
                capture_errors.append(
                    "Nsight Compute did not collect metrics for the selected --set/pages."
                )
            if not output_path.exists():
                capture_errors.append(f"Expected report file missing: {output_path}")
            elif replay_mode == 'app-range' and (
                not output_path.is_file()
                or output_path.stat().st_mtime_ns < capture_started_ns
            ):
                capture_errors.append("Nsight Compute app-range report is not fresh for this capture.")
            if capture_errors:
                hint_lines: List[str] = []
                if nvtx_includes:
                    hint_lines.append(
                        "NVTX include filters were set. Verify filter syntax and use "
                        "--profile-from-start off with cudaProfilerStart/Stop around the target region."
                    )
                if kernel_filter:
                    hint_lines.append(
                        "Kernel filter was set. Validate the filter against 'Available Kernels' shown by ncu output."
                    )
                if self._last_resolved_ncu_set:
                    hint_lines.append(f"Resolved metric set: {self._last_resolved_ncu_set}")
                self.last_error = " ".join(capture_errors + hint_lines)
                logger.error(f"Nsight Compute capture invalid: {self.last_error}")
                return None
            if replay_mode == 'app-range':
                from core.profiling.metrics_extractor import inspect_ncu_app_range_report

                try:
                    range_metrics, capture = inspect_ncu_app_range_report(
                        output_path,
                        expected_nvtx_range=ncu_cmd[ncu_cmd.index('--nvtx-include') + 1],
                        requested_metrics=MINIMAL_METRICS,
                    )
                except ValueError as exc:
                    self.last_error = f"Nsight Compute app-range capture invalid: {exc}"
                    logger.error(self.last_error)
                    return None
                self.last_run.update({"ncu_capture": capture, "ncu_metrics": range_metrics.to_dict()})
                output_path.with_suffix('.capture.json').write_text(
                    json.dumps(capture, indent=2, sort_keys=True) + '\n', encoding='utf-8'
                )
            return output_path
        except subprocess.TimeoutExpired as e:
            self.last_error = f"Nsight Compute timed out after {timeout_seconds}s"
            self.last_run.update(
                {
                    "timeout_hit": True,
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or "",
                }
            )
            logger.error(self.last_error)
            return None
        except subprocess.CalledProcessError as e:
            self.last_error = e.stderr or e.stdout or str(e)
            logger.error(f"Nsight Compute failed: {self.last_error}")
            if sanitize_python_startup and self._is_startup_sanitizer_issue(self.last_error):
                logger.warning(
                    "Retrying NCU capture with sanitize_python_startup=False due to "
                    "startup environment incompatibility."
                )
                return self.profile_ncu(
                    command,
                    output_name,
                    workload_type=workload_type,
                    kernel_filter=kernel_filter,
                    kernel_name_base=kernel_name_base,
                    nvtx_includes=nvtx_includes,
                    profile_from_start=profile_from_start,
                    force_lineinfo=force_lineinfo,
                    timeout_seconds=timeout_seconds,
                    sampling_interval=sampling_interval,
                    metric_set=metric_set,
                    launch_skip=launch_skip,
                    launch_count=launch_count,
                    replay_mode=replay_mode,
                    sanitize_python_startup=False,
                )
            return None
    
    def batch_profile(
        self,
        configs: List[Dict[str, Any]],
        base_command: List[str]
    ) -> List[Path]:
        """Run batch profiling with multiple configurations.
        
        Args:
            configs: List of config dicts with keys:
                - name: Output name
                - args: Additional command arguments
                - workload_type: Type for metric selection
            base_command: Base command (e.g., ['python', 'script.py'])
        
        Returns:
            output_paths: List of generated report paths
        """
        outputs = []
        
        for config in configs:
            name = config['name']
            args = config.get('args', [])
            workload_type = config.get('workload_type', 'memory_bound')
            
            # Build full command
            full_cmd = base_command + args
            
            logger.info(f"Profiling configuration: {name}")
            
            # Run Nsight Compute
            ncu_path = self.profile_ncu(
                full_cmd,
                f"{name}_ncu",
                workload_type=workload_type
            )
            
            if ncu_path:
                outputs.append(ncu_path)
        
        logger.info(f"Batch profiling complete: {len(outputs)} reports generated")
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Automated Nsight Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile with Nsight Systems
  python nsight_automation.py --tool nsys --output my_trace \\
    -- python ch01/optimized_performance.py
  
  # Profile with Nsight Compute (memory-bound)
  python nsight_automation.py --tool ncu --output my_profile \\
    --workload-type memory_bound -- python ch07/optimized_hbm3ecopy.py
  
  # Batch profiling
  python nsight_automation.py --batch-config configs.json
        """
    )
    
    parser.add_argument('--tool', type=str, choices=['nsys', 'ncu'],
                       help='Profiling tool')
    parser.add_argument('--output', type=str, required=True,
                       help='Output base name')
    parser.add_argument('--workload-type', type=str,
                       choices=list(NsightAutomation.METRIC_SETS.keys()),
                       default='memory_bound',
                       help='Workload type for metric selection')
    parser.add_argument('--kernel-filter', type=str,
                       help='Filter kernels by name pattern')
    parser.add_argument('--trace-cuda', action='store_true', default=True, help='Trace CUDA API (nsys)')
    parser.add_argument('--trace-nvtx', action='store_true', default=True, help='Trace NVTX ranges (nsys)')
    parser.add_argument('--trace-osrt', action='store_true', default=True, help='Trace OS runtime (nsys)')
    parser.add_argument('--full-timeline', action='store_true', default=False, help='Enable richer NSYS tracing (cuda-hw, cublas, cusolver, cusparse, cudnn)')
    parser.add_argument('--trace-forks', action='store_true', default=False, help='Trace child processes before exec (nsys)')
    parser.add_argument('--preset', type=str, default='light', choices=['light', 'full'],
                        help='NSYS preset: light (default, smaller/faster traces) or full (adds cuda-hw/cublas/cusolver/cusparse/cudnn and fork tracing)')
    parser.add_argument('--wait-mode', type=str, default='primary', choices=['primary', 'all'],
                        help="NSYS --wait mode (default: primary)")
    parser.add_argument('--finalize-grace-seconds', type=float, default=20.0,
                        help='Grace period after timeout SIGINT to let NSYS finalize report')
    parser.add_argument('--batch-config', type=Path,
                       help='JSON config for batch profiling')
    parser.add_argument('--timeout-seconds', type=float, default=None, help='Max runtime before aborting capture (seconds)')
    parser.add_argument('--force-lineinfo/--no-force-lineinfo', default=True, help='Force -lineinfo in NVCC/TORCH_NVCC_FLAGS for source mapping (default: on)')
    parser.add_argument('command', nargs='*',
                       help='Command to profile (after --)')
    
    args = parser.parse_args()
    
    # Create automation
    automation = NsightAutomation()
    
    # Batch mode
    if args.batch_config:
        with open(args.batch_config) as f:
            configs = json.load(f)
        
        outputs = automation.batch_profile(
            configs=configs['profiles'],
            base_command=configs['base_command']
        )
        
        print(f"\n{'='*60}")
        print(f"Batch Profiling Complete")
        print(f"{'='*60}")
        print(f"Reports generated: {len(outputs)}")
        for path in outputs:
            print(f"  - {path}")
        print(f"{'='*60}\n")
        return
    
    # Single profile mode
    if not args.command:
        parser.error("Command required (use -- before command)")
    
    if args.tool == 'nsys':
        output = automation.profile_nsys(
            args.command,
            args.output,
            trace_cuda=args.trace_cuda,
            trace_nvtx=args.trace_nvtx,
            trace_osrt=args.trace_osrt,
            full_timeline=args.full_timeline,
            trace_forks=args.trace_forks,
            preset=args.preset,
            force_lineinfo=args.force_lineinfo,
            timeout_seconds=args.timeout_seconds,
            wait_mode=args.wait_mode,
            finalize_grace_seconds=args.finalize_grace_seconds,
        )
    elif args.tool == 'ncu':
        output = automation.profile_ncu(
            args.command,
            args.output,
            workload_type=args.workload_type,
            kernel_filter=args.kernel_filter,
            force_lineinfo=args.force_lineinfo,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        parser.error("--tool required")
    
    if output:
        print(f"\n{'='*60}")
        print(f"Profiling Complete")
        print(f"{'='*60}")
        print(f"Output: {output}")
        if args.tool == 'nsys':
            print(f"Preset: {args.preset} (full_timeline={args.full_timeline or args.preset=='full'})")
            if args.preset == 'full' or args.full_timeline:
                print("Warning: NSYS full timeline enabled; captures run longer and produce larger traces. Ensure TMPDIR has space.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
