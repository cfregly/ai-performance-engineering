"""End-to-end benchmark orchestration for tier1, full sweep, and cluster eval."""

from __future__ import annotations

import contextlib
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import getpass
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

try:
    import fcntl  # POSIX-only; optional for portability.
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None  # type: ignore[assignment]

from core.benchmark.artifact_manager import build_run_id, validate_run_id
from core.benchmark.expectations import detect_expectation_key
from core.benchmark.run_manifest import get_git_info
from core.discovery import (
    chapter_slug,
    discover_all_chapters,
    discover_benchmarks,
    is_cuda_binary_benchmark_file,
)
from core.harness.progress import ProgressEvent, ProgressRecorder
from core.harness.validity_checks import detect_execution_environment
from core.harness.validity_profile import normalize_validity_profile

_STAGE_PROGRESS_POLL_SECONDS = 2.0
_STATE_HEARTBEAT_SECONDS = 60.0
_E2E_WATCHER_POLL_SECONDS = 15
_E2E_WATCHER_MAX_AUTO_RESUMES = 3
_E2E_WATCHER_SUPERVISED_ENV = "AISP_E2E_WATCHER_SUPERVISED"
_E2E_CONTRACT_SCHEMA_VERSION = "1.0"
_REPO_ROOT_LOCATOR = "<repo-root>"
_BENCH_ROOT_LOCATOR = "<bench-root>"
_ARTIFACTS_DIR_LOCATOR = "<artifacts-dir>"
_LOG_FILE_LOCATOR = "<log-file>"
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class _E2EAbort(BaseException):
    """Control-flow abort for the top-level e2e orchestrator.

    This intentionally subclasses BaseException so nested benchmark/harness
    code that uses broad ``except Exception`` cleanup paths cannot accidentally
    convert a process-level abort signal into an ordinary benchmark failure.
    """


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validated_clean_git_commit(
    *,
    repo_root: Path,
    git_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return the exact clean commit after an independent porcelain-status probe."""

    info = dict(git_info or get_git_info())
    commit = str(info.get("commit") or "").strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        return None, "Git provenance is missing a valid 40-hex commit"
    if info.get("dirty") is not False:
        return None, "Git provenance reports a dirty worktree"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(Path(repo_root).resolve()),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Independent Git status probe timed out"
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return None, f"Independent Git status probe failed: {type(exc).__name__}"
    if status.returncode != 0:
        return None, f"Independent Git status probe exited with code {status.returncode}"
    if str(status.stdout or "").strip():
        return None, "Independent Git status probe found a dirty worktree"
    return commit, None


def e2e_runs_root(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root or _repo_root()) / "artifacts" / "e2e_runs"


def e2e_run_dir(run_id: str, repo_root: Optional[Path] = None) -> Path:
    return e2e_runs_root(repo_root) / validate_run_id(run_id)


def e2e_progress_path(run_dir: Path) -> Path:
    return Path(run_dir) / "progress.json"


def e2e_checkpoint_path(run_dir: Path) -> Path:
    return Path(run_dir) / "checkpoint.json"


def e2e_watcher_status_path(run_dir: Path) -> Path:
    return Path(run_dir) / "watcher_status.json"


def e2e_watcher_launch_log_path(run_dir: Path, run_id: str) -> Path:
    return Path(run_dir) / f"{run_id}_watcher.launch.log"


def _watch_e2e_sweep_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "watch_e2e_sweep.py"


def resolve_e2e_run_id(run_id: Optional[str] = None, *, repo_root: Optional[Path] = None) -> str:
    if run_id and str(run_id).strip():
        return validate_run_id(str(run_id).strip())
    return validate_run_id(build_run_id("benchmark_e2e_sweep", base_dir=e2e_runs_root(repo_root)))


def resolve_latest_e2e_run_id(*, repo_root: Optional[Path] = None) -> Optional[str]:
    root = e2e_runs_root(repo_root)
    candidates = [path for path in root.iterdir()] if root.exists() else []
    candidates = [path for path in candidates if path.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return latest.name


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_utc_timestamp(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _age_seconds(value: Optional[str]) -> Optional[float]:
    parsed = _parse_utc_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _pid_is_live(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _append_event(events_path: Path, event: str, **fields: Any) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _utc_now(), "event": event, **_sanitize_persisted_value(fields)}
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _queue_lock_timeout_seconds() -> int:
    raw = os.environ.get("AISP_MCP_QUEUE_RUNNER_LOCK_TIMEOUT_SEC", "1800")
    try:
        value = int(raw)
    except Exception:
        value = 1800
    return max(0, value)


@contextlib.contextmanager
def _benchmark_queue_lock(stage_name: str, run_id: str, *, repo_root: Optional[Path] = None):
    if fcntl is None:
        yield
        return

    root = Path(repo_root or _repo_root())
    queue_dir = root / "artifacts" / "parallel_runs"
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / "queue.runner.lock"
    lock_path.touch(exist_ok=True)
    timeout = _queue_lock_timeout_seconds()
    started = time.monotonic()
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if timeout > 0 and (time.monotonic() - started) >= timeout:
                    raise RuntimeError(
                        f"Benchmark queue lock timeout after {timeout}s "
                        f"(stage={stage_name}, run_id={run_id})"
                    )
                time.sleep(1.0)
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "ts": _utc_now(),
                    "stage": stage_name,
                    "run_id": run_id,
                    "pid": os.getpid(),
                }
            )
            + "\n"
        )
        handle.flush()
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            handle.close()


def _is_local_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True

    local_hostnames = {
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
    }
    local_hostnames.update({name.split(".", 1)[0] for name in local_hostnames})
    return normalized in local_hostnames


def _normalize_cluster_hosts_and_labels(
    *,
    hosts: Optional[List[str]],
    labels: Optional[List[str]],
    ssh_user: Optional[str],
    ssh_key: Optional[str],
) -> Dict[str, Any]:
    normalized_hosts = [str(host).strip() for host in (hosts or []) if str(host).strip()]
    if not normalized_hosts:
        normalized_hosts = ["localhost"]

    normalized_labels = [str(label).strip() for label in (labels or []) if str(label).strip()]
    local_only = all(_is_local_host(host) for host in normalized_hosts)
    if not normalized_labels:
        if local_only:
            normalized_labels = ["localhost"] * len(normalized_hosts)
        else:
            normalized_labels = list(normalized_hosts)
    if len(normalized_labels) != len(normalized_hosts):
        raise ValueError("labels must match hosts count")

    effective_ssh_user = ssh_user
    if local_only and not effective_ssh_user:
        effective_ssh_user = getpass.getuser()
    if not local_only and (not ssh_user or not ssh_key):
        raise ValueError("Non-local hosts require explicit ssh_user and ssh_key for run-e2e.")

    return {
        "hosts": normalized_hosts,
        "labels": normalized_labels,
        "ssh_user": effective_ssh_user,
        "ssh_key": ssh_key,
        "local_only": local_only,
    }


def _with_e2e_cluster_extra_args(extra_args: Optional[List[str]]) -> Optional[List[str]]:
    merged = [str(arg) for arg in (extra_args or []) if str(arg).strip()]
    for raw_arg in merged:
        normalized = raw_arg.strip()
        if any(
            normalized == flag
            or normalized.startswith(f"{flag}=")
            or normalized.startswith(f"{flag} ")
            for flag in _SENSITIVE_EXTRA_ARG_FLAGS
        ):
            raise ValueError(
                "Credential flags are not allowed in extra_cluster_args. "
                "Use the named credential options instead."
            )
    render_flags = {"--render-localhost-report", "--skip-render-localhost-report"}
    if not any(flag in render_flags for flag in merged):
        merged.append("--skip-render-localhost-report")
    return merged or None


def _visible_gpu_count(*, single_gpu: bool) -> int:
    if single_gpu:
        return 1

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip():
        devices = [
            token.strip() for token in visible.split(",") if token.strip() and token.strip() != "-1"
        ]
        if devices:
            return len(devices)

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _validate_expectation_policy(
    *,
    validity_profile: str,
    allow_portable_expectations_update: bool,
    update_expectations: bool,
    accept_regressions: bool,
    allow_mixed_provenance: bool,
) -> Optional[str]:
    if validity_profile != "portable" or allow_portable_expectations_update:
        return None
    requested = []
    if update_expectations:
        requested.append("--update-expectations")
    if accept_regressions:
        requested.append("--accept-regressions")
    if allow_mixed_provenance:
        requested.append("--allow-mixed-provenance")
    if not requested:
        return None
    requested_summary = ", ".join(requested)
    return (
        "Invalid flag combination: "
        f"{requested_summary} requested with --validity-profile portable. "
        "Portable validity profile disables expectation writes by default. "
        "Add --allow-portable-expectations-update to enable writes in portable mode, "
        "or use --validity-profile strict."
    )


def _iter_discovered_targets(active_bench_root: Path) -> List[Dict[str, Any]]:
    from core.benchmark.bench_commands import _collect_benchmark_routing, _expectation_example_key

    discovered: Dict[str, Dict[str, Any]] = {}
    chapter_dirs = discover_all_chapters(active_bench_root, bench_roots=[active_bench_root])
    for chapter_dir in chapter_dirs:
        chapter_id = chapter_slug(chapter_dir, active_bench_root, bench_root=active_bench_root)
        benchmark_routing = _collect_benchmark_routing(chapter_dir)
        for baseline_path, _optimized_paths, example_name in discover_benchmarks(chapter_dir):
            bench_type = "cuda" if is_cuda_binary_benchmark_file(baseline_path) else "python"
            example_key = _expectation_example_key(example_name, bench_type)
            target = f"{chapter_id}:{example_name}"
            routing = benchmark_routing.get(example_key)
            minimum_gpu_count = int(routing.minimum_gpu_count if routing else 1)
            requires_torchrun = bool(routing.requires_torchrun if routing else False)
            discovered.setdefault(
                target,
                {
                    "target": target,
                    "chapter": chapter_id,
                    "example": example_name,
                    "bench_type": bench_type,
                    "multi_gpu": minimum_gpu_count >= 2,
                    "minimum_gpu_count": minimum_gpu_count,
                    "requires_torchrun": requires_torchrun,
                },
            )
    return [discovered[key] for key in sorted(discovered)]


def discover_benchmark_e2e_inventory(bench_root: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = _repo_root()
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
    discovered = _iter_discovered_targets(active_bench_root)
    single_gpu_targets = sorted(
        entry["target"] for entry in discovered if not entry.get("multi_gpu")
    )
    multi_gpu_targets = sorted(entry["target"] for entry in discovered if entry.get("multi_gpu"))
    return {
        "generated_at": _utc_now(),
        "bench_root_identity": _bench_root_identity(active_bench_root),
        "targets": discovered,
        "single_gpu": single_gpu_targets,
        "multi_gpu": multi_gpu_targets,
        "counts": {
            "total": len(discovered),
            "single_gpu": len(single_gpu_targets),
            "multi_gpu": len(multi_gpu_targets),
        },
    }


def _invoke_run_tier1_suite(**kwargs: Any) -> Dict[str, Any]:
    from core.benchmark.suites.tier1 import run_tier1_suite

    return run_tier1_suite(**kwargs)


def _invoke_execute_benchmarks(**kwargs: Any) -> Dict[str, Any]:
    from core.benchmark.bench_commands import _execute_benchmarks

    return _execute_benchmarks(**kwargs)


def _invoke_run_cluster_common_eval(**kwargs: Any) -> Dict[str, Any]:
    from core.cluster import run_cluster_common_eval

    return run_cluster_common_eval(**kwargs)


def _invoke_run_cluster_fabric_eval(**kwargs: Any) -> Dict[str, Any]:
    from core.cluster import run_cluster_fabric_eval

    return run_cluster_fabric_eval(**kwargs)


def _result_path_exists(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _read_json_object_artifact(
    path_value: Any,
    *,
    artifact_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    path = Path(str(path_value or ""))
    if not _result_path_exists(str(path)):
        return None, f"missing or empty required artifact: {artifact_name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"failed to parse {artifact_name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid {artifact_name}: expected a JSON object"
    return payload, None


def _configured_tier1_history_root(repo_root: Path) -> Path:
    config_path = Path(repo_root).resolve() / "configs" / "benchmark_suites" / "tier1.yaml"
    if not config_path.is_file():
        return Path(repo_root).resolve() / "artifacts" / "history" / "tier1"
    from core.benchmark.suites.tier1 import load_tier1_suite

    configured = Path(load_tier1_suite(config_path).history_root)
    if configured.is_absolute():
        return configured.resolve()
    return (Path(repo_root).resolve() / configured).resolve()


def _require_exact_file_path(
    path_value: Any,
    *,
    expected_path: Path,
    artifact_name: str,
) -> Optional[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return f"missing required artifact path: {artifact_name}"
    candidate = Path(raw)
    try:
        candidate_resolved = candidate.resolve(strict=True)
        expected_resolved = Path(expected_path).resolve(strict=True)
    except OSError:
        return f"missing required artifact: {artifact_name}"
    if candidate.is_symlink() or candidate_resolved != expected_resolved:
        return f"invalid {artifact_name}: path is not the expected run artifact"
    if not candidate_resolved.is_file() or candidate_resolved.stat().st_size <= 0:
        return f"missing or empty required artifact: {artifact_name}"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_manifest_identity_error(
    manifest: Dict[str, Any],
    *,
    expected_run_id: str,
    expected_git_commit: str,
) -> Optional[str]:
    if manifest.get("run_id") != expected_run_id:
        return "benchmark manifest run_id does not match the owning attempt"
    git_payload = manifest.get("git")
    if not isinstance(git_payload, dict):
        return "benchmark manifest is missing Git provenance"
    if str(git_payload.get("commit") or "") != expected_git_commit:
        return "benchmark manifest Git commit does not match the E2E commit"
    if git_payload.get("dirty") is not False:
        return "benchmark manifest Git provenance is dirty"
    entries = manifest.get("manifests")
    if not isinstance(entries, list) or not entries:
        return "benchmark manifest has no per-benchmark evidence"
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("manifest"), dict):
            return f"benchmark manifest entry {index} is malformed"
        entry_git = entry["manifest"].get("git")
        if not isinstance(entry_git, dict):
            return f"benchmark manifest entry {index} is missing Git provenance"
        if str(entry_git.get("commit") or "") != expected_git_commit:
            return f"benchmark manifest entry {index} has a different Git commit"
        if entry_git.get("dirty") is not False:
            return f"benchmark manifest entry {index} has dirty Git provenance"
    return None


def _validate_benchmark_run_manifest(
    *,
    expected_run_id: str,
    repo_root: Path,
    artifacts_dir: Optional[str],
    expected_git_commit: str,
) -> Optional[str]:
    expected_run_dir = _benchmark_run_dir(
        expected_run_id,
        repo_root=Path(repo_root).resolve(),
        artifacts_dir=artifacts_dir,
    )
    expected_path = expected_run_dir / "manifest.json"
    path_error = _require_exact_file_path(
        expected_path,
        expected_path=expected_path,
        artifact_name="benchmark manifest",
    )
    if path_error:
        return path_error
    payload, error = _read_json_object_artifact(
        expected_path,
        artifact_name="benchmark manifest",
    )
    if error:
        return error
    identity_error = _benchmark_manifest_identity_error(
        payload,
        expected_run_id=expected_run_id,
        expected_git_commit=expected_git_commit,
    )
    if identity_error:
        return identity_error
    output_path = expected_run_dir / "results" / "benchmark_test_results.json"
    output, output_error = _read_json_object_artifact(
        output_path,
        artifact_name="benchmark output JSON",
    )
    if output_error:
        return output_error
    if output.get("run_id") != expected_run_id:
        return "benchmark output run_id does not match the owning attempt"
    return None


def _validate_tier1_artifacts(
    result: Dict[str, Any],
    *,
    expected_run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    artifacts_dir: Optional[str] = None,
    expected_git_commit: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    binding_requested = any(
        value is not None for value in (expected_run_id, repo_root, expected_git_commit)
    )
    if binding_requested and (not expected_run_id or repo_root is None or not expected_git_commit):
        return None, None, "invalid Tier-1 evidence-binding context"

    expected_paths: Dict[str, Path] = {}
    if binding_requested:
        execution = result.get("execution")
        if not isinstance(execution, dict) or execution.get("run_id") != expected_run_id:
            return None, None, "Tier-1 execution run_id does not match the owning attempt"
        try:
            history_root = _configured_tier1_history_root(Path(repo_root))
        except Exception as exc:
            return None, None, f"failed to resolve configured Tier-1 history root: {exc}"
        raw_history_root = str(result.get("history_root") or "").strip()
        if not raw_history_root:
            return None, None, "Tier-1 result is missing history_root"
        try:
            if Path(raw_history_root).resolve(strict=True) != history_root.resolve(strict=True):
                return None, None, "Tier-1 history_root does not match the configured root"
        except OSError:
            return None, None, "Tier-1 history_root is missing"
        run_history_root = history_root / str(expected_run_id)
        if history_root.is_symlink() or run_history_root.is_symlink():
            return None, None, "Tier-1 history path may not be a symlink"
        expected_paths = {
            "summary_path": run_history_root / "summary.json",
            "regression_summary_path": run_history_root / "regression_summary.md",
            "regression_json_path": run_history_root / "regression_summary.json",
            "trend_snapshot_path": run_history_root / "trend_snapshot.json",
        }
        for field_name, expected_path in expected_paths.items():
            path_error = _require_exact_file_path(
                result.get(field_name),
                expected_path=expected_path,
                artifact_name=field_name,
            )
            if path_error:
                return None, None, path_error
        manifest_error = _validate_benchmark_run_manifest(
            expected_run_id=str(expected_run_id),
            repo_root=Path(repo_root),
            artifacts_dir=artifacts_dir,
            expected_git_commit=str(expected_git_commit),
        )
        if manifest_error:
            return None, None, manifest_error

    summary, error = _read_json_object_artifact(
        result.get("summary_path"), artifact_name="summary_path"
    )
    if error:
        return None, None, error
    if not isinstance(summary.get("targets"), list) or not isinstance(summary.get("summary"), dict):
        return None, None, "invalid summary_path: missing targets or summary"
    summary_counts = summary["summary"]
    for field_name in ("target_count", "succeeded", "failed", "skipped", "missing"):
        value = summary_counts.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None, None, f"invalid summary_path: {field_name} must be a nonnegative integer"
    targets = summary["targets"]
    if not targets:
        return None, None, "invalid summary_path: targets must not be empty"
    if any(not isinstance(target, dict) for target in targets):
        return None, None, "invalid summary_path: every target must be an object"
    observed_counts = {
        "succeeded": sum(target.get("status") == "succeeded" for target in targets),
        "failed": sum(str(target.get("status") or "").startswith("failed") for target in targets),
        "skipped": sum(str(target.get("status") or "").startswith("skipped") for target in targets),
        "missing": sum(target.get("status") == "missing" for target in targets),
    }
    if summary_counts.get("target_count") != len(targets):
        return None, None, "invalid summary_path: target_count does not match targets"
    for field_name, observed in observed_counts.items():
        if summary_counts.get(field_name) != observed:
            return None, None, f"invalid summary_path: {field_name} does not match target statuses"
    if sum(observed_counts.values()) != len(targets):
        unknown_count = len(targets) - sum(observed_counts.values())
        return None, None, f"{unknown_count} benchmark target(s) reported unknown"
    if binding_requested:
        if summary.get("run_id") != expected_run_id:
            return None, None, "Tier-1 summary run_id does not match the owning attempt"
        if str(summary.get("source_git_commit") or "") != expected_git_commit:
            return None, None, "Tier-1 summary source Git commit does not match the E2E commit"
        if str(summary.get("source_manifest_git_commit") or "") != expected_git_commit:
            return None, None, "Tier-1 summary manifest Git commit does not match the E2E commit"
        if summary.get("source_git_dirty") is not False:
            return None, None, "Tier-1 summary source Git provenance is dirty"

    comparison, error = _read_json_object_artifact(
        result.get("regression_json_path"), artifact_name="regression_json_path"
    )
    if error:
        return None, None, error
    for field_name in ("regressions", "missing_targets"):
        if not isinstance(comparison.get(field_name), list):
            return None, None, f"invalid regression_json_path: {field_name} must be a list"
    if binding_requested and comparison.get("current_run_id") != expected_run_id:
        return None, None, "Tier-1 comparison current_run_id does not match the owning attempt"

    trend, error = _read_json_object_artifact(
        result.get("trend_snapshot_path"), artifact_name="trend_snapshot_path"
    )
    if error:
        return None, None, error
    if (
        not isinstance(trend.get("run_count"), int)
        or isinstance(trend.get("run_count"), bool)
        or not isinstance(trend.get("history"), list)
        or not isinstance(trend.get("evidence_history"), list)
    ):
        return None, None, "invalid trend_snapshot_path: malformed trend snapshot"

    regression_summary_path = result.get("regression_summary_path")
    if not _result_path_exists(regression_summary_path):
        return None, None, "missing or empty required artifact: regression_summary_path"
    return summary, comparison, None


def _group_targets_by_unit(targets: List[str]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    index_by_name: Dict[str, int] = {}
    for target in targets:
        unit_name = _canonical_unit_name(
            str(target).split(":", 1)[0].strip() or str(target).strip()
        )
        if unit_name not in index_by_name:
            index_by_name[unit_name] = len(units)
            units.append({"name": unit_name, "targets": []})
        units[index_by_name[unit_name]]["targets"].append(str(target))
    return units


def _canonical_unit_name(unit_name: Optional[str]) -> str:
    value = str(unit_name or "").strip()
    if not value:
        return ""
    if value.startswith("labs_"):
        return f"labs/{value[len('labs_'):]}"
    return value


def _canonical_target_name(target: object) -> str:
    value = str(target or "").strip()
    if not value:
        return ""
    if ":" not in value:
        return _canonical_unit_name(value)
    unit_name, example = value.split(":", 1)
    return f"{_canonical_unit_name(unit_name)}:{example.strip()}"


def _completed_units_from_target_outcomes(
    targets: List[str],
    benchmark_summary: Optional[Dict[str, Any]],
) -> List[str]:
    if not isinstance(benchmark_summary, dict):
        return []
    canonical_targets = [_canonical_target_name(target) for target in targets]
    if any(not target for target in canonical_targets) or len(canonical_targets) != len(
        set(canonical_targets)
    ):
        return []
    raw_outcomes = benchmark_summary.get("target_outcomes")
    if not isinstance(raw_outcomes, list):
        return []
    outcome_rows = [outcome for outcome in raw_outcomes if isinstance(outcome, dict)]
    if len(outcome_rows) != len(raw_outcomes):
        return []
    observed_targets = [_canonical_target_name(outcome.get("target")) for outcome in outcome_rows]
    if (
        any(not target for target in observed_targets)
        or len(observed_targets) != len(set(observed_targets))
        or any(target not in set(canonical_targets) for target in observed_targets)
    ):
        return []
    status_by_target = {
        target: str(outcome.get("status") or "unknown")
        for target, outcome in zip(observed_targets, outcome_rows)
    }
    completed_units: List[str] = []
    for unit in _group_targets_by_unit(targets):
        expected_targets = [_canonical_target_name(target) for target in unit.get("targets", [])]
        if not expected_targets or any(
            status_by_target.get(target) != "succeeded" for target in expected_targets
        ):
            break
        completed_units.append(unit["name"])
    return completed_units


def _completed_units_from_attempts(
    attempts: List[Dict[str, Any]], *, frozen_targets: List[str]
) -> List[str]:
    canonical_frozen_targets = [_canonical_target_name(target) for target in frozen_targets]
    if any(not target for target in canonical_frozen_targets) or len(
        canonical_frozen_targets
    ) != len(set(canonical_frozen_targets)):
        return []
    frozen_units = _group_targets_by_unit(canonical_frozen_targets)
    completed_lookup: set[str] = set()
    for attempt in attempts:
        requested = attempt.get("verified_targets")
        benchmark_summary = attempt.get("benchmark_summary")
        if not isinstance(requested, list) or not isinstance(benchmark_summary, dict):
            completed_lookup.clear()
            continue
        canonical_requested = [_canonical_target_name(target) for target in requested]
        if any(not target for target in canonical_requested) or len(canonical_requested) != len(
            set(canonical_requested)
        ):
            completed_lookup.clear()
            continue

        start_index: Optional[int] = None
        for index in range(len(frozen_units)):
            suffix = [
                _canonical_target_name(target)
                for unit in frozen_units[index:]
                for target in unit.get("targets", [])
            ]
            if canonical_requested == suffix:
                start_index = index
                break
        if start_index is None:
            completed_lookup.clear()
            continue

        for unit in frozen_units[start_index:]:
            completed_lookup.discard(str(unit["name"]))
        completed_prefix = _completed_units_from_target_outcomes(
            canonical_requested,
            benchmark_summary,
        )
        for unit_name in completed_prefix:
            completed_lookup.add(_canonical_unit_name(unit_name))

    completed_units: List[str] = []
    for unit in frozen_units:
        unit_name = str(unit["name"])
        if unit_name not in completed_lookup:
            break
        completed_units.append(unit_name)
    return completed_units


def _verified_full_sweep_attempts(
    attempts: List[Dict[str, Any]],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
    expected_git_commit: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    verified: List[Dict[str, Any]] = []
    issues: List[str] = []
    for attempt in attempts:
        if not isinstance(attempt.get("verified_targets"), list) or not isinstance(
            attempt.get("benchmark_summary"), dict
        ):
            continue
        attempt_run_id = str(attempt.get("run_id") or "").strip()
        try:
            validated_attempt_run_id = validate_run_id(attempt_run_id)
        except ValueError:
            issues.append("full-sweep attempt has an unsafe run_id")
            continue
        evidence_error = _validate_benchmark_run_manifest(
            expected_run_id=validated_attempt_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            expected_git_commit=expected_git_commit,
        )
        if evidence_error:
            issues.append(f"{validated_attempt_run_id}: {evidence_error}")
            continue
        verified.append(attempt)
    return verified, issues


def _remaining_targets_after_completed_units(
    targets: List[str],
    *,
    completed_units: List[str],
) -> List[str]:
    grouped_units = _group_targets_by_unit(targets)
    completed_lookup = {
        _canonical_unit_name(unit) for unit in completed_units if _canonical_unit_name(unit)
    }
    first_incomplete_index: Optional[int] = None
    for index, unit in enumerate(grouped_units):
        if unit["name"] not in completed_lookup:
            first_incomplete_index = index
            break
    if first_incomplete_index is None:
        return []
    remaining: List[str] = []
    for unit in grouped_units[first_incomplete_index:]:
        remaining.extend([str(target) for target in unit.get("targets", [])])
    return remaining


def _remaining_units_after_completed_units(
    ordered_units: List[str],
    *,
    completed_units: List[str],
) -> List[str]:
    ordered_units = [_canonical_unit_name(unit) for unit in ordered_units]
    completed_lookup = {
        _canonical_unit_name(unit) for unit in completed_units if _canonical_unit_name(unit)
    }
    first_incomplete_index: Optional[int] = None
    for index, unit in enumerate(ordered_units):
        if unit not in completed_lookup:
            first_incomplete_index = index
            break
    if first_incomplete_index is None:
        return []
    return [str(unit) for unit in ordered_units[first_incomplete_index:]]


def _resolve_targets_for_units(
    available_targets: List[str],
    *,
    ordered_units: List[str],
) -> Tuple[List[str], List[str]]:
    grouped_targets: Dict[str, List[str]] = {}
    for target in available_targets:
        unit_name = _canonical_unit_name(
            str(target).split(":", 1)[0].strip() or str(target).strip()
        )
        grouped_targets.setdefault(unit_name, []).append(str(target))

    resolved_targets: List[str] = []
    missing_units: List[str] = []
    for unit_name in ordered_units:
        canonical_unit_name = _canonical_unit_name(unit_name)
        matches = grouped_targets.get(canonical_unit_name, [])
        if not matches:
            missing_units.append(canonical_unit_name)
            continue
        resolved_targets.extend(matches)
    return resolved_targets, missing_units


def _benchmark_stage_details_from_output(
    output_json_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not output_json_path or not Path(output_json_path).exists():
        return None
    try:
        payload = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    status_counts: Dict[str, int] = {}
    failed_benchmarks: List[Dict[str, Any]] = []
    skipped_benchmarks: List[Dict[str, Any]] = []
    target_outcomes: List[Dict[str, Any]] = []
    results = payload.get("results") or []
    for chapter_entry in results:
        chapter_name = _canonical_unit_name((chapter_entry or {}).get("chapter")) or None
        for benchmark in (chapter_entry or {}).get("benchmarks", []) or []:
            status = str((benchmark or {}).get("status") or "unknown").strip() or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            example = str((benchmark or {}).get("example") or "").strip()
            target = (
                f"{chapter_name}:{example}"
                if chapter_name and example
                else example or chapter_name or "<unknown>"
            )
            target_outcomes.append({"target": target, "status": status})
            if status.startswith("failed"):
                error_detail = str(
                    (benchmark or {}).get("error")
                    or (benchmark or {}).get("failure_reason")
                    or "benchmark target failed"
                )
                failure_entry = {
                    "target": target,
                    "status": status,
                    "error": error_detail,
                }
                for key in (
                    "best_speedup",
                    "best_memory_savings_pct",
                    "optimization_goal",
                    "minimum_required_speedup",
                ):
                    value = (benchmark or {}).get(key)
                    if value is not None:
                        failure_entry[key] = value
                failed_benchmarks.append(failure_entry)
            elif status == "skipped":
                skip_reason = str(
                    (benchmark or {}).get("error")
                    or (benchmark or {}).get("skip_reason")
                    or "benchmark target skipped"
                )
                skipped_benchmarks.append(
                    {
                        "target": target,
                        "status": status,
                        "reason": skip_reason,
                    }
                )

    return {
        "status_counts": status_counts,
        "failed_benchmarks": failed_benchmarks,
        "skipped_benchmarks": skipped_benchmarks,
        "target_outcomes": target_outcomes,
    }


def _benchmark_stage_details_from_suite_summary(
    summary_payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(summary_payload, dict):
        return None
    targets = summary_payload.get("targets")
    if not isinstance(targets, list):
        return None

    status_counts: Dict[str, int] = {}
    failed_benchmarks: List[Dict[str, Any]] = []
    skipped_benchmarks: List[Dict[str, Any]] = []
    target_outcomes: List[Dict[str, Any]] = []
    for target_payload in targets:
        if not isinstance(target_payload, dict):
            continue
        status = str(target_payload.get("status") or "unknown").strip() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        target = (
            str(target_payload.get("target") or target_payload.get("key") or "<unknown>").strip()
            or "<unknown>"
        )
        target_outcomes.append({"target": target, "status": status})
        if status.startswith("failed"):
            failure_entry = {
                "target": target,
                "status": status,
                "error": str(
                    target_payload.get("error")
                    or target_payload.get("failure_reason")
                    or f"{target} reported {status}"
                ),
            }
            for key in ("best_speedup", "best_memory_savings_pct", "optimization_goal"):
                value = target_payload.get(key)
                if value is not None:
                    failure_entry[key] = value
            failed_benchmarks.append(failure_entry)
        elif status == "skipped":
            skipped_benchmarks.append(
                {
                    "target": target,
                    "status": status,
                    "reason": str(
                        target_payload.get("error")
                        or target_payload.get("skip_reason")
                        or f"{target} reported {status}"
                    ),
                }
            )

    return {
        "status_counts": status_counts,
        "failed_benchmarks": failed_benchmarks,
        "skipped_benchmarks": skipped_benchmarks,
        "target_outcomes": target_outcomes,
    }


def _benchmark_stage_details(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    execution = result.get("execution")
    output_json_path = result.get("output_json")
    if output_json_path is None and isinstance(execution, dict):
        output_json_path = execution.get("output_json")
    benchmark_details = _benchmark_stage_details_from_output(
        str(output_json_path) if output_json_path else None
    )
    if benchmark_details is not None:
        return benchmark_details

    benchmark_details = _benchmark_stage_details_from_suite_summary(
        result.get("summary") if isinstance(result.get("summary"), dict) else None
    )
    if benchmark_details is not None:
        return benchmark_details

    summary_path = result.get("summary_path")
    if not _result_path_exists(summary_path):
        return None
    try:
        summary_payload = json.loads(Path(str(summary_path)).read_text(encoding="utf-8"))
    except Exception:
        return None
    return _benchmark_stage_details_from_suite_summary(
        summary_payload if isinstance(summary_payload, dict) else None
    )


def _benchmark_stage_status(
    result: Dict[str, Any],
    *,
    required_paths: List[str],
    required_targets: Optional[List[str]] = None,
    require_complete: bool = False,
    allow_comparison_regressions: bool = False,
    expected_run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    artifacts_dir: Optional[str] = None,
    expected_git_commit: Optional[str] = None,
) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    if result.get("error"):
        issues.append(str(result["error"]))
        return "failed", issues, None

    missing = [
        path_key for path_key in required_paths if not _result_path_exists(result.get(path_key))
    ]
    if missing:
        issues.append(f"missing required artifacts: {', '.join(missing)}")
        return "failed", issues, None

    validated_result = dict(result)
    if require_complete:
        summary, comparison, artifact_error = _validate_tier1_artifacts(
            result,
            expected_run_id=expected_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            expected_git_commit=expected_git_commit,
        )
        if artifact_error:
            return "failed", [artifact_error], None
        validated_result["summary"] = summary
        validated_result["comparison"] = comparison

    benchmark_details = _benchmark_stage_details(validated_result)
    execution = (
        validated_result.get("execution")
        if isinstance(validated_result.get("execution"), dict)
        else {}
    )
    if required_targets is not None:
        requested = [
            _canonical_target_name(target)
            for target in required_targets
            if _canonical_target_name(target)
        ]
        outcomes = (
            benchmark_details.get("target_outcomes", [])
            if isinstance(benchmark_details, dict)
            else []
        )
        observed = [
            _canonical_target_name(outcome.get("target"))
            for outcome in outcomes
            if isinstance(outcome, dict) and _canonical_target_name(outcome.get("target"))
        ]
        requested_counts = Counter(requested)
        observed_counts = Counter(observed)
        if observed_counts != requested_counts:
            missing_targets = sorted((requested_counts - observed_counts).elements())
            unexpected_targets = sorted((observed_counts - requested_counts).elements())
            if missing_targets:
                issues.append("missing terminal benchmark outcomes: " + ", ".join(missing_targets))
            if unexpected_targets:
                issues.append(
                    "unexpected terminal benchmark outcomes: " + ", ".join(unexpected_targets)
                )
            if not issues:
                issues.append("benchmark terminal outcomes did not match the requested targets")
            return "failed", issues, benchmark_details
        non_success = sorted(
            {
                f"{_canonical_target_name(outcome.get('target'))}={str(outcome.get('status') or 'unknown')}"
                for outcome in outcomes
                if isinstance(outcome, dict)
                and _canonical_target_name(outcome.get("target")) in requested_counts
                and str(outcome.get("status") or "unknown") != "succeeded"
            }
        )
        if non_success:
            issues.append("non-success terminal benchmark outcomes: " + ", ".join(non_success))
            return "failed", issues, benchmark_details
    if benchmark_details is not None:
        total_failed = sum(
            count
            for status, count in (benchmark_details.get("status_counts") or {}).items()
            if str(status).startswith("failed")
        )
        total_skipped = int((benchmark_details.get("status_counts") or {}).get("skipped", 0) or 0)
    else:
        total_failed = int(result.get("total_failed", 0) or 0)
        total_skipped = int(result.get("total_skipped", 0) or 0)
        if total_failed == 0 and execution:
            total_failed = int(execution.get("total_failed", 0) or 0)
        if total_skipped == 0 and execution:
            total_skipped = int(execution.get("total_skipped", 0) or 0)

    if total_failed > 0:
        failed_benchmarks = (benchmark_details or {}).get("failed_benchmarks") or []
        if failed_benchmarks:
            issues.extend(f"{entry['target']}: {entry['error']}" for entry in failed_benchmarks)
        else:
            issues.append(f"{total_failed} benchmark target(s) failed")
        return "failed", issues, benchmark_details

    if require_complete:
        from core.benchmark.bench_commands import _tier1_result_failure_count

        status_counts = (benchmark_details or {}).get("status_counts") or {}
        incomplete_statuses = {
            str(status): int(count or 0)
            for status, count in status_counts.items()
            if str(status) != "succeeded" and int(count or 0) > 0
        }
        reported_skipped = total_skipped
        for skipped_source in (result, execution):
            try:
                reported_skipped = max(
                    reported_skipped,
                    int(skipped_source.get("total_skipped", 0) or 0),
                )
            except (TypeError, ValueError):
                reported_skipped = max(reported_skipped, 1)
        failure_count = _tier1_result_failure_count(
            validated_result,
            allow_comparison_regressions=allow_comparison_regressions,
        )
        if failure_count > 0 or incomplete_statuses or reported_skipped > 0:
            for status, count in sorted(incomplete_statuses.items()):
                issues.append(f"{count} benchmark target(s) reported {status}")
            if reported_skipped > 0 and not any(
                status.startswith("skipped") for status in incomplete_statuses
            ):
                issues.append(f"{reported_skipped} benchmark target(s) skipped")

            comparison = validated_result.get("comparison")
            regressions = comparison.get("regressions") if isinstance(comparison, dict) else None
            if isinstance(regressions, list) and regressions:
                issues.append(f"{len(regressions)} benchmark comparison regression(s) detected")

            execution_failed = int(execution.get("total_failed", 0) or 0) if execution else 0
            if execution_failed > 0:
                issues.append(f"benchmark execution reported {execution_failed} failure(s)")

            if not issues:
                issues.append(
                    f"Tier-1 result reported {failure_count} failed or incomplete outcome(s)"
                )
            return "failed", issues, benchmark_details

    if total_skipped > 0:
        issues.append(f"{total_skipped} benchmark target(s) skipped")
        return "partial", issues, benchmark_details

    return "succeeded", issues, benchmark_details


def _fabric_scorecard_details(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    run_dir = result.get("run_dir")
    run_id = result.get("run_id")
    if not run_dir or not run_id:
        return None

    scorecard_path = Path(str(run_dir)) / "structured" / f"{run_id}_fabric_scorecard.json"
    if not scorecard_path.exists():
        return None

    payload, error = _read_json_object_artifact(
        scorecard_path,
        artifact_name="fabric scorecard",
    )
    if error:
        return {
            "path": str(scorecard_path),
            "status": "error",
            "error": error,
        }
    if (
        payload.get("schema_version") is None
        or str(payload.get("run_id") or "") != str(run_id)
        or not isinstance(payload.get("families"), dict)
        or not isinstance(payload.get("summary"), dict)
    ):
        return {
            "path": str(scorecard_path),
            "status": "error",
            "error": "invalid fabric scorecard: missing schema, run identity, families, or summary",
        }

    families = payload.get("families") or {}
    degraded = []
    for family_name, family_payload in families.items():
        completeness = str((family_payload or {}).get("completeness") or "unknown")
        if completeness in {"not_present", "not_configured"}:
            degraded.append({"family": family_name, "completeness": completeness})
    return {
        "path": str(scorecard_path),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "degraded_families": degraded,
    }


def _cluster_manifest_files_error(
    manifest: Dict[str, Any],
    *,
    run_dir: Path,
) -> Optional[str]:
    files = manifest.get("files")
    summary = manifest.get("summary")
    if not isinstance(files, list) or not isinstance(summary, dict):
        return "invalid manifest_path: missing files or summary"
    if any(not isinstance(raw_path, str) or not raw_path.strip() for raw_path in files):
        return "invalid manifest_path: every file must be a nonempty relative path"
    if len(files) != len(set(files)):
        return "invalid manifest_path: duplicate file entries"
    file_count = summary.get("file_count")
    if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count != len(files):
        return "invalid manifest_path: summary.file_count does not match files"
    hashes = summary.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(files):
        return "invalid manifest_path: summary.sha256 keys do not match files"

    canonical_run_dir = Path(run_dir).resolve()
    if not canonical_run_dir.is_dir() or Path(run_dir).is_symlink():
        return "invalid cluster run_dir"
    for raw_path in files:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            return f"invalid manifest file path: {raw_path}"
        candidate = canonical_run_dir / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(canonical_run_dir)
        except (OSError, ValueError):
            return f"manifest file is missing or outside run_dir: {raw_path}"
        if candidate.is_symlink() or not resolved.is_file():
            return f"manifest file is not a regular in-run artifact: {raw_path}"
        expected_digest = hashes.get(raw_path)
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest) is None
            or _sha256_file(resolved) != expected_digest.lower()
        ):
            return f"manifest SHA256 mismatch: {raw_path}"
    return None


def _cluster_stage_status(
    result: Dict[str, Any],
    *,
    require_scorecard: bool = False,
    expected_run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    expected_git_commit: Optional[str] = None,
) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    if not result.get("success", False):
        error = result.get("error") or result.get("stderr") or "cluster stage failed"
        issues.append(str(error))
        return "failed", issues, None

    run_dir = Path(str(result.get("run_dir") or ""))
    if expected_run_id is not None or repo_root is not None:
        if not expected_run_id or repo_root is None:
            return "failed", ["invalid cluster evidence-binding context"], None
        if result.get("run_id") != expected_run_id:
            return "failed", ["cluster result run_id does not match the owning attempt"], None
        expected_run_dir = Path(repo_root).resolve() / "cluster" / "runs" / str(expected_run_id)
        expected_manifest_path = expected_run_dir / "manifest.json"
        try:
            if run_dir.resolve(strict=True) != expected_run_dir.resolve(strict=True):
                return "failed", ["cluster run_dir does not match the owning attempt"], None
        except OSError:
            return "failed", ["cluster run_dir is missing"], None
        if run_dir.is_symlink():
            return "failed", ["cluster run_dir may not be a symlink"], None
        path_error = _require_exact_file_path(
            result.get("manifest_path"),
            expected_path=expected_manifest_path,
            artifact_name="manifest_path",
        )
        if path_error:
            return "failed", [path_error], None

    manifest, manifest_error = _read_json_object_artifact(
        result.get("manifest_path"),
        artifact_name="manifest_path",
    )
    if manifest_error:
        issues.append(manifest_error)
        return "failed", issues, None
    if (
        not isinstance(manifest.get("manifest_version"), int)
        or str(manifest.get("run_id") or "") != str(result.get("run_id") or "")
        or not isinstance(manifest.get("files"), list)
        or not isinstance(manifest.get("summary"), dict)
    ):
        issues.append("invalid manifest_path: missing version, run identity, files, or summary")
        return "failed", issues, None
    if manifest.get("finalized") is not True:
        issues.append("cluster manifest is not finalized")
        return "failed", issues, None
    manifest_status = str(manifest.get("status") or "").strip().lower()
    suite_status = str(manifest.get("suite_status") or "").strip().lower()
    if manifest_status != suite_status:
        issues.append("cluster manifest status and suite_status do not match")
        return "failed", issues, None
    if manifest_status not in {"succeeded", "partial"}:
        issues.append(f"cluster manifest is not terminal: {manifest_status or 'missing'}")
        return "failed", issues, None
    if manifest.get("success") is not True:
        issues.append("cluster manifest success must be true")
        return "failed", issues, None
    if expected_git_commit is not None:
        manifest_git = manifest.get("git")
        if not isinstance(manifest_git, dict):
            issues.append("cluster manifest is missing Git provenance")
            return "failed", issues, None
        if str(manifest_git.get("commit") or "") != expected_git_commit:
            issues.append("cluster manifest Git commit does not match the E2E commit")
            return "failed", issues, None
        if manifest_git.get("dirty") is not False:
            issues.append("cluster manifest Git provenance is dirty")
            return "failed", issues, None
    files_error = _cluster_manifest_files_error(manifest, run_dir=run_dir)
    if files_error:
        issues.append(files_error)
        return "failed", issues, None

    scorecard = _fabric_scorecard_details(result)
    if require_scorecard and scorecard is None:
        issues.append("missing required artifact: fabric scorecard")
        return "failed", issues, None
    if scorecard:
        scorecard_status = str(scorecard.get("status") or "").strip().lower()
        if scorecard_status in {"error", "failed"}:
            issues.append(str(scorecard.get("error") or "fabric scorecard reported a fatal status"))
            return "failed", issues, scorecard
        degraded = scorecard.get("degraded_families") or []
        if scorecard_status == "partial":
            if degraded:
                issues.append("fabric completeness is partial for one or more families")
            else:
                issues.append("fabric scorecard reported partial runtime verification")
            return "partial", issues, scorecard
        if scorecard_status not in {"", "ok"}:
            issues.append(
                str(
                    scorecard.get("error")
                    or f"unexpected fabric scorecard status: {scorecard_status}"
                )
            )
            return "failed", issues, scorecard
        if degraded:
            issues.append("fabric completeness is partial for one or more families")
            return "partial", issues, scorecard

    if manifest_status == "partial":
        issues.append("cluster manifest reported partial completion")
        return "partial", issues, scorecard
    return "succeeded", issues, scorecard


def _roll_up_overall_status(stage_statuses: List[str]) -> str:
    relevant = [status for status in stage_statuses if status not in {"skipped", "planned"}]
    if not relevant:
        return "failed"
    if any(
        status not in {"aborted", "failed", "partial", "skipped_duplicate", "succeeded"}
        for status in relevant
    ):
        return "failed"
    if any(status == "aborted" for status in relevant):
        return "aborted"
    if any(status == "failed" for status in relevant):
        return "failed"
    if any(status == "partial" for status in relevant):
        return "partial"
    if all(status == "skipped_duplicate" for status in relevant):
        return "skipped_duplicate"
    return "succeeded"


def _stage_entry(
    *,
    name: str,
    enabled: bool,
    stage_run_id: str,
    status: str,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
    attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": name,
        "enabled": enabled,
        "run_id": stage_run_id,
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    if command is not None:
        payload["command"] = command
    if returncode is not None:
        payload["returncode"] = returncode
    if result is not None:
        payload["result"] = result
    if artifacts:
        payload["artifacts"] = artifacts
    if issues:
        payload["issues"] = issues
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if attempts is not None:
        payload["attempts"] = attempts
    return payload


def _planned_stage_entries(
    *,
    run_tier1: bool,
    run_full_sweep: bool,
    run_cluster: bool,
    run_fabric: bool,
    cluster_preset: str,
    stage_run_ids: Dict[str, str],
) -> List[Dict[str, Any]]:
    stages = [
        _stage_entry(
            name="tier1",
            enabled=run_tier1,
            stage_run_id=stage_run_ids["tier1"],
            status="planned" if run_tier1 else "skipped",
            reason=None if run_tier1 else "disabled by flag",
        ),
        _stage_entry(
            name="full_sweep",
            enabled=run_full_sweep,
            stage_run_id=stage_run_ids["full_sweep"],
            status="planned" if run_full_sweep else "skipped",
            reason=None if run_full_sweep else "disabled by flag",
        ),
        _stage_entry(
            name="cluster",
            enabled=run_cluster,
            stage_run_id=stage_run_ids["cluster"],
            status="planned" if run_cluster else "skipped",
            reason=None if run_cluster else "disabled by flag",
        ),
    ]
    fabric_duplicate = (
        run_fabric and run_cluster and cluster_preset.strip().lower() == "fabric-systems"
    )
    fabric_status = "planned" if run_fabric else "skipped"
    fabric_reason = None if run_fabric else "disabled by flag"
    if fabric_duplicate:
        fabric_status = "skipped_duplicate"
        fabric_reason = "cluster preset already includes fabric evaluation"
    stages.append(
        _stage_entry(
            name="fabric",
            enabled=run_fabric,
            stage_run_id=stage_run_ids["fabric"],
            status=fabric_status,
            reason=fabric_reason,
        )
    )
    return stages


def _stage_index(stages: List[Dict[str, Any]], name: str) -> int:
    for idx, stage in enumerate(stages):
        if stage.get("name") == name:
            return idx
    raise KeyError(f"Unknown E2E stage '{name}'")


def _replace_stage(
    stages: List[Dict[str, Any]],
    *,
    name: str,
    status: str,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
    attempts: Optional[List[Dict[str, Any]]] = None,
) -> None:
    index = _stage_index(stages, name)
    existing = stages[index]
    stages[index] = _stage_entry(
        name=name,
        enabled=bool(existing.get("enabled")),
        stage_run_id=str(existing.get("run_id")),
        status=status,
        reason=reason,
        command=command,
        returncode=returncode,
        result=result,
        artifacts=artifacts,
        issues=issues,
        duration_ms=duration_ms,
        attempts=existing.get("attempts") if attempts is None else attempts,
    )


def _enabled_stages(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [stage for stage in stages if stage.get("enabled")]


def _completed_enabled_stage_count(stages: List[Dict[str, Any]]) -> int:
    return sum(
        1 for stage in _enabled_stages(stages) if stage.get("status") not in {"planned", "running"}
    )


def _current_stage_name(stages: List[Dict[str, Any]]) -> Optional[str]:
    for stage in _enabled_stages(stages):
        if stage.get("status") == "running":
            return str(stage["name"])
    for stage in _enabled_stages(stages):
        if stage.get("status") == "planned":
            return str(stage["name"])
    return None


def _progress_percent(
    stages: List[Dict[str, Any]],
    *,
    run_state: str,
    child_percent: Optional[float] = None,
) -> float:
    enabled = _enabled_stages(stages)
    if not enabled:
        return 100.0 if run_state == "completed" else 0.0
    if run_state == "completed":
        return 100.0
    completed = float(_completed_enabled_stage_count(stages))
    if child_percent is not None:
        bounded_child = max(0.0, min(100.0, float(child_percent)))
        return ((completed + (bounded_child / 100.0)) / float(len(enabled))) * 100.0
    return (completed / float(len(enabled))) * 100.0


def _load_progress_current(progress_path: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json_if_exists(progress_path)
    if payload is None:
        return None
    current = payload.get("current")
    if not isinstance(current, dict):
        return None
    return current


def _emit_live_progress(
    progress_recorder: Optional[ProgressRecorder],
    *,
    stages: List[Dict[str, Any]],
    run_state: str,
    overall_status: str,
    artifact_paths: Dict[str, Path],
    emit_lock: Optional[threading.Lock] = None,
    child_progress: Optional[Dict[str, Any]] = None,
    child_stage_name: Optional[str] = None,
    child_run_id: Optional[str] = None,
    child_bucket: Optional[str] = None,
    orchestrator_pid: Optional[int] = None,
) -> None:
    if progress_recorder is None:
        return

    enabled = _enabled_stages(stages)
    total_phases = max(1, len(enabled))
    current_stage = child_stage_name or _current_stage_name(stages)
    if current_stage and enabled:
        phase_index = next(
            (
                idx
                for idx, stage in enumerate(enabled, start=1)
                if stage.get("name") == current_stage
            ),
            1,
        )
    elif enabled:
        phase_index = len(enabled)
    else:
        phase_index = 1

    child_percent = None
    step = current_stage or ("complete" if run_state == "completed" else "idle")
    step_detail = f"run_state={run_state}, overall_status={overall_status}"
    if child_progress and run_state == "running" and current_stage:
        child_step = str(child_progress.get("step") or "").strip()
        child_detail = str(child_progress.get("step_detail") or "").strip()
        raw_child_percent = child_progress.get("percent_complete")
        if isinstance(raw_child_percent, (int, float)):
            child_percent = float(raw_child_percent)
        step_prefix = current_stage if not child_bucket else f"{current_stage}/{child_bucket}"
        if child_step:
            step = f"{step_prefix}:{child_step}"
        else:
            step = step_prefix
        if child_detail:
            step_detail = child_detail

    event = ProgressEvent(
        phase="e2e_sweep",
        phase_index=phase_index,
        total_phases=total_phases,
        step=step,
        step_detail=step_detail,
        percent_complete=_progress_percent(
            stages, run_state=run_state, child_percent=child_percent
        ),
        artifacts=[str(path) for path in artifact_paths.values()],
        metrics={
            "run_state": run_state,
            "overall_status": overall_status,
            "current_stage": current_stage,
            "current_stage_run_id": child_run_id,
            "current_bucket": child_bucket,
            "orchestrator_pid": orchestrator_pid,
            "completed_stages": _completed_enabled_stage_count(stages),
            "total_stages": len(enabled),
            "stages": _json_safe(stages),
            "child_progress": _json_safe(child_progress) if child_progress is not None else None,
        },
    )
    if emit_lock is None:
        progress_recorder.emit(event)
        return
    with emit_lock:
        progress_recorder.emit(event)


def _render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark E2E Sweep",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Success: `{summary['success']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Resume available: `{summary.get('resume_available', False)}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Run id | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for stage in summary.get("stages", []):
        notes = stage.get("reason") or "; ".join(stage.get("issues", [])) or ""
        lines.append(f"| `{stage['name']}` | `{stage['status']}` | `{stage['run_id']}` | {notes} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{summary['manifest_path']}`",
            f"- Summary JSON: `{summary['summary_path']}`",
            f"- Summary Markdown: `{summary['summary_markdown_path']}`",
            f"- Progress JSON: `{summary['progress_path']}`",
            f"- Checkpoint JSON: `{summary['checkpoint_path']}`",
            f"- Target inventory: `{summary['target_inventory_path']}`",
            f"- Events: `{summary['events_path']}`",
        ]
    )
    ledgers = dict(summary.get("ledgers") or {})
    if ledgers.get("active_issue_ledger_json") or ledgers.get("historical_failure_ledger_json"):
        lines.extend(
            [
                "",
                "## Ledgers",
                "",
            ]
        )
        if ledgers.get("active_issue_ledger_json"):
            lines.append(f"- Active issue ledger JSON: `{ledgers.get('active_issue_ledger_json')}`")
        if ledgers.get("active_issue_ledger_md"):
            lines.append(
                f"- Active issue ledger Markdown: `{ledgers.get('active_issue_ledger_md')}`"
            )
    historical_failure_ledger = summary.get("historical_failure_ledger")
    if isinstance(historical_failure_ledger, dict):
        ledger_summary = historical_failure_ledger.get("summary") or {}
        lines.extend(
            [
                "",
                "## Historical Failure Ledger",
                "",
                f"- Ledger JSON: `{historical_failure_ledger.get('json_path', '')}`",
                f"- Ledger Markdown: `{historical_failure_ledger.get('markdown_path', '')}`",
                f"- Total historical failures: `{ledger_summary.get('total_historical_failures', 0)}`",
                f"- Rechecked: `{ledger_summary.get('rechecked_count', 0)}`",
                f"- Resolved success: `{ledger_summary.get('resolved_success_count', 0)}`",
                f"- Resolved skip: `{ledger_summary.get('resolved_skip_count', 0)}`",
                f"- Still failing: `{ledger_summary.get('still_failing_count', 0)}`",
                f"- Not rerun: `{ledger_summary.get('not_rerun_count', 0)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _stage_attempt_entry(
    *,
    run_id: str,
    bucket: Optional[str] = None,
    status: str,
    targets: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
    completed_units: Optional[List[str]] = None,
    active_unit: Optional[str] = None,
    reason: Optional[str] = None,
    command: Optional[List[str]] = None,
    returncode: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    issues: Optional[List[str]] = None,
    duration_ms: Optional[int] = None,
    benchmark_summary: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    recovered: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "recovered": recovered,
    }
    if bucket:
        payload["bucket"] = bucket
    if targets is not None:
        payload["targets"] = list(targets)
    if units is not None:
        payload["units"] = list(units)
    if completed_units is not None:
        payload["completed_units"] = list(completed_units)
    if active_unit:
        payload["active_unit"] = active_unit
    if reason:
        payload["reason"] = reason
    if command is not None:
        payload["command"] = list(command)
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if result is not None:
        payload["result"] = result
    if artifacts:
        payload["artifacts"] = artifacts
    if issues:
        payload["issues"] = list(issues)
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    if benchmark_summary is not None:
        payload["benchmark_summary"] = benchmark_summary
    if started_at:
        payload["started_at"] = started_at
    if ended_at:
        payload["ended_at"] = ended_at
    return payload


_SENSITIVE_CONTRACT_FIELDS = (
    "ssh_key",
    "nmx_token",
    "ib_mgmt_ssh_key",
    "cumulus_ssh_key",
)

_SENSITIVE_EXTRA_ARG_FLAGS = (
    "--ssh-key",
    "--nmx-token",
    "--ib-mgmt-ssh-key",
    "--cumulus-ssh-key",
)

_PATH_CONTRACT_FIELDS = (
    "artifacts_dir",
    "log_file",
)

_E2E_CONTRACT_REQUIRED_FIELDS = (
    "schema_version",
    "run_tier1",
    "run_full_sweep",
    "run_cluster",
    "run_fabric",
    "cluster_preset",
    "hosts",
    "labels",
    "ssh_user",
    "ssh_key_configured",
    "oob_if",
    "socket_ifname",
    "nccl_ib_hca",
    "nmx_url",
    "nmx_token_configured",
    "ib_mgmt_host",
    "ib_mgmt_user",
    "ib_mgmt_ssh_key_configured",
    "cumulus_hosts",
    "cumulus_user",
    "cumulus_ssh_key_configured",
    "primary_label",
    "coverage_baseline_run_id",
    "extra_cluster_args",
    "bench_root_identity",
    "profile_type",
    "output_format",
    "suite_timeout",
    "full_sweep_suite_timeout",
    "timeout_multiplier",
    "timeout_seconds",
    "validity_profile",
    "allow_portable_expectations_update",
    "reproducible",
    "cold_start",
    "force_synchronize",
    "iterations",
    "warmup",
    "gpu_sm_clock_mhz",
    "gpu_mem_clock_mhz",
    "artifacts_dir_identity",
    "log_level",
    "log_file_identity",
    "single_gpu",
    "accept_regressions",
    "update_expectations",
    "allow_mixed_provenance",
    "ncu_metric_set",
    "ncu_replay_mode",
    "nsys_timeout_seconds",
    "ncu_timeout_seconds",
    "auto_resume",
    "max_auto_resumes",
    "watch_poll_interval_seconds",
)

_AUTO_RESUME_UNSUPPORTED_DEFAULTS: Dict[str, Any] = {
    "oob_if": None,
    "socket_ifname": None,
    "nccl_ib_hca": None,
    "nmx_url": None,
    "ib_mgmt_host": None,
    "ib_mgmt_user": None,
    "cumulus_hosts": [],
    "cumulus_user": None,
    "primary_label": None,
    "coverage_baseline_run_id": None,
    "extra_cluster_args": ["--skip-render-localhost-report"],
    "output_format": "both",
    "timeout_multiplier": 3.0,
    "reproducible": False,
    "cold_start": False,
    "force_synchronize": False,
    "log_level": "INFO",
    "artifacts_dir_identity": None,
    "log_file_identity": None,
    "ncu_metric_set": "minimal",
    "ncu_replay_mode": None,
    "nsys_timeout_seconds": None,
    "ncu_timeout_seconds": None,
}


def _bench_root_identity(value: Any) -> str:
    normalized = str(Path(str(value)).resolve())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _path_identity(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = str(Path(text).expanduser().resolve())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sanitize_e2e_contract_values(values: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(values)
    raw_bench_root = sanitized.pop("bench_root", None)
    if raw_bench_root is not None:
        sanitized["bench_root_identity"] = _bench_root_identity(raw_bench_root)
    for field_name in _PATH_CONTRACT_FIELDS:
        if field_name not in sanitized:
            continue
        raw_path = sanitized.pop(field_name)
        sanitized[f"{field_name}_identity"] = _path_identity(raw_path)
    for field_name in _SENSITIVE_CONTRACT_FIELDS:
        configured_name = f"{field_name}_configured"
        if field_name in sanitized:
            sanitized.setdefault(configured_name, bool(sanitized.pop(field_name)))
    return _json_safe(sanitized)


def _sanitize_persisted_value(
    value: Any,
    *,
    sensitive_values: Optional[List[Any]] = None,
    repo_root: Optional[Path] = None,
    private_bench_root: Optional[Path] = None,
    private_artifacts_dir: Optional[Path] = None,
    private_log_file: Optional[Path] = None,
) -> Any:
    secrets: List[str] = []
    for item in sensitive_values or []:
        text = str(item or "")
        if not text:
            continue
        secrets.append(text)
        if "/" in text or text.startswith((".", "~")):
            with contextlib.suppress(Exception):
                secrets.append(str(Path(text).expanduser().resolve()))
    secrets = list(dict.fromkeys(secrets))
    path_replacements: List[Tuple[str, str]] = []
    for raw_value, resolved_value, locator in (
        (
            private_artifacts_dir,
            private_artifacts_dir.resolve() if private_artifacts_dir else None,
            _ARTIFACTS_DIR_LOCATOR,
        ),
        (
            private_log_file,
            private_log_file.resolve() if private_log_file else None,
            _LOG_FILE_LOCATOR,
        ),
        (
            repo_root,
            repo_root.resolve() if repo_root else None,
            _REPO_ROOT_LOCATOR,
        ),
        (
            private_bench_root,
            private_bench_root.resolve() if private_bench_root else None,
            _BENCH_ROOT_LOCATOR,
        ),
    ):
        for candidate in (raw_value, resolved_value):
            text = str(candidate or "").strip()
            if text:
                path_replacements.append((text, locator))
    unique_path_replacements: Dict[str, str] = {}
    for private_path, locator in path_replacements:
        unique_path_replacements.setdefault(private_path, locator)
    path_replacements = sorted(
        unique_path_replacements.items(),
        key=lambda entry: len(entry[0]),
        reverse=True,
    )

    def _sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            public: Dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                if key in _SENSITIVE_CONTRACT_FIELDS:
                    public[f"{key}_configured"] = bool(raw_value)
                    continue
                if key == "bench_root":
                    if raw_value is not None:
                        public["bench_root_identity"] = _bench_root_identity(raw_value)
                    continue
                if key in _PATH_CONTRACT_FIELDS:
                    public[f"{key}_identity"] = _path_identity(raw_value)
                    continue
                public[key] = _sanitize(raw_value)
            return public
        if isinstance(item, (list, tuple)):
            return [_sanitize(entry) for entry in item]
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            sanitized = item
            for secret in secrets:
                sanitized = sanitized.replace(secret, "<redacted>")
            for private_path, locator in path_replacements:
                sanitized = sanitized.replace(private_path, locator)
            return sanitized
        return item

    return _json_safe(_sanitize(value))


def _restore_persisted_path_locators(
    value: Any,
    *,
    repo_root: Path,
    bench_root: Path,
    artifacts_dir: Optional[str],
    log_file: Optional[str],
) -> Any:
    replacements = {
        _REPO_ROOT_LOCATOR: str(Path(repo_root).resolve()),
        _BENCH_ROOT_LOCATOR: str(Path(bench_root).resolve()),
        _ARTIFACTS_DIR_LOCATOR: str(Path(artifacts_dir).expanduser().resolve())
        if artifacts_dir
        else None,
        _LOG_FILE_LOCATOR: str(Path(log_file).expanduser().resolve()) if log_file else None,
    }

    def _restore(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): _restore(raw_value) for key, raw_value in item.items()}
        if isinstance(item, list):
            return [_restore(entry) for entry in item]
        if isinstance(item, str):
            restored = item
            for locator, resolved_path in replacements.items():
                if locator not in restored:
                    continue
                if resolved_path is None:
                    raise ValueError(f"Cannot restore persisted path locator {locator}")
                restored = restored.replace(locator, resolved_path)
            return restored
        return item

    return _json_safe(_restore(value))


def _public_cluster_host_config(hosts: Dict[str, Any]) -> Dict[str, Any]:
    return _sanitize_persisted_value(hosts)


def _build_e2e_contract(**kwargs: Any) -> Dict[str, Any]:
    contract = _sanitize_e2e_contract_values(kwargs)
    contract["schema_version"] = _E2E_CONTRACT_SCHEMA_VERSION
    return contract


def _auto_resume_reconstruction_error(
    contract: Optional[Dict[str, Any]],
    *,
    repo_root: Optional[Path] = None,
) -> Optional[str]:
    stored = _sanitize_e2e_contract_values(dict(contract or {}))
    if stored.get("schema_version") != _E2E_CONTRACT_SCHEMA_VERSION:
        return "stored resume contract schema is missing or unsupported"
    missing_fields = [
        field_name for field_name in _E2E_CONTRACT_REQUIRED_FIELDS if field_name not in stored
    ]
    if missing_fields:
        return "stored resume contract is missing required field(s): " + ", ".join(missing_fields)
    enabled_remote_stage = bool(stored.get("run_cluster") or stored.get("run_fabric"))
    if enabled_remote_stage:
        configured_hosts = [str(host) for host in (stored.get("hosts") or [])]
        if configured_hosts and not all(_is_local_host(host) for host in configured_hosts):
            return "non-local hosts require an SSH key that is not persisted"
        required_credentials = [
            field_name
            for field_name in (f"{name}_configured" for name in _SENSITIVE_CONTRACT_FIELDS)
            if stored.get(field_name) is True
        ]
        if required_credentials:
            return "credentials are required but are not persisted: " + ", ".join(
                required_credentials
            )

    bench_root_identity = stored.get("bench_root_identity")
    if bench_root_identity:
        active_root = Path(repo_root or _repo_root()).resolve()
        if str(bench_root_identity) != _bench_root_identity(active_root):
            return "custom bench root cannot be reconstructed from its path-free identity"

    unsupported = []
    for field_name, default_value in _AUTO_RESUME_UNSUPPORTED_DEFAULTS.items():
        if field_name not in stored:
            continue
        if _json_safe(stored.get(field_name)) != _json_safe(default_value):
            unsupported.append(field_name)
    if unsupported:
        return "execution options cannot be reconstructed by run-e2e CLI: " + ", ".join(unsupported)
    return None


def _cli_shell_join(cmd: List[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def _append_cli_scalar_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.extend([flag, str(value)])


def _append_cli_repeatable_arg(cmd: List[str], flag: str, values: Optional[List[str]]) -> None:
    for value in values or []:
        if str(value).strip():
            cmd.extend([flag, str(value).strip()])


def build_benchmark_e2e_resume_command(
    run_id: str,
    *,
    contract: Optional[Dict[str, Any]],
    python_executable: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> List[str]:
    stored = dict(contract or {})
    if _auto_resume_reconstruction_error(stored, repo_root=repo_root):
        return []
    cmd: List[str] = [
        python_executable or sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "run-e2e",
        "--run-id",
        str(run_id),
        "--resume",
    ]

    boolean_pairs = [
        ("run_tier1", "--run-tier1", "--no-run-tier1", True),
        ("run_full_sweep", "--run-full-sweep", "--no-run-full-sweep", False),
        ("run_cluster", "--run-cluster", "--no-run-cluster", True),
        ("run_fabric", "--run-fabric", "--no-run-fabric", True),
        ("single_gpu", "--single-gpu", None, False),
        ("accept_regressions", "--accept-regressions", None, False),
        ("update_expectations", "--update-expectations", None, False),
        ("allow_mixed_provenance", "--allow-mixed-provenance", None, False),
        ("allow_portable_expectations_update", "--allow-portable-expectations-update", None, False),
        ("reproducible", "--reproducible", None, False),
        ("cold_start", "--cold-start", None, False),
        ("force_synchronize", "--force-synchronize", None, False),
        ("auto_resume", "--auto-resume", "--no-auto-resume", True),
    ]
    for field_name, positive_flag, negative_flag, default_value in boolean_pairs:
        value = stored.get(field_name)
        if value is None or bool(value) == bool(default_value):
            continue
        if bool(value):
            cmd.append(positive_flag)
        elif negative_flag:
            cmd.append(negative_flag)

    scalar_flags = [
        ("cluster_preset", "--cluster-preset"),
        ("ssh_user", "--ssh-user"),
        ("profile_type", "--profile"),
        ("suite_timeout", "--suite-timeout"),
        ("full_sweep_suite_timeout", "--full-sweep-suite-timeout"),
        ("timeout_seconds", "--timeout-seconds"),
        ("artifacts_dir", "--artifacts-dir"),
        ("validity_profile", "--validity-profile"),
        ("iterations", "--iterations"),
        ("warmup", "--warmup"),
        ("gpu_sm_clock_mhz", "--gpu-sm-clock-mhz"),
        ("gpu_mem_clock_mhz", "--gpu-mem-clock-mhz"),
        ("max_auto_resumes", "--max-auto-resumes"),
        ("watch_poll_interval_seconds", "--watch-poll-interval-seconds"),
    ]
    for field_name, flag in scalar_flags:
        _append_cli_scalar_arg(cmd, flag, stored.get(field_name))

    _append_cli_repeatable_arg(cmd, "--hosts", stored.get("hosts"))
    _append_cli_repeatable_arg(cmd, "--labels", stored.get("labels"))

    return cmd


def build_benchmark_e2e_status_actions(
    run_id: str,
    *,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    query = urlencode({"run_id": str(run_id)})
    status_command = [
        python_executable or sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "run-e2e-status",
        "--run-id",
        str(run_id),
    ]
    watch_command = status_command + ["--watch"]
    watcher_command = [
        python_executable or sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "watch-e2e",
        "--run-id",
        str(run_id),
    ]
    return _json_safe(
        {
            "status_command": status_command,
            "status_command_shell": _cli_shell_join(status_command),
            "watch_command": watch_command,
            "watch_command_shell": _cli_shell_join(watch_command),
            "watcher_command": watcher_command,
            "watcher_command_shell": _cli_shell_join(watcher_command),
            "status_api_path": f"/api/benchmark/e2e-status?{query}",
            "watcher_api_path": "/api/benchmark/e2e-watch",
            "dashboard_path": f"/e2e?{query}",
            "preferred_mcp_tool": "benchmark_e2e_status",
            "watcher_mcp_tool": "benchmark_e2e_watch",
        }
    )


def build_benchmark_e2e_progress_surface_hint(
    run_id: str,
    *,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    actions = build_benchmark_e2e_status_actions(run_id, python_executable=python_executable)
    return _json_safe(
        {
            "kind": "normalized_e2e_status",
            "label": "Normalized live e2e status",
            "reason": (
                "Prefer this surface for current progress. It merges top-level progress, child run progress, "
                "watcher state, recent events, and orchestrator liveness; raw summary/checkpoint files can lag "
                "during long child stages."
            ),
            "status_command_shell": actions.get("status_command_shell"),
            "status_api_path": actions.get("status_api_path"),
            "dashboard_path": actions.get("dashboard_path"),
            "preferred_mcp_tool": actions.get("preferred_mcp_tool"),
        }
    )


def attach_benchmark_e2e_status_hints(
    payload: Dict[str, Any], run_id: Optional[str]
) -> Dict[str, Any]:
    resolved_run_id = str(run_id or payload.get("run_id") or "").strip()
    if not resolved_run_id:
        return _json_safe(payload)
    enriched = dict(payload)
    enriched.setdefault("actions", build_benchmark_e2e_status_actions(resolved_run_id))
    enriched.setdefault(
        "preferred_progress_source", build_benchmark_e2e_progress_surface_hint(resolved_run_id)
    )
    return _json_safe(enriched)


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _stage_snapshot(stage: Dict[str, Any]) -> Dict[str, Any]:
    attempts = list(stage.get("attempts") or [])
    latest_attempt = attempts[-1] if attempts else None
    benchmark_summary = None
    if isinstance(latest_attempt, dict):
        benchmark_summary = latest_attempt.get("benchmark_summary")
    if (
        benchmark_summary is None
        and isinstance(latest_attempt, dict)
        and isinstance(latest_attempt.get("result"), dict)
    ):
        benchmark_summary = _benchmark_stage_details(latest_attempt["result"])
    if benchmark_summary is None:
        benchmark_summary = stage.get("benchmark_summary")
    if benchmark_summary is None and isinstance(stage.get("result"), dict):
        benchmark_summary = stage["result"].get("benchmark_summary")
    if benchmark_summary is None and isinstance(stage.get("result"), dict):
        benchmark_summary = _benchmark_stage_details(stage["result"])

    def _collect_attempt_entries(summary_key: str) -> List[Dict[str, Any]]:
        latest_by_target_bucket: Dict[tuple[str, str], Dict[str, Any]] = {}
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_summary = attempt.get("benchmark_summary")
            if not isinstance(attempt_summary, dict):
                continue
            bucket = str(attempt.get("bucket") or "").strip()
            run_id = str(attempt.get("run_id") or "").strip() or None
            attempt_artifacts = (
                attempt.get("artifacts") if isinstance(attempt.get("artifacts"), dict) else {}
            )
            entries = list(attempt_summary.get(summary_key) or [])
            if summary_key == "target_outcomes" and not entries:
                entries = list(attempt_summary.get("failed_benchmarks") or []) + list(
                    attempt_summary.get("skipped_benchmarks") or []
                )
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                merged = dict(entry)
                if bucket and not merged.get("bucket"):
                    merged["bucket"] = bucket
                if run_id and not merged.get("run_id"):
                    merged["run_id"] = run_id
                for artifact_key in ("events_path", "output_json", "progress_path", "run_dir"):
                    artifact_value = attempt_artifacts.get(artifact_key)
                    if artifact_value and not merged.get(artifact_key):
                        merged[artifact_key] = artifact_value
                target = str(merged.get("target") or "").strip()
                latest_by_target_bucket[(target, str(merged.get("bucket") or "").strip())] = merged
        return list(latest_by_target_bucket.values())

    aggregated_target_outcomes = _collect_attempt_entries("target_outcomes")
    aggregated_failed_benchmarks = [
        entry
        for entry in aggregated_target_outcomes
        if str(entry.get("status") or "").startswith("failed")
    ]
    aggregated_skipped_benchmarks = [
        entry for entry in aggregated_target_outcomes if str(entry.get("status") or "") == "skipped"
    ]
    aggregated_status_counts: Dict[str, int] = {}
    for entry in aggregated_target_outcomes:
        status = str(entry.get("status") or "unknown")
        aggregated_status_counts[status] = aggregated_status_counts.get(status, 0) + 1

    payload = {
        "name": stage.get("name"),
        "enabled": bool(stage.get("enabled", False)),
        "status": stage.get("status"),
        "run_id": stage.get("run_id"),
        "issues": _dedupe_strings(list(stage.get("issues") or [])),
        "attempt_count": len(attempts),
        "latest_attempt_run_id": latest_attempt.get("run_id")
        if isinstance(latest_attempt, dict)
        else None,
        "latest_attempt_status": latest_attempt.get("status")
        if isinstance(latest_attempt, dict)
        else None,
        "latest_attempt_bucket": latest_attempt.get("bucket")
        if isinstance(latest_attempt, dict)
        else None,
        "latest_attempt_active_unit": latest_attempt.get("active_unit")
        if isinstance(latest_attempt, dict)
        else None,
        "latest_attempt_completed_units": list(latest_attempt.get("completed_units") or [])
        if isinstance(latest_attempt, dict)
        else [],
    }
    if aggregated_target_outcomes:
        payload["status_counts"] = aggregated_status_counts
    elif isinstance(benchmark_summary, dict):
        payload["status_counts"] = dict(benchmark_summary.get("status_counts") or {})
    if aggregated_target_outcomes:
        payload["failed_benchmarks"] = aggregated_failed_benchmarks
    elif isinstance(benchmark_summary, dict):
        payload["failed_benchmarks"] = list(benchmark_summary.get("failed_benchmarks") or [])
    if aggregated_target_outcomes:
        payload["skipped_benchmarks"] = aggregated_skipped_benchmarks
    elif isinstance(benchmark_summary, dict):
        payload["skipped_benchmarks"] = list(benchmark_summary.get("skipped_benchmarks") or [])
    return payload


def _effective_stage_snapshot_status(snapshot: Dict[str, Any]) -> str:
    stored_status = str(snapshot.get("status") or "unknown").strip() or "unknown"
    if stored_status in {"running", "planned", "aborted", "skipped", "skipped_duplicate"}:
        return stored_status
    failed_benchmarks = list(snapshot.get("failed_benchmarks") or [])
    if failed_benchmarks:
        return "failed"
    skipped_benchmarks = list(snapshot.get("skipped_benchmarks") or [])
    if skipped_benchmarks:
        return "partial"
    return stored_status


def _apply_effective_stage_snapshot_statuses(
    stage_snapshots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    effective_snapshots: List[Dict[str, Any]] = []
    for snapshot in stage_snapshots:
        effective_status = _effective_stage_snapshot_status(snapshot)
        if effective_status == snapshot.get("status"):
            effective_snapshots.append(snapshot)
            continue
        updated = dict(snapshot)
        updated["stored_status"] = snapshot.get("status")
        updated["status"] = effective_status
        effective_snapshots.append(updated)
    return effective_snapshots


def _tail_records(path: Path, *, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    return _read_jsonl(path)[-limit:]


def _slugify_issue_component(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return text.strip("_") or "unknown"


def _event_target(payload: Dict[str, Any]) -> str:
    chapter = str(payload.get("chapter") or "").strip()
    example = str(payload.get("example") or "").strip()
    if chapter and example:
        return f"{chapter}:{example}"
    return example or chapter or "<unknown>"


def _reported_failures_from_child_events(
    events_path: Path,
    *,
    stage_name: Optional[str],
    bucket: Optional[str],
) -> List[Dict[str, Any]]:
    latest_by_target: Dict[str, Dict[str, Any]] = {}
    for payload in _read_jsonl(events_path):
        if payload.get("event_type") != "example_end":
            continue
        target = _event_target(payload)
        latest_by_target[target] = {
            "target": target,
            "stage": stage_name,
            "bucket": bucket,
            "status": str(payload.get("status") or "").strip() or "unknown",
            "error": str(payload.get("error") or payload.get("failure_reason") or "").strip()
            or None,
            "failure_reason": str(payload.get("failure_reason") or "").strip() or None,
            "best_speedup": payload.get("best_speedup"),
            "best_memory_savings_pct": payload.get("best_memory_savings_pct"),
            "optimization_goal": str(payload.get("optimization_goal") or "").strip() or None,
            "timestamp": payload.get("timestamp"),
            "run_id": payload.get("run_id"),
            "events_path": str(events_path),
            "source": "live_child_event",
        }
    failures = [
        entry
        for entry in latest_by_target.values()
        if str(entry.get("status") or "").startswith("failed")
    ]
    failures.sort(
        key=lambda entry: (str(entry.get("timestamp") or ""), str(entry.get("target") or ""))
    )
    return failures


def _benchmark_failure_issue_row(
    failure: Dict[str, Any],
    *,
    actions: Dict[str, Any],
    status_paths: Dict[str, Any],
) -> Dict[str, Any]:
    stage_name = str(failure.get("stage") or "unknown").strip() or "unknown"
    bucket = str(failure.get("bucket") or "").strip() or None
    target = str(failure.get("target") or "<unknown>").strip() or "<unknown>"
    issue_id = _benchmark_failure_issue_id(stage_name=stage_name, bucket=bucket, target=target)
    symptom = f"{target} reported `{failure.get('status')}`"
    error = str(failure.get("error") or "").strip()
    if error:
        symptom = f"{symptom}: {error}"
    evidence_paths: Dict[str, Any] = {
        "status_summary_path": status_paths.get("summary_path"),
        "status_checkpoint_path": status_paths.get("checkpoint_path"),
    }
    for key in ("events_path", "output_json", "progress_path", "run_dir"):
        value = failure.get(key)
        if value:
            evidence_paths[key] = value
    return {
        "issue_id": issue_id,
        "stage": stage_name if not bucket else f"{stage_name}/{bucket}",
        "status": "reported",
        "symptom": symptom,
        "root_cause": "Pending triage.",
        "fixes": [],
        "verification": {
            "status_command": actions.get("status_command_shell"),
            "status_api_path": actions.get("status_api_path"),
            "preferred_mcp_tool": actions.get("preferred_mcp_tool"),
        },
        "evidence_paths": evidence_paths,
        "target": target,
        "bucket": bucket,
        "error": failure.get("error"),
        "failure_reason": failure.get("failure_reason"),
        "benchmark_status": failure.get("status"),
        "benchmark_run_id": failure.get("run_id"),
        "optimization_goal": failure.get("optimization_goal"),
        "best_speedup": failure.get("best_speedup"),
        "best_memory_savings_pct": failure.get("best_memory_savings_pct"),
        "reported_at": failure.get("timestamp"),
        "source": failure.get("source") or "stage_snapshot",
    }


def _benchmark_failure_issue_id(
    *,
    stage_name: str,
    bucket: Optional[str],
    target: str,
) -> str:
    issue_id_parts = ["reported", stage_name]
    if bucket:
        issue_id_parts.append(bucket)
    issue_id_parts.append(target)
    return "_".join(_slugify_issue_component(part) for part in issue_id_parts)


def _issue_error_text(row: Dict[str, Any]) -> str:
    for key in ("error", "failure_reason"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    symptom = str(row.get("symptom") or "").strip()
    if ": " in symptom:
        return symptom.split(": ", 1)[1].strip()
    return symptom


def _issue_signature(error_text: str) -> Tuple[str, str, Optional[str]]:
    normalized = re.sub(r"\s+", " ", str(error_text or "").strip())
    lowered = normalized.lower()
    if not normalized:
        return "unknown_failure", "Unknown failure", None
    if "received sighup" in lowered:
        return (
            "received_sighup",
            "received SIGHUP",
            "A benchmark process terminated on SIGHUP. Inspect profiler subprocess ownership and benchmark lifecycle handling.",
        )
    if "[errno 5]" in lowered or "input/output error" in lowered:
        return (
            "errno5_input_output_error",
            "[Errno 5] Input/output error",
            "Input/output errors appearing after an earlier termination usually indicate a cascading process-lifecycle or descriptor cleanup failure.",
        )
    if "timeout" in lowered:
        return (
            "timeout",
            "timeout",
            "The benchmark or profiler timed out. Inspect timeout controls, launch replay count, and child-process liveness.",
        )
    if "unsupported" in lowered:
        return (
            "unsupported_capability",
            "unsupported capability",
            "This looks capability-gated. Validate the required hardware/software support before treating it as a benchmark regression.",
        )
    compact = re.sub(r"0x[0-9a-f]+", "<hex>", lowered)
    compact = re.sub(r"\b\d+\b", "<num>", compact)
    compact = re.sub(r"/\\S+", "<path>", compact)
    compact = compact[:160]
    return (f"raw_{_slugify_issue_component(compact)}", normalized[:160], None)


def _issue_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    reported_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("issue_id") or "").strip().startswith("reported_")
        and str(row.get("status") or "").strip() != "resolved"
    ]
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in reported_rows:
        stage = str(row.get("stage") or "unknown").strip() or "unknown"
        benchmark_status = (
            str(row.get("benchmark_status") or row.get("status") or "unknown").strip() or "unknown"
        )
        signature_key, signature_label, hint = _issue_signature(_issue_error_text(row))
        key = (stage, benchmark_status, signature_key)
        bucket = str(row.get("bucket") or "").strip() or None
        reported_at = str(row.get("reported_at") or "").strip() or None
        entry = grouped.setdefault(
            key,
            {
                "group_id": f"group_{_slugify_issue_component(stage)}_{_slugify_issue_component(signature_key)}",
                "stage": stage,
                "bucket": bucket,
                "benchmark_status": benchmark_status,
                "signature_key": signature_key,
                "signature": signature_label,
                "root_cause_hint": hint,
                "count": 0,
                "issue_ids": [],
                "sample_targets": [],
                "first_reported_at": reported_at,
                "last_reported_at": reported_at,
            },
        )
        entry["count"] += 1
        entry["issue_ids"].append(row.get("issue_id"))
        target = str(row.get("target") or "").strip()
        if target and target not in entry["sample_targets"] and len(entry["sample_targets"]) < 5:
            entry["sample_targets"].append(target)
        if reported_at:
            if not entry.get("first_reported_at") or reported_at < str(
                entry.get("first_reported_at")
            ):
                entry["first_reported_at"] = reported_at
            if not entry.get("last_reported_at") or reported_at > str(
                entry.get("last_reported_at")
            ):
                entry["last_reported_at"] = reported_at

    groups = sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("stage") or ""),
            -int(item.get("count") or 0),
            str(item.get("signature_key") or ""),
        ),
    )
    groups_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for group in groups:
        groups_by_stage.setdefault(str(group.get("stage") or "unknown"), []).append(group)
    for stage_groups in groups_by_stage.values():
        has_signal_group = any(
            str(group.get("signature_key") or "") == "received_sighup" for group in stage_groups
        )
        if not has_signal_group:
            continue
        for group in stage_groups:
            if str(group.get("signature_key") or "") != "errno5_input_output_error":
                continue
            group["root_cause_hint"] = (
                "This I/O error cluster follows a signal-terminated benchmark in the same stage. Treat it as a likely cascade, not hundreds of independent regressions."
            )
            group["cascade_from"] = "received_sighup"
    return groups


def _issue_summary(
    rows: List[Dict[str, Any]],
    *,
    run_id: str,
    active_issue_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    active_issue_ids = set(active_issue_ids or set())
    resolved_count = sum(1 for row in rows if str(row.get("status") or "").strip() == "resolved")
    unresolved_rows = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "").strip() != "resolved"
    ]
    active_rows: List[Dict[str, Any]] = []
    historical_rows: List[Dict[str, Any]] = []
    for row in unresolved_rows:
        issue_id = str(row.get("issue_id") or "").strip()
        if issue_id.startswith("reported_"):
            if issue_id in active_issue_ids:
                active_rows.append(row)
            else:
                historical_rows.append(row)
            continue
        active_rows.append(row)

    issue_groups = _issue_groups(rows)
    active_issue_groups = _issue_groups(active_rows)
    historical_issue_groups = _issue_groups(historical_rows)

    return {
        "summary": {
            "run_id": run_id,
            "issue_count": len(rows),
            "reported_issue_count": sum(
                1 for row in rows if str(row.get("issue_id") or "").strip().startswith("reported_")
            ),
            "resolved_count": resolved_count,
            "unresolved_count": len(unresolved_rows),
            "issue_group_count": len(issue_groups),
            "active_issue_count": len(active_rows),
            "active_unresolved_count": len(active_rows),
            "active_reported_issue_count": sum(
                1
                for row in active_rows
                if str(row.get("issue_id") or "").strip().startswith("reported_")
            ),
            "active_issue_group_count": len(active_issue_groups),
            "historical_issue_count": len(historical_rows),
            "historical_unresolved_count": len(historical_rows),
            "historical_reported_issue_count": sum(
                1
                for row in historical_rows
                if str(row.get("issue_id") or "").strip().startswith("reported_")
            ),
            "historical_issue_group_count": len(historical_issue_groups),
        },
        "issue_groups": issue_groups,
        "active_issue_groups": active_issue_groups,
        "historical_issue_groups": historical_issue_groups,
    }


def _render_issue_evidence(row: Dict[str, Any]) -> str:
    evidence: List[str] = []
    target = str(row.get("target") or "").strip()
    if target:
        evidence.append(f"`{target}`")
    benchmark_status = str(row.get("benchmark_status") or "").strip()
    if benchmark_status:
        evidence.append(f"`{benchmark_status}`")
    reported_at = str(row.get("reported_at") or "").strip()
    if reported_at:
        evidence.append(f"`{reported_at}`")
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    status_command = str(verification.get("status_command") or "").strip()
    if status_command:
        evidence.append("`run-e2e-status`")
    evidence_paths = (
        row.get("evidence_paths") if isinstance(row.get("evidence_paths"), dict) else {}
    )
    events_path = str(evidence_paths.get("events_path") or "").strip()
    if events_path:
        evidence.append(f"`{Path(events_path).name}`")
    return "; ".join(evidence)


def _render_active_issue_ledger_markdown(ledger: Dict[str, Any]) -> str:
    summary = dict(ledger.get("summary") or {})
    lines = [
        "# Active Issue Ledger",
        "",
        f"- E2E run: `{summary.get('run_id', '')}`",
        f"- Issue count: `{summary.get('issue_count', 0)}`",
        f"- Active issue count: `{summary.get('active_issue_count', summary.get('issue_count', 0))}`",
        f"- Historical unresolved count: `{summary.get('historical_issue_count', 0)}`",
        f"- Resolved: `{summary.get('resolved_count', 0)}`",
        f"- Unresolved: `{summary.get('unresolved_count', 0)}`",
        "",
    ]
    groups = list(ledger.get("issue_groups") or [])
    if groups:
        lines.extend(
            [
                "## Incident Groups",
                "",
                "| Group | Stage | Count | Signature | Hint |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for group in groups:
            lines.append(
                f"| `{group.get('group_id', '')}` | `{group.get('stage', '')}` | `{group.get('count', 0)}` | "
                f"{group.get('signature', '')} | {group.get('root_cause_hint', '') or '-'} |"
            )
        lines.append("")
    lines.extend(
        [
            "| Issue | Stage | Status | Symptom | Root cause | Evidence |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ledger.get("rows") or []:
        lines.append(
            f"| `{row.get('issue_id', '')}` | `{row.get('stage', '')}` | `{row.get('status', '')}` | "
            f"{row.get('symptom', '')} | {row.get('root_cause', '')} | {_render_issue_evidence(row)} |"
        )
    return "\n".join(lines) + "\n"


def _sync_active_issue_ledger(
    *,
    run_dir: Path,
    repo_root: Path,
    run_id: str,
    actions: Dict[str, Any],
    status_paths: Dict[str, Any],
    aggregate_failures: List[Dict[str, Any]],
    reported_failures: List[Dict[str, Any]],
    active_failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    active_issue_json = run_dir / "active_issue_ledger.json"
    active_issue_md = run_dir / "active_issue_ledger.md"
    historical_issue_json = run_dir / "historical_failure_ledger.json"
    historical_issue_md = run_dir / "historical_failure_ledger.md"

    existing_payload = _read_json_if_exists(active_issue_json) or {}
    existing_rows = list(existing_payload.get("rows") or [])
    existing_rows_by_id = {
        str(row.get("issue_id") or "").strip(): row
        for row in existing_rows
        if isinstance(row, dict) and str(row.get("issue_id") or "").strip()
    }
    manual_rows = []
    for row in existing_rows:
        issue_id = str(row.get("issue_id") or "").strip()
        if not issue_id.startswith("reported_"):
            manual_rows.append(row)
    auto_rows_by_id: Dict[str, Dict[str, Any]] = {}
    for failure in list(aggregate_failures or []) + list(reported_failures or []):
        row = _benchmark_failure_issue_row(
            failure,
            actions=actions,
            status_paths=status_paths,
        )
        issue_id = str(row["issue_id"])
        existing = existing_rows_by_id.get(issue_id)
        if isinstance(existing, dict):
            merged = dict(row)
            existing_status = str(existing.get("status") or "").strip()
            if existing_status and existing_status != "reported":
                merged["status"] = existing_status
            for field in (
                "root_cause",
                "fixes",
                "verification",
                "notes",
                "resolved_at",
                "resolved_by",
                "resolution_run_id",
            ):
                if existing.get(field):
                    merged[field] = existing[field]
            existing_symptom = str(existing.get("symptom") or "").strip()
            if existing_symptom and existing_symptom != str(row.get("symptom") or "").strip():
                merged["symptom"] = existing_symptom
            existing_evidence = existing.get("evidence_paths")
            if isinstance(existing_evidence, dict):
                merged_evidence = dict(row.get("evidence_paths") or {})
                merged_evidence.update(existing_evidence)
                merged["evidence_paths"] = merged_evidence
            row = merged
        auto_rows_by_id[issue_id] = row
    rows = manual_rows + [auto_rows_by_id[key] for key in sorted(auto_rows_by_id)]
    active_issue_ids = {
        _benchmark_failure_issue_id(
            stage_name=str(failure.get("stage") or "unknown").strip() or "unknown",
            bucket=str(failure.get("bucket") or "").strip() or None,
            target=str(failure.get("target") or "<unknown>").strip() or "<unknown>",
        )
        for failure in list(active_failures or [])
    }
    issue_summary = _issue_summary(rows, run_id=run_id, active_issue_ids=active_issue_ids)
    ledger_payload = _sanitize_persisted_value(
        {
            "schema_version": "1.0",
            "preferred_collection_key": "rows",
            "collection_aliases": {"issues": "rows"},
            "summary": dict(issue_summary["summary"]),
            "rows": rows,
            "issue_groups": list(issue_summary["issue_groups"]),
            "active_issue_groups": list(issue_summary["active_issue_groups"]),
            "historical_issue_groups": list(issue_summary["historical_issue_groups"]),
        },
        repo_root=repo_root,
    )
    _write_json(active_issue_json, ledger_payload)
    active_issue_md.write_text(
        _render_active_issue_ledger_markdown(ledger_payload), encoding="utf-8"
    )
    return _sanitize_persisted_value(
        {
            "active_issue_ledger_json": str(active_issue_json),
            "active_issue_ledger_md": str(active_issue_md),
            "historical_failure_ledger_json": str(historical_issue_json)
            if historical_issue_json.exists()
            else None,
            "historical_failure_ledger_md": str(historical_issue_md)
            if historical_issue_md.exists()
            else None,
            "summary": dict(ledger_payload["summary"]),
            "rows": rows,
            "issue_groups": list(ledger_payload["issue_groups"]),
            "active_issue_groups": list(ledger_payload["active_issue_groups"]),
            "historical_issue_groups": list(ledger_payload["historical_issue_groups"]),
        },
        repo_root=repo_root,
    )


def _existing_ledger_refs(run_dir: Path) -> Dict[str, Optional[str]]:
    return {
        "active_issue_ledger_json": str(run_dir / "active_issue_ledger.json")
        if (run_dir / "active_issue_ledger.json").exists()
        else None,
        "active_issue_ledger_md": str(run_dir / "active_issue_ledger.md")
        if (run_dir / "active_issue_ledger.md").exists()
        else None,
        "historical_failure_ledger_json": str(run_dir / "historical_failure_ledger.json")
        if (run_dir / "historical_failure_ledger.json").exists()
        else None,
        "historical_failure_ledger_md": str(run_dir / "historical_failure_ledger.md")
        if (run_dir / "historical_failure_ledger.md").exists()
        else None,
    }


def _watcher_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_if_exists(e2e_watcher_status_path(run_dir))


def inspect_benchmark_e2e_sweep_run(
    *,
    run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
    artifacts_dir: Optional[str] = None,
    recent_events_limit: int = 10,
) -> Dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()
    resolved_run_id = (
        resolve_e2e_run_id(run_id, repo_root=root)
        if run_id
        else resolve_latest_e2e_run_id(repo_root=root)
    )
    if not resolved_run_id:
        return {
            "success": False,
            "error": f"No e2e runs found under {e2e_runs_root(root)}",
            "run_state": "missing",
            "overall_status": "missing",
        }

    run_dir = e2e_run_dir(resolved_run_id, root)
    if not run_dir.exists():
        return {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "error": f"Missing run dir: {run_dir}",
            "run_state": "missing",
            "overall_status": "missing",
        }

    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.json"
    progress_path = e2e_progress_path(run_dir)
    events_path = run_dir / "events.jsonl"
    summary = _read_json_if_exists(summary_path) or {}
    checkpoint = _read_json_if_exists(checkpoint_path) or {}
    progress = _read_json_if_exists(progress_path) or {}
    current = progress.get("current") if isinstance(progress.get("current"), dict) else {}
    metrics = current.get("metrics") if isinstance(current.get("metrics"), dict) else {}
    persisted_run_state = checkpoint.get("run_state") or summary.get("run_state")
    terminal_persisted_state = str(persisted_run_state or "").strip() in {"completed", "aborted"}
    if terminal_persisted_state:
        stages = checkpoint.get("stages")
        if not isinstance(stages, list):
            stages = summary.get("stages")
        if not isinstance(stages, list):
            stages = metrics.get("stages")
    else:
        stages = metrics.get("stages")
        if not isinstance(stages, list):
            stages = checkpoint.get("stages")
        if not isinstance(stages, list):
            stages = summary.get("stages")
    if not isinstance(stages, list):
        stages = []

    contract = checkpoint.get("contract")
    if not isinstance(contract, dict):
        contract = summary.get("contract")
    if not isinstance(contract, dict):
        contract = {}

    run_state = (
        metrics.get("run_state")
        or checkpoint.get("run_state")
        or summary.get("run_state")
        or "unknown"
    )
    overall_status = (
        metrics.get("overall_status")
        or checkpoint.get("overall_status")
        or summary.get("overall_status")
        or "unknown"
    )
    stored_resume_available = bool(
        checkpoint.get("resume_available", summary.get("resume_available", False))
    )
    resume_available = stored_resume_available
    orchestrator_pid = metrics.get("orchestrator_pid")
    if orchestrator_pid is None:
        orchestrator_pid = checkpoint.get("orchestrator_pid", summary.get("orchestrator_pid"))
    orchestrator_live = _pid_is_live(orchestrator_pid)
    progress_timestamp = current.get("timestamp")
    progress_age_seconds = _age_seconds(progress_timestamp)
    summary_updated_at = summary.get("updated_at")
    checkpoint_updated_at = checkpoint.get("updated_at")
    summary_age_seconds = _age_seconds(summary_updated_at)
    checkpoint_age_seconds = _age_seconds(checkpoint_updated_at)
    summary_progress_lag_seconds = None
    if progress_age_seconds is not None and summary_age_seconds is not None:
        summary_progress_lag_seconds = max(0.0, summary_age_seconds - progress_age_seconds)
    checkpoint_progress_lag_seconds = None
    if progress_age_seconds is not None and checkpoint_age_seconds is not None:
        checkpoint_progress_lag_seconds = max(0.0, checkpoint_age_seconds - progress_age_seconds)

    inferred_state = "unknown"
    status_notes: List[str] = []
    if str(run_state) == "completed":
        inferred_state = "completed"
    elif str(run_state) == "aborted" and resume_available:
        inferred_state = "aborted_resume_available"
    elif str(run_state) == "running" and orchestrator_live:
        inferred_state = "running_live"
    elif str(run_state) == "running":
        inferred_state = "running_stale"
        status_notes.append("run is marked running but the orchestrator pid is not live")
        if not resume_available:
            # A stale running package with persisted state is manually resumable even if the
            # original checkpoint never flipped resume_available before the orchestrator died.
            resume_available = summary_path.exists() or checkpoint_path.exists()
            if resume_available:
                status_notes.append(
                    "stored resume_available was false. Corrected to true for stale running package"
                )
    elif str(run_state) == "aborted":
        inferred_state = "aborted_terminal"
    else:
        inferred_state = f"state_{run_state}"

    if summary_progress_lag_seconds and summary_progress_lag_seconds > 30.0:
        status_notes.append(
            f"summary/checkpoint persistence lags live progress by ~{summary_progress_lag_seconds:.0f}s"
        )

    current_stage_name = metrics.get("current_stage")
    current_child_run_id = metrics.get("current_stage_run_id")
    current_bucket = metrics.get("current_bucket")
    child_artifacts: Dict[str, Optional[str]] = {}
    recent_child_events: List[Dict[str, Any]] = []
    child_reported_failures: List[Dict[str, Any]] = []
    if isinstance(current_child_run_id, str) and current_child_run_id.strip():
        if current_stage_name in {"tier1", "full_sweep"}:
            benchmark_paths = _benchmark_run_event_paths(
                current_child_run_id,
                repo_root=root,
                artifacts_dir=artifacts_dir or contract.get("artifacts_dir"),
            )
            child_artifacts = {
                "run_dir": str(benchmark_paths["run_dir"]),
                "progress_path": str(benchmark_paths["progress"]),
                "events_path": str(benchmark_paths["events"]),
                "output_json": str(benchmark_paths["output_json"]),
            }
            if benchmark_paths["events"].exists():
                recent_child_events = _tail_records(
                    benchmark_paths["events"], limit=recent_events_limit
                )
                child_reported_failures = _reported_failures_from_child_events(
                    benchmark_paths["events"],
                    stage_name=current_stage_name,
                    bucket=current_bucket,
                )
        elif current_stage_name in {"cluster", "fabric"}:
            child_progress = _cluster_run_progress_path(current_child_run_id, repo_root=root)
            child_artifacts = {
                "run_dir": str(child_progress.parents[1]),
                "progress_path": str(child_progress),
            }

    watcher = dict(_watcher_summary(run_dir) or {})
    watcher_pid = watcher.get("watcher_pid")
    watcher_live = _pid_is_live(watcher_pid)
    stored_watch_state = watcher.get("watch_state")
    effective_watch_state = stored_watch_state
    if not watcher_live and str(stored_watch_state or "").strip() in {
        "watching",
        "resuming",
        "launched",
    }:
        effective_watch_state = "stale_dead"
        status_notes.append(
            f"stored watcher watch_state `{stored_watch_state}` corrected to `stale_dead` because the watcher pid is not live"
        )
    if watcher:
        watcher["stored_watch_state"] = stored_watch_state
        watcher["watch_state"] = effective_watch_state
        watcher["watcher_live"] = watcher_live
    actions = build_benchmark_e2e_status_actions(resolved_run_id, python_executable=sys.executable)

    stage_snapshots = _apply_effective_stage_snapshot_statuses(
        [_stage_snapshot(stage) for stage in stages if isinstance(stage, dict)]
    )
    aggregate_failures: List[Dict[str, Any]] = []
    for stage in stage_snapshots:
        for entry in stage.get("failed_benchmarks") or []:
            aggregate_failures.append({"stage": stage.get("name"), **entry})
    seen_failure_keys = {
        (
            str(entry.get("stage") or "").strip(),
            str(entry.get("bucket") or "").strip(),
            str(entry.get("target") or "").strip(),
        )
        for entry in aggregate_failures
    }
    for entry in child_reported_failures:
        key = (
            str(entry.get("stage") or "").strip(),
            str(entry.get("bucket") or "").strip(),
            str(entry.get("target") or "").strip(),
        )
        if key in seen_failure_keys:
            continue
        aggregate_failures.append(dict(entry))
        seen_failure_keys.add(key)

    progress_source_path = None
    if isinstance(child_artifacts, dict):
        progress_source_path = child_artifacts.get("progress_path")
    if not progress_source_path:
        progress_source_path = str(progress_path)
    child_progress_available = False
    child_progress_candidate = (
        child_artifacts.get("progress_path") if isinstance(child_artifacts, dict) else None
    )
    if isinstance(child_progress_candidate, str) and child_progress_candidate.strip():
        child_progress_available = Path(child_progress_candidate).exists()
    has_live_child_progress = bool(
        str(run_state) == "running"
        and current_child_run_id
        and progress_timestamp
        and (child_progress_available or recent_child_events)
    )
    surfaced_failures = list(child_reported_failures or [])
    if (
        not surfaced_failures
        and not has_live_child_progress
        and not (str(run_state) == "running" and orchestrator_live)
    ):
        surfaced_failures = list(aggregate_failures)

    recent_events = _tail_records(events_path, limit=recent_events_limit)
    resume_command = build_benchmark_e2e_resume_command(
        resolved_run_id,
        contract=contract,
        python_executable=sys.executable,
        repo_root=root,
    )
    resume_blocked_reason = _auto_resume_reconstruction_error(contract, repo_root=root)
    stored_provenance = checkpoint.get("provenance")
    if not isinstance(stored_provenance, dict):
        stored_provenance = summary.get("provenance")
    stored_git = stored_provenance.get("git") if isinstance(stored_provenance, dict) else None
    current_git = get_git_info()
    if isinstance(stored_git, dict) and bool(stored_git.get("dirty")):
        resume_blocked_reason = "stored worktree is dirty"
    elif bool(current_git.get("dirty")):
        resume_blocked_reason = "current worktree is dirty"
    if resume_blocked_reason:
        resume_command = []
        status_notes.append(f"auto-resume blocked: {resume_blocked_reason}")
    progress_source_kind = "checkpoint_summary"
    progress_source_label = "Checkpoint and summary"
    progress_source_reason = "Run is terminal or does not currently expose fresher child progress, so checkpoint/summary are the active status records."
    if has_live_child_progress:
        progress_source_kind = "live_child_progress"
        progress_source_label = "Live child progress"
        progress_source_reason = "Current progress is mirrored from the active child run and cross-checked with top-level progress plus orchestrator liveness."
    elif str(run_state) == "running" and progress_timestamp:
        progress_source_kind = "top_level_progress"
        progress_source_label = "Top-level progress"
        progress_source_reason = "Current progress comes from top-level progress.json because no child run progress path is attached for the active stage."
    progress_source = _json_safe(
        {
            "kind": progress_source_kind,
            "label": progress_source_label,
            "reason": progress_source_reason,
            "progress_timestamp": progress_timestamp,
            "progress_path": progress_source_path,
            "summary_progress_lag_seconds": summary_progress_lag_seconds,
            "checkpoint_progress_lag_seconds": checkpoint_progress_lag_seconds,
            "status_command_shell": actions.get("status_command_shell"),
            "status_api_path": actions.get("status_api_path"),
            "dashboard_path": actions.get("dashboard_path"),
            "preferred_mcp_tool": actions.get("preferred_mcp_tool"),
        }
    )

    status_paths = {
        "summary_path": str(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "progress_path": str(progress_path),
        "events_path": str(events_path),
        "watcher_status_path": str(e2e_watcher_status_path(run_dir)),
        "watcher_launch_log_path": str(e2e_watcher_launch_log_path(run_dir, resolved_run_id)),
    }
    ledgers = _sync_active_issue_ledger(
        run_dir=run_dir,
        repo_root=root,
        run_id=resolved_run_id,
        actions=actions,
        status_paths=status_paths,
        aggregate_failures=aggregate_failures,
        reported_failures=child_reported_failures,
        active_failures=surfaced_failures,
    )
    resolved_reported_issue_ids = {
        str(row.get("issue_id") or "").strip()
        for row in list(ledgers.get("rows") or [])
        if isinstance(row, dict)
        and str(row.get("status") or "").strip() == "resolved"
        and str(row.get("issue_id") or "").strip().startswith("reported_")
    }
    if resolved_reported_issue_ids:
        surfaced_failures = [
            entry
            for entry in surfaced_failures
            if _benchmark_failure_issue_id(
                stage_name=str(entry.get("stage") or "unknown").strip() or "unknown",
                bucket=str(entry.get("bucket") or "").strip() or None,
                target=str(entry.get("target") or "<unknown>").strip() or "<unknown>",
            )
            not in resolved_reported_issue_ids
        ]

    effective_overall_status = overall_status
    if str(run_state) != "running":
        corrected_stage_statuses = [
            str(stage.get("status") or "").strip()
            for stage in stage_snapshots
            if isinstance(stage, dict)
        ]
        rolled_up_status = _roll_up_overall_status(corrected_stage_statuses)
        if rolled_up_status and rolled_up_status != overall_status:
            status_notes.append(
                f"stored overall_status `{overall_status}` corrected to `{rolled_up_status}` from stage benchmark summaries"
            )
            effective_overall_status = rolled_up_status

    inspection_success = str(inferred_state) == "running_live" or (
        str(inferred_state) == "completed" and effective_overall_status == "succeeded"
    )

    return _sanitize_persisted_value(
        {
            "success": inspection_success,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "run_state": run_state,
            "overall_status": effective_overall_status,
            "stored_overall_status": overall_status,
            "stored_resume_available": stored_resume_available,
            "inferred_state": inferred_state,
            "resume_available": resume_available,
            "notes": status_notes,
            "progress_source": progress_source,
            "paths": {
                **status_paths,
            },
            "timestamps": {
                "progress_timestamp": progress_timestamp,
                "summary_updated_at": summary_updated_at,
                "checkpoint_updated_at": checkpoint_updated_at,
            },
            "ages_seconds": {
                "progress": progress_age_seconds,
                "summary": summary_age_seconds,
                "checkpoint": checkpoint_age_seconds,
                "summary_progress_lag": summary_progress_lag_seconds,
                "checkpoint_progress_lag": checkpoint_progress_lag_seconds,
            },
            "liveness": {
                "orchestrator_pid": orchestrator_pid,
                "orchestrator_live": orchestrator_live,
                "watcher_pid": watcher_pid,
                "watcher_live": watcher_live,
            },
            "contract": contract,
            "watcher": watcher or None,
            "current": {
                "stage": current_stage_name,
                "bucket": current_bucket,
                "child_run_id": current_child_run_id,
                "step": current.get("step"),
                "detail": current.get("step_detail"),
                "percent_complete": current.get("percent_complete"),
                "elapsed_seconds": current.get("elapsed_seconds"),
                "eta_seconds": current.get("eta_seconds"),
                "child_progress": metrics.get("child_progress"),
                "child_artifacts": child_artifacts or None,
                "recent_child_events": recent_child_events,
                "reported_failures": surfaced_failures,
            },
            "stages": stage_snapshots,
            "aggregate_failures": aggregate_failures,
            "issue_groups": ledgers.get("active_issue_groups") or [],
            "historical_issue_groups": ledgers.get("historical_issue_groups") or [],
            "recent_events": recent_events,
            "ledgers": ledgers,
            "actions": {
                **actions,
                "resume_command": resume_command,
                "resume_command_shell": _cli_shell_join(resume_command),
                "resume_blocked_reason": resume_blocked_reason,
            },
        },
        repo_root=root,
    )


def render_benchmark_e2e_status_text(status: Dict[str, Any]) -> str:
    if not status.get("run_id"):
        return json.dumps(status, indent=2, sort_keys=True)
    current = status.get("current") or {}
    liveness = status.get("liveness") or {}
    watcher = status.get("watcher") or {}
    lines = [
        f"run_id={status.get('run_id')}",
        f"state={status.get('inferred_state')} run_state={status.get('run_state')} overall_status={status.get('overall_status')}",
        f"orchestrator_pid={liveness.get('orchestrator_pid')} live={liveness.get('orchestrator_live')}",
    ]
    stored_overall_status = status.get("stored_overall_status")
    if stored_overall_status and stored_overall_status != status.get("overall_status"):
        lines.append(f"stored_overall_status={stored_overall_status}")
    stored_resume_available = status.get("stored_resume_available")
    if stored_resume_available is not None and stored_resume_available != status.get(
        "resume_available"
    ):
        lines.append(f"stored_resume_available={stored_resume_available}")
    progress_source = status.get("progress_source") or {}
    if progress_source.get("kind"):
        lines.append(
            f"progress_source={progress_source.get('kind')} ts={progress_source.get('progress_timestamp')} "
            f"path={progress_source.get('progress_path')}"
        )
    if current.get("stage"):
        lines.append(
            "current="
            + f"{current.get('stage')}"
            + (f"/{current.get('bucket')}" if current.get("bucket") else "")
            + f" step={current.get('step')} pct={current.get('percent_complete')}"
        )
    if current.get("detail"):
        lines.append(f"detail={current.get('detail')}")
    if watcher:
        lines.append(
            f"watcher_pid={watcher.get('watcher_pid')} watcher_state={watcher.get('watch_state')} "
            f"auto_resume_count={watcher.get('auto_resume_count')}"
        )
        if watcher.get("stored_watch_state") and watcher.get("stored_watch_state") != watcher.get(
            "watch_state"
        ):
            lines.append(f"stored_watcher_state={watcher.get('stored_watch_state')}")
    reported_failures = list(current.get("reported_failures") or [])
    if reported_failures:
        lines.append(f"reported_failures={len(reported_failures)}")
        preview = ", ".join(
            f"{entry.get('target')}[{entry.get('status')}]" for entry in reported_failures[:5]
        )
        if preview:
            lines.append(f"reported_failure_preview={preview}")
    ledgers = status.get("ledgers") or {}
    ledger_summary = dict(ledgers.get("summary") or {})
    if ledger_summary:
        active_issue_count = ledger_summary.get(
            "active_issue_count", ledger_summary.get("issue_count")
        )
        active_unresolved_count = ledger_summary.get(
            "active_unresolved_count", ledger_summary.get("unresolved_count")
        )
        lines.append(
            "active_issue_counts="
            + f"issues={active_issue_count} "
            + f"unresolved={active_unresolved_count}"
        )
        historical_issue_count = ledger_summary.get("historical_issue_count", 0)
        historical_unresolved_count = ledger_summary.get("historical_unresolved_count", 0)
        if historical_issue_count:
            lines.append(
                "historical_issue_counts="
                + f"issues={historical_issue_count} "
                + f"unresolved={historical_unresolved_count}"
            )
        lines.append(
            "ledger_totals="
            + f"issues={ledger_summary.get('issue_count')} "
            + f"resolved={ledger_summary.get('resolved_count')} "
            + f"unresolved={ledger_summary.get('unresolved_count')}"
        )
    issue_groups = list(status.get("issue_groups") or [])
    if issue_groups:
        lines.append(f"issue_groups={len(issue_groups)}")
        preview = "; ".join(
            f"{entry.get('stage')}:{entry.get('signature')} x{entry.get('count')}"
            for entry in issue_groups[:3]
        )
        if preview:
            lines.append(f"issue_group_preview={preview}")
    historical_issue_groups = list(status.get("historical_issue_groups") or [])
    if historical_issue_groups:
        lines.append(f"historical_issue_groups={len(historical_issue_groups)}")
        preview = "; ".join(
            f"{entry.get('stage')}:{entry.get('signature')} x{entry.get('count')}"
            for entry in historical_issue_groups[:3]
        )
        if preview:
            lines.append(f"historical_issue_group_preview={preview}")
    if ledgers.get("active_issue_ledger_json"):
        lines.append(f"active_issue_ledger={ledgers.get('active_issue_ledger_json')}")
    notes = list(status.get("notes") or [])
    for note in notes:
        lines.append(f"note={note}")
    lines.append(f"resume_available={status.get('resume_available')}")
    return "\n".join(lines)


def _watcher_status_payload(
    *,
    run_id: str,
    watcher_pid: int,
    watch_state: str,
    poll_interval_seconds: int,
    max_auto_resumes: int,
    auto_resume_count: int,
    last_action: Optional[Dict[str, Any]],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    current = dict(snapshot.get("current") or {})
    compact_current = {
        "stage": current.get("stage"),
        "bucket": current.get("bucket"),
        "child_run_id": current.get("child_run_id"),
        "step": current.get("step"),
        "detail": current.get("detail"),
        "percent_complete": current.get("percent_complete"),
        "elapsed_seconds": current.get("elapsed_seconds"),
        "eta_seconds": current.get("eta_seconds"),
    }
    compact_snapshot = {
        "run_id": snapshot.get("run_id"),
        "run_state": snapshot.get("run_state"),
        "overall_status": snapshot.get("overall_status"),
        "inferred_state": snapshot.get("inferred_state"),
        "resume_available": snapshot.get("resume_available"),
        "notes": list(snapshot.get("notes") or []),
        "liveness": dict(snapshot.get("liveness") or {}),
        "current": compact_current,
        "recent_events": list(snapshot.get("recent_events") or []),
        "ledgers": dict(snapshot.get("ledgers") or {}),
    }
    return _json_safe(
        {
            "run_id": run_id,
            "timestamp": _utc_now(),
            "watcher_pid": watcher_pid,
            "watch_state": watch_state,
            "poll_interval_seconds": poll_interval_seconds,
            "max_auto_resumes": max_auto_resumes,
            "auto_resume_count": auto_resume_count,
            "last_action": last_action,
            "last_snapshot": compact_snapshot,
        }
    )


def watch_benchmark_e2e_sweep_foreground(
    *,
    run_id: str,
    repo_root: Optional[Path] = None,
    poll_interval_seconds: int = _E2E_WATCHER_POLL_SECONDS,
    max_auto_resumes: int = _E2E_WATCHER_MAX_AUTO_RESUMES,
) -> Dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()

    def _public(payload: Dict[str, Any]) -> Dict[str, Any]:
        return _sanitize_persisted_value(payload, repo_root=root)

    run_dir = e2e_run_dir(run_id, root)
    if not run_dir.exists():
        return _public(
            {
                "success": False,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "error": f"Missing run dir: {run_dir}",
            }
        )

    watch_status_path = e2e_watcher_status_path(run_dir)
    events_path = run_dir / "events.jsonl"
    auto_resume_count = 0
    last_action: Optional[Dict[str, Any]] = None

    while True:
        snapshot = inspect_benchmark_e2e_sweep_run(run_id=run_id, repo_root=root)
        inferred_state = str(snapshot.get("inferred_state") or "unknown")
        watch_state = "watching"
        if inferred_state == "completed":
            watch_state = "completed"
            _write_json(
                watch_status_path,
                _watcher_status_payload(
                    run_id=run_id,
                    watcher_pid=os.getpid(),
                    watch_state=watch_state,
                    poll_interval_seconds=poll_interval_seconds,
                    max_auto_resumes=max_auto_resumes,
                    auto_resume_count=auto_resume_count,
                    last_action=last_action,
                    snapshot=snapshot,
                ),
            )
            return _public(
                {
                    "success": snapshot.get("success") is True,
                    "run_id": run_id,
                    "watch_state": watch_state,
                    "auto_resume_count": auto_resume_count,
                    "watcher_status_path": str(watch_status_path),
                    **(
                        {}
                        if snapshot.get("success") is True
                        else {
                            "error": (
                                "E2E run completed with terminal status "
                                f"{snapshot.get('overall_status') or 'unknown'}"
                            )
                        }
                    ),
                }
            )

        if inferred_state == "aborted_terminal":
            watch_state = "failed"
            _write_json(
                watch_status_path,
                _watcher_status_payload(
                    run_id=run_id,
                    watcher_pid=os.getpid(),
                    watch_state=watch_state,
                    poll_interval_seconds=poll_interval_seconds,
                    max_auto_resumes=max_auto_resumes,
                    auto_resume_count=auto_resume_count,
                    last_action=last_action,
                    snapshot=snapshot,
                ),
            )
            return _public(
                {
                    "success": False,
                    "run_id": run_id,
                    "watch_state": watch_state,
                    "auto_resume_count": auto_resume_count,
                    "watcher_status_path": str(watch_status_path),
                    "error": "E2E run ended in a non-resumable aborted state",
                }
            )

        if inferred_state in {"running_stale", "aborted_resume_available"}:
            if auto_resume_count >= max_auto_resumes:
                watch_state = "exhausted"
                last_action = {
                    "timestamp": _utc_now(),
                    "action": "auto_resume_skipped",
                    "reason": f"max_auto_resumes={max_auto_resumes} exhausted",
                }
                _append_event(
                    events_path,
                    "auto_resume_exhausted",
                    run_id=run_id,
                    max_auto_resumes=max_auto_resumes,
                )
                _write_json(
                    watch_status_path,
                    _watcher_status_payload(
                        run_id=run_id,
                        watcher_pid=os.getpid(),
                        watch_state=watch_state,
                        poll_interval_seconds=poll_interval_seconds,
                        max_auto_resumes=max_auto_resumes,
                        auto_resume_count=auto_resume_count,
                        last_action=last_action,
                        snapshot=snapshot,
                    ),
                )
                return _public(
                    {
                        "success": False,
                        "run_id": run_id,
                        "watch_state": watch_state,
                        "auto_resume_count": auto_resume_count,
                        "watcher_status_path": str(watch_status_path),
                        "error": f"max_auto_resumes={max_auto_resumes} exhausted",
                    }
                )

            resume_command = list(snapshot.get("actions", {}).get("resume_command") or [])
            if not resume_command:
                blocked_reason = str(
                    snapshot.get("actions", {}).get("resume_blocked_reason")
                    or "missing resume command"
                )
                watch_state = "resume_blocked"
                last_action = {
                    "timestamp": _utc_now(),
                    "action": "auto_resume_skipped",
                    "reason": blocked_reason,
                }
                _append_event(
                    events_path,
                    "auto_resume_blocked",
                    run_id=run_id,
                    reason=blocked_reason,
                )
                _write_json(
                    watch_status_path,
                    _watcher_status_payload(
                        run_id=run_id,
                        watcher_pid=os.getpid(),
                        watch_state=watch_state,
                        poll_interval_seconds=poll_interval_seconds,
                        max_auto_resumes=max_auto_resumes,
                        auto_resume_count=auto_resume_count,
                        last_action=last_action,
                        snapshot=snapshot,
                    ),
                )
                return _public(
                    {
                        "success": False,
                        "run_id": run_id,
                        "watch_state": watch_state,
                        "auto_resume_count": auto_resume_count,
                        "watcher_status_path": str(watch_status_path),
                        "error": blocked_reason,
                    }
                )
            else:
                watch_state = "resuming"
                auto_resume_count += 1
                last_action = {
                    "timestamp": _utc_now(),
                    "action": "auto_resume_started",
                    "attempt": auto_resume_count,
                    "command": resume_command,
                }
                _append_event(
                    events_path,
                    "auto_resume_started",
                    run_id=run_id,
                    attempt=auto_resume_count,
                    command=resume_command,
                )
                _write_json(
                    watch_status_path,
                    _watcher_status_payload(
                        run_id=run_id,
                        watcher_pid=os.getpid(),
                        watch_state=watch_state,
                        poll_interval_seconds=poll_interval_seconds,
                        max_auto_resumes=max_auto_resumes,
                        auto_resume_count=auto_resume_count,
                        last_action=last_action,
                        snapshot=snapshot,
                    ),
                )
                env = dict(os.environ)
                env[_E2E_WATCHER_SUPERVISED_ENV] = "1"
                returncode = subprocess.call(resume_command, cwd=str(root), env=env)
                last_action = {
                    "timestamp": _utc_now(),
                    "action": "auto_resume_finished",
                    "attempt": auto_resume_count,
                    "command": resume_command,
                    "returncode": returncode,
                }
                _append_event(
                    events_path,
                    "auto_resume_finished",
                    run_id=run_id,
                    attempt=auto_resume_count,
                    command=resume_command,
                    returncode=returncode,
                )
                continue

        _write_json(
            watch_status_path,
            _watcher_status_payload(
                run_id=run_id,
                watcher_pid=os.getpid(),
                watch_state=watch_state,
                poll_interval_seconds=poll_interval_seconds,
                max_auto_resumes=max_auto_resumes,
                auto_resume_count=auto_resume_count,
                last_action=last_action,
                snapshot=snapshot,
            ),
        )
        time.sleep(max(1, int(poll_interval_seconds)))


def watch_benchmark_e2e_sweep_run(
    *,
    run_id: str,
    repo_root: Optional[Path] = None,
    poll_interval_seconds: int = _E2E_WATCHER_POLL_SECONDS,
    max_auto_resumes: int = _E2E_WATCHER_MAX_AUTO_RESUMES,
) -> Dict[str, Any]:
    root = Path(repo_root or _repo_root()).resolve()
    run_dir = e2e_run_dir(run_id, root)
    if not run_dir.exists():
        return {"success": False, "error": f"Missing run dir: {run_dir}"}
    script = _watch_e2e_sweep_script()
    if not script.exists():
        return {"success": False, "error": f"Missing script: {script}"}

    status_path = e2e_watcher_status_path(run_dir)
    existing = _read_json_if_exists(status_path) or {}
    existing_pid = existing.get("watcher_pid")
    if _pid_is_live(existing_pid):
        return {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "watcher_pid": existing_pid,
            "watch_status_path": str(status_path),
            "launch_log_path": str(e2e_watcher_launch_log_path(run_dir, run_id)),
            "already_running": True,
        }

    launch_log_path = e2e_watcher_launch_log_path(run_dir, run_id)
    launch_log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(root),
        "--run-id",
        run_id,
        "--poll-interval-seconds",
        str(int(poll_interval_seconds)),
        "--max-auto-resumes",
        str(int(max_auto_resumes)),
    ]
    with launch_log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(
        status_path,
        {
            "run_id": run_id,
            "timestamp": _utc_now(),
            "watcher_pid": proc.pid,
            "watch_state": "launched",
            "poll_interval_seconds": int(poll_interval_seconds),
            "max_auto_resumes": int(max_auto_resumes),
            "auto_resume_count": 0,
        },
    )
    return {
        "success": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "watcher_pid": proc.pid,
        "watch_command": cmd,
        "watch_status_path": str(status_path),
        "launch_log_path": str(launch_log_path),
    }


def _find_stage(stages: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for stage in stages:
        if stage.get("name") == name:
            stage.setdefault("attempts", [])
            return stage
    raise KeyError(f"Unknown E2E stage '{name}'")


def _compute_stage_status_from_attempts(stage: Dict[str, Any]) -> str:
    attempts = list(stage.get("attempts") or [])
    if attempts:
        latest_status = str(attempts[-1].get("status") or "").strip()
        if latest_status:
            return latest_status
    return str(stage.get("status") or "planned")


def _bucket_attempt_run_id(stage_base_run_id: str, bucket: str, attempt_index: int) -> str:
    bucket_suffix = "single" if bucket == "single_gpu" else "multi"
    base_run_id = f"{stage_base_run_id}__{bucket_suffix}"
    if attempt_index <= 0:
        return base_run_id
    return f"{base_run_id}__resume{attempt_index}"


def _stage_attempt_run_id(stage_base_run_id: str, attempt_index: int) -> str:
    if attempt_index <= 0:
        return stage_base_run_id
    return f"{stage_base_run_id}__resume{attempt_index}"


def _bucket_attempts(stage: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    return [attempt for attempt in (stage.get("attempts") or []) if attempt.get("bucket") == bucket]


def _benchmark_run_dir(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Path:
    artifacts_root = (
        Path(artifacts_dir).resolve() if artifacts_dir else (repo_root / "artifacts" / "runs")
    )
    return artifacts_root / run_id


def _benchmark_run_event_paths(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Path]:
    run_dir = _benchmark_run_dir(run_id, repo_root=repo_root, artifacts_dir=artifacts_dir)
    return {
        "run_dir": run_dir,
        "events": run_dir / "logs" / "benchmark_events.jsonl",
        "output_json": run_dir / "results" / "benchmark_test_results.json",
        "progress": run_dir / "progress" / "run_progress.json",
    }


def _cluster_run_progress_path(
    run_id: str,
    *,
    repo_root: Path,
) -> Path:
    return repo_root / "cluster" / "runs" / run_id / "progress" / "run_progress.json"


def _load_benchmark_run_start(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Optional[Dict[str, Any]]:
    events_path = _benchmark_run_event_paths(
        run_id, repo_root=repo_root, artifacts_dir=artifacts_dir
    )["events"]
    for payload in _read_jsonl(events_path):
        if payload.get("event_type") == "run_start":
            return payload
    return None


def _load_benchmark_unit_progress(
    run_id: str,
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    events_path = _benchmark_run_event_paths(
        run_id, repo_root=repo_root, artifacts_dir=artifacts_dir
    )["events"]
    completed_units: List[str] = []
    started_units: List[str] = []
    for payload in _read_jsonl(events_path):
        event_type = str(payload.get("event_type") or "")
        unit_name = _canonical_unit_name(payload.get("chapter"))
        if not unit_name:
            continue
        if event_type == "chapter_start" and unit_name not in started_units:
            started_units.append(unit_name)
        elif event_type == "chapter_end" and unit_name not in completed_units:
            try:
                failed = int(payload.get("failed", -1))
                total_benchmarks = int(payload.get("total_benchmarks", -1))
                successful = int(payload.get("successful", -1))
                skipped_hardware = int(payload.get("skipped_hardware", -1))
                skipped_distributed = int(payload.get("skipped_distributed", -1))
                informational = int(payload.get("informational", -1))
            except (TypeError, ValueError):
                continue
            if (
                failed == 0
                and total_benchmarks > 0
                and successful == total_benchmarks
                and skipped_hardware == 0
                and skipped_distributed == 0
                and informational == 0
            ):
                completed_units.append(unit_name)
    active_unit = None
    for unit_name in started_units:
        if unit_name not in completed_units:
            active_unit = unit_name
            break
    return {
        "completed_units": completed_units,
        "active_unit": active_unit,
        "started_units": started_units,
    }


def _attach_benchmark_attempt_state(
    stage_name: str,
    attempt: Dict[str, Any],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> None:
    run_id = str(attempt.get("run_id") or "").strip()
    if not run_id:
        return
    benchmark_paths = _benchmark_run_event_paths(
        run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    attempt["artifacts"] = {
        **dict(attempt.get("artifacts") or {}),
        "run_dir": str(benchmark_paths["run_dir"]),
        "events_path": str(benchmark_paths["events"]),
        "output_json": str(benchmark_paths["output_json"]),
        "progress_path": str(benchmark_paths["progress"]),
    }
    attempt.pop("benchmark_summary", None)
    attempt.pop("verified_targets", None)
    run_start = _load_benchmark_run_start(
        run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    if isinstance(run_start, dict):
        requested_targets = run_start.get("targets")
        if isinstance(requested_targets, list) and all(
            isinstance(target, str) and target.strip() for target in requested_targets
        ):
            attempt["verified_targets"] = list(requested_targets)
    benchmark_summary = _benchmark_stage_details_from_output(str(benchmark_paths["output_json"]))
    if benchmark_summary is not None:
        attempt["benchmark_summary"] = benchmark_summary
    if stage_name == "full_sweep":
        unit_progress = _load_benchmark_unit_progress(
            run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        completed_units: List[str] = []
        if benchmark_summary is not None:
            completed_units = _completed_units_from_target_outcomes(
                list(attempt.get("targets") or []),
                benchmark_summary,
            )
        attempt["completed_units"] = completed_units
        attempt["active_unit"] = next(
            (
                unit
                for unit in unit_progress.get("started_units", [])
                if unit not in completed_units
            ),
            None,
        )


def _mark_attempt_aborted(
    stage_name: str,
    attempt: Dict[str, Any],
    *,
    reason: str,
    repo_root: Path,
    artifacts_dir: Optional[str],
    ended_at: Optional[str] = None,
) -> None:
    attempt["status"] = "aborted"
    attempt["ended_at"] = ended_at or _utc_now()
    attempt_issues = list(attempt.get("issues") or [])
    if reason not in attempt_issues:
        attempt_issues.append(reason)
    attempt["issues"] = attempt_issues
    _attach_benchmark_attempt_state(
        stage_name,
        attempt,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )


def _normalize_incomplete_attempts_for_resume(
    stages: List[Dict[str, Any]],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
    reason: str,
) -> None:
    for stage in stages:
        if not stage.get("enabled"):
            continue
        stage_name = str(stage.get("name") or "")
        changed = False
        for attempt in stage.get("attempts") or []:
            if stage_name == "full_sweep":
                _attach_benchmark_attempt_state(
                    stage_name,
                    attempt,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                )
            if str(attempt.get("status") or "") != "running":
                continue
            _mark_attempt_aborted(
                stage_name,
                attempt,
                reason=reason,
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
            )
            changed = True
        if changed and str(stage.get("status") or "") == "running":
            stage["status"] = _compute_stage_status_from_attempts(stage)


def _revalidate_full_sweep_stage_from_frozen_plan(
    stage: Dict[str, Any],
    frozen_plan: Dict[str, Any],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
    expected_git_commit: str,
) -> None:
    if not stage.get("enabled"):
        return
    exact_coverage = False
    saw_targets = False
    evidence_issues: List[str] = []
    for bucket_name, target_key in (
        ("single_gpu", "single_gpu_targets"),
        ("multi_gpu", "multi_gpu_targets"),
    ):
        frozen_targets = list(frozen_plan.get(target_key) or [])
        if not frozen_targets:
            continue
        saw_targets = True
        expected_units = [str(unit["name"]) for unit in _group_targets_by_unit(frozen_targets)]
        verified_attempts, bucket_evidence_issues = _verified_full_sweep_attempts(
            _bucket_attempts(stage, bucket_name),
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            expected_git_commit=expected_git_commit,
        )
        evidence_issues.extend(bucket_evidence_issues)
        completed_units = _completed_units_from_attempts(
            verified_attempts,
            frozen_targets=frozen_targets,
        )
        if completed_units != expected_units:
            exact_coverage = False
            break
        exact_coverage = True
    if saw_targets and exact_coverage:
        return
    if str(stage.get("status") or "") in {"succeeded", "skipped_duplicate"}:
        stage["status"] = "aborted"
        issue = "stored full-sweep success lacks exact frozen-target evidence"
        issues = list(stage.get("issues") or [])
        if issue not in issues:
            issues.append(issue)
        for evidence_issue in evidence_issues:
            if evidence_issue not in issues:
                issues.append(evidence_issue)
        stage["issues"] = issues


def _latest_stage_attempt(stage: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attempts = [attempt for attempt in (stage.get("attempts") or []) if isinstance(attempt, dict)]
    return attempts[-1] if attempts else None


def _downgrade_unverified_stage_success(
    stage: Dict[str, Any],
    *,
    reason: str,
) -> None:
    stage["status"] = "aborted"
    issues = list(stage.get("issues") or [])
    if reason not in issues:
        issues.append(reason)
    stage["issues"] = issues


def _revalidate_resumed_terminal_stages(
    stages: List[Dict[str, Any]],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
    expected_git_commit: str,
    allow_comparison_regressions: bool = False,
) -> None:
    for stage_name in ("tier1", "cluster", "fabric"):
        stage = _find_stage(stages, stage_name)
        if not stage.get("enabled") or str(stage.get("status") or "") != "succeeded":
            continue
        attempt = _latest_stage_attempt(stage)
        if attempt is None or str(attempt.get("status") or "") != "succeeded":
            _downgrade_unverified_stage_success(
                stage,
                reason=f"stored {stage_name} success lacks a successful attempt",
            )
            continue
        expected_attempt_run_id = str(attempt.get("run_id") or "").strip()
        try:
            expected_attempt_run_id = validate_run_id(expected_attempt_run_id)
        except ValueError:
            _downgrade_unverified_stage_success(
                stage,
                reason=f"stored {stage_name} success has an unsafe attempt run_id",
            )
            continue
        result = dict(stage.get("result") or {})
        result.update(dict(attempt.get("result") or {}))
        for source in (stage.get("artifacts"), attempt.get("artifacts")):
            if isinstance(source, dict):
                for key, value in source.items():
                    result.setdefault(str(key), value)

        if stage_name == "tier1":
            status, issues, _details = _benchmark_stage_status(
                result,
                required_paths=[
                    "summary_path",
                    "regression_summary_path",
                    "regression_json_path",
                    "trend_snapshot_path",
                ],
                require_complete=True,
                allow_comparison_regressions=allow_comparison_regressions,
                expected_run_id=expected_attempt_run_id,
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
                expected_git_commit=expected_git_commit,
            )
        else:
            status, issues, _details = _cluster_stage_status(
                result,
                require_scorecard=stage_name == "fabric",
                expected_run_id=expected_attempt_run_id,
                repo_root=repo_root,
                expected_git_commit=expected_git_commit,
            )
        if status == "succeeded":
            continue
        detail = ". ".join(issues) if issues else f"revalidated status was {status}"
        _downgrade_unverified_stage_success(
            stage,
            reason=f"stored {stage_name} success failed resume revalidation: {detail}",
        )


def _summarize_inventory_for_summary(inventory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "counts": dict(inventory.get("counts") or {}),
        "single_gpu": list(inventory.get("single_gpu") or []),
        "multi_gpu": list(inventory.get("multi_gpu") or []),
    }


def _build_frozen_full_sweep_plan(
    single_targets: List[str], multi_targets: List[str]
) -> Dict[str, Any]:
    return {
        "single_gpu_targets": list(single_targets),
        "single_gpu_units": [entry["name"] for entry in _group_targets_by_unit(single_targets)],
        "multi_gpu_targets": list(multi_targets),
        "multi_gpu_units": [entry["name"] for entry in _group_targets_by_unit(multi_targets)],
    }


def _build_frozen_plan(*, inventory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "full_sweep": _build_frozen_full_sweep_plan(
            list(inventory.get("single_gpu") or []),
            list(inventory.get("multi_gpu") or []),
        )
    }


def _stage_finish_event_map(events_path: Path) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for payload in _read_jsonl(events_path):
        if payload.get("event") != "stage_finished":
            continue
        stage_name = str(payload.get("stage") or "").strip()
        if stage_name:
            statuses[stage_name] = str(payload.get("status") or "unknown")
    return statuses


def _recover_legacy_resume_state(
    *,
    resolved_run_id: str,
    run_dir: Path,
    stage_run_ids: Dict[str, str],
    inventory: Dict[str, Any],
    planned_stages: List[Dict[str, Any]],
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    stage_statuses = _stage_finish_event_map(events_path)
    stages = _json_safe(planned_stages)
    tier1_stage = _find_stage(stages, "tier1")
    if "tier1" in stage_statuses:
        tier1_stage["status"] = stage_statuses["tier1"]
        tier1_stage["attempts"] = [
            _stage_attempt_entry(
                run_id=stage_run_ids["tier1"],
                status=stage_statuses["tier1"],
                recovered=True,
            )
        ]

    full_sweep_stage = _find_stage(stages, "full_sweep")
    single_run_id = _bucket_attempt_run_id(stage_run_ids["full_sweep"], "single_gpu", 0)
    single_run_start = _load_benchmark_run_start(
        single_run_id,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )
    single_targets = list((single_run_start or {}).get("targets") or [])
    if single_targets:
        unit_progress = _load_benchmark_unit_progress(
            single_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        benchmark_paths = _benchmark_run_event_paths(
            single_run_id,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        benchmark_summary = _benchmark_stage_details_from_output(
            str(benchmark_paths["output_json"])
        )
        completed_units = _completed_units_from_target_outcomes(
            single_targets,
            benchmark_summary,
        )
        active_unit = next(
            (
                unit
                for unit in unit_progress.get("started_units", [])
                if unit not in completed_units
            ),
            None,
        )
        full_sweep_stage["status"] = "aborted"
        full_sweep_stage["attempts"] = [
            _stage_attempt_entry(
                run_id=single_run_id,
                bucket="single_gpu",
                status="aborted",
                targets=single_targets,
                units=[entry["name"] for entry in _group_targets_by_unit(single_targets)],
                completed_units=completed_units,
                active_unit=active_unit,
                artifacts={
                    "run_dir": str(benchmark_paths["run_dir"]),
                    "events_path": str(benchmark_paths["events"]),
                    "output_json": str(benchmark_paths["output_json"]),
                    "progress_path": str(benchmark_paths["progress"]),
                },
                benchmark_summary=benchmark_summary,
                issues=[
                    entry["error"]
                    for entry in (benchmark_summary or {}).get("failed_benchmarks", [])
                ]
                or ["full_sweep single bucket aborted before stage completion"],
                recovered=True,
            )
        ]
    elif "full_sweep" in stage_statuses:
        full_sweep_stage["status"] = stage_statuses["full_sweep"]

    frozen_plan = {
        "full_sweep": _build_frozen_full_sweep_plan(
            single_targets or list(inventory.get("single_gpu") or []),
            list(inventory.get("multi_gpu") or []),
        )
    }
    legacy_manifest = _read_json_if_exists(run_dir / "manifest.json") or {}
    manifest_contract = legacy_manifest.get("contract")
    recovered_contract = (
        _sanitize_e2e_contract_values(dict(manifest_contract))
        if isinstance(manifest_contract, dict)
        else {}
    )
    if single_run_start:
        for key in (
            "profile_type",
            "validity_profile",
            "allow_portable_expectations_update",
            "timeout_multiplier",
            "ncu_metric_set",
            "ncu_replay_mode",
            "nsys_timeout_seconds",
            "ncu_timeout_seconds",
            "update_expectations",
            "allow_mixed_provenance",
        ):
            run_start_key = key
            if key == "profile_type":
                run_start_key = "profile_type"
            recovered_value = single_run_start.get(run_start_key)
            if recovered_value is not None:
                recovered_contract[key] = recovered_value
        recovered_contract["run_tier1"] = bool(
            stage_statuses.get("tier1")
            or any(
                payload.get("event") == "stage_started" and payload.get("stage") == "tier1"
                for payload in _read_jsonl(events_path)
            )
        )
        recovered_contract["run_full_sweep"] = True
    return {
        "generated_at": _utc_now(),
        "stages": stages,
        "contract": recovered_contract,
        "frozen_plan": frozen_plan,
        "provenance": _json_safe(legacy_manifest.get("provenance") or {}),
        "legacy_recovered": True,
    }


def _load_resume_state(
    *,
    run_dir: Path,
    resolved_run_id: str,
    stage_run_ids: Dict[str, str],
    inventory: Dict[str, Any],
    planned_stages: List[Dict[str, Any]],
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    checkpoint = _read_json_if_exists(e2e_checkpoint_path(run_dir))
    if checkpoint is not None:
        return checkpoint
    return _recover_legacy_resume_state(
        resolved_run_id=resolved_run_id,
        run_dir=run_dir,
        stage_run_ids=stage_run_ids,
        inventory=inventory,
        planned_stages=planned_stages,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
    )


def _normalize_stale_running_resume_state(
    resume_state: Dict[str, Any],
    *,
    repo_root: Path,
    artifacts_dir: Optional[str],
) -> Optional[str]:
    if str(resume_state.get("run_state") or "").strip() != "running":
        return None
    orchestrator_pid = resume_state.get("orchestrator_pid")
    if _pid_is_live(orchestrator_pid):
        return None

    stale_reason = (
        "orchestrator process exited without finalizing run state"
        if orchestrator_pid is None
        else f"orchestrator process {orchestrator_pid} exited without finalizing run state"
    )
    stages = _json_safe(resume_state.get("stages") or [])
    for stage in stages:
        if isinstance(stage, dict):
            stage.setdefault("attempts", [])
    _normalize_incomplete_attempts_for_resume(
        stages,
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        reason=stale_reason,
    )
    for stage in stages:
        if str(stage.get("status") or "") == "running":
            stage["status"] = _compute_stage_status_from_attempts(stage)
            stage_issues = list(stage.get("issues") or [])
            if stale_reason not in stage_issues:
                stage_issues.append(stale_reason)
            stage["issues"] = stage_issues
    crash = dict(resume_state.get("crash") or {})
    crash.setdefault("type", "orchestrator_exit")
    crash.setdefault("message", stale_reason)
    resume_state["crash"] = crash
    resume_state["run_state"] = "aborted"
    resume_state["overall_status"] = "aborted"
    resume_state["success"] = False
    resume_state["resume_available"] = True
    resume_state["error"] = stale_reason
    resume_state["stages"] = stages
    return stale_reason


def _validate_resume_contract(
    *,
    requested: Dict[str, Any],
    stored: Dict[str, Any],
) -> Optional[str]:
    requested = _sanitize_e2e_contract_values(requested)
    stored = _sanitize_e2e_contract_values(stored)

    schema_mismatches: List[str] = []
    for label, contract in (("requested", requested), ("stored", stored)):
        schema_version = contract.get("schema_version")
        if schema_version != _E2E_CONTRACT_SCHEMA_VERSION:
            rendered = "missing" if schema_version is None else repr(schema_version)
            schema_mismatches.append(
                f"{label} schema_version is {rendered}, expected {_E2E_CONTRACT_SCHEMA_VERSION!r}"
            )
    if schema_mismatches:
        return "Resume contract mismatch: " + ", ".join(schema_mismatches)

    missing_requested = [
        field_name for field_name in _E2E_CONTRACT_REQUIRED_FIELDS if field_name not in requested
    ]
    missing_stored = [
        field_name for field_name in _E2E_CONTRACT_REQUIRED_FIELDS if field_name not in stored
    ]
    if missing_requested or missing_stored:
        details: List[str] = []
        if missing_requested:
            details.append("requested missing required field(s): " + ", ".join(missing_requested))
        if missing_stored:
            details.append("stored missing required field(s): " + ", ".join(missing_stored))
        return "Resume contract mismatch: " + ". ".join(details)

    def _timeout_is_compatible(field_name: str, requested_value: Any, stored_value: Any) -> bool:
        if field_name != "suite_timeout":
            return False
        if requested_value == stored_value:
            return True
        if requested_value in (0, None):
            return True
        if stored_value in (0, None):
            return False
        try:
            return int(requested_value) >= int(stored_value)
        except (TypeError, ValueError):
            return False

    fields_to_validate = [
        field_name for field_name in _E2E_CONTRACT_REQUIRED_FIELDS if field_name != "schema_version"
    ]
    mismatches: List[str] = []
    for field_name in fields_to_validate:
        requested_value = _json_safe(requested.get(field_name))
        stored_value = _json_safe(stored.get(field_name))
        if _timeout_is_compatible(field_name, requested_value, stored_value):
            continue
        if requested_value != stored_value:
            mismatches.append(field_name)
    if not mismatches:
        return None
    return "Resume contract mismatch for field(s): " + ", ".join(mismatches)


def _validate_resume_provenance(
    *,
    current: Dict[str, Any],
    stored: Dict[str, Any],
) -> Optional[str]:
    fields = (
        "expectation_hardware_key",
        "execution_environment",
        "gpu_count",
        "bench_root_identity",
    )
    mismatches: List[str] = []
    current_git = current.get("git")
    stored_git = stored.get("git")
    if not isinstance(current_git, dict) or not isinstance(stored_git, dict):
        mismatches.append("git provenance is missing")
    else:
        if bool(stored_git.get("dirty")):
            mismatches.append("stored worktree is dirty")
        if bool(current_git.get("dirty")):
            mismatches.append("current worktree is dirty")
        for field_name in ("commit", "dirty"):
            if field_name not in stored_git or _json_safe(stored_git.get(field_name)) != _json_safe(
                current_git.get(field_name)
            ):
                mismatches.append(f"git.{field_name} changed")
    for field_name in fields:
        if field_name not in stored or _json_safe(stored.get(field_name)) != _json_safe(
            current.get(field_name)
        ):
            mismatches.append(f"{field_name} changed")
    if not mismatches:
        return None
    return "Resume provenance mismatch: " + ", ".join(mismatches)


def _build_checkpoint_payload(
    *,
    run_id: str,
    run_dir: Path,
    generated_at: str,
    updated_at: str,
    run_state: str,
    overall_status: str,
    success: bool,
    resume_available: bool,
    error: Optional[str],
    contract: Dict[str, Any],
    inventory: Dict[str, Any],
    frozen_plan: Dict[str, Any],
    hosts: Dict[str, Any],
    provenance: Dict[str, Any],
    stages: List[Dict[str, Any]],
    artifact_paths: Dict[str, Path],
    crash: Optional[Dict[str, Any]],
    orchestrator_pid: Optional[int],
    watcher: Optional[Dict[str, Any]],
    current_stage: Optional[str],
    current_stage_run_id: Optional[str],
    current_bucket: Optional[str],
) -> Dict[str, Any]:
    status_actions = build_benchmark_e2e_status_actions(run_id)
    ledger_refs = _existing_ledger_refs(run_dir)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "generated_at": generated_at,
        "updated_at": updated_at,
        "run_state": run_state,
        "overall_status": overall_status,
        "current_stage": current_stage,
        "current_stage_run_id": current_stage_run_id,
        "current_bucket": current_bucket,
        "success": success,
        "resume_available": resume_available,
        "error": error,
        "contract": contract,
        "inventory": _summarize_inventory_for_summary(inventory),
        "frozen_plan": frozen_plan,
        "hosts": hosts,
        "provenance": provenance,
        "stages": stages,
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "crash": crash,
        "orchestrator_pid": orchestrator_pid,
        "watcher": watcher,
        "ledgers": ledger_refs,
        "actions": status_actions,
        "preferred_progress_source": build_benchmark_e2e_progress_surface_hint(run_id),
    }


def _build_summary_payload(
    *,
    run_id: str,
    run_dir: Path,
    generated_at: str,
    updated_at: str,
    run_state: str,
    overall_status: str,
    success: bool,
    resume_available: bool,
    error: Optional[str],
    contract: Dict[str, Any],
    inventory: Dict[str, Any],
    hosts: Dict[str, Any],
    provenance: Dict[str, Any],
    stages: List[Dict[str, Any]],
    artifact_paths: Dict[str, Path],
    crash: Optional[Dict[str, Any]],
    orchestrator_pid: Optional[int],
    watcher: Optional[Dict[str, Any]],
    current_stage: Optional[str],
    current_stage_run_id: Optional[str],
    current_bucket: Optional[str],
) -> Dict[str, Any]:
    status_actions = build_benchmark_e2e_status_actions(run_id)
    ledger_refs = _existing_ledger_refs(run_dir)
    return {
        "success": success,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_state": run_state,
        "overall_status": overall_status,
        "current_stage": current_stage,
        "current_stage_run_id": current_stage_run_id,
        "current_bucket": current_bucket,
        "generated_at": generated_at,
        "updated_at": updated_at,
        "resume_available": resume_available,
        "error": error,
        "manifest_path": str(artifact_paths["manifest_path"]),
        "summary_path": str(artifact_paths["summary_path"]),
        "summary_markdown_path": str(artifact_paths["summary_markdown_path"]),
        "progress_path": str(artifact_paths["progress_path"]),
        "checkpoint_path": str(artifact_paths["checkpoint_path"]),
        "target_inventory_path": str(artifact_paths["target_inventory_path"]),
        "events_path": str(artifact_paths["events_path"]),
        "inventory": _summarize_inventory_for_summary(inventory),
        "hosts": hosts,
        "provenance": provenance,
        "contract": contract,
        "stages": stages,
        "crash": crash,
        "orchestrator_pid": orchestrator_pid,
        "watcher": watcher,
        "ledgers": ledger_refs,
        "actions": status_actions,
        "preferred_progress_source": build_benchmark_e2e_progress_surface_hint(run_id),
    }


def run_benchmark_e2e_sweep(
    *,
    run_tier1: bool = True,
    run_full_sweep: bool = False,
    run_cluster: bool = True,
    run_fabric: bool = True,
    cluster_preset: str = "common-answer-fast",
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
    extra_cluster_args: Optional[List[str]] = None,
    bench_root: Optional[Path] = None,
    profile_type: str = "minimal",
    output_format: str = "both",
    suite_timeout: Optional[int] = 14400,
    full_sweep_suite_timeout: Optional[int] = 0,
    timeout_multiplier: float = 3.0,
    validity_profile: str = "strict",
    allow_portable_expectations_update: bool = False,
    reproducible: bool = False,
    cold_start: bool = False,
    force_synchronize: bool = False,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    gpu_sm_clock_mhz: Optional[int] = None,
    gpu_mem_clock_mhz: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    single_gpu: bool = False,
    accept_regressions: bool = False,
    update_expectations: bool = False,
    allow_mixed_provenance: bool = False,
    ncu_metric_set: str = "minimal",
    ncu_replay_mode: Optional[str] = None,
    nsys_timeout_seconds: Optional[int] = None,
    ncu_timeout_seconds: Optional[int] = None,
    auto_resume: bool = True,
    max_auto_resumes: int = _E2E_WATCHER_MAX_AUTO_RESUMES,
    watch_poll_interval_seconds: int = _E2E_WATCHER_POLL_SECONDS,
    timeout_seconds: Optional[int] = None,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    repo_root = _repo_root()
    active_bench_root = Path(bench_root).resolve() if bench_root else repo_root
    private_bench_root = (
        active_bench_root if active_bench_root.resolve() != repo_root.resolve() else None
    )
    private_artifacts_dir = (
        Path(artifacts_dir).expanduser() if str(artifacts_dir or "").strip() else None
    )
    private_log_file = Path(log_file).expanduser() if str(log_file or "").strip() else None
    normalized_validity_profile = normalize_validity_profile(
        validity_profile, field_name="validity_profile"
    )
    if resume and not str(run_id or "").strip():
        return _json_safe(
            {
                "success": False,
                "overall_status": "failed",
                "run_state": "completed",
                "error": "resume=true requires an explicit run_id",
            }
        )
    resolved_run_id = resolve_e2e_run_id(run_id, repo_root=repo_root)
    run_dir = e2e_run_dir(resolved_run_id, repo_root)
    if not resume and (run_dir.is_symlink() or run_dir.exists()):
        return _sanitize_persisted_value(
            {
                "success": False,
                "run_id": resolved_run_id,
                "run_dir": str(run_dir),
                "run_state": "completed",
                "overall_status": "failed",
                "resume_available": False,
                "error": f"Refusing to overwrite existing E2E run {resolved_run_id!r}",
            },
            repo_root=repo_root,
            private_bench_root=private_bench_root,
            private_artifacts_dir=private_artifacts_dir,
            private_log_file=private_log_file,
        )
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    progress_path = e2e_progress_path(run_dir)
    checkpoint_path = e2e_checkpoint_path(run_dir)
    target_inventory_path = run_dir / "target_inventory.json"
    events_path = run_dir / "events.jsonl"
    sensitive_values = [ssh_key, nmx_token, ib_mgmt_ssh_key, cumulus_ssh_key]

    def _sanitize_run_value(value: Any) -> Any:
        return _sanitize_persisted_value(
            value,
            sensitive_values=sensitive_values,
            repo_root=repo_root,
            private_bench_root=private_bench_root,
            private_artifacts_dir=private_artifacts_dir,
            private_log_file=private_log_file,
        )

    def _append_run_event(event: str, **fields: Any) -> None:
        _append_event(
            events_path,
            event,
            **_sanitize_run_value(fields),
        )

    generated_at = _utc_now()
    progress_recorder = (
        None if dry_run else ProgressRecorder(run_id=resolved_run_id, progress_path=progress_path)
    )
    progress_emit_lock = threading.Lock()
    artifact_paths = {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "summary_markdown_path": summary_markdown_path,
        "progress_path": progress_path,
        "checkpoint_path": checkpoint_path,
        "target_inventory_path": target_inventory_path,
        "events_path": events_path,
    }
    stage_run_ids = {
        "tier1": f"{resolved_run_id}__tier1",
        "full_sweep": f"{resolved_run_id}__full_sweep",
        "cluster": f"{resolved_run_id}__cluster",
        "fabric": f"{resolved_run_id}__fabric",
    }
    planned_stages = _planned_stage_entries(
        run_tier1=run_tier1,
        run_full_sweep=run_full_sweep,
        run_cluster=run_cluster,
        run_fabric=run_fabric,
        cluster_preset=cluster_preset,
        stage_run_ids=stage_run_ids,
    )

    expectation_error = _validate_expectation_policy(
        validity_profile=normalized_validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        update_expectations=update_expectations,
        accept_regressions=accept_regressions,
        allow_mixed_provenance=allow_mixed_provenance,
    )
    inventory = discover_benchmark_e2e_inventory(active_bench_root)
    environment = detect_execution_environment()
    try:
        cluster_host_config = _normalize_cluster_hosts_and_labels(
            hosts=hosts,
            labels=labels,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
        )
    except Exception as exc:
        failure_payload = {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "run_state": "completed",
            "overall_status": "failed",
            "generated_at": generated_at,
            "updated_at": generated_at,
            "resume_available": False,
            "error": str(exc),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "checkpoint_path": str(checkpoint_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": _summarize_inventory_for_summary(inventory),
            "stages": _json_safe(planned_stages),
            "provenance": {
                "generated_at": generated_at,
                "git": get_git_info(),
                "bench_root_identity": _bench_root_identity(active_bench_root),
            },
        }
        safe_failure = _sanitize_run_value(failure_payload)
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(summary_path, safe_failure)
            summary_markdown_path.write_text(
                _render_summary_markdown(safe_failure), encoding="utf-8"
            )
            _write_json(manifest_path, {**safe_failure, "inventory": inventory})
        return attach_benchmark_e2e_status_hints(safe_failure, resolved_run_id)
    gpu_count = _visible_gpu_count(single_gpu=single_gpu)
    try:
        cluster_extra_args = _with_e2e_cluster_extra_args(extra_cluster_args)
    except ValueError as exc:
        cluster_extra_args = _with_e2e_cluster_extra_args(None)
        extra_args_error = str(exc)
        expectation_error = (
            f"{expectation_error}. {extra_args_error}" if expectation_error else extra_args_error
        )
    current_git_info = dict(get_git_info())
    validated_git_commit: Optional[str] = None
    git_preflight_error: Optional[str] = None
    if not dry_run:
        validated_git_commit, git_preflight_error = _validated_clean_git_commit(
            repo_root=repo_root,
            git_info=current_git_info,
        )
        if git_preflight_error:
            expectation_error = (
                f"{expectation_error}. {git_preflight_error}"
                if expectation_error
                else git_preflight_error
            )
    provenance = {
        "generated_at": generated_at,
        "git": current_git_info,
        "expectation_hardware_key": detect_expectation_key(),
        "execution_environment": {
            "kind": environment.kind,
            "virtualized": environment.virtualized,
            "dmi_product_name": environment.dmi_product_name,
        },
        "gpu_count": gpu_count,
        "bench_root_identity": _bench_root_identity(active_bench_root),
    }
    requested_contract = _build_e2e_contract(
        run_tier1=run_tier1,
        run_full_sweep=run_full_sweep,
        run_cluster=run_cluster,
        run_fabric=run_fabric,
        cluster_preset=cluster_preset,
        hosts=cluster_host_config["hosts"],
        labels=cluster_host_config["labels"],
        ssh_user=cluster_host_config["ssh_user"],
        ssh_key=cluster_host_config["ssh_key"],
        oob_if=oob_if,
        socket_ifname=socket_ifname,
        nccl_ib_hca=nccl_ib_hca,
        nmx_url=nmx_url,
        nmx_token=nmx_token,
        ib_mgmt_host=ib_mgmt_host,
        ib_mgmt_user=ib_mgmt_user,
        ib_mgmt_ssh_key=ib_mgmt_ssh_key,
        cumulus_hosts=list(cumulus_hosts or []),
        cumulus_user=cumulus_user,
        cumulus_ssh_key=cumulus_ssh_key,
        primary_label=primary_label,
        coverage_baseline_run_id=coverage_baseline_run_id,
        extra_cluster_args=list(cluster_extra_args or []),
        bench_root=str(active_bench_root),
        profile_type=profile_type,
        output_format=output_format,
        suite_timeout=suite_timeout,
        full_sweep_suite_timeout=full_sweep_suite_timeout,
        timeout_multiplier=timeout_multiplier,
        timeout_seconds=timeout_seconds,
        validity_profile=normalized_validity_profile,
        allow_portable_expectations_update=allow_portable_expectations_update,
        reproducible=reproducible,
        cold_start=cold_start,
        force_synchronize=force_synchronize,
        iterations=iterations,
        warmup=warmup,
        gpu_sm_clock_mhz=gpu_sm_clock_mhz,
        gpu_mem_clock_mhz=gpu_mem_clock_mhz,
        artifacts_dir=artifacts_dir,
        log_level=log_level,
        log_file=log_file,
        single_gpu=single_gpu,
        accept_regressions=accept_regressions,
        update_expectations=update_expectations,
        allow_mixed_provenance=allow_mixed_provenance,
        ncu_metric_set=ncu_metric_set,
        ncu_replay_mode=ncu_replay_mode,
        nsys_timeout_seconds=nsys_timeout_seconds,
        ncu_timeout_seconds=ncu_timeout_seconds,
        auto_resume=auto_resume,
        max_auto_resumes=max_auto_resumes,
        watch_poll_interval_seconds=watch_poll_interval_seconds,
    )

    if expectation_error:
        result = {
            "success": False,
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "run_state": "completed",
            "overall_status": "failed",
            "updated_at": generated_at,
            "resume_available": False,
            "error": expectation_error,
            "generated_at": generated_at,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "summary_markdown_path": str(summary_markdown_path),
            "progress_path": str(progress_path),
            "checkpoint_path": str(checkpoint_path),
            "target_inventory_path": str(target_inventory_path),
            "events_path": str(events_path),
            "inventory": _summarize_inventory_for_summary(inventory),
            "stages": _json_safe(planned_stages),
            "provenance": provenance,
            "contract": requested_contract,
        }
        if not dry_run:
            _append_run_event(
                "run_failed_preflight", error=expectation_error, run_id=resolved_run_id
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            safe_result = _sanitize_run_value(result)
            _write_json(summary_path, safe_result)
            summary_markdown_path.write_text(
                _render_summary_markdown(safe_result), encoding="utf-8"
            )
            _write_json(manifest_path, {**safe_result, "inventory": inventory})
        return attach_benchmark_e2e_status_hints(_sanitize_run_value(result), resolved_run_id)

    def _materialize_stages(
        loaded_stages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        loaded_by_name = {
            str(stage.get("name")): dict(stage)
            for stage in (loaded_stages or [])
            if str(stage.get("name", "")).strip()
        }
        materialized: List[Dict[str, Any]] = []
        for planned_stage in planned_stages:
            payload = dict(planned_stage)
            loaded = loaded_by_name.get(str(planned_stage.get("name")))
            if loaded:
                payload.update(loaded)
            payload["name"] = planned_stage["name"]
            payload.setdefault("enabled", planned_stage.get("enabled", False))
            payload.setdefault("run_id", planned_stage.get("run_id"))
            payload.setdefault("description", planned_stage.get("description"))
            payload.setdefault("attempts", [])
            materialized.append(payload)
        return materialized

    frozen_plan = _build_frozen_plan(inventory=inventory)
    if resume:
        if not checkpoint_path.exists() and not events_path.exists():
            return attach_benchmark_e2e_status_hints(
                _sanitize_run_value(
                    {
                        "success": False,
                        "run_id": resolved_run_id,
                        "run_dir": str(run_dir),
                        "run_state": "completed",
                        "overall_status": "failed",
                        "generated_at": generated_at,
                        "updated_at": generated_at,
                        "resume_available": False,
                        "error": f"No prior run state found for run_id={resolved_run_id!r}",
                        "manifest_path": str(manifest_path),
                        "summary_path": str(summary_path),
                        "summary_markdown_path": str(summary_markdown_path),
                        "progress_path": str(progress_path),
                        "checkpoint_path": str(checkpoint_path),
                        "target_inventory_path": str(target_inventory_path),
                        "events_path": str(events_path),
                        "inventory": _summarize_inventory_for_summary(inventory),
                        "stages": _json_safe(planned_stages),
                        "provenance": provenance,
                        "contract": requested_contract,
                    }
                ),
                resolved_run_id,
            )
        resume_state = _load_resume_state(
            run_dir=run_dir,
            resolved_run_id=resolved_run_id,
            stage_run_ids=stage_run_ids,
            inventory=inventory,
            planned_stages=planned_stages,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        mismatch_error = _validate_resume_contract(
            requested=requested_contract,
            stored=resume_state.get("contract") or {},
        )
        if mismatch_error is None:
            mismatch_error = _validate_resume_provenance(
                current=provenance,
                stored=resume_state.get("provenance") or {},
            )
        if mismatch_error is None:
            try:
                resume_state = _restore_persisted_path_locators(
                    resume_state,
                    repo_root=repo_root,
                    bench_root=active_bench_root,
                    artifacts_dir=artifacts_dir,
                    log_file=log_file,
                )
            except ValueError as exc:
                mismatch_error = f"Resume contract mismatch: {exc}"
        if mismatch_error:
            return attach_benchmark_e2e_status_hints(
                _sanitize_run_value(
                    {
                        "success": False,
                        "run_id": resolved_run_id,
                        "run_dir": str(run_dir),
                        "run_state": "completed",
                        "overall_status": "failed",
                        "generated_at": str(resume_state.get("generated_at") or generated_at),
                        "updated_at": generated_at,
                        "resume_available": True,
                        "error": mismatch_error,
                        "manifest_path": str(manifest_path),
                        "summary_path": str(summary_path),
                        "summary_markdown_path": str(summary_markdown_path),
                        "progress_path": str(progress_path),
                        "checkpoint_path": str(checkpoint_path),
                        "target_inventory_path": str(target_inventory_path),
                        "events_path": str(events_path),
                        "inventory": _summarize_inventory_for_summary(inventory),
                        "stages": _sanitize_run_value(resume_state.get("stages") or planned_stages),
                        "provenance": provenance,
                        "contract": requested_contract,
                    }
                ),
                resolved_run_id,
            )
        stale_reason = _normalize_stale_running_resume_state(
            resume_state,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
        )
        if stale_reason:
            _append_run_event(
                "run_recovered_aborted",
                run_id=resolved_run_id,
                error=stale_reason,
            )
        generated_at = str(resume_state.get("generated_at") or generated_at)
        frozen_plan = _json_safe(resume_state.get("frozen_plan") or frozen_plan)
        stages = _materialize_stages(resume_state.get("stages"))
        _normalize_incomplete_attempts_for_resume(
            stages,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            reason="resume superseded unfinished attempt",
        )
        if run_full_sweep and not dry_run:
            _revalidate_full_sweep_stage_from_frozen_plan(
                _find_stage(stages, "full_sweep"),
                dict(frozen_plan.get("full_sweep") or {}),
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
                expected_git_commit=str(validated_git_commit),
            )
        if not dry_run:
            _revalidate_resumed_terminal_stages(
                stages,
                repo_root=repo_root,
                artifacts_dir=artifacts_dir,
                expected_git_commit=str(validated_git_commit),
                allow_comparison_regressions=accept_regressions or update_expectations,
            )
    else:
        stages = _materialize_stages()

    run_state = "running"
    error: Optional[str] = None
    crash: Optional[Dict[str, Any]] = None
    resume_available = False
    run_finished_event_emitted = False
    current_stage_name: Optional[str] = None
    current_stage_started_at: Optional[float] = None
    current_stage_event_run_id: Optional[str] = None
    current_bucket: Optional[str] = None

    def _current_overall_status() -> str:
        if run_state == "aborted":
            return "aborted"
        statuses = [
            str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")
        ]
        if run_state == "running":
            if any(status == "aborted" for status in statuses):
                return "aborted"
            if any(status == "failed" for status in statuses):
                return "failed"
            if any(status == "partial" for status in statuses):
                return "partial"
            return "running"
        return _roll_up_overall_status(statuses)

    def _persist_state() -> Dict[str, Any]:
        updated_at = _utc_now()
        overall_status = _current_overall_status()
        success = run_state == "completed" and overall_status == "succeeded"
        watcher = _watcher_summary(run_dir)
        public_hosts = _sanitize_run_value(cluster_host_config)
        summary_payload = _build_summary_payload(
            run_id=resolved_run_id,
            run_dir=run_dir,
            generated_at=generated_at,
            updated_at=updated_at,
            run_state=run_state,
            overall_status=overall_status,
            success=success,
            resume_available=resume_available,
            error=error,
            contract=requested_contract,
            inventory=inventory,
            hosts=public_hosts,
            provenance=provenance,
            stages=stages,
            artifact_paths=artifact_paths,
            crash=crash,
            orchestrator_pid=os.getpid(),
            watcher=watcher,
            current_stage=current_stage_name or _current_stage_name(stages),
            current_stage_run_id=current_stage_event_run_id,
            current_bucket=current_bucket,
        )
        checkpoint_payload = _build_checkpoint_payload(
            run_id=resolved_run_id,
            run_dir=run_dir,
            generated_at=generated_at,
            updated_at=updated_at,
            run_state=run_state,
            overall_status=overall_status,
            success=success,
            resume_available=resume_available,
            error=error,
            contract=requested_contract,
            inventory=inventory,
            frozen_plan=frozen_plan,
            hosts=public_hosts,
            provenance=provenance,
            stages=stages,
            artifact_paths=artifact_paths,
            crash=crash,
            orchestrator_pid=os.getpid(),
            watcher=watcher,
            current_stage=current_stage_name or _current_stage_name(stages),
            current_stage_run_id=current_stage_event_run_id,
            current_bucket=current_bucket,
        )
        summary_payload = _sanitize_run_value(summary_payload)
        checkpoint_payload = _sanitize_run_value(checkpoint_payload)
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target_inventory_path, inventory)
            _write_json(checkpoint_path, _json_safe(checkpoint_payload))
            _write_json(summary_path, _json_safe(summary_payload))
            summary_markdown_path.write_text(
                _render_summary_markdown(_json_safe(summary_payload)), encoding="utf-8"
            )
            _write_json(
                manifest_path,
                {
                    **_json_safe(summary_payload),
                    "inventory": inventory,
                    "checkpoint": _json_safe(checkpoint_payload),
                    "frozen_plan": _sanitize_run_value(frozen_plan),
                },
            )
            _emit_live_progress(
                progress_recorder,
                stages=_sanitize_run_value(stages),
                run_state=run_state,
                overall_status=overall_status,
                artifact_paths=_sanitize_run_value(artifact_paths),
                emit_lock=progress_emit_lock,
                orchestrator_pid=os.getpid(),
            )
        return _json_safe(summary_payload)

    def _emit_stage_progress_snapshot(
        stage_name: str,
        child_run_id: str,
        child_progress_path: Path,
        *,
        bucket: Optional[str] = None,
    ) -> None:
        child_progress = _load_progress_current(child_progress_path)
        if child_progress is None:
            return
        _emit_live_progress(
            progress_recorder,
            stages=_sanitize_run_value(stages),
            run_state=run_state,
            overall_status=_current_overall_status(),
            artifact_paths=_sanitize_run_value(artifact_paths),
            emit_lock=progress_emit_lock,
            child_progress=_sanitize_run_value(child_progress),
            child_stage_name=stage_name,
            child_run_id=child_run_id,
            child_bucket=bucket,
            orchestrator_pid=os.getpid(),
        )

    def _run_with_stage_progress_mirror(
        stage_name: str,
        child_run_id: str,
        child_progress_path: Path,
        invoke,
        *,
        bucket: Optional[str] = None,
    ):
        if progress_recorder is None:
            return invoke()

        stop_event = threading.Event()
        last_state_persist = {"monotonic": time.monotonic()}

        def _mirror_worker() -> None:
            while True:
                _emit_stage_progress_snapshot(
                    stage_name,
                    child_run_id,
                    child_progress_path,
                    bucket=bucket,
                )
                now = time.monotonic()
                if now - last_state_persist["monotonic"] >= _STATE_HEARTBEAT_SECONDS:
                    _persist_state()
                    last_state_persist["monotonic"] = now
                if stop_event.wait(_STAGE_PROGRESS_POLL_SECONDS):
                    break

        thread = threading.Thread(
            target=_mirror_worker,
            name=f"e2e-progress-{stage_name}-{child_run_id}",
            daemon=True,
        )
        thread.start()
        try:
            return invoke()
        finally:
            stop_event.set()
            thread.join(timeout=5.0)
            _emit_stage_progress_snapshot(
                stage_name,
                child_run_id,
                child_progress_path,
                bucket=bucket,
            )

    def _start_stage(stage_name: str, event_run_id: str) -> None:
        nonlocal \
            current_stage_name, \
            current_stage_started_at, \
            current_stage_event_run_id, \
            current_bucket
        current_stage_name = stage_name
        current_stage_started_at = time.monotonic()
        current_stage_event_run_id = event_run_id
        current_bucket = None
        _replace_stage(stages, name=stage_name, status="running")
        _append_run_event("stage_started", stage=stage_name, run_id=event_run_id)
        _persist_state()

    def _finish_stage(
        stage_name: str,
        *,
        status: str,
        command: Optional[List[str]] = None,
        returncode: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        issues: Optional[List[str]] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        nonlocal \
            current_stage_name, \
            current_stage_started_at, \
            current_stage_event_run_id, \
            current_bucket
        stage = _find_stage(stages, stage_name)
        _replace_stage(
            stages,
            name=stage_name,
            status=status,
            command=command,
            returncode=returncode,
            result=result,
            artifacts=artifacts,
            issues=issues,
            duration_ms=duration_ms,
            attempts=stage.get("attempts") or [],
        )
        _append_run_event(
            "stage_finished",
            stage=stage_name,
            status=status,
            run_id=current_stage_event_run_id or stage.get("run_id"),
        )
        current_stage_name = None
        current_stage_started_at = None
        current_stage_event_run_id = None
        current_bucket = None
        _persist_state()

    def _abort_current_stage(message: str) -> None:
        nonlocal \
            current_stage_name, \
            current_stage_started_at, \
            current_stage_event_run_id, \
            current_bucket
        if not current_stage_name:
            return
        stage = _find_stage(stages, current_stage_name)
        attempts = stage.get("attempts") or []
        if attempts:
            latest_attempt = attempts[-1]
            if latest_attempt.get("status") == "running":
                _mark_attempt_aborted(
                    current_stage_name,
                    latest_attempt,
                    reason=message,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                    ended_at=_utc_now(),
                )
        issues = list(stage.get("issues") or [])
        if message not in issues:
            issues.append(message)
        _replace_stage(
            stages,
            name=current_stage_name,
            status="aborted",
            issues=issues,
            attempts=attempts,
            duration_ms=int((time.monotonic() - current_stage_started_at) * 1000)
            if current_stage_started_at is not None
            else stage.get("duration_ms"),
        )
        _append_run_event(
            "stage_finished",
            stage=current_stage_name,
            status="aborted",
            run_id=current_stage_event_run_id or stage.get("run_id"),
            error=message,
        )
        current_stage_name = None
        current_stage_started_at = None
        current_stage_event_run_id = None
        current_bucket = None
        _persist_state()

    if dry_run:
        return attach_benchmark_e2e_status_hints(
            _sanitize_run_value(
                {
                    "success": True,
                    "dry_run": True,
                    "run_id": resolved_run_id,
                    "run_dir": str(run_dir),
                    "run_state": "dry_run",
                    "overall_status": "dry_run",
                    "generated_at": generated_at,
                    "updated_at": generated_at,
                    "resume_available": bool(resume),
                    "manifest_path": str(manifest_path),
                    "summary_path": str(summary_path),
                    "summary_markdown_path": str(summary_markdown_path),
                    "progress_path": str(progress_path),
                    "checkpoint_path": str(checkpoint_path),
                    "target_inventory_path": str(target_inventory_path),
                    "events_path": str(events_path),
                    "inventory": _summarize_inventory_for_summary(inventory),
                    "hosts": _sanitize_run_value(cluster_host_config),
                    "provenance": provenance,
                    "contract": requested_contract,
                    "stages": stages,
                    "frozen_plan": frozen_plan,
                }
            ),
            resolved_run_id,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_inventory_path, inventory)
    _append_run_event("run_resumed" if resume else "run_started", run_id=resolved_run_id)
    _persist_state()
    if auto_resume and not dry_run and not os.environ.get(_E2E_WATCHER_SUPERVISED_ENV):
        watcher_info = watch_benchmark_e2e_sweep_run(
            run_id=resolved_run_id,
            repo_root=repo_root,
            poll_interval_seconds=watch_poll_interval_seconds,
            max_auto_resumes=max_auto_resumes,
        )
        if watcher_info.get("success"):
            _append_run_event(
                "watcher_armed",
                run_id=resolved_run_id,
                watcher_pid=watcher_info.get("watcher_pid"),
                already_running=bool(watcher_info.get("already_running", False)),
            )
        _persist_state()

    previous_handlers: Dict[int, Any] = {}
    abort_signal: Dict[str, Optional[str]] = {"signal": None}

    def _handle_abort_signal(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        abort_signal["signal"] = signame
        raise _E2EAbort(f"received {signame}")

    abort_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        abort_signals.append(signal.SIGHUP)

    for signum in abort_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_abort_signal)

    try:
        if run_tier1:
            tier1_stage = _find_stage(stages, "tier1")
            if tier1_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                attempt_index = len(tier1_stage.get("attempts") or [])
                tier1_run_id = _stage_attempt_run_id(stage_run_ids["tier1"], attempt_index)
                tier1_command = [
                    "python",
                    "-m",
                    "cli.aisp",
                    "bench",
                    "run-tier1",
                    "--run-id",
                    tier1_run_id,
                    "--profile",
                    profile_type,
                    "--validity-profile",
                    normalized_validity_profile,
                ]
                tier1_attempt = _stage_attempt_entry(
                    run_id=tier1_run_id,
                    status="running",
                    command=tier1_command,
                    started_at=_utc_now(),
                )
                tier1_stage.setdefault("attempts", []).append(tier1_attempt)
                _start_stage("tier1", tier1_run_id)
                tier1_progress_path = _benchmark_run_event_paths(
                    tier1_run_id,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                )["progress"]
                with _benchmark_queue_lock("tier1", tier1_run_id, repo_root=repo_root):
                    tier1_result = _run_with_stage_progress_mirror(
                        "tier1",
                        tier1_run_id,
                        tier1_progress_path,
                        lambda: _invoke_run_tier1_suite(
                            bench_root=active_bench_root,
                            profile_type=profile_type,
                            output_format=output_format,
                            suite_timeout=suite_timeout,
                            timeout_multiplier=timeout_multiplier,
                            validity_profile=normalized_validity_profile,
                            allow_portable_expectations_update=allow_portable_expectations_update,
                            reproducible=reproducible,
                            cold_start=cold_start,
                            force_synchronize=force_synchronize,
                            iterations=iterations,
                            warmup=warmup,
                            gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                            gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                            artifacts_dir=artifacts_dir,
                            run_id=tier1_run_id,
                            log_level=log_level,
                            log_file=log_file,
                            single_gpu=single_gpu,
                            accept_regressions=accept_regressions,
                            update_expectations=update_expectations,
                            allow_mixed_provenance=allow_mixed_provenance,
                            ncu_metric_set=ncu_metric_set,
                            ncu_replay_mode=ncu_replay_mode,
                            nsys_timeout_seconds=nsys_timeout_seconds,
                            ncu_timeout_seconds=ncu_timeout_seconds,
                        ),
                    )
                tier1_status, tier1_issues, tier1_benchmark_summary = _benchmark_stage_status(
                    tier1_result,
                    required_paths=[
                        "summary_path",
                        "regression_summary_path",
                        "regression_json_path",
                        "trend_snapshot_path",
                    ],
                    require_complete=True,
                    allow_comparison_regressions=accept_regressions or update_expectations,
                    expected_run_id=tier1_run_id,
                    repo_root=repo_root,
                    artifacts_dir=artifacts_dir,
                    expected_git_commit=str(validated_git_commit),
                )
                tier1_attempt.update(
                    {
                        "status": tier1_status,
                        "ended_at": _utc_now(),
                        "returncode": 1 if tier1_status == "failed" else 0,
                        "result": tier1_result,
                        "artifacts": {
                            "summary_path": tier1_result.get("summary_path"),
                            "regression_summary_path": tier1_result.get("regression_summary_path"),
                            "regression_json_path": tier1_result.get("regression_json_path"),
                            "trend_snapshot_path": tier1_result.get("trend_snapshot_path"),
                            "history_root": tier1_result.get("history_root"),
                        },
                        "issues": tier1_issues,
                        "duration_ms": int(
                            (time.monotonic() - (current_stage_started_at or time.monotonic()))
                            * 1000
                        ),
                    }
                )
                if tier1_benchmark_summary is not None:
                    tier1_attempt["benchmark_summary"] = tier1_benchmark_summary
                _finish_stage(
                    "tier1",
                    status=tier1_status,
                    command=tier1_command,
                    returncode=1 if tier1_status == "failed" else 0,
                    result=tier1_result,
                    artifacts=tier1_attempt.get("artifacts"),
                    issues=tier1_issues,
                    duration_ms=tier1_attempt.get("duration_ms"),
                )

        if run_full_sweep:
            full_stage = _find_stage(stages, "full_sweep")
            if full_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                _start_stage("full_sweep", stage_run_ids["full_sweep"])
                full_stage_issues: List[str] = []
                full_stage_result: Dict[str, Any] = {"buckets": {}}
                bucket_outcomes: List[str] = []
                single_targets = list(
                    (frozen_plan.get("full_sweep") or {}).get("single_gpu_targets") or []
                )
                multi_targets = list(
                    (frozen_plan.get("full_sweep") or {}).get("multi_gpu_targets") or []
                )
                if not single_targets and not multi_targets:
                    full_stage_issues.append("no benchmark targets discovered for full sweep")
                    full_status = "failed"
                else:
                    full_status = "succeeded"
                    for bucket_name, all_targets in (
                        ("single_gpu", single_targets),
                        ("multi_gpu", multi_targets),
                    ):
                        if not all_targets:
                            continue
                        current_bucket_targets = list(
                            inventory.get(
                                "single_gpu" if bucket_name == "single_gpu" else "multi_gpu"
                            )
                            or []
                        )
                        all_units = list(
                            (frozen_plan.get("full_sweep") or {}).get(
                                "single_gpu_units"
                                if bucket_name == "single_gpu"
                                else "multi_gpu_units"
                            )
                            or [entry["name"] for entry in _group_targets_by_unit(all_targets)]
                        )
                        bucket_attempts = _bucket_attempts(full_stage, bucket_name)
                        verified_bucket_attempts, _prior_evidence_issues = (
                            _verified_full_sweep_attempts(
                                bucket_attempts,
                                repo_root=repo_root,
                                artifacts_dir=artifacts_dir,
                                expected_git_commit=str(validated_git_commit),
                            )
                        )
                        completed_units = _completed_units_from_attempts(
                            verified_bucket_attempts,
                            frozen_targets=all_targets,
                        )
                        remaining_targets = _remaining_targets_after_completed_units(
                            all_targets,
                            completed_units=completed_units,
                        )
                        current_target_lookup = {
                            _canonical_target_name(target) for target in current_bucket_targets
                        }
                        missing_units = list(
                            dict.fromkeys(
                                _canonical_unit_name(target.split(":", 1)[0])
                                for target in remaining_targets
                                if _canonical_target_name(target) not in current_target_lookup
                            )
                        )
                        if bucket_name == "multi_gpu" and gpu_count < 2:
                            skip_reason = f"requires >=2 visible GPUs; detected {gpu_count}"
                            if not bucket_attempts:
                                skip_attempt = _stage_attempt_entry(
                                    run_id=_bucket_attempt_run_id(
                                        stage_run_ids["full_sweep"], bucket_name, 0
                                    ),
                                    bucket=bucket_name,
                                    status="skipped",
                                    targets=all_targets,
                                    units=all_units,
                                    completed_units=completed_units,
                                    reason=skip_reason,
                                    issues=[skip_reason],
                                    started_at=_utc_now(),
                                    ended_at=_utc_now(),
                                )
                                full_stage.setdefault("attempts", []).append(skip_attempt)
                                bucket_attempts = _bucket_attempts(full_stage, bucket_name)
                                _persist_state()
                            full_stage_result["buckets"][bucket_name] = {
                                "targets": all_targets,
                                "status": "skipped",
                                "reason": skip_reason,
                                "attempts": bucket_attempts,
                            }
                            full_stage_issues.append(
                                f"multi-GPU bucket skipped because only {gpu_count} visible GPU(s) were detected"
                            )
                            bucket_outcomes.append("partial")
                            continue
                        if missing_units:
                            missing_issue = f"resume could not resolve current benchmark targets for unit(s): {', '.join(missing_units)}"
                            full_stage_result["buckets"][bucket_name] = {
                                "targets": all_targets,
                                "status": "failed",
                                "reason": missing_issue,
                                "attempts": bucket_attempts,
                                "missing_units": missing_units,
                            }
                            full_stage_issues.append(missing_issue)
                            bucket_outcomes.append("failed")
                            continue
                        if not remaining_targets:
                            latest_attempt = bucket_attempts[-1] if bucket_attempts else None
                            full_stage_result["buckets"][bucket_name] = {
                                "targets": all_targets,
                                "status": "succeeded",
                                "attempts": bucket_attempts,
                                "latest_attempt_run_id": (
                                    latest_attempt.get("run_id") if latest_attempt else None
                                ),
                            }
                            bucket_outcomes.append("succeeded")
                            continue

                        attempt_index = len(bucket_attempts)
                        bucket_run_id = _bucket_attempt_run_id(
                            stage_run_ids["full_sweep"], bucket_name, attempt_index
                        )
                        bucket_command = [
                            "python",
                            "-m",
                            "cli.aisp",
                            "bench",
                            "run",
                            "--run-id",
                            bucket_run_id,
                            "--profile",
                            profile_type,
                            "--validity-profile",
                            normalized_validity_profile,
                            *sum([["-t", target] for target in remaining_targets], []),
                        ]
                        if full_sweep_suite_timeout is not None:
                            bucket_command.extend(
                                ["--suite-timeout", str(full_sweep_suite_timeout)]
                            )
                        attempt_units = [
                            entry["name"] for entry in _group_targets_by_unit(remaining_targets)
                        ]
                        bucket_attempt = _stage_attempt_entry(
                            run_id=bucket_run_id,
                            bucket=bucket_name,
                            status="running",
                            targets=remaining_targets,
                            units=attempt_units,
                            completed_units=[],
                            active_unit=attempt_units[0] if attempt_units else None,
                            command=bucket_command,
                            started_at=_utc_now(),
                        )
                        full_stage.setdefault("attempts", []).append(bucket_attempt)
                        current_bucket = bucket_name
                        _persist_state()
                        bucket_progress_path = _benchmark_run_event_paths(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )["progress"]
                        with _benchmark_queue_lock(
                            f"full_sweep_{bucket_name}", bucket_run_id, repo_root=repo_root
                        ):
                            bucket_result = _run_with_stage_progress_mirror(
                                "full_sweep",
                                bucket_run_id,
                                bucket_progress_path,
                                partial(
                                    _invoke_execute_benchmarks,
                                    targets=remaining_targets,
                                    bench_root=active_bench_root,
                                    output_format=output_format,
                                    profile_type=profile_type,
                                    suite_timeout=full_sweep_suite_timeout,
                                    timeout_multiplier=timeout_multiplier,
                                    validity_profile=normalized_validity_profile,
                                    allow_portable_expectations_update=allow_portable_expectations_update,
                                    reproducible=reproducible,
                                    cold_start=cold_start,
                                    force_synchronize=force_synchronize,
                                    iterations=iterations,
                                    warmup=warmup,
                                    gpu_sm_clock_mhz=gpu_sm_clock_mhz,
                                    gpu_mem_clock_mhz=gpu_mem_clock_mhz,
                                    artifacts_dir=artifacts_dir,
                                    run_id=bucket_run_id,
                                    log_level=log_level,
                                    log_file=log_file,
                                    single_gpu=single_gpu,
                                    accept_regressions=accept_regressions,
                                    update_expectations=update_expectations,
                                    allow_mixed_provenance=allow_mixed_provenance,
                                    ncu_metric_set=ncu_metric_set,
                                    ncu_replay_mode=ncu_replay_mode,
                                    nsys_timeout_seconds=nsys_timeout_seconds,
                                    ncu_timeout_seconds=ncu_timeout_seconds,
                                    exit_on_failure=False,
                                    enforce_external_assets=False,
                                ),
                                bucket=bucket_name,
                            )
                        bucket_status, bucket_issues, bucket_benchmark_summary = (
                            _benchmark_stage_status(
                                bucket_result,
                                required_paths=["output_json"],
                                required_targets=remaining_targets,
                            )
                        )
                        unit_progress = _load_benchmark_unit_progress(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )
                        completed_units = _completed_units_from_target_outcomes(
                            remaining_targets,
                            bucket_benchmark_summary,
                        )
                        benchmark_paths = _benchmark_run_event_paths(
                            bucket_run_id,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )
                        bucket_attempt.update(
                            {
                                "status": bucket_status,
                                "ended_at": _utc_now(),
                                "returncode": 1 if bucket_status == "failed" else 0,
                                "result": bucket_result,
                                "artifacts": {
                                    "run_dir": str(benchmark_paths["run_dir"]),
                                    "events_path": str(benchmark_paths["events"]),
                                    "output_json": str(benchmark_paths["output_json"]),
                                    "progress_path": str(benchmark_paths["progress"]),
                                },
                                "issues": bucket_issues,
                                "duration_ms": int(
                                    (
                                        time.monotonic()
                                        - (current_stage_started_at or time.monotonic())
                                    )
                                    * 1000
                                ),
                                "completed_units": completed_units,
                                "active_unit": (
                                    unit_progress.get("active_unit")
                                    if unit_progress.get("active_unit") not in completed_units
                                    else None
                                ),
                            }
                        )
                        if bucket_benchmark_summary is not None:
                            bucket_attempt["benchmark_summary"] = bucket_benchmark_summary
                        _attach_benchmark_attempt_state(
                            "full_sweep",
                            bucket_attempt,
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                        )
                        verified_bucket_attempts, evidence_issues = _verified_full_sweep_attempts(
                            _bucket_attempts(full_stage, bucket_name),
                            repo_root=repo_root,
                            artifacts_dir=artifacts_dir,
                            expected_git_commit=str(validated_git_commit),
                        )
                        exact_completed_units = _completed_units_from_attempts(
                            verified_bucket_attempts,
                            frozen_targets=all_targets,
                        )
                        expected_completed_units = [
                            str(unit["name"]) for unit in _group_targets_by_unit(all_targets)
                        ]
                        if exact_completed_units != expected_completed_units:
                            bucket_status = "failed"
                            exact_issue = (
                                "full-sweep bucket lacks one successful terminal outcome for "
                                "every frozen target"
                            )
                            if exact_issue not in bucket_issues:
                                bucket_issues.append(exact_issue)
                            for evidence_issue in evidence_issues:
                                if evidence_issue not in bucket_issues:
                                    bucket_issues.append(evidence_issue)
                            bucket_attempt["status"] = "failed"
                            bucket_attempt["returncode"] = 1
                            bucket_attempt["issues"] = bucket_issues
                        full_stage_result["buckets"][bucket_name] = {
                            "targets": all_targets,
                            "status": bucket_status,
                            "attempts": _bucket_attempts(full_stage, bucket_name),
                            "latest_attempt_run_id": bucket_run_id,
                        }
                        full_stage_issues.extend(bucket_issues)
                        bucket_outcomes.append(bucket_status)
                        _persist_state()
                        current_bucket = None

                    if any(status == "aborted" for status in bucket_outcomes):
                        full_status = "aborted"
                    elif any(status == "failed" for status in bucket_outcomes):
                        full_status = "failed"
                    elif any(status in {"partial", "skipped"} for status in bucket_outcomes):
                        full_status = "partial"
                    elif bucket_outcomes:
                        full_status = "succeeded"
                    else:
                        full_status = "failed"
                        full_stage_issues.append("no full-sweep bucket produced a terminal result")
                _finish_stage(
                    "full_sweep",
                    status=full_status,
                    returncode=1 if full_status in {"failed", "aborted"} else 0,
                    result=full_stage_result,
                    artifacts={"target_inventory_path": str(target_inventory_path)},
                    issues=full_stage_issues,
                    duration_ms=int(
                        (time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000
                    ),
                )

        if run_cluster:
            cluster_stage = _find_stage(stages, "cluster")
            if cluster_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
                attempt_index = len(cluster_stage.get("attempts") or [])
                cluster_run_id = _stage_attempt_run_id(stage_run_ids["cluster"], attempt_index)
                cluster_attempt = _stage_attempt_entry(
                    run_id=cluster_run_id,
                    status="running",
                    started_at=_utc_now(),
                )
                cluster_stage.setdefault("attempts", []).append(cluster_attempt)
                _start_stage("cluster", cluster_run_id)
                cluster_result = _run_with_stage_progress_mirror(
                    "cluster",
                    cluster_run_id,
                    _cluster_run_progress_path(cluster_run_id, repo_root=repo_root),
                    lambda: _invoke_run_cluster_common_eval(
                        preset=cluster_preset,
                        run_id=cluster_run_id,
                        hosts=cluster_host_config["hosts"],
                        labels=cluster_host_config["labels"],
                        ssh_user=cluster_host_config["ssh_user"],
                        ssh_key=cluster_host_config["ssh_key"],
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
                        extra_args=cluster_extra_args,
                        timeout_seconds=timeout_seconds,
                    ),
                )
                cluster_status, cluster_issues, cluster_scorecard = _cluster_stage_status(
                    cluster_result,
                    expected_run_id=cluster_run_id,
                    repo_root=repo_root,
                    expected_git_commit=str(validated_git_commit),
                )
                cluster_attempt.update(
                    {
                        "status": cluster_status,
                        "ended_at": _utc_now(),
                        "command": cluster_result.get("command"),
                        "returncode": int(cluster_result.get("returncode", 0) or 0),
                        "result": cluster_result,
                        "artifacts": {
                            "run_dir": cluster_result.get("run_dir"),
                            "manifest_path": cluster_result.get("manifest_path"),
                            "fabric_scorecard": cluster_scorecard,
                        },
                        "issues": cluster_issues,
                        "duration_ms": int(
                            (time.monotonic() - (current_stage_started_at or time.monotonic()))
                            * 1000
                        ),
                    }
                )
                _finish_stage(
                    "cluster",
                    status=cluster_status,
                    command=cluster_result.get("command"),
                    returncode=int(cluster_result.get("returncode", 0) or 0),
                    result=cluster_result,
                    artifacts=cluster_attempt.get("artifacts"),
                    issues=cluster_issues,
                    duration_ms=cluster_attempt.get("duration_ms"),
                )

        fabric_duplicate = (
            run_fabric and run_cluster and cluster_preset.strip().lower() == "fabric-systems"
        )
        fabric_stage = _find_stage(stages, "fabric")
        if fabric_duplicate:
            if fabric_stage.get("status") != "skipped_duplicate":
                duplicate_attempt = _stage_attempt_entry(
                    run_id=stage_run_ids["fabric"],
                    status="skipped_duplicate",
                    reason="fabric stage duplicated by cluster preset fabric-systems",
                    issues=["fabric stage duplicated by cluster preset fabric-systems"],
                    started_at=_utc_now(),
                    ended_at=_utc_now(),
                )
                fabric_stage.setdefault("attempts", []).append(duplicate_attempt)
                _finish_stage(
                    "fabric",
                    status="skipped_duplicate",
                    issues=["fabric stage duplicated by cluster preset fabric-systems"],
                    duration_ms=0,
                )
        elif run_fabric and fabric_stage.get("status") not in {"succeeded", "skipped_duplicate"}:
            attempt_index = len(fabric_stage.get("attempts") or [])
            fabric_run_id = _stage_attempt_run_id(stage_run_ids["fabric"], attempt_index)
            fabric_attempt = _stage_attempt_entry(
                run_id=fabric_run_id,
                status="running",
                started_at=_utc_now(),
            )
            fabric_stage.setdefault("attempts", []).append(fabric_attempt)
            _start_stage("fabric", fabric_run_id)
            fabric_result = _run_with_stage_progress_mirror(
                "fabric",
                fabric_run_id,
                _cluster_run_progress_path(fabric_run_id, repo_root=repo_root),
                lambda: _invoke_run_cluster_fabric_eval(
                    run_id=fabric_run_id,
                    hosts=cluster_host_config["hosts"],
                    labels=cluster_host_config["labels"],
                    ssh_user=cluster_host_config["ssh_user"],
                    ssh_key=cluster_host_config["ssh_key"],
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
                    extra_args=cluster_extra_args,
                    timeout_seconds=timeout_seconds,
                ),
            )
            fabric_status, fabric_issues, fabric_scorecard = _cluster_stage_status(
                fabric_result,
                require_scorecard=True,
                expected_run_id=fabric_run_id,
                repo_root=repo_root,
                expected_git_commit=str(validated_git_commit),
            )
            fabric_attempt.update(
                {
                    "status": fabric_status,
                    "ended_at": _utc_now(),
                    "command": fabric_result.get("command"),
                    "returncode": int(fabric_result.get("returncode", 0) or 0),
                    "result": fabric_result,
                    "artifacts": {
                        "run_dir": fabric_result.get("run_dir"),
                        "manifest_path": fabric_result.get("manifest_path"),
                        "fabric_scorecard": fabric_scorecard,
                    },
                    "issues": fabric_issues,
                    "duration_ms": int(
                        (time.monotonic() - (current_stage_started_at or time.monotonic())) * 1000
                    ),
                }
            )
            _finish_stage(
                "fabric",
                status=fabric_status,
                command=fabric_result.get("command"),
                returncode=int(fabric_result.get("returncode", 0) or 0),
                result=fabric_result,
                artifacts=fabric_attempt.get("artifacts"),
                issues=fabric_issues,
                duration_ms=fabric_attempt.get("duration_ms"),
            )

        final_git_commit, final_git_error = _validated_clean_git_commit(
            repo_root=repo_root,
            git_info=dict(get_git_info()),
        )
        if final_git_error:
            raise RuntimeError(f"Final Git provenance validation failed: {final_git_error}")
        if final_git_commit != validated_git_commit:
            raise RuntimeError("Git commit changed during the E2E run")

        run_state = "completed"
        resume_available = False
        _append_run_event(
            "run_finished",
            run_id=resolved_run_id,
            overall_status=_roll_up_overall_status(
                [str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")]
            ),
            success=_roll_up_overall_status(
                [str(stage.get("status") or "planned") for stage in stages if stage.get("enabled")]
            )
            == "succeeded",
        )
        run_finished_event_emitted = True
        return attach_benchmark_e2e_status_hints(_persist_state(), resolved_run_id)
    except (_E2EAbort, KeyboardInterrupt) as exc:
        error = str(exc)
        crash = {
            "type": type(exc).__name__,
            "message": str(exc),
            "signal": abort_signal.get("signal"),
            "traceback": traceback.format_exc(),
        }
        run_state = "aborted"
        resume_available = True
        _abort_current_stage(error)
    except Exception as exc:
        error = str(exc)
        crash = {
            "type": type(exc).__name__,
            "message": str(exc),
            "signal": abort_signal.get("signal"),
            "traceback": traceback.format_exc(),
        }
        run_state = "aborted"
        resume_available = True
        _abort_current_stage(error)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if not run_finished_event_emitted:
        _append_run_event(
            "run_finished",
            run_id=resolved_run_id,
            overall_status="aborted",
            success=False,
            error=error,
        )
    return attach_benchmark_e2e_status_hints(_persist_state(), resolved_run_id)
