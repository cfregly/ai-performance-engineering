"""Small artifact bundles for diagnostic tools, separate from benchmark claims."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def capture(argv: list[str], timeout: float = 15) -> dict[str, Any]:
    """Run an argument vector without a shell and preserve failures as evidence."""
    finite_number(timeout, "timeout", positive=True)
    started = time.time()
    monotonic = time.monotonic()
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        status = "ok" if process.returncode == 0 else "failed"
        stdout, stderr, rc = process.stdout, process.stderr, process.returncode
    except FileNotFoundError as exc:
        status, stdout, stderr, rc = "unavailable", "", str(exc), None
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        status, stdout, stderr, rc = "timeout", decode(exc.stdout), decode(exc.stderr), None
    return {"argv": argv, "status": status, "returncode": rc, "stdout": stdout,
            "stderr": stderr, "start_unix_s": started, "duration_s": time.monotonic() - monotonic}


def write_bundle(run_dir: Path, tool: str, raw: Any, report: dict[str, Any]) -> Path:
    """Write tool-specific files and retain other tools in the run manifest."""
    import fcntl

    if not tool or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in tool):
        raise ValueError("Tool artifact name must contain only lowercase letters, digits, hyphens or underscores")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".diagnostic-bundle.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _write_bundle_locked(run_dir, tool, raw, report)


def _atomic_text(path: Path, content: str) -> None:
    """Readers see either the previous complete file or the new complete file."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _write_bundle_locked(run_dir: Path, tool: str, raw: Any, report: dict[str, Any]) -> Path:
    for name in ("raw", "structured", "reports"):
        (run_dir / name).mkdir(exist_ok=True)
    raw_path = run_dir / "raw" / f"{tool}.json"
    report_path = run_dir / "structured" / f"{tool}.json"
    markdown_path = run_dir / "reports" / f"{tool}.md"
    if any(p.exists() for p in (raw_path, report_path, markdown_path)):
        raise ValueError(f"{tool} artifacts already exist in {run_dir}. Choose a new run directory.")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema": "aisp.diagnostic-run.v1", "run_id": run_dir.name,
        "created_unix_s": time.time(), "platform": platform.system(), "tools": {},
        "source_revision": capture(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"])["stdout"].strip(),
    }
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tools", {}), dict):
        raise ValueError("Existing diagnostic manifest must be an object with a tools object")
    report = {**report, "raw_artifact": f"raw/{tool}.json", "valid_for_performance_claim": False}
    raw_text = json.dumps(raw, indent=2, allow_nan=False) + "\n"
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    lines = [f"# {tool.replace('-', ' ').capitalize()}", "", f"Status: {report['status']}", "",
             "Diagnostic evidence. This report does not establish a benchmark speedup.", ""]
    for finding in report.get("findings", []):
        lines.append(f"- {finding['summary']} Next: {finding['next_measurement']}")
    lines += ["", f"Structured evidence: [report](../structured/{tool}.json)",
              f"Raw evidence: [capture](../raw/{tool}.json)", ""]
    _atomic_text(raw_path, raw_text)
    _atomic_text(report_path, report_text)
    _atomic_text(markdown_path, "\n".join(lines))
    manifest.setdefault("tools", {})[tool] = {
        "status": report["status"], "artifacts": [{"path": str(p.relative_to(run_dir)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (raw_path, report_path, markdown_path)]}
    code_root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__).resolve()]
    entry = {"network-diagnose": "ch03/network_diagnosis_tool.py",
             "cross-layer-diagnose": "core/analysis/cross_layer_diagnosis.py"}.get(tool)
    if entry:
        source_files.append(code_root / entry)
    manifest["tools"][tool]["source_files"] = [
        {"path": str(p.relative_to(code_root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in source_files]
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return report_path
