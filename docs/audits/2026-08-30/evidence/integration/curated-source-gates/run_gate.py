"""Capture one source-only gate with isolated caches and no overwrite."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GATES = {
    "make-lint": ["make", "lint", f"PYTHON={sys.executable}"],
    "make-validate": ["make", "validate"],
    "strict-contract-verbose": [sys.executable, "-m", "core.scripts.linting.check_benchmarks", "--include-unpaired", "--fail-on-warnings", "--verbose"],
    "strict-fallbacks": [sys.executable, "core/scripts/audit_silent_fallbacks.py", "--fail-on-findings", "--categories", "global_warning_filter", "stderr_reassignment", "stdout_reassignment", "stdio_dup2_hijack", "syntax_error", "read_error"],
    "import-edges": [sys.executable, "core/scripts/check_import_edges.py"],
    "repository-ruff": ["ruff", "check", ".", "--select", "E9,F63,F7,F82,B006,B023"],
    "focused-ruff": ["ruff", "check", "core/optimization/campaign.py", "core/optimization/campaign_evidence.py", "core/optimization/evidence_validation.py", "core/harness/llm_patch_worker.py", "core/optimization/search/mcts_optimizer.py", "core/analysis/llm_patch_promotion.py", "tests/test_optimization_campaign.py", "tests/test_campaign_evidence.py", "tests/test_llm_patch_worker.py", "tests/test_mcts_optimizer.py", "tests/test_llm_patch_promotion.py", "tests/test_repository_configuration.py"],
    "focused-mypy": [sys.executable, "-m", "mypy", "core/optimization/campaign.py", "core/analysis/llm_patch_promotion.py"],
}


def main() -> None:
    name = sys.argv[1]
    command = GATES[name]
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[5]
    destination = evidence / name
    destination.mkdir()
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
           "PYTHONDONTWRITEBYTECODE": "1", "RUFF_CACHE_DIR": str(destination / "ruff-cache"),
           "MYPY_CACHE_DIR": str(destination / "mypy-cache")}
    started = time.monotonic()
    record = {"command": command, "cwd": str(root / "code"),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": 180,
              "environment_overrides": {key: env[key] for key in ("PATH", "PYTHONDONTWRITEBYTECODE", "RUFF_CACHE_DIR", "MYPY_CACHE_DIR")}}
    (destination / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    with (destination / "output.txt").open("xb") as output:
        child = subprocess.Popen(command, cwd=root / "code", env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = child.pid
        try:
            record["exit_code"] = child.wait(timeout=180)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            record["exit_code"] = child.returncode
            record["timed_out"] = True
    record["elapsed_seconds"] = time.monotonic() - started
    (destination / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"gate": name, "exit_code": record["exit_code"], "timed_out": record["timed_out"], "elapsed_seconds": record["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
