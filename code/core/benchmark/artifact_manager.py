"""Artifact manager for standardized benchmark artifact organization.

Manages directory structure for benchmark runs, ensuring consistent organization
of results, profiles, reports, logs, and manifests.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
import re

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_run_id(run_id: str) -> str:
    """Return a filesystem-safe run id or raise before any artifact write."""
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "Run id must start with a letter or digit and contain only letters, digits, dots, "
            "underscores, or hyphens"
        )
    return run_id


def default_artifacts_root(repo_root: Optional[Path] = None) -> Path:
    """Return the standard artifacts root for run outputs."""
    root = repo_root or Path.cwd()
    return root / "artifacts" / "runs"


def slugify(value: str, max_len: int = 80) -> str:
    """Return a filesystem-safe, ASCII-only slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "run"
    return cleaned[:max_len].strip("_")


def build_run_id(
    kind: str,
    label: Optional[str] = None,
    base_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Build a predictable, self-describing run id."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, slugify(kind)]
    if label:
        parts.append(slugify(label))
    run_id = "__".join(parts)
    if base_dir is None:
        return run_id

    candidate = run_id
    counter = 2
    while (base_dir / candidate).exists():
        candidate = f"{run_id}__{counter}"
        counter += 1
    return candidate


def build_bench_run_label(targets: Sequence[str], profile_type: Optional[str]) -> str:
    parts = []
    if profile_type:
        parts.append(f"profile-{slugify(profile_type)}")
    if not targets:
        parts.append("targets-all")
    elif len(targets) == 1:
        parts.append(f"targets-{slugify(targets[0])}")
    else:
        parts.append(f"targets-{len(targets)}")
    return "__".join(parts)


def build_tool_run_label(tool_key: str, label: str) -> str:
    label_part = slugify(label) if label else "run"
    return f"{slugify(tool_key)}__{label_part}"


class ArtifactManager:
    """Manages artifact directory structure for benchmark runs.
    
    Creates and manages a standardized directory structure:
    artifacts/runs/<run_id>/
      - results/          # JSON results
      - profiles/         # nsys-rep, ncu-rep, torch traces
      - reports/          # markdown reports
      - logs/             # structured logs
      - manifest.json     # run manifest
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        run_kind: Optional[str] = None,
        run_label: Optional[str] = None,
    ):
        """Initialize artifact manager.
        
        Args:
            base_dir: Base directory for artifacts (defaults to ./artifacts/runs)
            run_id: Optional run ID (defaults to a descriptive timestamped id)
            run_kind: Optional run kind used to generate a default run_id
            run_label: Optional run label used to generate a default run_id
        """
        if base_dir is None:
            base_dir = Path("artifacts") / "runs"
        
        if run_id is None:
            run_id = build_run_id(run_kind or "run", run_label, base_dir=Path(base_dir))
        run_id = validate_run_id(run_id)
        
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        
        # Create directory structure
        self._create_structure()
    
    def _create_structure(self) -> None:
        """Create the directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "results").mkdir(exist_ok=True)
        (self.run_dir / "profiles").mkdir(exist_ok=True)
        (self.run_dir / "reports").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "progress").mkdir(exist_ok=True)
    
    @property
    def results_dir(self) -> Path:
        """Get path to results directory."""
        return self.run_dir / "results"
    
    @property
    def profiles_dir(self) -> Path:
        """Get path to profiles directory."""
        return self.run_dir / "profiles"
    
    @property
    def reports_dir(self) -> Path:
        """Get path to reports directory."""
        return self.run_dir / "reports"
    
    @property
    def logs_dir(self) -> Path:
        """Get path to logs directory."""
        return self.run_dir / "logs"

    @property
    def progress_dir(self) -> Path:
        """Get path to progress directory."""
        return self.run_dir / "progress"

    @property
    def manifest_path(self) -> Path:
        """Get path to manifest.json file."""
        return self.run_dir / "manifest.json"
    
    def get_result_path(self, filename: str = "benchmark_test_results.json") -> Path:
        """Get path for a result file.
        
        Args:
            filename: Name of the result file
        
        Returns:
            Path to result file
        """
        return self.results_dir / filename
    
    def get_profile_path(self, profiler: str, benchmark_name: str) -> Path:
        """Get path for a profiling artifact.
        
        Args:
            profiler: Profiler name ('nsys', 'ncu', 'torch')
            benchmark_name: Name of the benchmark
        
        Returns:
            Path to profile artifact
        """
        if profiler == "nsys":
            return self.profiles_dir / f"{benchmark_name}.nsys-rep"
        elif profiler == "ncu":
            return self.profiles_dir / f"{benchmark_name}.ncu-rep"
        elif profiler == "torch":
            return self.profiles_dir / f"{benchmark_name}_torch_trace.json"
        else:
            return self.profiles_dir / f"{benchmark_name}_{profiler}"
    
    def get_report_path(self, filename: str = "benchmark_report.md") -> Path:
        """Get path for a report file.
        
        Args:
            filename: Name of the report file
        
        Returns:
            Path to report file
        """
        return self.reports_dir / filename
    
    def get_log_path(self, filename: str = "benchmark.log") -> Path:
        """Get path for a log file.
        
        Args:
            filename: Name of the log file
        
        Returns:
            Path to log file
        """
        return self.logs_dir / filename
    
    def __str__(self) -> str:
        """String representation."""
        return str(self.run_dir)
    
    def __repr__(self) -> str:
        """Representation."""
        return f"ArtifactManager(base_dir={self.base_dir}, run_id={self.run_id})"
