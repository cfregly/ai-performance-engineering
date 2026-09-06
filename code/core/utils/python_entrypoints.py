"""Helpers for launching repo Python entrypoints without local sys.path hacks."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


def _dedupe_entries(entries: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for entry in entries:
        text = str(entry).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def build_repo_python_env(
    repo_root: Path,
    *,
    base_env: Optional[Mapping[str, str]] = None,
    extra_pythonpath: Optional[Sequence[str | Path]] = None,
) -> Dict[str, str]:
    """Return an env dict with ``repo_root`` prepended to ``PYTHONPATH``."""
    env = dict(os.environ if base_env is None else base_env)
    entries: List[str] = [str(Path(repo_root).resolve())]
    for entry in extra_pythonpath or ():
        entries.append(str(Path(entry).resolve()))
    existing = env.get("PYTHONPATH", "")
    if existing:
        entries.extend(
            str(Path(part).resolve()) for part in existing.split(os.pathsep) if part.strip()
        )
    env["PYTHONPATH"] = os.pathsep.join(_dedupe_entries(entries))
    return env


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from ``start`` (or cwd)."""
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    while current.parent != current:
        if (current / ".git").exists() or (current / "core" / "common").exists():
            return current
        current = current.parent
    return current


@contextmanager
def temporary_sys_path(*entries: str | Path) -> Iterator[None]:
    """Temporarily prepend resolved entries to ``sys.path`` and restore it."""
    original = list(sys.path)
    try:
        resolved_entries = [str(Path(entry).resolve()) for entry in entries]
        sys.path[:] = _dedupe_entries([*resolved_entries, *original])
        yield
    finally:
        sys.path[:] = original


def install_local_module_override(module_name: str, package_path: str | Path) -> ModuleType:
    """Load ``module_name`` directly from a local package path without mutating ``sys.path``."""
    package_dir = Path(package_path).resolve()
    init_py = package_dir / "__init__.py"
    spec = importlib_util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_name} from {package_dir}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_module_from_path(
    module_name: str,
    module_path: str | Path,
    *,
    extra_sys_path: Optional[Sequence[str | Path]] = None,
) -> ModuleType:
    """Load a module from disk and register it in ``sys.modules`` before exec.

    Registering first is required for Python 3.12 features like ``@dataclass``
    that consult ``sys.modules`` while the module body is still executing.
    """
    resolved_path = Path(module_path).resolve()
    spec = importlib_util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_name} from {resolved_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        search_path = [resolved_path.parent, *(extra_sys_path or ())]
        with temporary_sys_path(*search_path):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def build_python_entry_command(
    *,
    module_name: Optional[str] = None,
    script_path: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build ``python`` command for either a module or a script path."""
    if bool(module_name) == bool(script_path):
        raise ValueError("Specify exactly one of module_name or script_path.")
    python = python_executable or sys.executable
    if module_name:
        return [python, "-m", module_name, *(argv or [])]
    return [python, str(Path(script_path).resolve()), *(argv or [])]


def build_torchrun_entry_command(
    *,
    module_name: Optional[str] = None,
    script_path: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
    nproc_per_node: int,
    nnodes: Optional[int] = None,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> List[str]:
    """Launch distributed workers using the selected Python environment."""
    if bool(module_name) == bool(script_path):
        raise ValueError("Specify exactly one of module_name or script_path.")
    cmd = [
        python_executable or sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(int(nproc_per_node)),
    ]
    if nnodes is not None:
        cmd.extend(["--nnodes", str(int(nnodes))])
    if rdzv_backend is not None:
        cmd.extend(["--rdzv_backend", str(rdzv_backend)])
    if rdzv_endpoint is not None:
        cmd.extend(["--rdzv_endpoint", str(rdzv_endpoint)])
    if module_name:
        cmd.extend(["-m", module_name])
    else:
        cmd.append(str(Path(script_path).resolve()))
    cmd.extend(argv or [])
    return cmd
