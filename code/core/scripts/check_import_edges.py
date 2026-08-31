#!/usr/bin/env python3
"""Check import edges to enforce the new core.* namespace.

Rules:
  - Migrated analysis, profiling, optimization and common modules use core.*.
  - Operational modules that actually live in code/scripts retain their scripts.*
    namespace. Migrated core/scripts modules must use core.scripts.*.
  - Core code may not import from labs/* or ch* (to prevent back-edges).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRS = [
    "core",
    "scripts",
    *(f"ch{idx:02d}" for idx in range(1, 21)),
    *(f"ch{idx}" for idx in range(1, 10)),  # Retain legacy unpadded directories.
    "labs",
    "cli",
    "dashboard",
    "mcp",
    "tests",
]
SKIP_DIRS = {
    "third_party",
    "vendor",
    "book",
    ".git",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".next",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "tmp-tebuild",
    "tmp-tebuild.*",
}
BANNED_TOP_LEVEL = {"analysis", "profiling", "optimization", "common"}


def iter_python_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in SKIP_DIRS:
                continue
            if entry.is_dir():
                stack.append(entry)
                continue
            if entry.suffix == ".py":
                yield entry


def _module_toplevel(module: str) -> str:
    return module.split(".")[0] if module else ""


def _is_operational_script(module: str, root: Path) -> bool:
    """Allow a real repository module, never a blanket scripts.* exemption."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "scripts":
        return False
    candidate = root.joinpath(*parts)
    # Namespace packages are valid too; resolve within the actual scripts tree.
    for target in (candidate.with_suffix(".py"), candidate):
        if not target.exists():
            continue
        if not target.resolve().is_relative_to((root / "scripts").resolve()):
            continue
        if (target == candidate.with_suffix(".py") and target.is_file()) or target.is_dir():
            return True
    return False


def check_file(path: Path, root: Path = REPO_ROOT) -> List[Tuple[int, str]]:
    errors: List[Tuple[int, str]] = []
    try:
        tree = ast.parse(path.read_text())
    except Exception as exc:  # pragma: no cover - parse errors handled as failures
        return [(0, f"Failed to parse: {exc}")]

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return [(0, "Source file is outside the requested scan root")]
    in_core = relative.parts[0] == "core"

    def _check_module(module: str, lineno: int) -> None:
        top = _module_toplevel(module)
        if top in BANNED_TOP_LEVEL:
            errors.append(
                (lineno, f"Use core.{top}.* instead of importing '{module}' directly")
            )
        if top == "scripts" and not _is_operational_script(module, root):
            errors.append((lineno, f"Use core.scripts.* for migrated helpers; '{module}' is not an explicit repository scripts module"))
        is_chapter = top.startswith("ch") and top[2:].isdigit()
        if in_core and (is_chapter or top == "labs"):
            errors.append((lineno, f"Core code must not import '{module}' (labs/ch back-edge)"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative imports stay inside their package. Do not confuse
                # core.common with the migrated top-level common namespace.
                package = relative.parts[:-1]
                if node.level > len(package):
                    errors.append((node.lineno, "Relative import escapes the top-level package"))
                    continue
                prefix = package[:len(package) - node.level + 1]
                module = ".".join((*prefix, *(node.module or "").split("."))).rstrip(".")
                _check_module(module, node.lineno)
                continue
            if node.module == "scripts":
                for alias in node.names:
                    _check_module(f"scripts.{alias.name}", node.lineno)
                continue
            if node.module:
                _check_module(node.module, node.lineno)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (default: project root)",
    )
    args = parser.parse_args()

    all_errors: List[Tuple[Path, int, str]] = []
    for rel_dir in TARGET_DIRS:
        base = args.root / rel_dir
        if not base.exists():
            continue
        for path in iter_python_files(base):
            for lineno, msg in check_file(path, args.root):
                all_errors.append((path.relative_to(args.root), lineno, msg))

    if all_errors:
        for path, lineno, msg in sorted(all_errors):
            location = f"{path}:{lineno}" if lineno else f"{path}"
            print(f"{location}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
