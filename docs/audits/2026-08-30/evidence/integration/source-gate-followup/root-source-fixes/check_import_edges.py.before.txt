#!/usr/bin/env python3
"""Check import edges to enforce the new core.* namespace.

Rules:
  - Banned top-level modules: analysis, profiling, optimization, scripts, common
    (must be imported via core.<module>).
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
    *(f"ch{idx}" for idx in range(1, 21)),
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
BANNED_TOP_LEVEL = {"analysis", "profiling", "optimization", "scripts", "common"}
CHAPTER_PREFIXES = {f"ch{idx}" for idx in range(1, 21)}


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


def check_file(path: Path) -> List[Tuple[int, str]]:
    errors: List[Tuple[int, str]] = []
    try:
        tree = ast.parse(path.read_text())
    except Exception as exc:  # pragma: no cover - parse errors handled as failures
        return [(0, f"Failed to parse: {exc}")]

    in_core = path.parts[0] == "core"

    def _check_module(module: str, lineno: int) -> None:
        top = _module_toplevel(module)
        if top in BANNED_TOP_LEVEL:
            errors.append(
                (lineno, f"Use core.{top}.* instead of importing '{module}' directly")
            )
        if in_core and (top in CHAPTER_PREFIXES or top == "labs"):
            errors.append((lineno, f"Core code must not import '{module}' (labs/ch back-edge)"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                # Relative import of names, skip
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
            for lineno, msg in check_file(path):
                all_errors.append((path.relative_to(args.root), lineno, msg))

    if all_errors:
        for path, lineno, msg in sorted(all_errors):
            location = f"{path}:{lineno}" if lineno else f"{path}"
            print(f"{location}: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
