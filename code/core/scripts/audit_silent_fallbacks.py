#!/usr/bin/env python3
"""Audit public/shared Python code for silent fallback and global suppression patterns."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_SCAN_PATTERNS = ("ch*", "labs", "core")
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "build",
    "dist",
    "venv",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    category: str
    message: str

    def to_dict(self, repo_root: Path) -> dict[str, object]:
        return {
            "path": str(self.path.relative_to(repo_root)),
            "line": self.line,
            "column": self.column,
            "category": self.category,
            "message": self.message,
        }


def _matches_attr(node: ast.AST, module: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == module
    )


def _is_catch_warnings_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _matches_attr(node.func, "warnings", "catch_warnings")


def _is_filterwarnings_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _matches_attr(node.func, "warnings", "filterwarnings")


def _is_os_dup2_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _matches_attr(node.func, "os", "dup2")


def _is_sys_stream(node: ast.AST, stream_name: str) -> bool:
    return _matches_attr(node, "sys", stream_name)


def _is_exception_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "Exception"


def _is_exception_tuple(node: ast.AST) -> bool:
    return isinstance(node, ast.Tuple) and any(_is_exception_name(item) for item in node.elts)


class SilentFallbackAudit(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.findings: List[Finding] = []
        self._ancestors: List[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        self._ancestors.append(node)
        super().generic_visit(node)
        self._ancestors.pop()

    def _record(self, node: ast.AST, *, category: str, message: str) -> None:
        self.findings.append(
            Finding(
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                category=category,
                message=message,
            )
        )

    def _inside_catch_warnings(self) -> bool:
        for ancestor in reversed(self._ancestors):
            if isinstance(ancestor, (ast.With, ast.AsyncWith)):
                for item in ancestor.items:
                    if _is_catch_warnings_call(item.context_expr):
                        return True
        return False

    def _inside_none_stdio_guard(self, stream_name: str) -> bool:
        for ancestor in reversed(self._ancestors):
            if not isinstance(ancestor, ast.If):
                continue
            test = ancestor.test
            if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
                continue
            left = test.left
            right = test.comparators[0]
            if isinstance(test.ops[0], ast.Is):
                if _is_sys_stream(left, stream_name) and isinstance(right, ast.Constant) and right.value is None:
                    return True
                if _is_sys_stream(right, stream_name) and isinstance(left, ast.Constant) and left.value is None:
                    return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if _is_filterwarnings_call(node) and not self._inside_catch_warnings():
            self._record(
                node,
                category="global_warning_filter",
                message="warnings.filterwarnings(...) mutates process-global warning state outside warnings.catch_warnings()",
            )
        if _is_os_dup2_call(node):
            self._record(
                node,
                category="stdio_dup2_hijack",
                message="os.dup2(...) hijacks process stdio and should not be used for warning suppression",
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if _is_sys_stream(target, "stderr") and not self._inside_none_stdio_guard("stderr"):
                self._record(
                    node,
                    category="stderr_reassignment",
                    message="sys.stderr is reassigned outside a sys.stderr is None recovery guard",
                )
                break
            if _is_sys_stream(target, "stdout") and not self._inside_none_stdio_guard("stdout"):
                self._record(
                    node,
                    category="stdout_reassignment",
                    message="sys.stdout is reassigned outside a sys.stdout is None recovery guard",
                )
                break
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        exception_matches = (
            node.type is None
            or _is_exception_name(node.type)
            or _is_exception_tuple(node.type)
        )
        if exception_matches and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self._record(
                node,
                category="silent_except_pass",
                message="except Exception: pass silently hides failures; emit a warning or fail fast instead",
            )
        self.generic_visit(node)


def _iter_python_files(repo_root: Path, scan_patterns: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in scan_patterns:
        for root in sorted(repo_root.glob(pattern)):
            if root.is_file() and root.suffix == ".py":
                resolved = root.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
                continue
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                relative_parts = path.relative_to(repo_root).parts
                if any(part in EXCLUDED_DIR_NAMES for part in relative_parts):
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def audit_file(path: Path) -> List[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [
            Finding(
                path=path,
                line=1,
                column=1,
                category="read_error",
                message=f"Failed to read {path}: {exc}",
            )
        ]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            Finding(
                path=path,
                line=exc.lineno or 1,
                column=(exc.offset or 1),
                category="syntax_error",
                message=f"Failed to parse {path}: {exc.msg}",
            )
        ]

    visitor = SilentFallbackAudit(path)
    visitor.visit(tree)
    return visitor.findings


def collect_findings(repo_root: Path, scan_patterns: Sequence[str]) -> List[Finding]:
    findings: List[Finding] = []
    for path in _iter_python_files(repo_root, scan_patterns):
        findings.extend(audit_file(path))
    return sorted(findings, key=lambda item: (str(item.path), item.line, item.column, item.category))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Python code for silent fallbacks and process-global warning suppression",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional repo-root-relative scan patterns or paths (default: ch*, labs, core)",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit non-zero when any finding is detected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional finding categories to include (default: all).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[2]
    scan_patterns = tuple(args.paths) if args.paths else DEFAULT_SCAN_PATTERNS
    findings = collect_findings(repo_root, scan_patterns)
    if args.categories:
        allowed = set(args.categories)
        findings = [finding for finding in findings if finding.category in allowed]

    if args.json:
        payload = {
            "scan_patterns": list(scan_patterns),
            "categories": list(args.categories) if args.categories else None,
            "finding_count": len(findings),
            "findings": [finding.to_dict(repo_root) for finding in findings],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Scanned patterns: {', '.join(scan_patterns)}")
        print(f"Findings: {len(findings)}")
        for finding in findings:
            rel_path = finding.path.relative_to(repo_root)
            print(
                f"{rel_path}:{finding.line}:{finding.column}: "
                f"[{finding.category}] {finding.message}"
            )

    if args.fail_on_findings and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
