"""Source-level checks for benchmark hot-path anti-patterns."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import inspect
from pathlib import Path
import sys
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple


_HOT_PATH_REPO_ROOT = Path(__file__).resolve().parents[1]

_RANDOM_INPUT_CALLS = {
    ("torch", "rand"),
    ("torch", "rand_like"),
    ("torch", "randint"),
    ("torch", "randn"),
    ("torch", "randn_like"),
}

_ALLOCATION_CALLS = {
    ("torch", "empty"),
    ("torch", "empty_like"),
    ("torch", "full"),
    ("torch", "full_like"),
    ("torch", "ones"),
    ("torch", "ones_like"),
    ("torch", "zeros"),
    ("torch", "zeros_like"),
}

_TENSOR_CONSTRUCTION_CALLS = {
    ("torch", "as_tensor"),
    ("torch", "from_numpy"),
    ("torch", "tensor"),
}

_NUMPY_IO_CALLS = {
    ("np", "load"),
    ("np", "save"),
    ("numpy", "load"),
    ("numpy", "save"),
}

_LOGGING_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warning",
}

_COMPILE_CALLS = {
    ("torch", "compile"),
    ("compile_callable",),
    ("torch", "utils", "cpp_extension", "load"),
    ("torch", "utils", "cpp_extension", "load_inline"),
    ("triton", "compile"),
}

_PROFILER_CALLS = {
    ("torch", "profiler", "profile"),
    ("torch", "cuda", "profiler", "start"),
    ("torch", "cuda", "profiler", "stop"),
}

_SUBPROCESS_OR_NETWORK_CALLS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "request"),
    ("subprocess", "Popen"),
    ("subprocess", "check_output"),
    ("subprocess", "run"),
}

_PATH_IO_METHODS = {
    "open",
    "read_bytes",
    "read_text",
    "write_bytes",
    "write_text",
}

_HOST_TRANSFER_METHODS = {
    "cpu": "benchmark_fn() transfers tensors to CPU via .cpu() "
    "(line {line}); keep host transfers out of the timed hot path",
    "item": "benchmark_fn() materializes a host scalar via .item() "
    "(line {line}); keep host round-trips out of the timed hot path",
    "numpy": "benchmark_fn() converts tensors to NumPy via .numpy() "
    "(line {line}); keep host conversions out of the timed hot path",
    "tolist": "benchmark_fn() materializes tensors as Python lists via .tolist() "
    "(line {line}); keep host conversions out of the timed hot path",
}

_Finding = Tuple[str, str]
_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _ImportRef:
    module_name: Optional[str]
    symbol_name: Optional[str]


@dataclass(frozen=True)
class _ModuleContext:
    module_path: Path
    module_name: str
    tree: ast.Module
    classes: Dict[str, ast.ClassDef]
    functions: Dict[str, _FunctionNode]
    imports: Dict[str, _ImportRef]


@dataclass(frozen=True)
class _ClassRef:
    module_path: Path
    module_name: str
    class_name: str


@dataclass(frozen=True)
class _ScopeRef:
    module_path: Path
    module_name: str
    function_name: str
    class_name: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, str]:
        return (
            str(self.module_path),
            self.class_name or "",
            self.function_name,
        )


def _function_node_map(
    class_node: ast.ClassDef,
) -> Dict[str, _FunctionNode]:
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _self_method_call_name(node: ast.Call) -> Optional[str]:
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "self":
        return None
    return node.func.attr


def _reachable_same_class_functions(
    class_node: ast.ClassDef,
    *,
    root_name: str = "benchmark_fn",
) -> List[_FunctionNode]:
    methods = _function_node_map(class_node)
    root = methods.get(root_name)
    if root is None:
        return []

    visited: set[str] = set()
    ordered: List[_FunctionNode] = []

    def visit(method_name: str) -> None:
        if method_name in visited:
            return
        method_node = methods.get(method_name)
        if method_node is None:
            return
        visited.add(method_name)
        ordered.append(method_node)
        for child in ast.walk(method_node):
            if not isinstance(child, ast.Call):
                continue
            callee = _self_method_call_name(child)
            if callee is None or callee == method_name:
                continue
            visit(callee)

    visit(root_name)
    return ordered


def _dedupe_messages(messages: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        deduped.append(message)
    return deduped


def _attribute_chain(node: ast.AST) -> Tuple[str, ...]:
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _call_target_chain(node: ast.AST) -> Tuple[str, ...]:
    if isinstance(node, ast.Call):
        return _call_target_chain(node.func)
    return _attribute_chain(node)


def _has_stream_or_event_hint(parts: Tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in parts)
    return any("stream" in part or "event" in part for part in lowered)


def _assigned_name_targets(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_assigned_name_targets(elt))
        return names
    return []


def _is_stream_or_event_factory(node: ast.AST) -> bool:
    target = _call_target_chain(node)
    if _has_stream_or_event_hint(target):
        return True
    if not target:
        return False
    tail = target[-1].lower()
    return tail in {"event", "stream", "current_stream", "default_stream"}


def _benchmark_fn_sync_like_names(
    function_node: _FunctionNode,
) -> set[str]:
    known: set[str] = set()
    pending: List[Tuple[List[str], ast.AST]] = []

    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            pending.append(
                (
                    [name for target in node.targets for name in _assigned_name_targets(target)],
                    node.value,
                )
            )
        elif isinstance(node, ast.AnnAssign):
            pending.append((_assigned_name_targets(node.target), node.value))

    changed = True
    while changed:
        changed = False
        for targets, value in pending:
            if not targets or value is None:
                continue
            is_sync_like = False
            if _is_stream_or_event_factory(value):
                is_sync_like = True
            elif isinstance(value, ast.Name) and value.id in known:
                is_sync_like = True
            elif _has_stream_or_event_hint(_attribute_chain(value)):
                is_sync_like = True
            if not is_sync_like:
                continue
            for name in targets:
                if name not in known:
                    known.add(name)
                    changed = True
    return known


def _is_cpu_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() == "cpu"
    if isinstance(node, ast.Call):
        target = _attribute_chain(node.func)
        if target == ("torch", "device") and node.args:
            return _is_cpu_target(node.args[0])
    return False


def _is_hot_path_logging_call(target: Tuple[str, ...]) -> bool:
    if target == ("print",):
        return True
    if not target or target[-1] not in _LOGGING_METHODS:
        return False
    owner = target[:-1]
    return bool(owner) and (owner[0] == "logging" or owner[-1] in {"log", "logger"})


def _normalize_allowed_codes(allowed_codes: Optional[Iterable[str]]) -> set[str]:
    return {str(code).strip().lower() for code in (allowed_codes or ()) if str(code).strip()}


def _filter_findings(findings: List[_Finding], *, allowed_codes: Optional[Iterable[str]] = None) -> List[str]:
    allowed = _normalize_allowed_codes(allowed_codes)
    return [message for code, message in findings if code not in allowed]


def _module_name_for_file(file_path: Path) -> str:
    resolved = file_path.resolve()
    try:
        relative = resolved.relative_to(_HOT_PATH_REPO_ROOT)
    except ValueError:
        return resolved.stem
    if relative.name == "__init__.py":
        return ".".join(relative.parts[:-1])
    return ".".join(relative.with_suffix("").parts)


def _module_path_for_name(module_name: Optional[str]) -> Optional[Path]:
    if not module_name:
        return None
    candidate = _HOT_PATH_REPO_ROOT / Path(*module_name.split("."))
    module_file = candidate.with_suffix(".py")
    if module_file.exists():
        return module_file.resolve()
    package_init = candidate / "__init__.py"
    if package_init.exists():
        return package_init.resolve()
    return None


def _absolute_import_module(current_module: str, module: Optional[str], level: int) -> Optional[str]:
    if level == 0:
        return module

    package_parts = current_module.split(".")[:-1]
    prefix_len = len(package_parts) - (level - 1)
    if prefix_len < 0:
        return None

    prefix = package_parts[:prefix_len]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


@lru_cache(maxsize=None)
def _load_module_context(module_path_str: str) -> Optional[_ModuleContext]:
    module_path = Path(module_path_str)
    if not module_path.exists() or module_path.suffix != ".py":
        return None

    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    module_name = _module_name_for_file(module_path)
    imports: Dict[str, _ImportRef] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            absolute_module = _absolute_import_module(module_name, node.module, node.level)
            for alias in node.names:
                imports[alias.asname or alias.name] = _ImportRef(
                    module_name=absolute_module,
                    symbol_name=alias.name,
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = _ImportRef(
                    module_name=alias.name,
                    symbol_name=None,
                )

    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return _ModuleContext(
        module_path=module_path.resolve(),
        module_name=module_name,
        tree=tree,
        classes=classes,
        functions=functions,
        imports=imports,
    )


def _resolve_module_symbol(
    module_name: Optional[str],
    symbol_name: str,
    *,
    seen: Optional[set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef | _ScopeRef]:
    if not module_name:
        return None
    key = (module_name, symbol_name)
    if seen is not None and key in seen:
        return None
    next_seen = set(seen or set())
    next_seen.add(key)
    module_path = _module_path_for_name(module_name)
    if module_path is None:
        return None
    module_ctx = _load_module_context(str(module_path))
    if module_ctx is None:
        return None
    return _resolve_symbol(module_ctx, symbol_name, seen=next_seen)


def _resolve_symbol(
    module_ctx: _ModuleContext,
    symbol_name: str,
    *,
    seen: Optional[set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef | _ScopeRef]:
    if symbol_name in module_ctx.functions:
        return _ScopeRef(
            module_path=module_ctx.module_path,
            module_name=module_ctx.module_name,
            function_name=symbol_name,
        )
    if symbol_name in module_ctx.classes:
        return _ClassRef(
            module_path=module_ctx.module_path,
            module_name=module_ctx.module_name,
            class_name=symbol_name,
        )

    import_ref = module_ctx.imports.get(symbol_name)
    if import_ref is None or import_ref.symbol_name is None:
        return None
    return _resolve_module_symbol(import_ref.module_name, import_ref.symbol_name, seen=seen)


def _resolve_attr_symbol(
    module_ctx: _ModuleContext,
    parts: Tuple[str, ...],
    *,
    seen: Optional[set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef | _ScopeRef]:
    if len(parts) < 2:
        return None

    import_ref = module_ctx.imports.get(parts[0])
    if import_ref is None or import_ref.module_name is None or import_ref.symbol_name is not None:
        return None

    module_name = import_ref.module_name
    rest = list(parts[1:])
    imported_suffix = module_name.split(".")[1:]
    if rest[:-1] and imported_suffix == rest[:-1]:
        return _resolve_module_symbol(module_name, rest[-1], seen=seen)
    if rest[:-1]:
        module_name = ".".join([module_name, *rest[:-1]])
    return _resolve_module_symbol(module_name, rest[-1], seen=seen)


def _resolve_expr_symbol(
    module_ctx: _ModuleContext,
    expr: ast.AST,
    *,
    seen: Optional[set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef | _ScopeRef]:
    if isinstance(expr, ast.Name):
        return _resolve_symbol(module_ctx, expr.id, seen=seen)
    parts = _attribute_chain(expr)
    if not parts:
        return None
    return _resolve_attr_symbol(module_ctx, parts, seen=seen)


def _resolve_expr_class(
    module_ctx: _ModuleContext,
    expr: ast.AST,
) -> Optional[_ClassRef]:
    resolved = _resolve_expr_symbol(module_ctx, expr)
    return resolved if isinstance(resolved, _ClassRef) else None


def _resolve_scope_ref_node(scope_ref: _ScopeRef) -> Tuple[Optional[_ModuleContext], Optional[_FunctionNode]]:
    module_ctx = _load_module_context(str(scope_ref.module_path))
    if module_ctx is None:
        return None, None
    if scope_ref.class_name:
        class_node = module_ctx.classes.get(scope_ref.class_name)
        if class_node is None:
            return module_ctx, None
        return module_ctx, _function_node_map(class_node).get(scope_ref.function_name)
    return module_ctx, module_ctx.functions.get(scope_ref.function_name)


def _collect_self_attr_instance_bindings(
    class_node: ast.ClassDef,
    module_ctx: _ModuleContext,
) -> Dict[str, _ClassRef]:
    bindings: Dict[str, _ClassRef] = {}
    for method_node in _function_node_map(class_node).values():
        for node in ast.walk(method_node):
            if isinstance(node, ast.Assign):
                value = node.value
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                targets = [node.target]
            else:
                continue
            if value is None:
                continue
            class_ref = None
            if isinstance(value, ast.Call):
                class_ref = _resolve_expr_class(module_ctx, value.func)
            elif isinstance(value, ast.Attribute):
                owner = value.value
                if isinstance(owner, ast.Name) and owner.id == "self":
                    class_ref = bindings.get(value.attr)
            if class_ref is None:
                continue
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    bindings[target.attr] = class_ref
    return bindings


def _collect_local_instance_bindings(
    function_node: _FunctionNode,
    module_ctx: _ModuleContext,
    *,
    self_attr_bindings: Optional[Dict[str, _ClassRef]] = None,
) -> Dict[str, _ClassRef]:
    bindings: Dict[str, _ClassRef] = {}
    self_attr_bindings = self_attr_bindings or {}
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        else:
            continue
        if value is None:
            continue
        class_ref = None
        if isinstance(value, ast.Call):
            class_ref = _resolve_expr_class(module_ctx, value.func)
        elif isinstance(value, ast.Name):
            class_ref = bindings.get(value.id)
        elif isinstance(value, ast.Attribute):
            owner = value.value
            if isinstance(owner, ast.Name):
                if owner.id == "self":
                    class_ref = self_attr_bindings.get(value.attr)
                else:
                    class_ref = bindings.get(owner.id)
        if class_ref is None:
            continue
        for target in targets:
            for name in _assigned_name_targets(target):
                bindings[name] = class_ref
    return bindings


def _resolve_method_ref_for_class(class_ref: _ClassRef, method_name: str) -> Optional[_ScopeRef]:
    module_ctx = _load_module_context(str(class_ref.module_path))
    if module_ctx is None:
        return None
    class_node = module_ctx.classes.get(class_ref.class_name)
    if class_node is None:
        return None
    method_node = _function_node_map(class_node).get(method_name)
    if method_node is None:
        return None
    return _ScopeRef(
        module_path=module_ctx.module_path,
        module_name=module_ctx.module_name,
        class_name=class_ref.class_name,
        function_name=method_node.name,
    )


def _reachable_scopes_for_class(
    class_node: ast.ClassDef,
    *,
    module_path: Optional[Path] = None,
    class_name: Optional[str] = None,
    root_name: str = "benchmark_fn",
) -> List[_ScopeRef]:
    if module_path is None or not module_path.exists():
        return []

    module_ctx = _load_module_context(str(module_path.resolve()))
    if module_ctx is None:
        return []

    effective_class_name = class_name or class_node.name
    effective_class_node = module_ctx.classes.get(effective_class_name)
    if effective_class_node is None:
        return []

    self_attr_bindings = _collect_self_attr_instance_bindings(effective_class_node, module_ctx)
    visited: set[Tuple[str, str, str]] = set()
    ordered: List[_ScopeRef] = []

    def visit(scope_ref: _ScopeRef) -> None:
        if scope_ref.key in visited:
            return
        visited.add(scope_ref.key)
        resolved_module_ctx, function_node = _resolve_scope_ref_node(scope_ref)
        if resolved_module_ctx is None or function_node is None:
            return
        ordered.append(scope_ref)

        local_bindings = _collect_local_instance_bindings(
            function_node,
            resolved_module_ctx,
            self_attr_bindings=self_attr_bindings if scope_ref.class_name == effective_class_name else None,
        )
        for child in ast.walk(function_node):
            if not isinstance(child, ast.Call):
                continue

            if scope_ref.class_name == effective_class_name:
                callee = _self_method_call_name(child)
                if callee is not None and callee != scope_ref.function_name:
                    visit(
                        _ScopeRef(
                            module_path=resolved_module_ctx.module_path,
                            module_name=resolved_module_ctx.module_name,
                            class_name=scope_ref.class_name,
                            function_name=callee,
                        )
                    )
                    continue

            if isinstance(child.func, ast.Attribute):
                owner = child.func.value
                if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == "self":
                    class_ref = self_attr_bindings.get(owner.attr)
                    if class_ref is not None:
                        method_ref = _resolve_method_ref_for_class(class_ref, child.func.attr)
                        if method_ref is not None:
                            visit(method_ref)
                            continue
                if isinstance(owner, ast.Name):
                    class_ref = local_bindings.get(owner.id)
                    if class_ref is not None:
                        method_ref = _resolve_method_ref_for_class(class_ref, child.func.attr)
                        if method_ref is not None:
                            visit(method_ref)
                            continue

            resolved = _resolve_expr_symbol(resolved_module_ctx, child.func)
            if isinstance(resolved, _ScopeRef):
                visit(resolved)

    visit(
        _ScopeRef(
            module_path=module_ctx.module_path,
            module_name=module_ctx.module_name,
            class_name=effective_class_name,
            function_name=root_name,
        )
    )
    return ordered


def _iter_function_nodes_for_class(
    class_node: ast.ClassDef,
    *,
    module_path: Optional[Path] = None,
    class_name: Optional[str] = None,
    root_name: str = "benchmark_fn",
) -> List[_FunctionNode]:
    scopes = _reachable_scopes_for_class(
        class_node,
        module_path=module_path,
        class_name=class_name,
        root_name=root_name,
    )
    if not scopes:
        return _reachable_same_class_functions(class_node, root_name=root_name)

    ordered: List[_FunctionNode] = []
    seen_nodes: set[Tuple[int, int, int]] = set()
    for scope_ref in scopes:
        _, function_node = _resolve_scope_ref_node(scope_ref)
        if function_node is None:
            continue
        node_key = (function_node.lineno, function_node.col_offset, function_node.end_lineno or -1)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        ordered.append(function_node)
    return ordered


def benchmark_fn_sync_warnings(
    function_node: _FunctionNode,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return source-level warnings for explicit synchronization in benchmark_fn()."""
    findings: List[_Finding] = []
    sync_like_names = _benchmark_fn_sync_like_names(function_node)
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_synchronize":
            findings.append((
                "sync",
                "benchmark_fn() contains _synchronize() "
                f"(line {node.lineno}); this inflates harness timings and blocks CUDA graph capture",
            ))
            continue
        if _attribute_chain(node.func) == ("torch", "cuda", "synchronize"):
            findings.append((
                "sync",
                "benchmark_fn() contains torch.cuda.synchronize() "
                f"(line {node.lineno}); this inflates harness timings and blocks CUDA graph capture",
            ))
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize":
            owner_chain = _call_target_chain(node.func.value)
            owner_name = owner_chain[0] if owner_chain else ""
            if owner_chain and (
                _has_stream_or_event_hint(owner_chain) or owner_name in sync_like_names
            ):
                findings.append((
                    "sync",
                    "benchmark_fn() contains stream/event synchronize() "
                    f"(line {node.lineno}); move this post-timing or use stream dependencies instead",
                ))
    return _filter_findings(findings, allowed_codes=allowed_codes)


def benchmark_fn_sync_warnings_for_class(
    class_node: ast.ClassDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
    module_path: Optional[Path] = None,
    class_name: Optional[str] = None,
) -> List[str]:
    messages: List[str] = []
    for function_node in _iter_function_nodes_for_class(
        class_node,
        module_path=module_path,
        class_name=class_name,
    ):
        messages.extend(
            benchmark_fn_sync_warnings(
                function_node,
                allowed_codes=allowed_codes,
            )
        )
    return _dedupe_messages(messages)


def benchmark_fn_antipattern_warnings(
    function_node: _FunctionNode,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return source-level warnings for common hot-path anti-patterns."""
    findings: List[_Finding] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue

        target = _attribute_chain(node.func)
        if target in _RANDOM_INPUT_CALLS:
            findings.append((
                "random_input_regeneration",
                "benchmark_fn() regenerates random inputs via "
                f"{'.'.join(target)}() (line {node.lineno}); allocate inputs in setup() "
                "and reuse/mutate buffers during timing",
            ))
            continue
        if target in _ALLOCATION_CALLS:
            findings.append((
                "allocation",
                "benchmark_fn() allocates tensors via "
                f"{'.'.join(target)}() (line {node.lineno}); preallocate reusable buffers in setup() "
                "unless allocation cost is the benchmarked behavior",
            ))
            continue
        if target in _TENSOR_CONSTRUCTION_CALLS:
            findings.append((
                "tensor_construction",
                "benchmark_fn() constructs a tensor view or copy via "
                f"{'.'.join(target)}() (line {node.lineno}); create or wrap stable inputs in "
                "setup() unless construction/materialization is the benchmarked behavior",
            ))
            continue
        if target in _NUMPY_IO_CALLS:
            findings.append((
                "io",
                "benchmark_fn() performs NumPy file I/O via "
                f"{'.'.join(target)}() (line {node.lineno}); keep storage I/O out of the timed "
                "hot path unless it is the benchmarked behavior",
            ))
            continue
        if _is_hot_path_logging_call(target):
            findings.append((
                "logging",
                "benchmark_fn() emits output or invokes logging via "
                f"{'.'.join(target)}() (line {node.lineno}); capture structured state and report "
                "it after timing instead",
            ))
            continue
        if target in _COMPILE_CALLS:
            findings.append((
                "compile",
                "benchmark_fn() triggers compilation via "
                f"{'.'.join(target)}() (line {node.lineno}); compile kernels/functions in setup()",
            ))
            continue
        if target in _PROFILER_CALLS:
            findings.append((
                "profiling",
                "benchmark_fn() starts/stops profiling via "
                f"{'.'.join(target)}() (line {node.lineno}); profiling setup must stay out of the timed hot path",
            ))
            continue
        if target in _SUBPROCESS_OR_NETWORK_CALLS:
            findings.append((
                "io",
                "benchmark_fn() performs subprocess/network I/O via "
                f"{'.'.join(target)}() (line {node.lineno}); external I/O invalidates timing measurements",
            ))
            continue

        if isinstance(node.func, ast.Name) and node.func.id == "open":
            findings.append((
                "io",
                "benchmark_fn() performs file I/O via open() "
                f"(line {node.lineno}); file reads/writes invalidate timing measurements",
            ))
            continue

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in _HOST_TRANSFER_METHODS:
                findings.append(("host_transfer", _HOST_TRANSFER_METHODS[attr].format(line=node.lineno)))
                continue
            if attr in _PATH_IO_METHODS:
                findings.append((
                    "io",
                    "benchmark_fn() performs file I/O via "
                    f".{attr}() (line {node.lineno}); file reads/writes invalidate timing measurements",
                ))
                continue
            if attr == "to":
                if node.args and _is_cpu_target(node.args[0]):
                    findings.append((
                        "host_transfer",
                        "benchmark_fn() transfers tensors to CPU via .to('cpu') "
                        f"(line {node.lineno}); keep host transfers out of the timed hot path",
                    ))
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "device" and keyword.value is not None and _is_cpu_target(
                        keyword.value
                    ):
                        findings.append((
                            "host_transfer",
                            "benchmark_fn() transfers tensors to CPU via .to(device='cpu') "
                            f"(line {node.lineno}); keep host transfers out of the timed hot path",
                        ))
                        break
    return _filter_findings(findings, allowed_codes=allowed_codes)


def benchmark_fn_antipattern_warnings_for_class(
    class_node: ast.ClassDef,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
    module_path: Optional[Path] = None,
    class_name: Optional[str] = None,
) -> List[str]:
    messages: List[str] = []
    for function_node in _iter_function_nodes_for_class(
        class_node,
        module_path=module_path,
        class_name=class_name,
    ):
        messages.extend(
            benchmark_fn_antipattern_warnings(
                function_node,
                allowed_codes=allowed_codes,
            )
        )
    return _dedupe_messages(messages)


def _parse_benchmark_fn(benchmark_fn: Any) -> Optional[_FunctionNode]:
    try:
        source = textwrap.dedent(inspect.getsource(benchmark_fn))
    except (OSError, TypeError):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )


def _runtime_class_source_context(
    benchmark_fn: Any,
) -> Tuple[Optional[ast.ClassDef], Optional[Path], Optional[str]]:
    owner = getattr(benchmark_fn, "__self__", None)
    if owner is None:
        return None, None, None
    cls = owner if inspect.isclass(owner) else owner.__class__
    source_file: Optional[str] = None
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is not None:
        source_file = getattr(module, "__file__", None)
    if not source_file:
        func_obj = getattr(benchmark_fn, "__func__", benchmark_fn)
        code_obj = getattr(func_obj, "__code__", None)
        if code_obj is not None:
            source_file = code_obj.co_filename
    if not source_file:
        try:
            source_file = inspect.getsourcefile(cls)
        except (OSError, TypeError):
            source_file = None
    if source_file:
        module_path = Path(source_file).resolve()
        module_ctx = _load_module_context(str(module_path))
        if module_ctx is not None:
            class_node = module_ctx.classes.get(cls.__name__)
            if class_node is not None:
                return class_node, module_path, cls.__name__

    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return None, None, None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None, None

    class_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__
        ),
        None,
    )
    return class_node, None, cls.__name__


def check_benchmark_fn_sync_calls(
    benchmark_fn: Any,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Detect explicit CUDA synchronization calls in benchmark_fn source."""
    class_node, module_path, class_name = _runtime_class_source_context(benchmark_fn)
    if class_node is not None:
        findings = benchmark_fn_sync_warnings_for_class(
            class_node,
            allowed_codes=allowed_codes,
            module_path=module_path,
            class_name=class_name,
        )
        return len(findings) == 0, findings
    function_node = _parse_benchmark_fn(benchmark_fn)
    if function_node is None:
        return True, []
    findings = benchmark_fn_sync_warnings(function_node, allowed_codes=allowed_codes)
    return len(findings) == 0, findings


def check_benchmark_fn_antipatterns(
    benchmark_fn: Any,
    *,
    allowed_codes: Optional[Iterable[str]] = None,
) -> Tuple[bool, List[str]]:
    """Detect common performance anti-patterns in benchmark_fn source."""
    class_node, module_path, class_name = _runtime_class_source_context(benchmark_fn)
    if class_node is not None:
        findings = benchmark_fn_antipattern_warnings_for_class(
            class_node,
            allowed_codes=allowed_codes,
            module_path=module_path,
            class_name=class_name,
        )
        return len(findings) == 0, findings
    function_node = _parse_benchmark_fn(benchmark_fn)
    if function_node is None:
        return True, []
    findings = benchmark_fn_antipattern_warnings(function_node, allowed_codes=allowed_codes)
    return len(findings) == 0, findings
