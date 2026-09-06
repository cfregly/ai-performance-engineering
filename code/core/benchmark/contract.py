"""Benchmark contract definition and validation.

Defines the required interface that all benchmarks must implement,
and provides utilities for validating benchmark compliance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.hot_path_checks import (
    benchmark_fn_antipattern_warnings_for_class,
    benchmark_fn_sync_warnings_for_class,
)


_CONTRACT_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_BENCHMARK_QUALNAME = "core.harness.benchmark_harness.BaseBenchmark"
_VERIFICATION_PAYLOAD_MIXIN_QUALNAME = "core.benchmark.verification_mixin.VerificationPayloadMixin"


@lru_cache(maxsize=1)
def _registered_standalone_paths() -> frozenset[Path]:
    """Read core registry metadata without importing any workload modules."""
    from core.demos.demos_commands import DEMOS
    from core.tools.tools_commands import TOOLS

    return frozenset(
        spec.script_path.resolve()
        for registry in (DEMOS, TOOLS)
        for spec in registry.values()
    )


@dataclass(frozen=True)
class _ImportRef:
    module_name: Optional[str]
    symbol_name: Optional[str]


@dataclass(frozen=True)
class _ClassRef:
    module_path: Path
    class_name: str
    qualified_name: str


@dataclass(frozen=True)
class _ModuleContext:
    module_path: Path
    module_name: str
    tree: ast.Module
    classes: Dict[str, ast.ClassDef]
    imports: Dict[str, _ImportRef]


@dataclass(frozen=True)
class _AstClassInfo:
    ref: _ClassRef
    own_methods: Set[str]
    method_origins: Dict[str, Set[str]]
    ancestor_names: Set[str]


def _attribute_chain(node: ast.AST) -> Tuple[str, ...]:
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _function_nodes(class_node: ast.ClassDef) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        item for item in class_node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _method_names(class_node: ast.ClassDef) -> Set[str]:
    return {item.name for item in _function_nodes(class_node)}


def _literal_string_collection(
    class_node: ast.ClassDef,
    attribute_name: str,
) -> Set[str]:
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            if len(item.targets) != 1 or not isinstance(item.targets[0], ast.Name):
                continue
            if item.targets[0].id != attribute_name:
                continue
            value_node = item.value
        elif isinstance(item, ast.AnnAssign):
            if not isinstance(item.target, ast.Name):
                continue
            if item.target.id != attribute_name or item.value is None:
                continue
            value_node = item.value
        else:
            continue
        try:
            value = ast.literal_eval(value_node)
        except Exception:
            return set()
        if isinstance(value, (list, tuple, set, frozenset)):
            return {str(entry).strip().lower() for entry in value if str(entry).strip()}
        return set()
    return set()


def _module_name_for_file(file_path: Path) -> str:
    resolved = file_path.resolve()
    try:
        relative = resolved.relative_to(_CONTRACT_REPO_ROOT)
    except ValueError:
        return resolved.stem

    if relative.name == "__init__.py":
        return ".".join(relative.parts[:-1])
    return ".".join(relative.with_suffix("").parts)


def _module_path_for_name(module_name: Optional[str]) -> Optional[Path]:
    if not module_name:
        return None
    candidate = _CONTRACT_REPO_ROOT / Path(*module_name.split("."))
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
    return _ModuleContext(
        module_path=module_path.resolve(),
        module_name=module_name,
        tree=tree,
        classes=classes,
        imports=imports,
    )


def _make_class_ref(module_ctx: _ModuleContext, class_name: str) -> _ClassRef:
    return _ClassRef(
        module_path=module_ctx.module_path,
        class_name=class_name,
        qualified_name=f"{module_ctx.module_name}.{class_name}",
    )


def _resolve_module_symbol(
    module_name: Optional[str],
    symbol_name: str,
    seen: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef]:
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
    return _resolve_symbol(module_ctx, symbol_name, next_seen)


def _resolve_symbol(
    module_ctx: _ModuleContext,
    symbol_name: str,
    seen: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef]:
    if symbol_name in module_ctx.classes:
        return _make_class_ref(module_ctx, symbol_name)

    import_ref = module_ctx.imports.get(symbol_name)
    if import_ref is None or import_ref.symbol_name is None:
        return None

    return _resolve_module_symbol(import_ref.module_name, import_ref.symbol_name, seen)


def _resolve_attr_base_expr(
    module_ctx: _ModuleContext,
    parts: Tuple[str, ...],
    seen: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef]:
    if len(parts) < 2:
        return None

    import_ref = module_ctx.imports.get(parts[0])
    if import_ref is None or import_ref.module_name is None or import_ref.symbol_name is not None:
        return None

    module_name = import_ref.module_name
    rest = list(parts[1:])
    imported_suffix = module_name.split(".")[1:]
    if rest[:-1] and imported_suffix == rest[:-1]:
        return _resolve_module_symbol(module_name, rest[-1], seen)
    if rest[:-1]:
        module_name = ".".join([module_name, *rest[:-1]])
    return _resolve_module_symbol(module_name, rest[-1], seen)


def _resolve_base_expr(
    module_ctx: _ModuleContext,
    base_expr: ast.expr,
    seen: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[_ClassRef]:
    if isinstance(base_expr, ast.Name):
        return _resolve_symbol(module_ctx, base_expr.id, seen)
    parts = _attribute_chain(base_expr)
    if not parts:
        return None
    return _resolve_attr_base_expr(module_ctx, parts, seen)


@lru_cache(maxsize=None)
def _get_ast_class_info(module_path_str: str, class_name: str) -> Optional[_AstClassInfo]:
    module_ctx = _load_module_context(module_path_str)
    if module_ctx is None:
        return None
    class_node = module_ctx.classes.get(class_name)
    if class_node is None:
        return None

    class_ref = _make_class_ref(module_ctx, class_name)
    own_methods = _method_names(class_node)
    method_origins: Dict[str, Set[str]] = {
        method_name: {class_ref.qualified_name}
        for method_name in own_methods
    }
    ancestor_names: Set[str] = {class_ref.qualified_name}

    for base in class_node.bases:
        base_ref = _resolve_base_expr(module_ctx, base)
        if base_ref is None:
            continue
        ancestor_names.add(base_ref.qualified_name)
        base_info = _get_ast_class_info(str(base_ref.module_path), base_ref.class_name)
        if base_info is None:
            continue
        ancestor_names.update(base_info.ancestor_names)
        for method_name, providers in base_info.method_origins.items():
            method_origins.setdefault(method_name, set()).update(providers)

    return _AstClassInfo(
        ref=class_ref,
        own_methods=own_methods,
        method_origins=method_origins,
        ancestor_names=ancestor_names,
    )


def _resolve_return_expr_to_class_refs(
    module_ctx: _ModuleContext,
    expr: ast.expr,
    assignments: Dict[str, ast.expr],
    seen_locals: Optional[Set[str]] = None,
) -> List[_ClassRef]:
    local_seen = set(seen_locals or set())

    if isinstance(expr, ast.Name):
        assigned = assignments.get(expr.id)
        if assigned is not None and expr.id not in local_seen:
            local_seen.add(expr.id)
            return _resolve_return_expr_to_class_refs(module_ctx, assigned, assignments, local_seen)
        class_ref = _resolve_symbol(module_ctx, expr.id)
        return [class_ref] if class_ref is not None else []

    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name) and expr.func.id == "attach_benchmark_metadata" and expr.args:
            return _resolve_return_expr_to_class_refs(module_ctx, expr.args[0], assignments, local_seen)

        class_ref = _resolve_base_expr(module_ctx, expr.func)
        return [class_ref] if class_ref is not None else []

    return []


def _returned_benchmark_class_refs(module_ctx: _ModuleContext) -> List[_ClassRef]:
    refs: List[_ClassRef] = []
    seen: Set[Tuple[Path, str]] = set()

    for node in module_ctx.tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "get_benchmark":
            continue

        assignments: Dict[str, ast.expr] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    continue
                assignments[stmt.targets[0].id] = stmt.value
                continue
            if isinstance(stmt, ast.AnnAssign):
                if not isinstance(stmt.target, ast.Name) or stmt.value is None:
                    continue
                assignments[stmt.target.id] = stmt.value
                continue
            if not isinstance(stmt, ast.Return) or stmt.value is None:
                continue
            for class_ref in _resolve_return_expr_to_class_refs(module_ctx, stmt.value, assignments):
                key = (class_ref.module_path, class_ref.class_name)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(class_ref)

    return refs


def _method_has_non_base_provider(class_info: _AstClassInfo, method_name: str) -> bool:
    providers = class_info.method_origins.get(method_name, set())
    return any(provider != _BASE_BENCHMARK_QUALNAME for provider in providers)


def _candidate_benchmark_class_refs(module_ctx: _ModuleContext) -> List[_ClassRef]:
    explicit = _returned_benchmark_class_refs(module_ctx)
    if explicit:
        return explicit

    inferred: List[_ClassRef] = []
    for class_name in sorted(module_ctx.classes):
        class_info = _get_ast_class_info(str(module_ctx.module_path), class_name)
        if class_info is None:
            continue
        if _method_has_non_base_provider(class_info, "benchmark_fn"):
            inferred.append(_make_class_ref(module_ctx, class_name))
    return inferred


class BenchmarkContract:
    """Defines the contract that all benchmarks must follow.
    
    The contract includes required methods that must be implemented by all benchmarks.
    
    Verification Enforcement:
    - get_input_signature(): Returns workload description for equivalence checking
    - validate_result(): Returns error message or None for output validation
    - get_verify_output(): Returns output tensor(s) for correctness comparison
    
    See core.benchmark.verification for enforcement phase configuration.
    """
    
    # Required methods that must be implemented
    # These are enforced based on the current EnforcementPhase (DETECT, QUARANTINE, GATE)
    REQUIRED_METHODS: Set[str] = {
        "setup",
        "benchmark_fn",
        "teardown",
    }
    
    # Verification methods - required for correctness validation
    # Missing these methods will quarantine the benchmark in QUARANTINE/GATE phases
    VERIFICATION_REQUIRED_METHODS: Set[str] = {
        "get_input_signature",  # Returns workload description dict
        "validate_result",  # Returns error message or None
        "get_verify_output",  # Returns output tensor(s) for comparison
        "get_output_tolerance",  # Returns ToleranceSpec for custom tolerances
    }
    
    # Optional methods that enhance verification but have sensible defaults
    RECOMMENDED_METHODS: Set[str] = {
        "get_config",
        "get_workload_metadata",  # Returns WorkloadMetadata for invariant checking
        "get_verify_inputs",  # Returns input tensor(s) for aliasing detection
    }
    
    # Methods for verification skip control (use sparingly with justification)
    VERIFICATION_SKIP_METHODS: Set[str] = {
        "skip_input_verification",
        "skip_output_verification",
    }
    
    # Required attributes (if using BaseBenchmark)
    REQUIRED_ATTRIBUTES: Set[str] = {
        "device",  # Set by BaseBenchmark.__init__
    }
    
    @staticmethod
    def validate_benchmark_class_ast(
        class_node: ast.ClassDef,
        class_info: Optional[_AstClassInfo] = None,
    ) -> Tuple[List[str], List[str]]:
        """Validate benchmark class using AST (side-effect free).
        
        Args:
            class_node: AST ClassDef node
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        if class_info is None:
            synthetic_ref = _ClassRef(
                module_path=Path("<ast>"),
                class_name=class_node.name,
                qualified_name=class_node.name,
            )
            own_methods = _method_names(class_node)
            class_info = _AstClassInfo(
                ref=synthetic_ref,
                own_methods=own_methods,
                method_origins={name: {synthetic_ref.qualified_name} for name in own_methods},
                ancestor_names={synthetic_ref.qualified_name},
            )

        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if method_name == "benchmark_fn":
                if not _method_has_non_base_provider(class_info, method_name):
                    errors.append(f"Missing required method: {method_name}()")
                continue
            if method_name not in class_info.method_origins:
                errors.append(f"Missing required method: {method_name}()")

        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            providers = class_info.method_origins.get(method_name, set())
            if method_name == "validate_result":
                if not providers:
                    errors.append(f"Missing verification method: {method_name}()")
                continue
            if not providers or not _method_has_non_base_provider(class_info, method_name):
                errors.append(f"Missing verification method: {method_name}()")

        for item in _function_nodes(class_node):
            method_name = item.name
            if method_name not in BenchmarkContract.REQUIRED_METHODS:
                continue
            args = item.args.args
            if len(args) > 1:
                has_var_args = item.args.vararg is not None
                has_var_kwargs = item.args.kwarg is not None
                if not (has_var_args or has_var_kwargs):
                    errors.append(f"{method_name}() should take no arguments (except self)")

        for item in _function_nodes(class_node):
            if item.name == "benchmark_fn":
                allowed_antipatterns = _literal_string_collection(
                    class_node,
                    "allowed_benchmark_fn_antipatterns",
                )
                warnings.extend(
                    benchmark_fn_sync_warnings_for_class(
                        class_node,
                        allowed_codes=allowed_antipatterns,
                        module_path=(
                            class_info.ref.module_path
                            if class_info.ref.module_path.exists()
                            else None
                        ),
                        class_name=class_info.ref.class_name,
                    )
                )
                warnings.extend(
                    benchmark_fn_antipattern_warnings_for_class(
                        class_node,
                        allowed_codes=allowed_antipatterns,
                        module_path=(
                            class_info.ref.module_path
                            if class_info.ref.module_path.exists()
                            else None
                        ),
                        class_name=class_info.ref.class_name,
                    )
                )
                break

        return errors, warnings
    
    @staticmethod
    def validate_benchmark_class(cls: type) -> Tuple[bool, List[str]]:
        """Validate that a benchmark class follows the contract.
        
        Args:
            cls: Benchmark class to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required methods
        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if not hasattr(cls, method_name):
                errors.append(f"Missing required method: {method_name}()")
                continue
            
            method = getattr(cls, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
                continue
            
            # Check method signature (should take no args except self)
            sig = inspect.signature(method)
            params = list(sig.parameters.values())
            if len(params) > 1:  # More than just 'self'
                # Allow *args and **kwargs
                has_var_args = any(p.kind == p.VAR_POSITIONAL for p in params)
                has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
                if not (has_var_args or has_var_kwargs):
                    errors.append(f"{method_name}() should take no arguments (except self)")
        
        # Check verification-required methods
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(cls, method_name):
                errors.append(f"Missing verification method: {method_name}()")
                continue
            method = getattr(cls, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
                continue
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_benchmark_instance(benchmark: Any, run_setup: bool = False) -> Tuple[bool, List[str]]:
        """Validate that a benchmark instance follows the contract.
        
        Args:
            benchmark: Benchmark instance to validate
            run_setup: If True, actually call setup() to test execution (default: False).
                       WARNING: This will allocate GPU memory and run code - only use
                       when explicitly needed, not in pre-commit hooks or CI linting.
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required methods exist and are callable
        for method_name in BenchmarkContract.REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                errors.append(f"Missing required method: {method_name}()")
                continue
            
            method = getattr(benchmark, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
        
        # Verification-required methods must also be present
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                errors.append(f"Missing verification method: {method_name}()")
                continue
            method = getattr(benchmark, method_name)
            if not callable(method):
                errors.append(f"{method_name} is not callable")
        
        # Only call setup() if explicitly requested (structural validation only by default)
        if run_setup and hasattr(benchmark, "setup") and callable(benchmark.setup):
            try:
                benchmark.setup()
            except Exception as e:
                errors.append(f"setup() raised exception: {type(e).__name__}: {e}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_verification_compliance(benchmark: Any) -> Tuple[bool, List[str], List[str]]:
        """Check if a benchmark is compliant with verification enforcement requirements.
        
        This is separate from validate_benchmark_instance as verification requirements
        are enforced according to the enforcement phase (DETECT, QUARANTINE, GATE).
        
        Args:
            benchmark: Benchmark instance to check
            
        Returns:
            Tuple of (is_compliant, errors, warnings)
            - In DETECT phase: all issues are warnings
            - In QUARANTINE/GATE phase: missing methods are errors
        """
        from core.benchmark.verification import get_enforcement_phase, EnforcementPhase
        
        errors: List[str] = []
        warnings: List[str] = []
        phase = get_enforcement_phase()
        
        # Check verification required methods
        for method_name in BenchmarkContract.VERIFICATION_REQUIRED_METHODS:
            if not hasattr(benchmark, method_name):
                msg = f"Missing verification method: {method_name}()"
                if phase == EnforcementPhase.DETECT:
                    warnings.append(msg)
                else:
                    errors.append(msg)
            else:
                method = getattr(benchmark, method_name)
                if not callable(method):
                    msg = f"Verification method {method_name} is not callable"
                    if phase == EnforcementPhase.DETECT:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        
        # Check for skip flags (always warn - these should have justification)
        for skip_method in BenchmarkContract.VERIFICATION_SKIP_METHODS:
            if hasattr(benchmark, skip_method):
                try:
                    method = getattr(benchmark, skip_method)
                    if callable(method) and method():
                        # Check for justification
                        justification = getattr(benchmark, f"{skip_method}_reason", None)
                        if not justification:
                            warnings.append(
                                f"{skip_method}() returns True without justification attribute"
                            )
                except Exception:
                    pass
        
        # Check for legacy skip flags (deprecated)
        for legacy_flag in ["skip_output_check", "skip_input_check", "skip_verification"]:
            if hasattr(benchmark, legacy_flag):
                value = getattr(benchmark, legacy_flag)
                if value:
                    msg = f"Legacy skip flag '{legacy_flag}' is deprecated; use skip_*_verification() methods"
                    if phase == EnforcementPhase.DETECT:
                        warnings.append(msg)
                    else:
                        errors.append(msg)
        
        return len(errors) == 0, errors, warnings


def get_benchmark_class_from_module(module_path: Path) -> Optional[type]:
    """Extract benchmark class from a Python module file.
    
    Args:
        module_path: Path to Python module file
        
    Returns:
        Benchmark class if found, None otherwise
    """
    try:
        # Read and parse the file
        source = module_path.read_text()
        tree = ast.parse(source, filename=str(module_path))
        
        # Find classes that have benchmark_fn method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class has benchmark_fn method
                has_benchmark_fn = any(
                    isinstance(item, ast.FunctionDef) and item.name == "benchmark_fn"
                    for item in node.body
                )
                if has_benchmark_fn:
                    # Try to import and return the class
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return getattr(module, node.name, None)
        
        return None
    except Exception:
        return None


def check_benchmark_file_ast(file_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Check benchmark file using AST parsing (side-effect free).
    
    Args:
        file_path: Path to benchmark Python file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"], []
    
    if not file_path.suffix == ".py":
        return False, [f"Not a Python file: {file_path}"], []
    
    try:
        module_ctx = _load_module_context(str(file_path.resolve()))
        if module_ctx is None:
            errors.append(f"Failed to parse file: unable to load module context for {file_path}")
            return False, errors, warnings

        tree = module_ctx.tree

        has_get_benchmark = False
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "get_benchmark":
                continue
            has_get_benchmark = True
            if len(node.args.args) > 0 or node.args.vararg is not None or node.args.kwarg is not None:
                errors.append("get_benchmark() should take no arguments")

        def _is_name_main_expr(expr: ast.AST) -> bool:
            if isinstance(expr, ast.Name):
                return expr.id == "__name__"
            if not isinstance(expr, ast.Call):
                return False
            if not isinstance(expr.func, ast.Attribute) or expr.func.attr != "get":
                return False
            if len(expr.args) != 1 or expr.keywords:
                return False
            if not isinstance(expr.args[0], ast.Constant) or expr.args[0].value != "__name__":
                return False
            globals_call = expr.func.value
            return (
                isinstance(globals_call, ast.Call)
                and isinstance(globals_call.func, ast.Name)
                and globals_call.func.id == "globals"
                and not globals_call.args
                and not globals_call.keywords
            )

        def _is_main_guard_test(test: ast.AST) -> bool:
            if not isinstance(test, ast.Compare):
                return False
            if len(test.ops) != 1 or len(test.comparators) != 1:
                return False
            left = test.left
            right = test.comparators[0]
            if not isinstance(right, ast.Constant) or right.value != "__main__":
                return False
            return _is_name_main_expr(left) and isinstance(test.ops[0], (ast.Eq, ast.NotEq))

        def _is_main_guard(stmt: ast.stmt) -> bool:
            return isinstance(stmt, ast.If) and _is_main_guard_test(stmt.test)

        def _invokes_main_guard_helper(module_tree: ast.Module) -> bool:
            helper_names: Set[str] = set()
            for stmt in module_tree.body:
                if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any(
                    isinstance(node, ast.If) and _is_main_guard_test(node.test)
                    for node in ast.walk(stmt)
                ):
                    helper_names.add(stmt.name)
            if not helper_names:
                return False
            for stmt in module_tree.body:
                if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                    continue
                if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in helper_names:
                    return True
            return False

        benchmark_class_refs = _candidate_benchmark_class_refs(module_ctx)
        for class_ref in benchmark_class_refs:
            class_module_ctx = _load_module_context(str(class_ref.module_path))
            if class_module_ctx is None:
                errors.append(
                    f"Failed to parse file: unable to load benchmark class module for {class_ref.qualified_name}"
                )
                continue
            class_node = class_module_ctx.classes.get(class_ref.class_name)
            if class_node is None:
                errors.append(f"Failed to parse file: benchmark class not found: {class_ref.qualified_name}")
                continue
            class_info = _get_ast_class_info(str(class_ref.module_path), class_ref.class_name)
            class_errors, class_warnings = BenchmarkContract.validate_benchmark_class_ast(
                class_node,
                class_info=class_info,
            )
            errors.extend(class_errors)
            warnings.extend(class_warnings)

        if not has_get_benchmark and not benchmark_class_refs:
            errors.append("No get_benchmark() function or benchmark class found")
        elif (
            any(_is_main_guard(node) for node in tree.body) or _invokes_main_guard_helper(tree)
        ) and (
            file_path.name.startswith(("baseline_", "optimized_"))
            or file_path.resolve() not in _registered_standalone_paths()
        ):
            errors.append(
                "Benchmark modules must not define __main__ blocks; run them via compare.py or cli.aisp bench run"
            )

    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
    except Exception as e:
        errors.append(f"Failed to parse file: {type(e).__name__}: {e}")
    
    return len(errors) == 0, errors, warnings


def check_benchmark_file(file_path: Path, run_setup: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Check if a benchmark file follows the contract.
    
    By default, uses AST parsing for side-effect free validation.
    Only imports and instantiates if run_setup=True.
    
    Args:
        file_path: Path to benchmark Python file
        run_setup: If True, actually import and instantiate benchmark (default: False).
                   WARNING: This will execute module-level code and constructors,
                   which may require CUDA. Use only when explicitly needed.
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    # By default, use AST parsing (side-effect free)
    if not run_setup:
        return check_benchmark_file_ast(file_path)
    
    # If run_setup=True, do full validation with instantiation
    errors = []
    warnings = []
    
    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"], []
    
    if not file_path.suffix == ".py":
        return False, [f"Not a Python file: {file_path}"], []
    
    # Try to find get_benchmark function or benchmark class
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for get_benchmark function
            if hasattr(module, "get_benchmark"):
                get_benchmark = getattr(module, "get_benchmark")
                if callable(get_benchmark):
                    try:
                        benchmark = get_benchmark()
                        is_valid, issues = BenchmarkContract.validate_benchmark_instance(benchmark, run_setup=run_setup)
                        if not is_valid:
                            errors.extend(issues)
                        else:
                            # Separate warnings from errors
                            for issue in issues:
                                if issue.startswith("Missing recommended") or "should have" in issue:
                                    warnings.append(issue)
                                else:
                                    errors.append(issue)
                    except Exception as e:
                        errors.append(f"get_benchmark() raised exception: {type(e).__name__}: {e}")
                else:
                    errors.append("get_benchmark is not callable")
            else:
                # Try to find benchmark class
                benchmark_class = get_benchmark_class_from_module(file_path)
                if benchmark_class:
                    is_valid, issues = BenchmarkContract.validate_benchmark_class(benchmark_class)
                    if not is_valid:
                        errors.extend(issues)
                    else:
                        # Separate warnings from errors
                        for issue in issues:
                            if issue.startswith("Missing recommended") or "should have" in issue:
                                warnings.append(issue)
                            else:
                                errors.append(issue)
                else:
                    errors.append("No get_benchmark() function or benchmark class found")
    except Exception as e:
        errors.append(f"Failed to load module: {type(e).__name__}: {e}")
    
    return len(errors) == 0, errors, warnings
