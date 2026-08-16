#!/usr/bin/env python3
"""Audit verification compliance by actually instantiating benchmarks.

This script properly detects inherited methods (unlike grep-based approaches)
by importing and inspecting each benchmark class.

Usage:
    python -m core.scripts.audit_verification_compliance [--chapter ch10] [--lab decode_optimization]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.abc
import importlib.util
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import torch

from core.benchmark.verification import InputSignature

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _capture_python_output() -> List[str]:
    """Capture Python stream writes without replacing process file descriptors.

    Native writes and child-process output remain attached to the caller's file
    descriptors. This avoids process-wide descriptor mutation during an audit.
    """
    captured: List[str] = []
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield captured
    combined = stdout_buffer.getvalue() + stderr_buffer.getvalue()
    captured.extend(line.strip() for line in combined.splitlines() if line.strip())


def _module_name_for_path(filepath: Path) -> str:
    """Resolve a benchmark file to its repo package module path."""
    resolved = filepath.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.stem
    return ".".join(relative.with_suffix("").parts)


class _SiblingImportFinder(importlib.abc.MetaPathFinder):
    """Resolve sibling benchmark imports without mutating the global sys.path."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
        if path is not None or "." in fullname:
            return None

        module_path = self.root / f"{fullname}.py"
        if module_path.is_file():
            return importlib.util.spec_from_file_location(fullname, module_path)

        package_init = self.root / fullname / "__init__.py"
        if package_init.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                package_init,
                submodule_search_locations=[str(package_init.parent)],
            )

        return None


@contextlib.contextmanager
def _with_sibling_import_finder(path: Path):
    finder = _SiblingImportFinder(path)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        for index, candidate in enumerate(sys.meta_path):
            if candidate is finder:
                sys.meta_path.pop(index)
                break


def _should_retry_with_multigpu_floor(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "requires >=2 gpus",
        "need >=2 gpus",
        "requires multiple gpus",
        "distributed benchmark requires multiple gpus",
    )
    return any(marker in message for marker in markers)


@contextlib.contextmanager
def _audit_gpu_floor(min_gpus: int = 2):
    """Expose a device-count floor for load-only audit retries on single-GPU hosts."""
    if not torch.cuda.is_available():
        yield
        return
    available = torch.cuda.device_count()
    if available >= min_gpus:
        yield
        return
    with mock.patch("torch.cuda.device_count", return_value=min_gpus):
        yield


def _scan_source_compliance(filepath: Path) -> Dict[str, bool]:
    """Static checks that keep benchmark_fn() hot path clean."""
    flags = {
        "no_seed_setting_in_benchmark_fn": True,
        "no_payload_set_in_benchmark_fn": True,
        # Best-practice checks.
        # NOTE: This is a *lint-style* signal; it should not run benchmark code.
        "backend_toggles_present": False,
        "no_backend_toggles": True,
        "determinism_toggles_present": False,
        "no_determinism_enable_without_justification": True,
    }

    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except Exception:
        # If the file cannot be parsed, treat as non-compliant for source checks.
        flags["no_seed_setting_in_benchmark_fn"] = False
        flags["no_payload_set_in_benchmark_fn"] = False
        flags["no_determinism_enable_without_justification"] = False
        return flags

    def _is_seed_call(call: ast.Call) -> bool:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return False
        # torch.manual_seed(...)
        if func.attr == "manual_seed" and isinstance(func.value, ast.Name) and func.value.id == "torch":
            return True
        # torch.cuda.manual_seed_all(...)
        if func.attr == "manual_seed_all" and isinstance(func.value, ast.Attribute):
            base = func.value
            if base.attr == "cuda" and isinstance(base.value, ast.Name) and base.value.id == "torch":
                return True
        # random.seed(...)
        if func.attr == "seed" and isinstance(func.value, ast.Name) and func.value.id == "random":
            return True
        # np.random.seed(...) / numpy.random.seed(...)
        if func.attr == "seed" and isinstance(func.value, ast.Attribute):
            base = func.value
            if base.attr == "random" and isinstance(base.value, ast.Name) and base.value.id in {"np", "numpy"}:
                return True
        return False

    def _is_payload_set_call(call: ast.Call) -> bool:
        func = call.func
        return isinstance(func, ast.Attribute) and func.attr == "_set_verification_payload"

    def _dotted_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        else:
            return None
        return ".".join(reversed(parts))

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: List[str] = []
            self._determinism_enable_found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            fn = self._stack[-1] if self._stack else ""
            if fn == "benchmark_fn":
                if _is_seed_call(node):
                    flags["no_seed_setting_in_benchmark_fn"] = False
                if _is_payload_set_call(node):
                    flags["no_payload_set_in_benchmark_fn"] = False
            call_name = _dotted_name(node.func)
            if call_name in {"torch.use_deterministic_algorithms", "torch.set_deterministic_debug_mode"}:
                flags["determinism_toggles_present"] = True
                if node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value is False:
                        pass
                    else:
                        self._determinism_enable_found = True
                else:
                    self._determinism_enable_found = True
            if call_name in {
                "torch.set_float32_matmul_precision",
                "enable_tf32",
                "configure_tf32",
                "restore_tf32",
                "tf32_override",
            }:
                flags["backend_toggles_present"] = True
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                target_name = _dotted_name(target)
                if target_name == "torch.backends.cudnn.deterministic":
                    flags["determinism_toggles_present"] = True
                    value = node.value
                    if isinstance(value, ast.Constant) and value.value is False:
                        pass
                    else:
                        self._determinism_enable_found = True
                if target_name in {
                    "torch.backends.cudnn.benchmark",
                    "torch.backends.cuda.matmul.allow_tf32",
                    "torch.backends.cudnn.allow_tf32",
                    "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
                    "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
                }:
                    flags["backend_toggles_present"] = True
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    flags["no_backend_toggles"] = not flags["backend_toggles_present"]
    allowlist = "aisp: allow_determinism" in source
    if visitor._determinism_enable_found and not allowlist:
        flags["no_determinism_enable_without_justification"] = False
    return flags


def _declares_get_benchmark(filepath: Path) -> bool:
    """Return True when the source declares a top-level get_benchmark() factory."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except Exception:
        return True
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "get_benchmark"
        for node in tree.body
    )


def load_benchmark_class(filepath: Path) -> Optional[Tuple[Any, str, List[str]]]:
    """Load benchmark class from a file.
    
    Returns:
        Tuple of (benchmark_instance, class_name, captured_output) or None if failed
    """
    module_name = _module_name_for_path(filepath)
    previous_module = sys.modules.get(module_name)
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _with_sibling_import_finder(filepath.parent):
            with _capture_python_output() as captured_output:
                spec.loader.exec_module(module)

                # Look for get_benchmark factory function
                if hasattr(module, "get_benchmark"):
                    try:
                        benchmark = module.get_benchmark()
                    except RuntimeError as exc:
                        if not _should_retry_with_multigpu_floor(exc):
                            raise
                        with _audit_gpu_floor():
                            benchmark = module.get_benchmark()
                    return (benchmark, type(benchmark).__name__, captured_output)
        
        return None
    except Exception as e:
        # Silently skip files that can't be loaded
        return None
    finally:
        # Clean up
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


def check_compliance(benchmark: Any) -> Dict[str, bool]:
    """Check if a benchmark has required verification methods.
    
    Returns dict with compliance status for each required method.
    """
    compliance = {
        "get_verify_output": False,
        "get_input_signature": False,
        "get_verify_inputs": False,
        "get_output_tolerance": False,
        "validate_result": False,
        # Jitter exemptions are NOT allowed; any exemption is a failure.
        "jitter_exemption_reason": False,
        "register_workload_metadata_called": False,
        # Hot-path hygiene (static analysis) populated by caller.
        "no_seed_setting_in_benchmark_fn": False,
        "no_payload_set_in_benchmark_fn": False,
        "determinism_toggles_present": False,
        "no_determinism_enable_without_justification": False,
    }
    
    # Check methods (including inherited)
    if hasattr(benchmark, "get_verify_output"):
        # Check if it's a real implementation, not just NotImplementedError
        try:
            method = getattr(benchmark, "get_verify_output")
            # Check if the method is defined in the class (not just BaseBenchmark)
            if hasattr(method, "__func__"):
                defining_class = method.__func__.__qualname__.split(".")[0]
                if defining_class != "BaseBenchmark":
                    compliance["get_verify_output"] = True
            # Also check CudaBinaryBenchmark
            if "CudaBinaryBenchmark" in str(type(benchmark).__mro__):
                compliance["get_verify_output"] = True
        except Exception:
            pass
    
    if hasattr(benchmark, "get_input_signature"):
        try:
            sig = benchmark.get_input_signature()
            if isinstance(sig, InputSignature):
                compliance["get_input_signature"] = not sig.validate(strict=True)
            elif isinstance(sig, dict):
                shapes = sig.get("shapes", {})
                dtypes = sig.get("dtypes", {})
                batch = sig.get("batch_size", None)
                params = sig.get("parameter_count", None)
                compliance["get_input_signature"] = bool(
                    shapes and dtypes and batch is not None and params is not None
                )
        except RuntimeError:
            # Accept RuntimeError (e.g., setup not called) as an implemented method
            compliance["get_input_signature"] = True
        except Exception:
            pass

    if hasattr(benchmark, "get_verify_inputs"):
        try:
            inp = benchmark.get_verify_inputs()
            if isinstance(inp, torch.Tensor):
                compliance["get_verify_inputs"] = True
            elif isinstance(inp, dict):
                compliance["get_verify_inputs"] = any(isinstance(v, torch.Tensor) for v in inp.values())
        except RuntimeError:
            compliance["get_verify_inputs"] = True
        except Exception:
            pass
    
    if hasattr(benchmark, "get_output_tolerance"):
        try:
            tol = benchmark.get_output_tolerance()
            compliance["get_output_tolerance"] = tol is not None
        except RuntimeError:
            # Payload-backed benchmarks raise until capture_verification_payload() runs.
            compliance["get_output_tolerance"] = True
        except Exception:
            pass
    
    if hasattr(benchmark, "validate_result"):
        compliance["validate_result"] = True
    
    # Jitter exemptions are no longer permitted.
    if hasattr(benchmark, "jitter_exemption_reason"):
        reason = getattr(benchmark, "jitter_exemption_reason")
        compliance["jitter_exemption_reason"] = not bool(reason)
    elif hasattr(benchmark, "non_jitterable_reason"):
        reason = getattr(benchmark, "non_jitterable_reason")
        compliance["jitter_exemption_reason"] = not bool(reason)
    else:
        compliance["jitter_exemption_reason"] = True
    
    # Check if workload metadata was registered
    if hasattr(benchmark, "_workload_registered") and benchmark._workload_registered:
        compliance["register_workload_metadata_called"] = True
    elif hasattr(benchmark, "_workload") and benchmark._workload is not None:
        compliance["register_workload_metadata_called"] = True
    
    return compliance


def audit_directory(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Audit all benchmark files in a directory.
    
    Returns:
        Dict mapping filepath to compliance info
    """
    results = {}
    
    skip_parts = {
        "__pycache__",
        "llm_patches",
        "llm_patches_test",
        ".venv",
        "venv",
        "site-packages",
        "dist-packages",
        "node_modules",
    }
    for filepath in sorted(directory.rglob("*.py")):
        if any(part in skip_parts for part in filepath.parts):
            continue
        if not (filepath.name.startswith("baseline_") or filepath.name.startswith("optimized_")):
            continue
        if not _declares_get_benchmark(filepath):
            continue

        source_flags = _scan_source_compliance(filepath)
        
        result = load_benchmark_class(filepath)
        if result is None:
            results[str(filepath)] = {
                "status": "error",
                "error": "Could not load benchmark",
                "class_name": None,
                "compliance": None,
            }
            continue
        
        benchmark, class_name, python_load_output = result
        compliance = check_compliance(benchmark)
        compliance.update(source_flags)
        
        # Determine overall status
        critical_methods = [
            "get_verify_output",
            "get_input_signature",
            "get_output_tolerance",
            "validate_result",
            "jitter_exemption_reason",
            "no_seed_setting_in_benchmark_fn",
            "no_payload_set_in_benchmark_fn",
            "no_backend_toggles",
            "no_determinism_enable_without_justification",
        ]
        is_compliant = all(compliance.get(m, False) for m in critical_methods)

        warnings: List[str] = []
        if python_load_output:
            warnings.append("Benchmark emitted Python output during audit load.")
        if compliance.get("backend_toggles_present", False):
            warnings.append("Backend policy toggles detected in benchmark file.")
        if compliance.get("determinism_toggles_present", False):
            warnings.append("Determinism toggles detected in benchmark file.")
        if not compliance.get("no_determinism_enable_without_justification", True):
            warnings.append("Determinism enabled without allowlist comment (# aisp: allow_determinism ...).")

        results[str(filepath)] = {
            "status": "compliant" if is_compliant else "needs_work",
            "class_name": class_name,
            "compliance": compliance,
            "warnings": warnings,
            "python_load_output": python_load_output,
        }
    
    return results


def print_summary(results: Dict[str, Dict[str, Any]], title: str) -> Tuple[int, int, int]:
    """Print summary of audit results.
    
    Returns:
        Tuple of (compliant_count, needs_work_count, error_count)
    """
    compliant = [f for f, r in results.items() if r["status"] == "compliant"]
    needs_work = [f for f, r in results.items() if r["status"] == "needs_work"]
    errors = [f for f, r in results.items() if r["status"] == "error"]
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"✅ Compliant: {len(compliant)}")
    print(f"⚠️  Needs work: {len(needs_work)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"Total: {len(results)}")
    
    if needs_work:
        print(f"\n--- Files needing work ---")
        for filepath in needs_work[:10]:  # Show first 10
            r = results[filepath]
            missing = [k for k, v in r["compliance"].items() if not v]
            print(f"  {Path(filepath).name}: missing {missing}")
        if len(needs_work) > 10:
            print(f"  ... and {len(needs_work) - 10} more")

    if errors:
        print(f"\n--- Errors (failed to load) ---")
        for filepath in errors[:10]:  # Show first 10
            r = results[filepath]
            err = r.get("error") or "Unknown error"
            print(f"  {Path(filepath).name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    backend_toggles = [
        f for f, r in results.items() if (r.get("compliance") or {}).get("backend_toggles_present", False)
    ]
    if backend_toggles:
        print(f"\n--- Backend toggles (error) ---")
        print(f"Files with backend-related toggles: {len(backend_toggles)}")
        for filepath in backend_toggles[:10]:
            r = results[filepath]
            warnings = r.get("warnings") or []
            details = f" ({'; '.join(warnings)})" if warnings else ""
            print(f"  {Path(filepath).name}{details}")
        if len(backend_toggles) > 10:
            print(f"  ... and {len(backend_toggles) - 10} more")
    
    return len(compliant), len(needs_work), len(errors)


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark verification compliance")
    parser.add_argument("--chapter", type=str, help="Specific chapter to audit (e.g., ch10)")
    parser.add_argument("--lab", type=str, help="Specific lab to audit (e.g., decode_optimization)")
    parser.add_argument("--all", action="store_true", help="Audit all chapters and labs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    code_dir = REPO_ROOT
    
    total_compliant = 0
    total_needs_work = 0
    total_errors = 0
    
    # Audit chapters
    if args.chapter:
        chapters = [args.chapter]
    elif args.all or not args.lab:
        chapters = [f"ch{i:02d}" for i in range(1, 21)]
    else:
        chapters = []
    
    for chapter in chapters:
        chapter_dir = code_dir / chapter
        if not chapter_dir.exists():
            continue
        
        results = audit_directory(chapter_dir)
        if results:
            c, n, e = print_summary(results, f"{chapter.upper()}")
            total_compliant += c
            total_needs_work += n
            total_errors += e
    
    # Audit labs
    if args.lab:
        labs = [args.lab]
    elif args.all or not args.chapter:
        labs_dir = code_dir / "labs"
        if labs_dir.exists():
            labs = [d.name for d in labs_dir.iterdir() if d.is_dir()]
        else:
            labs = []
    else:
        labs = []
    
    for lab in sorted(labs):
        lab_dir = code_dir / "labs" / lab
        if not lab_dir.exists():
            continue
        
        results = audit_directory(lab_dir)
        if results:
            c, n, e = print_summary(results, f"LAB: {lab}")
            total_compliant += c
            total_needs_work += n
            total_errors += e
    
    # Grand total
    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"{'='*60}")
    print(f"✅ Compliant: {total_compliant}")
    print(f"⚠️  Needs work: {total_needs_work}")
    print(f"❌ Errors: {total_errors}")
    print(f"Total: {total_compliant + total_needs_work + total_errors}")
    
    coverage = (total_compliant / max(1, total_compliant + total_needs_work)) * 100
    print(f"\n📊 Coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
