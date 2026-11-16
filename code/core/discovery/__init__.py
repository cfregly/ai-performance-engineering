"""Benchmark discovery utilities.

Provides functions to discover benchmarks across chapters and CUDA benchmarks.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Repository root (…/code)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
# Support both legacy ch01-9 and zero-padded ch01-09
CHAPTER_ALIAS_MAP: Dict[str, str] = {f"ch{i}": f"ch{i:02d}" for i in range(1, 10)}


def get_bench_roots(repo_root: Optional[Path] = None, bench_root: Optional[Path] = None) -> List[Path]:
    """
    Resolve benchmark search roots.

    Flags/explicit args control the root; environment variables are intentionally
    ignored to avoid hidden implicit overrides.
    """
    if bench_root:
        return [Path(bench_root).expanduser().resolve()]
    root = Path(repo_root or DEFAULT_REPO_ROOT).expanduser().resolve()
    return [root]


def _has_get_benchmark(file_path: Path) -> bool:
    """Quick check if a Python file has get_benchmark() function.
    
    Does a simple text search without importing the module.
    """
    import ast

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Failed to read benchmark file {file_path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        raise SyntaxError(f"Syntax error in benchmark file {file_path}: {exc}") from exc

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_benchmark":
            return True
    return False


def _has_benchmark_decorator(file_path: Path) -> bool:
    """Return True when a Python file uses benchmark registration decorators."""
    import ast

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Failed to read benchmark file {file_path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        raise SyntaxError(f"Syntax error in benchmark file {file_path}: {exc}") from exc

    decorator_names = {"export_benchmark", "register_benchmark"}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in decorator_names:
                return True
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id in decorator_names:
                    return True
    return False


def is_benchmark_entrypoint_file(file_path: Path) -> bool:
    """Return True when a Python file exposes a benchmark entrypoint."""
    if file_path.suffix != ".py":
        return False
    return _has_get_benchmark(file_path) or _has_benchmark_decorator(file_path)


def is_generated_benchmark_copy(file_path: Path) -> bool:
    """Return True for generated exploration copies that should not be auto-discovered."""
    stem = file_path.stem
    for prefix in ("baseline_", "optimized_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return bool(re.search(r"_mcp_copy(?:_\d+)?$", stem))


def should_ignore_benchmark_candidate(file_path: Path) -> bool:
    """Return True for baseline_/optimized_ files reserved for non-harness helper flows."""
    stem = file_path.stem
    return stem in {
        "baseline_submission",
        "optimized_submission",
        "reference_submission",
    } or stem.startswith("optimized_submission_") or is_generated_benchmark_copy(file_path)


SUPPRESSED_VARIANT_ALIAS_TARGETS = {
    "add_cuda_parallel",
}


def is_cuda_binary_benchmark_file(file_path: Path) -> bool:
    """Return True if a Python benchmark subclasses CudaBinaryBenchmark.

    This is used to treat CUDA binaries as Python-wrapped benchmarks when
    enforcing wrapper-only execution (no direct .cu discovery).
    """
    if file_path.suffix != ".py":
        return False

    import ast

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Failed to read benchmark file {file_path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        raise SyntaxError(f"Syntax error in benchmark file {file_path}: {exc}") from exc

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "CudaBinaryBenchmark":
                return True
            if isinstance(base, ast.Attribute) and base.attr == "CudaBinaryBenchmark":
                return True
    return False


def validate_benchmark_file(file_path: Path, warn: bool = True) -> bool:
    """Validate that a benchmark file has get_benchmark().
    
    Args:
        file_path: Path to benchmark file
        warn: If True, emit a warning for missing get_benchmark()
        
    Returns:
        True if file has get_benchmark(), False otherwise
    """
    if not file_path.suffix == ".py":
        return True  # Skip non-Python files
    
    has_fn = is_benchmark_entrypoint_file(file_path)
    
    if not has_fn and warn:
        warnings.warn(
            f"Benchmark file '{file_path.name}' is missing get_benchmark() function. "
            f"Add: def get_benchmark() -> BaseBenchmark: return YourClass()",
            UserWarning,
            stacklevel=2
        )
    
    return has_fn


def discover_benchmark_entrypoints(
    repo_root: Path,
    bench_roots: Optional[List[Path]] = None,
    *,
    include_unpaired: bool = False,
) -> List[Path]:
    """Discover benchmark entrypoint files for linting and audit workflows."""
    roots = bench_roots or get_bench_roots(repo_root=repo_root)
    benchmark_files: List[Path] = []
    seen: Set[Path] = set()

    for chapter_dir in discover_all_chapters(repo_root, bench_roots=roots):
        for baseline, optimized_list, _ in discover_benchmarks(
            chapter_dir,
            warn_missing=False,
        ):
            for candidate in [baseline, *optimized_list]:
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                benchmark_files.append(candidate)

        if not include_unpaired:
            continue

        for candidate in sorted(chapter_dir.rglob("*.py")):
            if "__pycache__" in candidate.parts or candidate.name.startswith("test_"):
                continue
            if not is_benchmark_entrypoint_file(candidate):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            benchmark_files.append(candidate)

    return benchmark_files

# Shorthand aliases for common labs (optional convenience)
LAB_ALIASES: Dict[str, str] = {
    "capstone": "labs/fullstack_cluster",
    "moe_journey": "labs/moe_optimization_journey",
}

def _labs_root(repo_root: Path, bench_root: Optional[Path] = None) -> Path:
    root = bench_root or get_bench_roots(repo_root=repo_root, bench_root=bench_root)[0]
    return root / "labs"


def _lab_dirs(repo_root: Path, bench_root: Optional[Path] = None) -> Iterable[Path]:
    """Auto-discover all lab directories that contain benchmark files."""
    labs_root = _labs_root(repo_root, bench_root=bench_root)
    if not labs_root.is_dir():
        return []
    
    lab_dirs = []
    for p in labs_root.iterdir():
        if not p.is_dir() or p.name.startswith(('_', '.')):
            continue
        # Check if directory has any baseline_*.py files (benchmark convention)
        has_benchmarks = any(
            path for path in p.glob("baseline_*.py")
            if not is_generated_benchmark_copy(path)
        ) or any(p.glob("level*.py"))
        if has_benchmarks:
            lab_dirs.append(p)
    
    return sorted(lab_dirs, key=lambda p: p.name)


def _iter_benchmark_dirs(bench_root: Path) -> Iterable[Path]:
    """Yield directories that contain baseline_* benchmarks."""
    ignore_dirs = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        ".torch_inductor",
        ".torch_extensions",
        ".next",
        ".turbo",
        "build",
        "dist",
        "out",
        "artifacts",
        "book",
        "docs",
        "eval_datasets",
        "examples",
        "mcp",
        "monitoring",
        "scripts",
        "tools",
        "profiling_results",
        "hta_output",
        "gpt-oss-20b",
        "mixtral-8x7b",
        "phi-3.5-moe",
        "kernels",
        "third_party",
        "vendor",
    }
    for current, dirnames, filenames in os.walk(bench_root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in ignore_dirs
            and not d.startswith(".")
            and not d.startswith("artifacts")
            and not d.startswith("pymp-")
        ]
        if any(
            fname.startswith("baseline_") and fname.endswith(".py")
            and not is_generated_benchmark_copy(Path(current) / fname)
            for fname in filenames
        ):
            yield Path(current)


def chapter_slug(chapter_dir: Path, repo_root: Path, bench_root: Optional[Path] = None) -> str:
    """Return a consistent chapter identifier relative to the benchmark root."""
    roots = [bench_root] if bench_root else get_bench_roots(repo_root=repo_root)
    for root in roots:
        try:
            return chapter_dir.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            continue
    try:
        return chapter_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return chapter_dir.name


def _parse_examples(examples: str) -> List[str]:
    """Split example payloads like 'a,b c' into distinct names."""
    tokens = examples.replace(",", " ").split()
    return [tok.strip() for tok in tokens if tok.strip()]


def normalize_chapter_token(
    token: str,
    repo_root: Optional[Path] = None,
    bench_root: Optional[Path] = None,
) -> str:
    """Normalize chapter token (CLI arg or alias) to a relative path slug.

    Examples:
      - 'ch10' -> 'ch10'
      - '10' -> 'ch10'
      - 'labs/blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'lab_blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'blackwell_matmul' -> 'labs/blackwell_matmul'
      - 'capstone2' -> 'labs/blackwell_matmul'
    """
    chapter = token.strip().lower()
    if not chapter:
        raise ValueError("Chapter token cannot be empty.")

    def _normalize_ch_slug(slug: str) -> str:
        # Accept bare numbers, legacy ch01-9, and zero-pad to ch01-ch09
        if slug.isdigit():
            return f"ch{int(slug):02d}"
        if slug in CHAPTER_ALIAS_MAP:
            return CHAPTER_ALIAS_MAP[slug]
        if slug.startswith("ch") and slug[2:].isdigit():
            num = int(slug[2:])
            if 1 <= num <= 9:
                return f"ch{num:02d}"
        return slug
    
    roots = [bench_root] if bench_root else get_bench_roots(repo_root=repo_root or DEFAULT_REPO_ROOT)
    primary_root = roots[0]

    # Absolute path or explicit relative path
    candidate_path = Path(chapter).expanduser()
    if candidate_path.is_absolute() and candidate_path.is_dir():
        try:
            return candidate_path.resolve().relative_to(primary_root.resolve()).as_posix()
        except Exception:
            return str(candidate_path.resolve())
    chapter = _normalize_ch_slug(chapter)
    candidate_relative = (primary_root / chapter)
    if candidate_relative.is_dir():
        return Path(chapter).as_posix()

    chapter = chapter.replace("labs.", "labs/").replace("\\", "/")

    if chapter in LAB_ALIASES:
        return LAB_ALIASES[chapter]

    # Get repo root for auto-discovery
    if repo_root is None:
        repo_root = DEFAULT_REPO_ROOT
    
    # Auto-discover valid lab names from filesystem
    discovered_labs = {p.name for p in _lab_dirs(repo_root, bench_root=primary_root)}

    if chapter.startswith("lab_"):
        trimmed = chapter[len("lab_") :]
        if trimmed in discovered_labs:
            return f"labs/{trimmed}"

    if chapter.startswith("labs/"):
        _, _, suffix = chapter.partition("/")
        if suffix in discovered_labs:
            return chapter

    if chapter in discovered_labs:
        return f"labs/{chapter}"

    if chapter.startswith("ch") and chapter[2:].isdigit():
        return chapter

    raise ValueError(
        f"Invalid chapter identifier '{token}'. Expected formats like "
        "'ch03', 'labs/blackwell_matmul', or 'blackwell_matmul'."
    )


def discover_benchmarks(
    chapter_dir: Path,
    validate: bool = True,
    warn_missing: bool = True,
) -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark modules by looking for baseline_*.py files with matching optimized_*.py.
    
    Note: This function only discovers Python benchmarks (.py files).
    CUDA benchmarks (.cu files) should be discovered separately via discover_cuda_benchmarks()
    in core.harness.run_benchmarks to avoid trying to load .cu files as Python modules.
    
    Args:
        chapter_dir: Path to chapter directory (e.g., Path('ch16'))
        validate: Deprecated (kept for API compatibility). Discovery always excludes
            Python files missing get_benchmark(), because they cannot be loaded by
            the harness or the verification tooling.
        warn_missing: If True, emit warnings for files missing get_benchmark().
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
        Example: (Path('ch16/baseline_moe_dense.py'), [Path('ch16/optimized_moe_sparse.py')], 'moe')
    """
    pairs = []
    # Only discover Python files - CUDA benchmarks are handled by discover_cuda_benchmarks()
    baseline_files = sorted(
        path
        for path in chapter_dir.glob("baseline_*.py")
        if not should_ignore_benchmark_candidate(path)
    )

    example_names = {
        baseline_file.stem.replace("baseline_", "")
        for baseline_file in baseline_files
    }
    
    for baseline_file in baseline_files:
        # Always exclude Python files missing get_benchmark(); warnings are optional.
        if not validate_benchmark_file(baseline_file, warn=warn_missing):
            continue
        
        # Extract example name using the entire suffix after "baseline_"
        # This preserves variants like "moe_dense" instead of collapsing everything to "moe".
        example_name = baseline_file.stem.replace("baseline_", "")
        optimized_files: List[Path] = []
        variant_aliases: List[Tuple[str, Path]] = []
        ext = baseline_file.suffix or ".py"
        
        # Pattern 1: optimized_{name}_*.{ext} (e.g., optimized_moe_sparse.py)
        pattern1 = chapter_dir / f"optimized_{example_name}_*{ext}"
        for opt_path in pattern1.parent.glob(pattern1.name):
            if should_ignore_benchmark_candidate(opt_path):
                continue
            # Always exclude Python files missing get_benchmark(); warnings are optional.
            if not validate_benchmark_file(opt_path, warn=warn_missing):
                continue

            optimized_name = opt_path.stem.replace("optimized_", "", 1)
            if any(
                other != example_name
                and len(other) > len(example_name)
                and (optimized_name == other or optimized_name.startswith(f"{other}_"))
                for other in example_names
            ):
                # Prefer the most specific baseline match (e.g., baseline_ddp_multigpu_compression_* over baseline_ddp_multigpu)
                continue
            
            suffix = opt_path.stem.replace(f"optimized_{example_name}_", "", 1)
            candidate_name = f"{example_name}_{suffix}"
            if candidate_name in example_names:
                continue
            optimized_files.append(opt_path)
            if candidate_name not in SUPPRESSED_VARIANT_ALIAS_TARGETS:
                variant_aliases.append((candidate_name, opt_path))
        
        # Pattern 2: optimized_{name}.{ext} (e.g., optimized_moe.py / optimized_moe.cu)
        pattern2 = chapter_dir / f"optimized_{example_name}{ext}"
        if pattern2.exists():
            if should_ignore_benchmark_candidate(pattern2):
                continue
            if validate_benchmark_file(pattern2, warn=warn_missing):
                optimized_files.append(pattern2)

        if optimized_files:
            pairs.append((baseline_file, optimized_files, example_name))
            for variant_name, opt_path in variant_aliases:
                pairs.append((baseline_file, [opt_path], variant_name))
    
    return pairs



def discover_cuda_benchmarks(repo_root: Path) -> List[Path]:
    """Discover CUDA benchmark files (files with .cu extension or in cuda/ directories)."""
    cuda_benchmarks: List[Path] = []

    def _collect_from_dir(root: Path) -> None:
        cuda_benchmarks.extend(
            path for path in root.glob("*.cu") if not is_generated_benchmark_copy(path)
        )
        cuda_subdir = root / "cuda"
        if cuda_subdir.exists() and cuda_subdir.is_dir():
            cuda_benchmarks.extend(
                path for path in cuda_subdir.glob("*.cu") if not is_generated_benchmark_copy(path)
            )

    for chapter_dir in discover_all_chapters(repo_root):
        _collect_from_dir(chapter_dir)

    return sorted(set(cuda_benchmarks))


def discover_all_chapters(repo_root: Path, bench_roots: Optional[List[Path]] = None) -> List[Path]:
    """Discover all directories that contain benchmark pairs."""
    roots = bench_roots or get_bench_roots(repo_root=repo_root)
    chapter_dirs: List[Path] = []
    seen = set()

    for bench_root in roots:
        if not bench_root.exists():
            continue

        # Legacy ch* and labs/* paths (keep natural ordering for these)
        for ch_dir in sorted(
            [
                d
                for d in bench_root.iterdir()
                if d.is_dir() and d.name.startswith("ch") and d.name[2:].isdigit()
            ],
            key=lambda p: int(p.name[2:]) if p.name[2:].isdigit() else 0,
        ):
            if ch_dir.resolve() not in seen:
                seen.add(ch_dir.resolve())
                chapter_dirs.append(ch_dir)

        for lab_dir in _lab_dirs(repo_root, bench_root=bench_root):
            if lab_dir.resolve() not in seen:
                seen.add(lab_dir.resolve())
                chapter_dirs.append(lab_dir)

        # Generic discovery: any directory with baseline_* files
        for dir_with_benchmark in _iter_benchmark_dirs(bench_root):
            resolved = dir_with_benchmark.resolve()
            if resolved not in seen:
                seen.add(resolved)
                chapter_dirs.append(resolved)

    # Stable, human-friendly ordering
    chapter_dirs.sort(key=lambda p: p.as_posix())
    return chapter_dirs


def resolve_target_chapters(
    targets: Optional[List[str]],
    bench_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[List[Path], Dict[str, Set[str]]]:
    """
    Translate CLI target tokens into chapter directories + per-chapter filters.

    Args:
        targets: List like ["ch07", "ch07:memory_access"] (None/"all" -> every chapter)

    Returns:
        (chapter_dirs, chapter_filters)
          chapter_dirs: ordered list of chapter paths to run
          chapter_filters: map of chapter slug -> set of example names to include
    """
    repo_root = Path(repo_root or DEFAULT_REPO_ROOT)
    chapter_filters: Dict[str, Set[str]] = {}

    roots = [Path(bench_root).resolve()] if bench_root else get_bench_roots(repo_root=repo_root)
    primary_root = roots[0]

    # Default: run everything
    if not targets or any(str(t).lower() == "all" for t in targets):
        return discover_all_chapters(primary_root, bench_roots=roots), chapter_filters

    chapter_dirs: List[Path] = []
    for raw_target in targets:
        if not raw_target:
            continue

        target = str(raw_target).strip()
        if not target:
            continue

        chapter_token, sep, examples = target.partition(":")
        normalized = normalize_chapter_token(chapter_token, repo_root=repo_root, bench_root=primary_root)
        chapter_dir = Path(normalized)
        if not chapter_dir.is_absolute():
            chapter_dir = (primary_root / normalized).resolve()

        # Expand top-level labs selector into concrete lab chapters so
        # `-t labs` behaves like "all labs/*" instead of an empty top-level scan.
        if chapter_dir.resolve() == (primary_root / "labs").resolve():
            if sep and examples.strip():
                raise ValueError(
                    "Target 'labs:<example>' is ambiguous. "
                    "Use 'labs/<lab>:<example>' to select a specific lab example."
                )
            for lab_dir in _lab_dirs(repo_root, bench_root=primary_root):
                if lab_dir not in chapter_dirs:
                    chapter_dirs.append(lab_dir)
            continue

        if not chapter_dir.is_dir():
            raise FileNotFoundError(f"Chapter '{normalized}' not found at {chapter_dir}")

        if chapter_dir not in chapter_dirs:
            chapter_dirs.append(chapter_dir)

        # Collect per-chapter example filters when provided
        if sep:
            slug = chapter_slug(chapter_dir, repo_root, bench_root=primary_root)
            allowed = {example for _, _, example in discover_benchmarks(chapter_dir)}
            for example in _parse_examples(examples):
                if example not in allowed:
                    raise ValueError(
                        f"Example '{example}' not found in {slug}. "
                        f"Available: {', '.join(sorted(allowed)) or 'none'}"
                    )
                chapter_filters.setdefault(slug, set()).add(example)

    if not chapter_dirs:
        raise ValueError("No valid chapters resolved from targets.")

    return chapter_dirs, chapter_filters


def discover_benchmark_pairs(repo_root: Path, chapter: str = "all") -> List[Tuple[Path, List[Path], str]]:
    """Discover benchmark pairs across chapters.
    
    Args:
        repo_root: Path to repository root
        chapter: Chapter identifier ('all' or specific chapter like 'ch12' or '12')
        
    Returns:
        List of tuples: (baseline_path, [optimized_paths], example_name)
    """
    all_pairs = []
    bench_roots = get_bench_roots(repo_root=repo_root)
    primary_root = bench_roots[0]
    
    if chapter == "all":
        chapter_dirs = discover_all_chapters(repo_root, bench_roots=bench_roots)
    else:
        try:
            normalized = normalize_chapter_token(chapter, repo_root=repo_root, bench_root=primary_root)
        except ValueError:
            chapter_dirs = []
        else:
            chapter_path = Path(normalized)
            if not chapter_path.is_absolute():
                chapter_path = (primary_root / normalized).resolve()
            chapter_dirs = [chapter_path] if chapter_path.exists() else []
    
    for chapter_dir in chapter_dirs:
        pairs = discover_benchmarks(chapter_dir)
        all_pairs.extend(pairs)
    
    return all_pairs
