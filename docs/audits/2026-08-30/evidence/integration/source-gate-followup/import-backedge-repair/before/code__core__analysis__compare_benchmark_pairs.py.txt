"""Generic baseline/optimized comparator using discovery + harness.

Example:
  python -m core.analysis.compare_benchmark_pairs --chapter labs/moe_parallelism
  python -m core.analysis.compare_benchmark_pairs --chapter ch15 --targets expert_parallelism
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

from core.discovery import discover_benchmarks, normalize_chapter_token
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from labs.moe_parallelism.plan import PlanEvaluator


def _load_pairs(chapter_slug: str) -> List[Tuple[Path, Path, str]]:
    root = Path(REPO_ROOT)
    chapter_dir = (root / chapter_slug).resolve()
    pairs: List[Tuple[Path, Path, str]] = []
    for baseline_path, optimized_paths, example_name in discover_benchmarks(chapter_dir):
        for opt in optimized_paths:
            pairs.append((baseline_path, opt, example_name))
    return pairs


def _run(target: Path) -> Tuple[str, float, float]:
    name = target.stem
    module_path = str(target)
    import importlib.util

    spec_obj = importlib.util.spec_from_file_location(target.stem, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Could not load module {module_path}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)  # type: ignore[arg-type]

    # Prefer analytic path if module exposes plan + evaluator inputs
    if hasattr(module, "build_plan") and hasattr(module, "CLUSTER") and hasattr(module, "MODEL"):
        plan = module.build_plan()
        evaluator = PlanEvaluator(module.CLUSTER, module.MODEL)
        report = evaluator.analyze(plan)
        return name, report.estimated_step_ms, report.throughput_tokens_per_s

    if not hasattr(module, "get_benchmark"):
        raise ImportError(f"Module {module_path} missing get_benchmark() or plan hooks")
    benchmark = module.get_benchmark()
    # Minimal config: single iteration, no subprocess
    config = BenchmarkConfig(iterations=1, warmup=0)
    config.use_subprocess = False
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    result = harness.benchmark(benchmark)
    step_ms = result.timing.mean_ms if result.timing else 0.0
    return name, step_ms, 0.0


def format_table(rows: List[Tuple[str, float, float, float, float]]) -> str:
    header = f"{'target':<28}{'step_ms':>12}{'throughput':>16}{'opt/base':>12}{'thrpt ratio':>14}"
    lines = [header, "-" * len(header)]
    for name, b_ms, b_tp, o_ms, o_tp in rows:
        ratio_ms = o_ms / b_ms if b_ms else 0.0
        ratio_tp = o_tp / b_tp if b_tp else 0.0
        lines.append(
            f"{name:<28}{b_ms:>12.1f}{b_tp:>16,.0f}{ratio_ms:>12.2f}{ratio_tp:>14.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", default="labs/moe_parallelism", help="Chapter/lab token (e.g., ch15, labs/moe_parallelism)")
    parser.add_argument("--targets", nargs="*", help="Optional example names to filter (comma or space separated)")
    args = parser.parse_args()

    chapter_slug = normalize_chapter_token(args.chapter)
    pairs = _load_pairs(chapter_slug)
    if args.targets:
        wanted = set()
        for t in args.targets:
            wanted.update(t.split(","))
        pairs = [p for p in pairs if p[2] in wanted]
    if not pairs:
        print("No baseline/optimized pairs found")
        return

    rows = []
    for baseline, optimized, name in pairs:
        _, b_ms, b_tp = _run(baseline)
        _, o_ms, o_tp = _run(optimized)
        rows.append((name, b_ms, b_tp, o_ms, o_tp))
    print(format_table(rows))


if __name__ == "__main__":
    main()
