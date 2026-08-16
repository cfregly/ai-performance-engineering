#!/usr/bin/env python3
"""Generate file registry for all baseline_/optimized_ pairs across chapters.

This script auto-discovers all baseline_/optimized_ file pairs and generates
a registry that can be used by coverage verification tools.

Usage:
    python -m core.scripts.generate_file_registry [--output registry.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

code_root = Path(__file__).resolve().parents[2]

from core.utils.chapter_compare_template import discover_benchmarks


def extract_concept_from_filename(filename: str) -> str:
    """Extract concept name from filename.
    
    Examples:
        baseline_cublas_vs_cutlass.py -> cublas
        baseline_shared_memory.py -> shared_memory
        optimized_moe_sparse.py -> moe
    """
    # Remove baseline_ or optimized_ prefix
    name = filename.replace("baseline_", "").replace("optimized_", "")
    # Remove .py extension
    name = name.replace(".py", "")
    # Extract first concept (split by _)
    concept = name.split("_")[0]
    return concept


def discover_all_benchmarks(repo_root: Path) -> Dict[str, List[Dict]]:
    """Discover all baseline_/optimized_ pairs across all chapters.
    
    Returns:
        Dictionary mapping chapter IDs to lists of benchmark pairs with metadata
    """
    registry = {}
    
    # Find all chapter directories
    chapter_dirs = sorted(repo_root.glob("ch[0-9]*"))
    
    for ch_dir in chapter_dirs:
        ch_id = ch_dir.name
        
        # Discover benchmarks using the standard discovery function
        pairs = discover_benchmarks(ch_dir)
        
        chapter_registry = []
        for baseline_path, optimized_paths, example_name in pairs:
            # Extract concept from baseline filename
            concept = extract_concept_from_filename(baseline_path.name)
            
            # Build registry entry
            entry = {
                "concept": concept,
                "example_name": example_name,
                "baseline_file": baseline_path.name,
                "baseline_path": str(baseline_path.relative_to(repo_root)),
                "optimized_files": [p.name for p in optimized_paths],
                "optimized_paths": [str(p.relative_to(repo_root)) for p in optimized_paths],
            }
            
            # Try to extract metadata from files
            try:
                baseline_content = baseline_path.read_text()
                if "Benchmark" in baseline_content:
                    entry["implements_benchmark"] = True
                if "nvtx" in baseline_content.lower():
                    entry["uses_nvtx"] = True
            except OSError:
                pass  # File read error
            
            chapter_registry.append(entry)
        
        if chapter_registry:
            registry[ch_id] = chapter_registry
    
    return registry


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate file registry for baseline_/optimized_ pairs")
    parser.add_argument(
        "--output",
        type=Path,
        default=code_root / "file_registry.json",
        help="Output JSON file path (default: file_registry.json)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING FILE REGISTRY")
    print("=" * 80)
    print()
    
    # Discover all benchmarks
    registry = discover_all_benchmarks(code_root)
    
    # Count statistics
    total_chapters = len(registry)
    total_pairs = sum(len(pairs) for pairs in registry.values())
    total_files = sum(
        len(pairs) + sum(len(p["optimized_files"]) for p in pairs)
        for pairs in registry.values()
    )
    
    print(f"Discovered:")
    print(f"  - Chapters: {total_chapters}")
    print(f"  - Baseline/Optimized pairs: {total_pairs}")
    print(f"  - Total files: {total_files}")
    print()
    
    # Write output
    if args.format == "json":
        with open(args.output, "w") as f:
            json.dump({
                "metadata": {
                    "generated_by": "generate_file_registry.py",
                    "total_chapters": total_chapters,
                    "total_pairs": total_pairs,
                    "total_files": total_files,
                },
                "chapters": registry
            }, f, indent=2)
        print(f"[OK] Registry written to {args.output}")
    else:
        try:
            import yaml
            with open(args.output, "w") as f:
                yaml.dump({
                    "metadata": {
                        "generated_by": "generate_file_registry.py",
                        "total_chapters": total_chapters,
                        "total_pairs": total_pairs,
                        "total_files": total_files,
                    },
                    "chapters": registry
                }, f, default_flow_style=False)
            print(f"[OK] Registry written to {args.output}")
        except ImportError:
            print("ERROR: YAML format requires PyYAML. Install with: pip install pyyaml")
            sys.exit(1)
    
    # Print summary by chapter
    print()
    print("Summary by chapter:")
    for ch_id in sorted(registry.keys()):
        pairs = registry[ch_id]
        print(f"  {ch_id}: {len(pairs)} pairs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
