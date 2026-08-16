#!/usr/bin/env python3
"""Generate concept_code_mapping.json from discovered baseline_/optimized_ pairs.

This script auto-generates the concept-to-code mapping by:
1. Extracting concepts from book markdown files
2. Discovering baseline_/optimized_ file pairs
3. Matching concepts to file pairs
4. Preserving existing notes/metadata where possible

Usage:
    python -m core.scripts.generate_concept_mapping [--output concept_code_mapping.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

code_root = Path(__file__).resolve().parents[2]
from core.utils.chapter_compare_template import discover_benchmarks

CHAPTER_EXCLUSIONS: Dict[int, Set[str]] = {}
CONCEPT_KEYWORDS: Dict[str, tuple[str, ...]] = {}
_GENERIC_CONCEPTS = {
    "baseline",
    "benchmark",
    "benchmarks",
    "comparison",
    "concept",
    "concepts",
    "example",
    "examples",
    "introduction",
    "optimized",
    "performance",
    "summary",
}


def _slugify_concept(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def extract_key_concepts_from_text(text: str) -> Set[str]:
    """Extract plausible concept tokens from headings and inline code."""
    concepts: Set[str] = set()

    for match in re.finditer(r"^#+\s+(.+)$", text, flags=re.MULTILINE):
        concept = _slugify_concept(match.group(1))
        if concept and concept not in _GENERIC_CONCEPTS:
            concepts.add(concept)

    for match in re.finditer(r"`([A-Za-z][A-Za-z0-9_\- ]+)`", text):
        concept = _slugify_concept(match.group(1))
        if concept and concept not in _GENERIC_CONCEPTS:
            concepts.add(concept)

    return concepts


def should_exclude_concept(chapter_num: int, concept: str) -> bool:
    if len(concept) < 3:
        return True
    if concept in _GENERIC_CONCEPTS:
        return True
    return concept in CHAPTER_EXCLUSIONS.get(chapter_num, set())


def extract_concept_from_filename(filename: str) -> str:
    """Extract concept name from filename.
    
    Handles multi-word concepts like 'ai_optimization', 'kv_cache', etc.
    Returns the full concept name, not just the first word.
    """
    name = filename.replace("baseline_", "").replace("optimized_", "").replace(".py", "")
    # Return full name (e.g., 'ai_optimization', 'kv_cache_management')
    # This handles multi-word concepts correctly
    return name


def get_chapter_name(ch_id: str) -> str:
    """Get chapter name from ID."""
    chapter_names = {
        "ch01": "Introduction and AI Systems Overview",
        "ch02": "Hardware Overview",
        "ch03": "Infrastructure and OS Tuning",
        "ch04": "Distributed Networking Communication",
        "ch05": "Storage and Input Pipelines",
        "ch06": "GPU Architecture and CUDA Fundamentals",
        "ch07": "Memory Access and Data Movement",
        "ch08": "Occupancy and Profiling",
        "ch09": "Arithmetic Intensity and Kernel Efficiency",
        "ch10": "Intra-Kernel Pipelining",
        "ch11": "Streams and Concurrency",
        "ch12": "Dynamic Scheduling and CUDA Graphs",
        "ch13": "PyTorch Profiling and Scaling",
        "ch14": "Compilers, Graph Breaks, and Triton",
        "ch15": "Multinode and Disaggregated Inference",
        "ch16": "Production Inference and Serving",
        "ch17": "Disaggregated Prefill and Decode",
        "ch18": "Attention Optimization and Decode Kernels",
        "ch19": "Dynamic and Adaptive Inference",
        "ch20": "AI-Assisted Performance Optimization",
    }
    return chapter_names.get(ch_id, ch_id.replace("ch", "Chapter "))


def extract_notes_from_file(file_path: Path) -> Optional[str]:
    """Extract notes from file docstring or comments."""
    try:
        content = file_path.read_text()
        # Look for docstring notes
        if '"""' in content:
            docstring_start = content.find('"""')
            docstring_end = content.find('"""', docstring_start + 3)
            if docstring_end > docstring_start:
                docstring = content[docstring_start + 3:docstring_end]
                # Extract meaningful notes (lines with "demonstrates", "shows", etc.)
                lines = [l.strip() for l in docstring.split('\n') if l.strip()]
                note_lines = [l for l in lines if any(
                    word in l.lower() for word in ['demonstrates', 'shows', 'optimized', 'baseline', 'benefit', 'speedup']
                )]
                if note_lines:
                    return ' '.join(note_lines[:2])  # First 2 relevant lines
    except OSError:
        pass  # File read error
    return None


def generate_concept_mapping(repo_root: Path, existing_json: Optional[Dict] = None) -> Dict:
    """Generate concept mapping JSON structure."""
    mapping = {
        "metadata": {
            "total_chapters": 20,
            "total_concepts": 0,
            "chapters_with_gaps": [],
            "excluded_concepts": list(set().union(*CHAPTER_EXCLUSIONS.values()) if CHAPTER_EXCLUSIONS else []),
        },
        "chapters": {}
    }
    
    # Preserve existing notes if provided
    existing_notes = {}
    if existing_json and "chapters" in existing_json:
        for ch_id, ch_data in existing_json["chapters"].items():
            if "concepts" in ch_data:
                for concept, concept_data in ch_data["concepts"].items():
                    if isinstance(concept_data, dict) and "notes" in concept_data:
                        existing_notes[f"{ch_id}.{concept}"] = concept_data["notes"]
    
    # Process each chapter
    for ch_num in range(1, 21):
        ch_id = f"ch{ch_num}"
        ch_dir = repo_root / ch_id
        md_file = repo_root / "book" / f"{ch_id}.md"
        
        # Extract concepts from book
        book_concepts = set()
        if md_file.exists():
            book_text = md_file.read_text()
            all_concepts = extract_key_concepts_from_text(book_text)
            # Filter excluded concepts
            for concept in all_concepts:
                if not should_exclude_concept(ch_num, concept):
                    book_concepts.add(concept)
        
        # Discover baseline_/optimized_ pairs
        file_pairs = {}
        if ch_dir.exists():
            pairs = discover_benchmarks(ch_dir)
            for baseline_path, optimized_paths, example_name in pairs:
                concept = extract_concept_from_filename(baseline_path.name)
                file_pairs[concept] = {
                    "baseline_file": baseline_path.name,
                    "optimized_file": optimized_paths[0].name if optimized_paths else None,
                    "status": "covered" if optimized_paths else "missing",
                }
                # Extract notes
                note = extract_notes_from_file(baseline_path)
                if not note and optimized_paths:
                    note = extract_notes_from_file(optimized_paths[0])
                if note:
                    file_pairs[concept]["notes"] = note
                # Preserve existing notes if available
                if f"{ch_id}.{concept}" in existing_notes:
                    file_pairs[concept]["notes"] = existing_notes[f"{ch_id}.{concept}"]
        
        # Build chapter entry
        chapter_entry = {
            "name": get_chapter_name(ch_id),
            "concepts": {},
        }
        
        # Add all book concepts
        for concept in sorted(book_concepts):
            if concept in file_pairs:
                # Concept has implementation
                chapter_entry["concepts"][concept] = file_pairs[concept]
            else:
                # Concept is missing
                chapter_entry["concepts"][concept] = {
                    "baseline_file": None,
                    "optimized_file": None,
                    "status": "missing",
                }
        
        # Add concepts found in files but not in book (for completeness)
        for concept, pair_data in file_pairs.items():
            if concept not in chapter_entry["concepts"]:
                chapter_entry["concepts"][concept] = pair_data
        
        mapping["chapters"][ch_id] = chapter_entry
    
    # Update metadata
    total_concepts = sum(len(ch["concepts"]) for ch in mapping["chapters"].values())
    mapping["metadata"]["total_concepts"] = total_concepts
    
    chapters_with_gaps = [
        ch_id for ch_id, ch_data in mapping["chapters"].items()
        if any(c.get("status") == "missing" for c in ch_data["concepts"].values())
    ]
    mapping["metadata"]["chapters_with_gaps"] = chapters_with_gaps
    
    return mapping


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate concept_code_mapping.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=code_root / "concept_code_mapping.json",
        help="Output JSON file path (default: concept_code_mapping.json)"
    )
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Preserve existing JSON file and merge notes/metadata"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING concept_code_mapping.json")
    print("=" * 80)
    print()
    
    # Load existing JSON if preserving
    existing_json = None
    if args.preserve_existing and args.output.exists():
        try:
            with open(args.output) as f:
                existing_json = json.load(f)
            print(f"Loaded existing JSON from {args.output}")
        except Exception as e:
            print(f"Could not load existing JSON: {e}")
    
    # Generate mapping
    mapping = generate_concept_mapping(code_root, existing_json)
    
    # Write output
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=False)
    
    print(f"Generated concept_code_mapping.json at {args.output}")
    print()
    print(f"Summary:")
    print(f"  - Chapters: {len(mapping['chapters'])}")
    print(f"  - Total concepts: {mapping['metadata']['total_concepts']}")
    print(f"  - Chapters with gaps: {len(mapping['metadata']['chapters_with_gaps'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
