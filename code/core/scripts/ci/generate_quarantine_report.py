#!/usr/bin/env python3
"""Generate CI summary report for quarantined benchmarks.

This script generates a human-readable report of all quarantined benchmarks,
grouped by quarantine reason. Designed for CI output and GitHub Actions summaries.

Usage:
    python -m core.scripts.ci.generate_quarantine_report
    python -m core.scripts.ci.generate_quarantine_report --format markdown
    python -m core.scripts.ci.generate_quarantine_report --format json --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parents[3]

from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import QuarantineReason


def generate_text_report(manager: QuarantineManager) -> str:
    """Generate plain text report."""
    records = manager.get_all_records()
    
    if not records:
        return "✅ No quarantined benchmarks - all benchmarks are compliant!\n"
    
    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK QUARANTINE REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")
    
    # Group by reason
    by_reason: Dict[str, List[str]] = defaultdict(list)
    for path, record in records.items():
        reason_value = record.quarantine_reason.value if hasattr(record.quarantine_reason, 'value') else str(record.quarantine_reason)
        by_reason[reason_value].append(path)
    
    # Summary
    total = len(records)
    lines.append(f"SUMMARY: {total} benchmark(s) quarantined")
    lines.append("")
    lines.append("By Reason:")
    for reason, paths in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {reason}: {len(paths)}")
    lines.append("")
    lines.append("-" * 70)
    
    # Details by reason
    for reason in sorted(by_reason.keys()):
        paths = by_reason[reason]
        lines.append("")
        lines.append(f"## {reason.upper()} ({len(paths)} benchmarks)")
        lines.append("")
        for path in sorted(paths):
            record = records[path]
            lines.append(f"  • {path}")
            if record.details:
                for key, value in record.details.items():
                    lines.append(f"      {key}: {value}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("To fix quarantined benchmarks:")
    lines.append("  1. Add missing verification methods (get_input_signature, validate_result, get_verify_output)")
    lines.append("  2. Remove skip flags or add justification")
    lines.append("  3. Fix output comparison failures")
    lines.append("  4. Run: python -m core.scripts.audit_verification_compliance.py")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def generate_markdown_report(manager: QuarantineManager) -> str:
    """Generate GitHub-flavored Markdown report."""
    records = manager.get_all_records()
    
    if not records:
        return "## ✅ Quarantine Report\n\nNo quarantined benchmarks - all benchmarks are compliant!\n"
    
    lines = []
    lines.append("## ⚠️ Quarantine Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().isoformat()}*")
    lines.append("")
    
    # Group by reason
    by_reason: Dict[str, List[str]] = defaultdict(list)
    for path, record in records.items():
        reason_value = record.quarantine_reason.value if hasattr(record.quarantine_reason, 'value') else str(record.quarantine_reason)
        by_reason[reason_value].append(path)
    
    # Summary table
    total = len(records)
    lines.append(f"**{total} benchmark(s) quarantined**")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, paths in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        lines.append(f"| `{reason}` | {len(paths)} |")
    lines.append("")
    
    # Details by reason
    for reason in sorted(by_reason.keys()):
        paths = by_reason[reason]
        lines.append(f"### {reason}")
        lines.append("")
        lines.append("<details>")
        lines.append(f"<summary>{len(paths)} benchmark(s)</summary>")
        lines.append("")
        for path in sorted(paths):
            record = records[path]
            lines.append(f"- `{path}`")
            if record.details:
                for key, value in record.details.items():
                    lines.append(f"  - {key}: `{value}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    
    # Action items
    lines.append("### 🔧 How to Fix")
    lines.append("")
    lines.append("1. **Missing verification methods**: Implement `get_input_signature()`, `validate_result()`, `get_verify_output()`")
    lines.append("2. **Skip flags**: Remove skip flags or add justification attribute")
    lines.append("3. **Output mismatch**: Fix numerical accuracy issues or adjust tolerances")
    lines.append("4. **Run audit**: `python -m core.scripts.audit_verification_compliance`")
    lines.append("")
    
    return "\n".join(lines)


def generate_json_report(manager: QuarantineManager) -> str:
    """Generate JSON report."""
    records = manager.get_all_records()
    
    # Build report structure
    by_reason: Dict[str, List[Dict]] = defaultdict(list)
    for path, record in records.items():
        reason_value = record.quarantine_reason.value if hasattr(record.quarantine_reason, 'value') else str(record.quarantine_reason)
        by_reason[reason_value].append({
            "path": path,
            "timestamp": record.quarantine_timestamp.isoformat() if record.quarantine_timestamp else None,
            "details": record.details or {},
        })
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_quarantined": len(records),
        "summary": {reason: len(paths) for reason, paths in by_reason.items()},
        "by_reason": dict(by_reason),
    }
    
    return json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CI summary report for quarantined benchmarks"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--quarantine-file",
        type=str,
        default=None,
        help="Path to quarantine.json (default: artifacts/verify_cache/quarantine.json)"
    )
    
    args = parser.parse_args()
    
    # Initialize quarantine manager
    if args.quarantine_file:
        cache_dir = Path(args.quarantine_file).parent
    else:
        cache_dir = PROJECT_ROOT / "artifacts" / "verify_cache"
    
    manager = QuarantineManager(cache_dir=cache_dir)
    
    # Generate report
    if args.format == "text":
        report = generate_text_report(manager)
    elif args.format == "markdown":
        report = generate_markdown_report(manager)
    elif args.format == "json":
        report = generate_json_report(manager)
    else:
        report = generate_text_report(manager)
    
    # Output
    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
    
    # Exit with non-zero if there are quarantined benchmarks
    records = manager.get_all_records()
    if records:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()




