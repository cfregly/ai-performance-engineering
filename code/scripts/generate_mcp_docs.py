#!/usr/bin/env python3
"""Generate MCP tool catalog in docs/mcp_tools.md."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from mcp.mcp_server import TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent

START_MARKER = "<!-- BEGIN MCP TOOL LIST -->"
END_MARKER = "<!-- END MCP TOOL LIST -->"

CATEGORY_ORDER = [
    "GPU",
    "System",
    "Profiling",
    "Analyze",
    "Optimize",
    "Distributed",
    "Inference",
    "Benchmark",
    "AI",
    "Export",
    "Hardware",
    "HuggingFace",
    "Cluster/Cost",
    "Tools",
    "Utility",
    "Other",
]


def _short_desc(description: str) -> str:
    desc = (description or "").strip()
    if desc.startswith("Tags:"):
        parts = desc.split(". ", 1)
        if len(parts) > 1:
            desc = parts[1]
        else:
            desc = desc[len("Tags:"):].strip()
    # Use first sentence as summary
    sentence = desc.split(". ", 1)[0].strip()
    return (sentence or desc).replace("; ", ", ")


def _classify(name: str) -> str:
    if name.startswith("gpu_"):
        return "GPU"
    if name.startswith("system_"):
        return "System"
    if name == "clock_lock_check":
        return "System"
    if name.startswith("profile_") or name in {"compare_nsys", "compare_ncu", "nsys_summary", "ncu_summary"}:
        return "Profiling"
    if name.startswith("analyze_") or name.startswith("predict_"):
        return "Analyze"
    if name in {"optimize", "recommend", "optimize_roi", "optimize_techniques"}:
        return "Optimize"
    if name.startswith("distributed_") or name == "launch_plan":
        return "Distributed"
    if name.startswith("cluster_") or name == "cost_estimate":
        return "Cluster/Cost"
    if name.startswith("inference_"):
        return "Inference"
    if name.startswith("benchmark_") or name in {"run_benchmarks", "list_chapters"}:
        return "Benchmark"
    if name in {"ask", "explain", "ai_status", "ai_troubleshoot", "suggest_tools"}:
        return "AI"
    if name.startswith("export_"):
        return "Export"
    if name.startswith("hw_"):
        return "Hardware"
    if name == "hf":
        return "HuggingFace"
    if name.startswith("tools_"):
        return "Tools"
    if name in {"status", "triage", "context_summary", "context_full", "job_status"}:
        return "Utility"
    return "Other"


def render_mcp_tool_block() -> str:
    category_map: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_ORDER}
    for tool_name in TOOLS:
        category = _classify(tool_name)
        category_map.setdefault(category, []).append(tool_name)

    lines: List[str] = []
    lines.append("## Tool Catalog (generated)")
    lines.append("")
    lines.append("Generated from `mcp.mcp_server.TOOLS`. Run `python -m scripts.generate_mcp_docs` to refresh.")
    lines.append("")

    for category in CATEGORY_ORDER:
        tools = sorted(category_map.get(category, []))
        if not tools:
            continue
        lines.append(f"### {category} ({len(tools)})")
        for name in tools:
            desc = _short_desc(TOOLS[name].description)
            lines.append(f"- `{name}`: {desc}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def update_doc(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        raise ValueError("MCP tool list markers not found in docs/mcp_tools.md")
    start_idx = content.index(START_MARKER) + len(START_MARKER)
    end_idx = content.index(END_MARKER)
    generated = render_mcp_tool_block()
    new_content = content[:start_idx] + "\n\n" + generated + "\n" + content[end_idx:]
    path.write_text(new_content, encoding="utf-8")


def main() -> None:
    doc_path = REPO_ROOT / "docs" / "mcp_tools.md"
    update_doc(doc_path)


if __name__ == "__main__":
    main()
