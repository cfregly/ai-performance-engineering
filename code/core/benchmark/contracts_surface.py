"""Shared benchmark methodology and warehouse contract surfaces.

This module provides one small summary object that can be reused by CLI, MCP,
without duplicating path or schema logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "2026-03-13"

_SURFACES = {
    "methodology": {
        "path": REPO_ROOT / "docs" / "benchmark_methodology.md",
        "kind": "doc",
        "description": "Repo-wide benchmark methodology, evidence policy, and bottleneck model.",
    },
    "warehouse": {
        "path": REPO_ROOT / "docs" / "performance_warehouse.md",
        "kind": "doc",
        "description": "Planet-scale warehouse design, schema, telemetry joins, and retention policy.",
    },
    "workload_spec": {
        "path": REPO_ROOT / "templates" / "benchmark_workload_spec.yaml",
        "kind": "yaml",
        "description": "Frozen workload definition and serving fairness contract.",
    },
    "benchmark_run": {
        "path": REPO_ROOT / "templates" / "benchmark_run.yaml",
        "kind": "yaml",
        "description": "Declarative BenchmarkRun contract with observability and sinks.",
    },
    "warehouse_contract": {
        "path": REPO_ROOT / "templates" / "performance_warehouse_contract.yaml",
        "kind": "yaml",
        "description": "Raw-vs-curated warehouse contract, telemetry sources, and retention tiers.",
    },
    "kubernetes_service": {
        "path": REPO_ROOT / "cluster" / "docs" / "kubernetes_benchmark_service.md",
        "kind": "doc",
        "description": "Kubernetes-native operator and control-loop design for BenchmarkRun.",
    },
    "benchmark_run_crd": {
        "path": REPO_ROOT / "cluster" / "configs" / "benchmarkrun-crd.yaml",
        "kind": "yaml",
        "description": "CRD sketch for BenchmarkRun.",
    },
}

_INTERFACE_ENTRIES = (
    {
        "id": "cli",
        "label": "CLI",
        "transport": "cli",
        "entrypoint": "python -m cli.aisp tools benchmark-contracts",
        "description": "Print the shared contract summary as JSON.",
    },
    {
        "id": "mcp",
        "label": "MCP",
        "transport": "mcp",
        "entrypoint": "benchmark_contracts",
        "description": "MCP tool exposing the same summary object to remote clients.",
    },
)

_GENERATOR_CHOICES = {
    "benchmarkClass": ["publication_grade", "realism_grade"],
    "workloadType": ["training", "inference", "mixed"],
    "cadence": ["canary", "nightly", "pre_release"],
    "comparisonVariable": [
        "hardware_generation",
        "runtime_version",
        "scheduler_path",
        "control_plane_path",
        "driver_stack",
        "network_topology",
        "storage_stack",
    ],
}

_GENERATOR_RENDER_INTERFACE_ENTRIES = (
    {
        "id": "cli",
        "label": "CLI",
        "transport": "cli",
        "entrypoint": "python -m cli.aisp tools benchmark-run-render -- --overrides-json '{\"name\":\"publication-inference-stack-b200\"}'",
        "description": "Render BenchmarkRun YAML through the shared Python renderer.",
    },
    {
        "id": "mcp",
        "label": "MCP",
        "transport": "mcp",
        "entrypoint": "render_benchmark_run",
        "description": "MCP tool that renders BenchmarkRun YAML with the same backend function.",
    },
)


def _summarize_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: Dict[str, Any] = {"top_level_keys": sorted(payload.keys())}
    spec = payload.get("spec")
    if isinstance(spec, dict):
        summary["spec_keys"] = sorted(spec.keys())
        if "layers" in spec and isinstance(spec["layers"], list):
            summary["enabled_layers"] = [
                layer.get("name")
                for layer in spec["layers"]
                if isinstance(layer, dict) and layer.get("enabled")
            ]
        if "observability" in spec:
            summary["has_observability"] = True
        if "sinks" in spec:
            summary["has_sinks"] = True
    return summary


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _benchmark_run_generator_defaults() -> Dict[str, str]:
    benchmark_run_path = Path(_SURFACES["benchmark_run"]["path"])
    payload = _load_yaml_mapping(benchmark_run_path)
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    spec = payload.get("spec", {}) if isinstance(payload.get("spec"), dict) else {}
    intent = spec.get("intent", {}) if isinstance(spec.get("intent"), dict) else {}
    workload = spec.get("workload", {}) if isinstance(spec.get("workload"), dict) else {}
    comparison = spec.get("comparison", {}) if isinstance(spec.get("comparison"), dict) else {}
    return {
        "name": str(metadata.get("name", "publication-inference-stack-b200")),
        "benchmarkClass": str(intent.get("benchmarkClass", "publication_grade")),
        "workloadType": str(intent.get("workloadType", "inference")),
        "schedulerPath": str(intent.get("schedulerPath", "slinky-kueue")),
        "cadence": str(intent.get("cadence", "pre_release")),
        "model": str(workload.get("model", "openai/gpt-oss-20b")),
        "precision": str(workload.get("precision", "bf16")),
        "batchingPolicy": str(workload.get("batchingPolicy", "continuous")),
        "concurrencyModel": str(workload.get("concurrencyModel", "closed_loop")),
        "comparisonVariable": str(comparison.get("variableUnderTest", "runtime_version")),
    }


def _normalize_generator_values(overrides: Dict[str, Any] | None = None) -> Dict[str, str]:
    values = dict(_benchmark_run_generator_defaults())
    for key, raw_value in (overrides or {}).items():
        if key not in values:
            raise ValueError(f"Unknown BenchmarkRun generator field: {key}")
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            values[key] = value
    for field, allowed in _GENERATOR_CHOICES.items():
        value = values[field]
        if value not in allowed:
            allowed_display = ", ".join(allowed)
            raise ValueError(f"{field} must be one of: {allowed_display}")
    return values


def _render_benchmark_run_document(values: Dict[str, str]) -> Dict[str, Any]:
    benchmark_run_path = Path(_SURFACES["benchmark_run"]["path"])
    payload = _load_yaml_mapping(benchmark_run_path)
    metadata = payload.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    spec = payload.setdefault("spec", {})
    intent = spec.setdefault("intent", {})
    workload = spec.setdefault("workload", {})
    comparison = spec.setdefault("comparison", {})
    controls = comparison.setdefault("controls", {})
    fixed = controls.setdefault("fixed", {})
    metrics = spec.setdefault("metrics", {})
    training = metrics.setdefault("training", {})
    inference = metrics.setdefault("inference", {})
    execution_policy = spec.setdefault("executionPolicy", {})
    realism = execution_policy.setdefault("realismGrade", {})

    metadata["name"] = values["name"]
    labels["aisp.dev/benchmark-class"] = values["benchmarkClass"]

    intent["benchmarkClass"] = values["benchmarkClass"]
    intent["workloadType"] = values["workloadType"]
    intent["schedulerPath"] = values["schedulerPath"]
    intent["cadence"] = values["cadence"]

    workload["model"] = values["model"]
    workload["precision"] = values["precision"]
    workload["batchingPolicy"] = values["batchingPolicy"]
    workload["concurrencyModel"] = values["concurrencyModel"]

    comparison["variableUnderTest"] = values["comparisonVariable"]
    fixed["model"] = values["model"]
    fixed["precision"] = values["precision"]
    fixed["batchingPolicy"] = values["batchingPolicy"]
    fixed["concurrencyModel"] = values["concurrencyModel"]

    training["enabled"] = values["workloadType"] in {"training", "mixed"}
    inference["enabled"] = values["workloadType"] in {"inference", "mixed"}
    realism["multiTenantScenarios"] = values["benchmarkClass"] == "realism_grade"
    return payload


def render_benchmark_run_yaml(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Render BenchmarkRun YAML using the repo template as the source of truth."""
    values = _normalize_generator_values(overrides)
    payload = _render_benchmark_run_document(values)
    rendered_yaml = yaml.safe_dump(payload, sort_keys=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "template_path": str(_SURFACES["benchmark_run"]["path"]),
        "applied_values": values,
        "rendered_yaml": rendered_yaml,
    }


def get_benchmark_run_generator_config() -> Dict[str, Any]:
    """Return generator metadata and a default preview rendered by the shared backend."""
    preview = render_benchmark_run_yaml()
    interfaces = {entry["id"]: entry["entrypoint"] for entry in _GENERATOR_RENDER_INTERFACE_ENTRIES}
    return {
        "template_path": preview["template_path"],
        "defaults": preview["applied_values"],
        "choices": _GENERATOR_CHOICES,
        "preview_yaml": preview["rendered_yaml"],
        "render_interfaces": {
            "cli": interfaces["cli"],
            "mcp_tool": interfaces["mcp"],
        },
        "render_interface_entries": list(_GENERATOR_RENDER_INTERFACE_ENTRIES),
    }


def _surface_entry(name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(meta["path"])
    exists = path.exists()
    entry: Dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": exists,
        "kind": meta["kind"],
        "description": meta["description"],
    }
    if exists and meta["kind"] == "yaml":
        entry["summary"] = _summarize_yaml(path)
    return entry


def get_benchmark_contracts_summary() -> Dict[str, Any]:
    """Return the repo-exposed benchmark methodology and warehouse surfaces."""
    contract_names = list(_SURFACES.keys())
    contracts = {name: _surface_entry(name, meta) for name, meta in _SURFACES.items()}
    missing_surfaces = [name for name in contract_names if not contracts[name]["exists"]]
    interfaces = {entry["id"]: entry["entrypoint"] for entry in _INTERFACE_ENTRIES}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "available": not missing_surfaces,
        "repo_root": str(REPO_ROOT),
        "surface_order": contract_names,
        "surface_count": len(contract_names),
        "missing_surface_count": len(missing_surfaces),
        "missing_surfaces": missing_surfaces,
        "contracts": contracts,
        "interfaces": {
            "cli": interfaces["cli"],
            "mcp_tool": interfaces["mcp"],
        },
        "interface_entries": list(_INTERFACE_ENTRIES),
        "generator": get_benchmark_run_generator_config(),
    }
