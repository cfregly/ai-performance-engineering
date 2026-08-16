from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2026-03-16"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent / "nvidia-advanced-networking-for-ai-infra"
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "fabric_command_catalog.json"

_COMMAND_PREFIXES = (
    "ib",
    "saquery",
    "perfquery",
    "watch ",
    "curl ",
    "nv ",
    "vtysh ",
    "ip ",
    "cat ",
    "systemctl ",
)
_TIP_PATTERNS = (
    "look for",
    "verify",
    "expected",
    "must ",
    "requires",
    "only supported",
    "common values",
    "used internally",
    "cannot participate",
    "degraded_bw",
    "disable may be irreversible",
    "not actual port bandwidth",
    "lossless",
    "lossy",
    "expected in ibmgtsim",
    "real-time insights",
    "near-infiniband performance",
)
_TOPIC_TO_PLANE = {
    "address resolution": "control_plane",
    "adaptive routing": "control_plane",
    "bgp unnumbered": "control_plane",
    "comprehensive diagnostics": "control_plane",
    "contents": "reference",
    "domain discovery": "management_plane",
    "fabric discovery": "control_plane",
    "introduction": "reference",
    "ipv4 addressing": "control_plane",
    "ipv4 unicast in bgp": "control_plane",
    "lab topology": "inventory",
    "partition management": "management_plane",
    "performance counters": "control_plane",
    "port state & link mgmt": "control_plane",
    "references": "reference",
    "roce": "control_plane",
    "rocev2 on cumulus linux": "control_plane",
    "routing verification": "control_plane",
    "summary": "reference",
    "telemetry monitoring": "management_plane",
    "topology exploration": "management_plane",
    "w-ecmp": "control_plane",
    "w-ecmp on cumulus linux": "control_plane",
}
_SUPPLEMENTAL_ENTRIES = (
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Topology Exploration",
        "plane": "management_plane",
        "text": "curl -k <nmx-base>/switch-nodes | jq",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Topology Exploration",
        "plane": "management_plane",
        "text": "curl -k <nmx-base>/chassis | jq",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Topology Exploration",
        "plane": "management_plane",
        "text": "curl -k <nmx-base>/ports | jq",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Topology Exploration",
        "plane": "management_plane",
        "text": "echo \"GPUs: $(curl -sk <nmx-base>/gpus | jq length)\"; echo \"Switches: $(curl -sk <nmx-base>/switches | jq length)\"",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Partition Management",
        "plane": "management_plane",
        "text": "curl -k -X PUT <nmx-base>/partitions/<partition-id> -H \"Content-Type: application/json\" -d '{\"DomainUUID\":\"<DomainUUID>\",\"Members\":{\"locations\":[\"<chassis.slot.host.gpu>\"]}}' | jq",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^switch_temperature\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^PortXmitDataExtended\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^PortRcvDataExtended\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^PortLocalPhysicalErrors\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^CableInfoTemperature\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^CableInfoRxPower\" | head -5",
        "entry_type": "command",
    },
    {
        "source_doc": "supplemental/nmx_scenarios",
        "family": "nvlink",
        "topic": "Telemetry Monitoring",
        "plane": "management_plane",
        "text": "curl -sk <nmx-base>/metrics | grep \"^CableInfoTxPower\" | head -5",
        "entry_type": "command",
    },
)


def _relative_doc_path(path: Path, source_root: Path) -> str:
    return str(path.resolve().relative_to(source_root.resolve()))


def _family_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent in {"infiniband-cli", "infiniband-ufm-plugins", "ufm-webui"}:
        return "infiniband"
    if parent == "netq-nvlink":
        return "nvlink"
    return "spectrum-x"


def _topic_from_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace(" — Accelerated Networking for AI Infrastructure", "")
    stem = stem.replace("_", ": ")
    return stem.strip()


def _plane_for(topic: str) -> str:
    return _TOPIC_TO_PLANE.get(topic.lower(), "reference")


def _html_to_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"<script\b.*?</script>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in raw_lines if line]


def _normalize_command(line: str) -> str:
    cmd = line[2:].strip() if line.startswith("$ ") else line.strip()
    cmd = cmd.replace("\xa0", " ")
    cmd = re.sub(r"\s+", " ", cmd).strip()
    cmd = re.sub(r"https://(?:seat##-)?nvlink\.nvacademy\.dev/nmx/v1", "<nmx-base>", cmd)
    cmd = re.sub(r"<nmx-base>/services/[0-9a-f]+", "<nmx-base>/services/<service-id>", cmd)
    return cmd


def _looks_like_command(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.startswith("$ "):
        return True
    lowered = line.lower()
    if any(lowered.startswith(prefix) for prefix in _COMMAND_PREFIXES):
        return True
    return " -c " in lowered and lowered.startswith("vtysh")


def _looks_like_tip(line: str) -> bool:
    lowered = line.lower()
    if _looks_like_command(line):
        return False
    if len(line) < 20 or len(line) > 240:
        return False
    return any(pattern in lowered for pattern in _TIP_PATTERNS)


def _mutates_state(command: str) -> bool:
    lowered = command.lower()
    if lowered.startswith("nv set ") or lowered == "nv config apply":
        return True
    if " -x delete " in lowered or lowered.startswith("curl ") and " delete " in lowered:
        return True
    if re.search(r"\bibportstate\b.*\b(disable|enable)\b", lowered):
        return True
    if lowered.startswith("curl ") and any(token in lowered for token in (" -x post ", " -x patch ", " -x put ")):
        return True
    return False


def _expected_signal(command: str, family: str, topic: str, entry_type: str) -> str:
    if entry_type == "tip":
        return topic
    lowered = command.lower()
    if lowered.startswith("ibswitches"):
        return "switch inventory"
    if lowered.startswith("ibhosts"):
        return "host inventory"
    if lowered.startswith("iblinkinfo"):
        return "link topology"
    if lowered.startswith("ibnetdiscover"):
        return "full fabric topology"
    if lowered.startswith("ibstat"):
        return "local HCA state"
    if lowered.startswith("ibv_devinfo"):
        return "RDMA device capabilities"
    if lowered.startswith("saquery"):
        return "subnet manager database reachability"
    if lowered.startswith("ibaddr"):
        return "LID/GUID/GID mapping"
    if lowered.startswith("ibtracert"):
        return "path trace between endpoints"
    if lowered.startswith("ibroute"):
        return "switch forwarding table"
    if lowered.startswith("perfquery"):
        return "port counters and error state"
    if lowered.startswith("ibdiagnet"):
        return "comprehensive InfiniBand diagnostics"
    if "/nmx/v1/services" in lowered:
        return "NVLink service inventory"
    if "/nmx/v1/compute-nodes" in lowered:
        return "NVLink compute node inventory"
    if "/nmx/v1/gpus" in lowered:
        return "NVLink GPU inventory and health"
    if "/nmx/v1/switches" in lowered:
        return "NVSwitch inventory and health"
    if "/nmx/v1/switch-nodes" in lowered:
        return "NVSwitch tray inventory"
    if "/nmx/v1/chassis" in lowered:
        return "NVLink chassis inventory"
    if "/nmx/v1/ports" in lowered:
        return "NVLink port inventory"
    if "/nmx/v1/partitions" in lowered:
        if " -x post " in lowered:
            return "NVLink partition create request"
        if " -x put " in lowered:
            return "NVLink partition update request"
        if " -x delete " in lowered:
            return "NVLink partition delete request"
        return "NVLink partition inventory"
    if "/nmx/v1/operations" in lowered:
        return "NVLink partition operation status"
    if "/nmx/v1/metrics" in lowered:
        return "NVLink telemetry metrics"
    if lowered.startswith("nv show qos roce"):
        return "RoCE QoS state"
    if lowered.startswith("nv show router adaptive-routing"):
        return "adaptive routing global state"
    if "router adaptive-routing" in lowered:
        return "adaptive routing interface state"
    if "router bgp" in lowered or lowered.startswith("vtysh"):
        return "BGP control-plane state"
    if lowered.startswith("ip route") or "/proc/net/fib_trie" in lowered:
        return "routing table state"
    if lowered.startswith("ip a show") or "ipv4" in lowered:
        return "interface addressing state"
    return f"{family} {topic.lower()}"


def _preconditions(command: str, family: str, plane: str, entry_type: str) -> str:
    if entry_type == "tip":
        return "raw HTML source available"
    lowered = command.lower()
    if family == "infiniband":
        if any(lowered.startswith(prefix) for prefix in ("ib", "saquery", "perfquery")):
            return "--ib-mgmt-host provided and OFED/rdma-core tools installed on that host"
    if family == "nvlink":
        return "--nmx-url provided and endpoint reachable"
    if family == "spectrum-x":
        if lowered.startswith("nv ") or lowered.startswith("vtysh"):
            return "--cumulus-hosts provided and NVUE/vtysh available on the target switch"
    return f"{plane} access available"


def _artifact_target(family: str, plane: str, entry_type: str) -> str:
    if entry_type == "tip":
        return f"fabric_handbook.{family}"
    return f"fabric_verification.{family}.{plane}"


def _entry_id(source_doc: str, text: str, entry_type: str) -> str:
    token = hashlib.sha1(f"{source_doc}|{entry_type}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{entry_type}_{token}"


def _build_entry(
    *,
    source_doc: str,
    family: str,
    topic: str,
    plane: str,
    text: str,
    entry_type: str,
) -> dict[str, Any]:
    command = text if entry_type == "command" else ""
    mutates = _mutates_state(command)
    notes = text if entry_type == "tip" else f"Extracted from {topic}"
    return {
        "id": _entry_id(source_doc, text, entry_type),
        "entry_type": entry_type,
        "source_doc": source_doc,
        "fabric_family": family,
        "topic": topic,
        "plane": plane,
        "command": command,
        "mutates_state": mutates,
        "lab_only": mutates,
        "expected_signal": _expected_signal(command or text, family, topic, entry_type),
        "preconditions": _preconditions(command or text, family, plane, entry_type),
        "artifact_target": _artifact_target(family, plane, entry_type),
        "notes": notes,
    }


def _dedupe_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for entry in entries:
        key = (entry["source_doc"], entry["entry_type"], entry["command"] or entry["notes"])
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def generate_catalog_entries(source_root: Path | None = None) -> list[dict[str, Any]]:
    source = (source_root or DEFAULT_SOURCE_ROOT).resolve()
    if not source.exists():
        raise FileNotFoundError(f"fabric source root not found: {source}")
    entries: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*.html")):
        rel = _relative_doc_path(path, source)
        family = _family_from_path(path)
        topic = _topic_from_path(path)
        plane = _plane_for(topic)
        lines = _html_to_lines(path)
        for line in lines:
            if _looks_like_command(line):
                entries.append(
                    _build_entry(
                        source_doc=rel,
                        family=family,
                        topic=topic,
                        plane=plane,
                        text=_normalize_command(line),
                        entry_type="command",
                    )
                )
            elif _looks_like_tip(line):
                entries.append(
                    _build_entry(
                        source_doc=rel,
                        family=family,
                        topic=topic,
                        plane=plane,
                        text=line,
                        entry_type="tip",
                    )
                )
    for entry in _SUPPLEMENTAL_ENTRIES:
        entries.append(
            _build_entry(
                source_doc=str(entry["source_doc"]),
                family=str(entry["family"]),
                topic=str(entry["topic"]),
                plane=str(entry["plane"]),
                text=str(entry["text"]),
                entry_type=str(entry["entry_type"]),
            )
        )
    return _dedupe_entries(entries)


def _load_committed_catalog(*, run_id: str) -> dict[str, Any]:
    if not DEFAULT_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"committed fabric catalog not found: {DEFAULT_CATALOG_PATH}")
    payload = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"committed fabric catalog schema does not match {SCHEMA_VERSION}: {DEFAULT_CATALOG_PATH}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"committed fabric catalog has no entries: {DEFAULT_CATALOG_PATH}")
    payload["run_id"] = run_id or ""
    payload["catalog_source"] = "committed_snapshot"
    return payload


def generate_catalog_payload(source_root: Path | None = None, *, run_id: str = "") -> dict[str, Any]:
    if source_root is None and not DEFAULT_SOURCE_ROOT.exists():
        return _load_committed_catalog(run_id=run_id)

    source = (source_root or DEFAULT_SOURCE_ROOT).resolve()
    entries = generate_catalog_entries(source)
    family_counts = Counter(entry["fabric_family"] for entry in entries)
    entry_type_counts = Counter(entry["entry_type"] for entry in entries)
    lab_only = sum(1 for entry in entries if entry["lab_only"])
    docs = sorted({entry["source_doc"] for entry in entries})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or "",
        "catalog_source": "archived_source_extract",
        "fabric_family": "all",
        "collection_mode": "catalog_extract",
        "status": "ok",
        "completeness": "catalog_ready",
        "evidence_refs": docs,
        "recommendations": [
            "Use the committed catalog as the canonical command/tip index and snapshot it into run artifacts for auditability.",
            "Treat entries marked lab_only=true as training material only; do not execute them in canonical verification paths.",
        ],
        "summary": {
            "entry_count": len(entries),
            "doc_count": len(docs),
            "family_counts": dict(sorted(family_counts.items())),
            "entry_type_counts": dict(sorted(entry_type_counts.items())),
            "lab_only_count": lab_only,
        },
        "entries": entries,
    }


def write_catalog_payload(out_path: Path, payload: dict[str, Any]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
