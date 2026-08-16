from __future__ import annotations

import csv
import json
import os
import re
import ssl
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from cluster.fabric.catalog import SCHEMA_VERSION, generate_catalog_payload

CommandRunner = Callable[..., dict[str, Any]]

_COMPLETENESS_ORDER = {
    "not_present": 0,
    "present_unverified": 1,
    "runtime_verified": 2,
    "full_stack_verified": 3,
}


@dataclass(frozen=True)
class ManagementConfig:
    ib_mgmt_host: str | None
    ib_mgmt_user: str | None
    ib_mgmt_ssh_key: str | None
    nmx_url: str | None
    nmx_token: str | None
    cumulus_hosts: tuple[str, ...]
    cumulus_user: str | None
    cumulus_ssh_key: str | None


def _clean_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = [str(item) for item in value]
    return [item.strip() for item in items if item and item.strip()]


def make_management_config(
    *,
    ib_mgmt_host: str | None = None,
    ib_mgmt_user: str | None = None,
    ib_mgmt_ssh_key: str | None = None,
    nmx_url: str | None = None,
    nmx_token: str | None = None,
    cumulus_hosts: str | list[str] | tuple[str, ...] | None = None,
    cumulus_user: str | None = None,
    cumulus_ssh_key: str | None = None,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
) -> ManagementConfig:
    return ManagementConfig(
        ib_mgmt_host=(ib_mgmt_host or "").strip() or None,
        ib_mgmt_user=(ib_mgmt_user or ssh_user or "").strip() or None,
        ib_mgmt_ssh_key=(ib_mgmt_ssh_key or ssh_key or "").strip() or None,
        nmx_url=(nmx_url or "").strip() or None,
        nmx_token=(nmx_token or "").strip() or None,
        cumulus_hosts=tuple(_clean_csv(cumulus_hosts)),
        cumulus_user=(cumulus_user or ssh_user or "cumulus").strip() or None,
        cumulus_ssh_key=(cumulus_ssh_key or ssh_key or "").strip() or None,
    )


def _is_local_host(host: str | None) -> bool:
    if not host:
        return True
    host = host.strip()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    current = {os.uname().nodename}
    current.add(os.uname().nodename.split(".")[0])
    return host in current


def default_command_runner(
    command: str,
    *,
    host: str | None = None,
    user: str | None = None,
    ssh_key: str | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    if _is_local_host(host):
        cmd = ["bash", "-lc", command]
    else:
        ssh_cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ConnectionAttempts=2",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none",
        ]
        if ssh_key:
            ssh_cmd.extend(["-i", ssh_key])
        target = f"{user}@{host}" if user else str(host)
        ssh_cmd.extend([target, f"bash -lc {subprocess.list2cmdline([command])}"])
        cmd = ssh_cmd
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "host": host or "localhost",
            "command": command,
            "returncode": None,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
            "error": f"timeout after {timeout}s",
        }
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "host": host or "localhost",
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "error": "" if proc.returncode == 0 else f"returncode={proc.returncode}",
    }


def _relative_to_run_dir(run_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _command_stdout(meta_payload: dict[str, Any], name: str) -> str:
    commands = meta_payload.get("commands") or {}
    payload = commands.get(name) or {}
    return str(payload.get("stdout") or "")


def _max_nccl_metric(path: Path, key: str) -> float:
    payload = _load_json_optional(path) or {}
    rows = payload.get("results") or []
    best = 0.0
    for row in rows:
        try:
            best = max(best, float(row.get(key) or 0.0))
        except Exception:
            continue
    return best


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _overall_completeness(families: dict[str, dict[str, Any]]) -> str:
    present_levels = [
        _COMPLETENESS_ORDER.get(str(payload.get("completeness")), 0)
        for payload in families.values()
        if bool(payload.get("present"))
    ]
    if not present_levels:
        return "not_present"
    min_level = min(present_levels)
    for label, value in _COMPLETENESS_ORDER.items():
        if value == min_level:
            return label
    return "present_unverified"


def _result_state(results: list[dict[str, Any]]) -> str:
    if not results:
        return "not_configured"
    if all(result.get("status") == "ok" for result in results):
        return "ok"
    if any(result.get("status") == "ok" for result in results):
        return "partial"
    return "error"


def _command_record(name: str, result: dict[str, Any], *, notes: str = "") -> dict[str, Any]:
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    return {
        "name": name,
        "status": result.get("status", "error"),
        "host": result.get("host"),
        "command": result.get("command"),
        "returncode": result.get("returncode"),
        "stdout_excerpt": "\n".join(stdout.splitlines()[:20]),
        "stderr_excerpt": "\n".join(stderr.splitlines()[:20]),
        "stdout_line_count": len([line for line in stdout.splitlines() if line.strip()]),
        "stderr_line_count": len([line for line in stderr.splitlines() if line.strip()]),
        "error": result.get("error"),
        "notes": notes,
    }


def _http_json(url: str, token: str | None = None, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("X-Auth-Token", token)
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read().decode("utf-8", errors="ignore")
            return {
                "status": "ok",
                "http_status": response.status,
                "json": json.loads(raw) if raw else {},
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {"status": "error", "http_status": exc.code, "json": {}, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - exercised via command-path tests instead
        return {"status": "error", "http_status": None, "json": {}, "error": str(exc)}


def _http_text(url: str, token: str | None = None, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("X-Auth-Token", token)
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read().decode("utf-8", errors="ignore")
            return {
                "status": "ok",
                "http_status": response.status,
                "text": raw,
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {"status": "error", "http_status": exc.code, "text": "", "error": str(exc)}
    except Exception as exc:  # pragma: no cover - exercised via command-path tests instead
        return {"status": "error", "http_status": None, "text": "", "error": str(exc)}


def _nmx_api_base(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/nmx/v1"):
        return base
    if base.endswith("/nmx"):
        return f"{base}/v1"
    return f"{base}/nmx/v1"


def _json_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _str_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _safe_len(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value)
    return 0


def _location_info(payload: dict[str, Any]) -> dict[str, Any]:
    location = payload.get("LocationInfo") or {}
    if not isinstance(location, dict):
        location = {}
    return {
        "chassis_id": location.get("ChassisID"),
        "slot_id": location.get("SlotID"),
        "host_id": location.get("HostID"),
        "tray_index": location.get("TrayIndex"),
        "chassis_serial_number": location.get("ChassisSerialNumber"),
    }


def _location_key(payload: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    location = _location_info(payload)
    return (
        location.get("chassis_id"),
        location.get("slot_id"),
        location.get("host_id"),
        location.get("tray_index"),
    )


def _gpu_location_string(gpu: dict[str, Any]) -> str | None:
    location = _location_info(gpu)
    parts = [
        location.get("chassis_id"),
        location.get("slot_id"),
        location.get("host_id"),
        gpu.get("DeviceID"),
    ]
    if any(part in (None, "") for part in parts):
        return None
    return ".".join(str(part) for part in parts)


def _health_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        health = str(item.get("Health") or "UNKNOWN")
        counts[health] = counts.get(health, 0) + 1
    return dict(sorted(counts.items()))


def _sample_gpu_fields(gpus: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not gpus:
        return None
    gpu = gpus[0]
    return {
        "id": gpu.get("ID"),
        "name": gpu.get("Name"),
        "device_id": gpu.get("DeviceID"),
        "device_uid": gpu.get("DeviceUID"),
        "system_uid": gpu.get("SystemUID"),
        "domain_uuid": gpu.get("DomainUUID"),
        "partition_id": gpu.get("PartitionID"),
        "partition_name": gpu.get("PartitionName"),
        "health": gpu.get("Health"),
        "internal_description": gpu.get("InternalDescription"),
        "description": gpu.get("Description"),
        "alid_count": _safe_len(gpu.get("ALIDList")),
        "port_count": _safe_len(gpu.get("PortIDList")),
        "location": _location_info(gpu),
    }


def _sample_compute_node_fields(compute_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not compute_nodes:
        return None
    node = compute_nodes[0]
    return {
        "id": node.get("ID") or node.get("id"),
        "name": node.get("Name"),
        "system_uid": node.get("SystemUID"),
        "device_uid": node.get("DeviceUID"),
        "gpu_id_count": _safe_len(node.get("GpuIDList")),
        "location": _location_info(node),
    }


def _sample_switch_fields(switches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not switches:
        return None
    switch = switches[0]
    return {
        "id": switch.get("ID"),
        "device_id": switch.get("DeviceID"),
        "domain_uuid": switch.get("DomainUUID"),
        "health": switch.get("Health"),
        "port_count": _safe_len(switch.get("PortIDList")),
        "location": _location_info(switch),
    }


def _sample_chassis_fields(chassis: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not chassis:
        return None
    item = chassis[0]
    return {
        "id": item.get("ID") or item.get("id"),
        "name": item.get("Name"),
        "domain_uuid": item.get("DomainUUID"),
        "health": item.get("Health"),
        "location": _location_info(item),
    }


def _candidate_team_allocations(node_gpu_map: list[dict[str, Any]], *, team_size: int = 4) -> dict[str, Any]:
    remaining = [dict(item) for item in node_gpu_map]

    def take_nodes(team_name: str) -> dict[str, Any]:
        selected_nodes: list[dict[str, Any]] = []
        gpu_locations: list[str] = []
        for node in remaining:
            if len(gpu_locations) >= team_size:
                break
            selected_nodes.append(
                {
                    "node": node.get("node_name"),
                    "system_uid": node.get("system_uid"),
                    "gpu_count": node.get("gpu_count"),
                }
            )
            gpu_locations.extend(node.get("gpu_locations") or [])
        used = set(item.get("system_uid") for item in selected_nodes if item.get("system_uid"))
        if used:
            remaining[:] = [node for node in remaining if node.get("system_uid") not in used]
        return {
            "team": team_name,
            "target_gpu_count": team_size,
            "selected_gpu_count": min(len(gpu_locations), team_size),
            "satisfied": len(gpu_locations) >= team_size,
            "nodes": selected_nodes,
            "gpu_locations": gpu_locations[:team_size],
        }

    alpha = take_nodes("Alpha")
    beta = take_nodes("Beta")
    return {
        "team_alpha": alpha,
        "team_beta": beta,
        "can_support_two_teams": alpha["satisfied"] and beta["satisfied"],
    }


def _member_locations(partition: dict[str, Any]) -> list[str]:
    members = partition.get("Members") or {}
    if isinstance(members, dict):
        locations = members.get("locations")
        if isinstance(locations, list):
            return [str(item) for item in locations if item not in (None, "")]
        gpu_ids = members.get("gpuIds")
        if isinstance(gpu_ids, list):
            return [str(item) for item in gpu_ids if item not in (None, "")]
    for key in ("MemberLocationList", "MemberGPUIdList", "GPUList"):
        value = partition.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
    return []


def _build_nmx_summary(
    *,
    api_base: str,
    services: list[dict[str, Any]],
    compute_nodes: list[dict[str, Any]],
    gpus: list[dict[str, Any]],
    switches: list[dict[str, Any]],
    switch_nodes: list[dict[str, Any]],
    chassis: list[dict[str, Any]],
    ports: list[dict[str, Any]],
    partitions: list[dict[str, Any]],
    metrics_text: str,
) -> dict[str, Any]:
    service_ids = sorted(
        str(item.get("ID") or item.get("id") or item.get("ServiceID") or "")
        for item in services
        if item.get("ID") or item.get("id") or item.get("ServiceID")
    )
    domain_uuids = sorted(
        {
            str(item.get("DomainUUID"))
            for item in [*services, *gpus, *switches, *switch_nodes, *chassis, *partitions]
            if item.get("DomainUUID")
        }
    )
    chassis_rows = [
        {
            "id": item.get("ID") or item.get("id"),
            "name": item.get("Name"),
            "health": item.get("Health"),
            "location": _location_info(item),
        }
        for item in chassis
    ]
    chassis_serial_numbers = sorted(
        {
            str(location.get("chassis_serial_number"))
            for location in (_location_info(item) for item in [*compute_nodes, *gpus, *switches, *switch_nodes, *chassis])
            if location.get("chassis_serial_number")
        }
    )
    node_gpu_grouping_field = "GpuIDList" if any(node.get("GpuIDList") for node in compute_nodes) else "SystemUID"

    gpu_by_id: dict[str, dict[str, Any]] = {
        str(gpu.get("ID")): gpu for gpu in gpus if gpu.get("ID") not in (None, "")
    }
    gpu_groups: dict[str, list[dict[str, Any]]] = {}
    for gpu in gpus:
        system_uid = _str_or_none(gpu.get("SystemUID")) or _str_or_none(gpu.get("ComputeNodeID")) or _str_or_none(gpu.get("SystemId"))
        if not system_uid:
            continue
        gpu_groups.setdefault(system_uid, []).append(gpu)

    node_gpu_map: list[dict[str, Any]] = []
    for node in compute_nodes:
        system_uid = _str_or_none(node.get("SystemUID")) or _str_or_none(node.get("DeviceUID")) or _str_or_none(node.get("ID"))
        node_gpu_ids = [str(item) for item in (node.get("GpuIDList") or []) if item not in (None, "")]
        if node_gpu_ids:
            node_gpus = [gpu_by_id[gpu_id] for gpu_id in node_gpu_ids if gpu_id in gpu_by_id]
        else:
            node_gpus = gpu_groups.get(system_uid or "", [])
        node_location = _location_info(node)
        node_name = _str_or_none(node.get("Name")) or _str_or_none(node.get("Description")) or system_uid or "unknown-node"
        node_gpu_map.append(
            {
                "node_name": node_name,
                "system_uid": system_uid,
                "location": node_location,
                "gpu_count": len(node_gpus),
                "gpu_device_ids": sorted(str(gpu.get("DeviceID")) for gpu in node_gpus if gpu.get("DeviceID") not in (None, "")),
                "gpu_locations": [loc for gpu in node_gpus if (loc := _gpu_location_string(gpu))],
            }
        )
    node_gpu_map.sort(key=lambda item: (item["location"].get("chassis_id") or 0, item["location"].get("slot_id") or 0, item["node_name"]))

    switch_node_map: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {
        _location_key(node): node for node in switch_nodes
    }
    switch_groups: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for switch in switches:
        switch_groups.setdefault(_location_key(switch), []).append(switch)

    switch_trays: list[dict[str, Any]] = []
    for key, members in sorted(switch_groups.items(), key=lambda item: tuple(v or 0 for v in item[0])):
        switch_node = switch_node_map.get(key) or {}
        location = _location_info(members[0]) if members else {}
        switch_trays.append(
            {
                "location": location,
                "switch_asic_count": len(members),
                "switch_device_ids": sorted(str(item.get("DeviceID")) for item in members if item.get("DeviceID") not in (None, "")),
                "switch_ids": sorted(str(item.get("ID")) for item in members if item.get("ID") not in (None, "")),
                "switch_node_id": switch_node.get("ID"),
                "switch_node_switch_id_list": switch_node.get("SwitchIDList") or [],
            }
        )
    sample_switch_tray = next((tray for tray in switch_trays if int(tray.get("switch_asic_count") or 0) >= 2), switch_trays[0] if switch_trays else None)

    gpu_facing_ports = [port for port in ports if str(port.get("Type") or "").upper() == "GPU"]
    switch_facing_ports = [port for port in ports if str(port.get("Type") or "").upper() == "SWITCH_ACCESS"]
    expected_total_ports = len(gpus) * len(switches) * 2 if gpus and switches else 0
    ports_summary = {
        "total_ports": len(ports),
        "gpu_facing_ports": len(gpu_facing_ports),
        "switch_facing_ports": len(switch_facing_ports),
        "gpu_ports_with_base_lid": sum(1 for port in gpu_facing_ports if port.get("BaseLID") not in (None, "")),
        "switch_ports_with_base_lid": sum(1 for port in switch_facing_ports if port.get("BaseLID") not in (None, "")),
        "gpu_facing_ports_have_base_lid": bool(gpu_facing_ports) and all(port.get("BaseLID") not in (None, "") for port in gpu_facing_ports),
        "switch_facing_ports_have_base_lid": bool(switch_facing_ports) and all(port.get("BaseLID") not in (None, "") for port in switch_facing_ports),
        "expected_total_ports": expected_total_ports,
        "expected_formula": f"2 * gpu_count({len(gpus)}) * switch_asic_count({len(switches)}) = {expected_total_ports}" if expected_total_ports else "n/a",
        "matches_expected_formula": bool(expected_total_ports) and len(ports) == expected_total_ports,
    }

    partition_rows: list[dict[str, Any]] = []
    default_partition: dict[str, Any] | None = None
    for partition in partitions:
        locations = _member_locations(partition)
        row = {
            "id": partition.get("ID") or partition.get("id"),
            "partition_id": partition.get("PartitionID"),
            "name": partition.get("Name"),
            "type": partition.get("Type"),
            "member_count": len(locations),
            "member_locations": locations,
        }
        partition_rows.append(row)
        name = str(partition.get("Name") or "").lower()
        if partition.get("PartitionID") == 32766 or name == "default" or "default" in name:
            default_partition = row
    unassigned_gpu_count = sum(
        1
        for gpu in gpus
        if gpu.get("PartitionID") in (None, "", 0) or str(gpu.get("PartitionName") or "").lower() == "unassigned"
    )
    unassigned_gpu_locations = [
        location
        for gpu in gpus
        if gpu.get("PartitionID") in (None, "", 0) or str(gpu.get("PartitionName") or "").lower() == "unassigned"
        if (location := _gpu_location_string(gpu))
    ]

    metric_patterns = {
        "switch_temperature_series": r"^switch_temperature\b",
        "tx_throughput_series": r"^PortXmitDataExtended\b",
        "rx_throughput_series": r"^PortRcvDataExtended\b",
        "physical_error_series": r"^PortLocalPhysicalErrors\b",
        "cable_temperature_series": r"^CableInfoTemperature\b",
        "cable_rx_power_series": r"^CableInfoRxPower\b",
        "cable_tx_power_series": r"^CableInfoTxPower\b",
    }
    telemetry_summary = {name: len(re.findall(pattern, metrics_text, flags=re.MULTILINE)) for name, pattern in metric_patterns.items()}
    telemetry_summary["sample_queries"] = [
        'curl -sk <nmx_api_base>/metrics | grep "^switch_temperature" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^PortXmitDataExtended" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^PortRcvDataExtended" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^PortLocalPhysicalErrors" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^CableInfoTemperature" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^CableInfoRxPower" | head -5',
        'curl -sk <nmx_api_base>/metrics | grep "^CableInfoTxPower" | head -5',
    ]

    allocations = _candidate_team_allocations(node_gpu_map, team_size=4)
    return {
        "api_base": api_base,
        "services": {
            "service_count": len(services),
            "service_ids": service_ids[:8],
            "domain_uuids": domain_uuids,
        },
        "topology": {
            "chassis_count": len(chassis_rows) or len({item.get("location", {}).get("chassis_id") for item in node_gpu_map if item.get("location", {}).get("chassis_id") is not None}),
            "chassis_serial_numbers": chassis_serial_numbers,
            "compute_node_count": len(compute_nodes),
            "gpu_count": len(gpus),
            "switch_asic_count": len(switches),
            "switch_tray_count": len(switch_trays) if switch_trays else len(switch_nodes),
            "port_count": len(ports),
            "gpu_health_counts": _health_counts(gpus),
            "switch_health_counts": _health_counts(switches),
            "compute_nodes": node_gpu_map,
            "chassis": chassis_rows,
            "switch_trays": switch_trays,
            "ports": ports_summary,
            "sample_compute_node": _sample_compute_node_fields(compute_nodes),
            "sample_gpu": _sample_gpu_fields(gpus),
            "sample_switch": _sample_switch_fields(switches),
            "sample_chassis": _sample_chassis_fields(chassis),
            "scenario_answers": {
                "can_support_two_concurrent_4gpu_workloads": allocations["can_support_two_teams"],
                "team_alpha_candidate": allocations["team_alpha"],
                "team_beta_candidate": allocations["team_beta"],
                "node_gpu_grouping_field": node_gpu_grouping_field,
                "switch_asic_distinguishing_field": "DeviceID",
                "switch_tray_grouping_fields": [
                    "LocationInfo.ChassisID",
                    "LocationInfo.SlotID",
                    "LocationInfo.HostID",
                    "LocationInfo.TrayIndex",
                ],
                "sample_switch_tray": sample_switch_tray,
                "port_formula_matches_inventory": ports_summary["matches_expected_formula"],
            },
        },
        "partitions": {
            "partition_count": len(partitions),
            "default_partition": default_partition,
            "unassigned_gpu_count": unassigned_gpu_count,
            "unassigned_gpu_locations": unassigned_gpu_locations,
            "partitions": partition_rows,
            "scenario_answers": {
                "ready_for_new_partition_create": unassigned_gpu_count > 0,
                "default_partition_present": default_partition is not None,
                "default_partition_member_count": int((default_partition or {}).get("member_count") or 0),
                "inspect_command": f"curl -k {api_base}/partitions | jq",
                "create_command": (
                    "curl -k -X POST <nmx_api_base>/partitions "
                    "-H \"Content-Type: application/json\" "
                    "-d '{\"Name\":\"<partition-name>\",\"DomainUUID\":\"<DomainUUID>\","
                    "\"Members\":{\"locations\":[\"<chassis.slot.host.gpu>\"]}}' | jq"
                ),
                "update_command": (
                    "curl -k -X PUT <nmx_api_base>/partitions/<partition-id> "
                    "-H \"Content-Type: application/json\" "
                    "-d '{\"DomainUUID\":\"<DomainUUID>\","
                    "\"Members\":{\"locations\":[\"<chassis.slot.host.gpu>\"]}}' | jq"
                ),
                "delete_command": "curl -k -X DELETE <nmx_api_base>/partitions/<partition-id> | jq",
                "operation_poll_path": f"{api_base}/operations/<operation-id>",
                "update_flow": [
                    "Remove GPUs from the source partition first.",
                    "Poll the returned operationId until status=completed.",
                    "Update the target partition only after the GPUs are free.",
                    "DELETE frees the partition's GPUs back to the unassigned/default pool.",
                ],
            },
        },
        "telemetry": {
            **telemetry_summary,
            "metrics_endpoint": f"{api_base}/metrics",
        },
    }


def _first_numbers_from_text(text: str, *, pattern: str, limit: int = 2) -> list[str]:
    out: list[str] = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        out.append(match.group(1))
        if len(out) >= limit:
            break
    return out


def _meta_records(structured_dir: Path, run_id: str, labels: list[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    preferred = set(labels or [])
    for path in sorted(structured_dir.glob(f"{run_id}_*_meta.json")):
        if path.name.endswith("_cluster_meta.json"):
            continue
        label = path.name[len(f"{run_id}_") : -len("_meta.json")]
        if preferred and label not in preferred:
            continue
        records.append({"label": label, "path": path, "payload": _load_json(path)})
    return records


def _detect_nvlink(record: dict[str, Any], run_id: str, run_dir: Path) -> dict[str, Any]:
    label = str(record["label"])
    summary_path = run_dir / "structured" / f"{run_id}_{label}_meta_nvlink_topology.json"
    summary_payload = _load_json_optional(summary_path) or {}
    summary = summary_payload.get("summary") or {}
    topo = _command_stdout(record["payload"], "nvidia_smi_topo")
    pair_count = int(summary.get("nvlink_pair_count") or 0)
    present = pair_count > 0 or " NV" in topo or "\tNV" in topo
    evidence_refs = [_relative_to_run_dir(run_dir, record["path"])]
    if summary_path.exists():
        evidence_refs.append(_relative_to_run_dir(run_dir, summary_path))
    return {
        "present": present,
        "gpu_count": int(summary.get("gpu_count") or 0),
        "nvlink_pair_count": pair_count,
        "management_plane_expected": present,
        "evidence_refs": evidence_refs,
        "signals": {
            "topology_contains_nvlink": " NV" in topo or "\tNV" in topo,
            "nvlink_summary_present": summary_path.exists(),
        },
    }


def _detect_infiniband(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    payload = record["payload"]
    ibstat = _command_stdout(payload, "ibstat")
    rdma_link = _command_stdout(payload, "rdma_link")
    ibv_devinfo = _command_stdout(payload, "ibv_devinfo")
    present = bool(
        re.search(r"link layer:\s*infiniband", ibstat, flags=re.IGNORECASE)
        or re.search(r"\bmlx5_\d+\b", rdma_link)
        or re.search(r"\bmlx5_\d+\b", ibv_devinfo)
    )
    hcas = sorted(set(re.findall(r"\bmlx5_\d+\b", "\n".join([ibstat, rdma_link, ibv_devinfo]))))
    return {
        "present": present,
        "hcas": hcas,
        "management_plane_expected": present,
        "evidence_refs": [_relative_to_run_dir(run_dir, record["path"])],
        "signals": {
            "ibstat_active": "active" in ibstat.lower(),
            "rdma_link_present": bool(rdma_link.strip()),
            "ibv_devinfo_present": bool(ibv_devinfo.strip()),
        },
    }


def _detect_spectrum_x(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    payload = record["payload"]
    ibstat = _command_stdout(payload, "ibstat")
    rdma_link = _command_stdout(payload, "rdma_link")
    ibv_devinfo = _command_stdout(payload, "ibv_devinfo")
    ethtool = _command_stdout(payload, "ethtool")
    high_speed_eth = any(int(speed) >= 100000 for speed in re.findall(r"Speed:\s*(\d+)Mb/s", ethtool))
    roce_like = "ethernet" in ibstat.lower() or "netdev" in rdma_link.lower()
    present = bool(high_speed_eth and (roce_like or "mlx5" in "\n".join([rdma_link, ibv_devinfo, ethtool]).lower()))
    return {
        "present": present,
        "management_plane_expected": present,
        "evidence_refs": [_relative_to_run_dir(run_dir, record["path"])],
        "signals": {
            "high_speed_eth": high_speed_eth,
            "rdma_over_ethernet": roce_like,
        },
    }


def _build_capability_matrix(
    *,
    run_id: str,
    structured_dir: Path,
    run_dir: Path,
    labels: list[str] | None,
    management: ManagementConfig,
) -> dict[str, Any]:
    records = _meta_records(structured_dir, run_id, labels)
    if not records:
        raise FileNotFoundError(f"no meta artifacts found under {structured_dir} for run_id={run_id}")

    primary = records[0]
    families = {
        "nvlink": _detect_nvlink(primary, run_id, run_dir),
        "infiniband": _detect_infiniband(primary, run_dir),
        "spectrum-x": _detect_spectrum_x(primary, run_dir),
    }
    families["nvlink"]["management_plane_configured"] = bool(management.nmx_url)
    families["infiniband"]["management_plane_configured"] = bool(management.ib_mgmt_host)
    families["spectrum-x"]["management_plane_configured"] = bool(management.cumulus_hosts)

    for payload in families.values():
        payload["completeness"] = "present_unverified" if payload["present"] else "not_present"

    evidence_refs = []
    for payload in families.values():
        evidence_refs.extend(payload.get("evidence_refs", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "fabric_family": "all",
        "collection_mode": "capability_discovery",
        "status": "ok",
        "completeness": _overall_completeness(families),
        "evidence_refs": sorted(set(evidence_refs)),
        "recommendations": [],
        "families": families,
    }


def _collect_ib_control_plane(
    *,
    present: bool,
    management: ManagementConfig,
    runner: CommandRunner,
    ssh_user: str | None,
    ssh_key: str | None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not present:
        return [], "not_present", []
    host = management.ib_mgmt_host
    if not host:
        return [], "not_configured", ["Pass --ib-mgmt-host to unlock InfiniBand fabric CLI verification."]

    user = management.ib_mgmt_user or ssh_user
    key = management.ib_mgmt_ssh_key or ssh_key
    checks: list[tuple[str, str]] = [
        ("ibstat", "ibstat"),
        ("ibswitches", "ibswitches"),
        ("ibhosts", "ibhosts"),
        ("iblinkinfo", "iblinkinfo | head -n 80"),
        ("ibnetdiscover", "ibnetdiscover | head -n 80"),
        ("saquery", "saquery"),
        ("ibdiagnet", "ibdiagnet -r"),
    ]
    results: list[dict[str, Any]] = []
    host_lids: list[str] = []
    switch_lids: list[str] = []
    for name, command in checks:
        result = runner(command, host=host, user=user, ssh_key=key, timeout=25)
        results.append(_command_record(name, result))
        stdout = str(result.get("stdout") or "")
        if name == "ibhosts":
            host_lids.extend(_first_numbers_from_text(stdout, pattern=r"\blid\s+(\d+)\b", limit=4))
        if name == "ibswitches":
            switch_lids.extend(_first_numbers_from_text(stdout, pattern=r"\blid\s+(\d+)\b", limit=2))
    if host_lids:
        result = runner(f"ibaddr -l {host_lids[0]}", host=host, user=user, ssh_key=key, timeout=20)
        results.append(_command_record("ibaddr", result))
    if len(host_lids) >= 2:
        result = runner(f"ibtracert {host_lids[0]} {host_lids[1]}", host=host, user=user, ssh_key=key, timeout=20)
        results.append(_command_record("ibtracert", result))
    if switch_lids:
        route_result = runner(f"ibroute {switch_lids[0]}", host=host, user=user, ssh_key=key, timeout=20)
        results.append(_command_record("ibroute", route_result))
        perf_result = runner(f"perfquery -x {switch_lids[0]} 1", host=host, user=user, ssh_key=key, timeout=20)
        results.append(_command_record("perfquery", perf_result))
    status = _result_state(results)
    recommendations = []
    if status != "ok":
        recommendations.append("Use a UFM/IB management node with OFED tools installed to complete InfiniBand routing and counter verification.")
    return results, status, recommendations


def fetch_nmx_snapshot(nmx_url: str, nmx_token: str | None = None) -> dict[str, Any]:
    api_base = _nmx_api_base(nmx_url)
    endpoints = [
        ("services", f"{api_base}/services"),
        ("compute_nodes", f"{api_base}/compute-nodes"),
        ("gpus", f"{api_base}/gpus"),
        ("switches", f"{api_base}/switches"),
        ("switch_nodes", f"{api_base}/switch-nodes"),
        ("chassis", f"{api_base}/chassis"),
        ("ports", f"{api_base}/ports"),
        ("partitions", f"{api_base}/partitions"),
    ]
    records: list[dict[str, Any]] = []
    payloads: dict[str, list[dict[str, Any]]] = {}
    for name, url in endpoints:
        result = _http_json(url, nmx_token)
        payload = _json_list(result.get("json"))
        payloads[name] = payload
        records.append(
            {
                "name": name,
                "status": result["status"],
                "url": url,
                "http_status": result.get("http_status"),
                "result_size": len(payload),
                "error": result.get("error", ""),
            }
        )
    metrics_result = _http_text(f"{api_base}/metrics", nmx_token)
    metrics_text = str(metrics_result.get("text") or "")
    records.append(
        {
            "name": "metrics",
            "status": metrics_result["status"],
            "url": f"{api_base}/metrics",
            "http_status": metrics_result.get("http_status"),
            "result_size": len(metrics_text.splitlines()) if metrics_text else 0,
            "error": metrics_result.get("error", ""),
        }
    )
    status = _result_state(records)
    summary = _build_nmx_summary(
        api_base=api_base,
        services=payloads.get("services") or [],
        compute_nodes=payloads.get("compute_nodes") or [],
        gpus=payloads.get("gpus") or [],
        switches=payloads.get("switches") or [],
        switch_nodes=payloads.get("switch_nodes") or [],
        chassis=payloads.get("chassis") or [],
        ports=payloads.get("ports") or [],
        partitions=payloads.get("partitions") or [],
        metrics_text=metrics_text,
    )
    recommendations = []
    if status != "ok":
        recommendations.append("Verify the NMX endpoint, token, and TLS reachability for NetQ NVLink management-plane checks.")
    if not payloads.get("compute_nodes") or not payloads.get("gpus") or not payloads.get("switches"):
        recommendations.append("NMX topology exploration is incomplete; capacity-planning answers require compute-nodes, gpus, and switches endpoints.")
    if not payloads.get("partitions"):
        recommendations.append("NMX partition inventory is unavailable; tenant carve-up workflows remain unverified until /partitions responds.")
    if metrics_result.get("status") != "ok":
        recommendations.append("NMX telemetry metrics are unavailable; switch temperature, throughput, BER, and cable diagnostics remain unverified.")
    return {
        "api_base": api_base,
        "records": records,
        "payloads": payloads,
        "metrics_text": metrics_text,
        "status": status,
        "summary": summary,
        "recommendations": sorted(dict.fromkeys(recommendations)),
    }


def _collect_nvlink_management(
    management: ManagementConfig,
) -> tuple[list[dict[str, Any]], str, list[str], dict[str, Any] | None]:
    if not management.nmx_url:
        return [], "not_configured", ["Pass --nmx-url to the fabric CLI entrypoint to enable NVLink NetQ/NMX verification."], None
    snapshot = fetch_nmx_snapshot(management.nmx_url, management.nmx_token)
    return snapshot["records"], snapshot["status"], snapshot["recommendations"], snapshot["summary"]


def _partition_post_command(*, api_base: str, name: str, domain_uuid: str, locations: list[str]) -> str:
    members = json.dumps({"locations": locations}, separators=(",", ":"))
    return (
        f"curl -k -X POST {api_base}/partitions "
        f'-H "Content-Type: application/json" '
        f"-d '{json.dumps({'Name': name, 'DomainUUID': domain_uuid, 'Members': json.loads(members)}, separators=(',', ':'))}' | jq"
    )


def _partition_put_command(*, api_base: str, partition_id: str, domain_uuid: str, locations: list[str]) -> str:
    return (
        f"curl -k -X PUT {api_base}/partitions/{partition_id} "
        f'-H "Content-Type: application/json" '
        f"-d '{json.dumps({'DomainUUID': domain_uuid, 'Members': {'locations': locations}}, separators=(',', ':'))}' | jq"
    )


def _partition_delete_command(*, api_base: str, partition_id: str) -> str:
    return f"curl -k -X DELETE {api_base}/partitions/{partition_id} | jq"


def build_nmx_partition_lab_payload(
    *,
    nmx_url: str,
    nmx_token: str | None = None,
    alpha_name: str = "AlphaPartition",
    beta_name: str = "BetaPartition",
    alpha_size: int = 4,
    beta_size: int = 4,
) -> dict[str, Any]:
    snapshot = fetch_nmx_snapshot(nmx_url, nmx_token)
    summary = snapshot["summary"]
    topology = summary.get("topology") or {}
    partitions = summary.get("partitions") or {}
    api_base = str(snapshot["api_base"])
    domain_uuid = ""
    service_info = summary.get("services") or {}
    domain_uuids = service_info.get("domain_uuids") or []
    if domain_uuids:
        domain_uuid = str(domain_uuids[0])

    unassigned = list(partitions.get("unassigned_gpu_locations") or [])
    alpha_locations = unassigned[:alpha_size]
    beta_locations = unassigned[alpha_size : alpha_size + beta_size]
    ready_for_create = len(alpha_locations) == alpha_size and len(beta_locations) == beta_size and bool(domain_uuid)

    existing_partitions = partitions.get("partitions") or []
    beta_source = next((item for item in existing_partitions if str(item.get("name") or "").lower() == beta_name.lower()), None)
    alpha_source = next((item for item in existing_partitions if str(item.get("name") or "").lower() == alpha_name.lower()), None)
    borrow_count = min(2, len(beta_locations))
    borrowed_locations = beta_locations[:borrow_count]
    beta_after_borrow = beta_locations[borrow_count:]
    alpha_after_borrow = alpha_locations + borrowed_locations

    recommendations = list(snapshot.get("recommendations") or [])
    if not ready_for_create:
        recommendations.append(
            "Not enough unassigned GPUs were visible to create both lab partitions directly; free GPUs from the default/source partition first."
        )
    if not domain_uuid:
        recommendations.append("No DomainUUID was discovered from NMX services/inventory; fill it explicitly before issuing lab partition commands.")

    commands = {
        "inspect_partitions": f"curl -k {api_base}/partitions | jq",
        "poll_operation": f"curl -k {api_base}/operations/<operation-id> | jq",
        "verify_partitions": f"curl -k {api_base}/partitions | jq",
        "verify_gpus": f"curl -k {api_base}/gpus | jq",
    }
    if ready_for_create:
        commands["create_alpha"] = _partition_post_command(
            api_base=api_base,
            name=alpha_name,
            domain_uuid=domain_uuid,
            locations=alpha_locations,
        )
        commands["create_beta"] = _partition_post_command(
            api_base=api_base,
            name=beta_name,
            domain_uuid=domain_uuid,
            locations=beta_locations,
        )
        commands["update_beta_after_borrow"] = _partition_put_command(
            api_base=api_base,
            partition_id=str((beta_source or {}).get("id") or "<beta-partition-id>"),
            domain_uuid=domain_uuid,
            locations=beta_after_borrow,
        )
        commands["update_alpha_after_borrow"] = _partition_put_command(
            api_base=api_base,
            partition_id=str((alpha_source or {}).get("id") or "<alpha-partition-id>"),
            domain_uuid=domain_uuid,
            locations=alpha_after_borrow,
        )
        commands["delete_beta"] = _partition_delete_command(
            api_base=api_base,
            partition_id=str((beta_source or {}).get("id") or "<beta-partition-id>"),
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "fabric_family": "nvlink",
        "collection_mode": "nmx_partition_lab",
        "status": "ok" if snapshot["status"] in {"ok", "partial"} else "error",
        "completeness": "lab_plan_ready" if ready_for_create else "inventory_only",
        "evidence_refs": [record["url"] for record in snapshot["records"]],
        "recommendations": sorted(dict.fromkeys(recommendations)),
        "lab_only": True,
        "api_base": api_base,
        "domain_uuid": domain_uuid,
        "topology": {
            "chassis_count": topology.get("chassis_count", 0),
            "compute_node_count": topology.get("compute_node_count", 0),
            "gpu_count": topology.get("gpu_count", 0),
            "switch_asic_count": topology.get("switch_asic_count", 0),
            "switch_tray_count": topology.get("switch_tray_count", 0),
            "port_count": topology.get("port_count", 0),
            "ports": topology.get("ports") or {},
            "sample_compute_node": topology.get("sample_compute_node"),
            "team_alpha_candidate": (topology.get("scenario_answers") or {}).get("team_alpha_candidate") or {},
            "team_beta_candidate": (topology.get("scenario_answers") or {}).get("team_beta_candidate") or {},
            "node_gpu_grouping_field": (topology.get("scenario_answers") or {}).get("node_gpu_grouping_field"),
            "switch_asic_distinguishing_field": (topology.get("scenario_answers") or {}).get("switch_asic_distinguishing_field"),
            "switch_tray_grouping_fields": (topology.get("scenario_answers") or {}).get("switch_tray_grouping_fields") or [],
            "sample_switch_tray": (topology.get("scenario_answers") or {}).get("sample_switch_tray") or {},
            "sample_gpu": topology.get("sample_gpu"),
            "sample_switch": topology.get("sample_switch"),
            "sample_chassis": topology.get("sample_chassis"),
        },
        "partitions": {
            "existing": existing_partitions,
            "default_partition": partitions.get("default_partition"),
            "unassigned_gpu_count": partitions.get("unassigned_gpu_count", 0),
            "unassigned_gpu_locations": unassigned,
            "alpha_seed_locations": alpha_locations,
            "beta_seed_locations": beta_locations,
            "borrowed_locations": borrowed_locations,
            "alpha_after_borrow_locations": alpha_after_borrow,
            "beta_after_borrow_locations": beta_after_borrow,
            "ready_for_create": ready_for_create,
        },
        "commands": commands,
    }


def _collect_spectrum_control_plane(
    *,
    present: bool,
    management: ManagementConfig,
    runner: CommandRunner,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if not present:
        return [], "not_present", []
    if not management.cumulus_hosts:
        return [], "not_configured", ["Pass --cumulus-hosts to enable Spectrum-X / Cumulus verification."]

    commands = [
        ("adaptive_routing", "nv show router adaptive-routing"),
        ("roce", "nv show qos roce"),
        ("bgp_neighbors", "nv show vrf default router bgp neighbor"),
        ("bgp_summary", "vtysh -c \"show bgp ipv4 unicast summary\""),
        ("bgp_routes", "vtysh -c \"show ip route vrf default bgp\""),
    ]
    records: list[dict[str, Any]] = []
    for host in management.cumulus_hosts:
        for name, command in commands:
            result = runner(command, host=host, user=management.cumulus_user, ssh_key=management.cumulus_ssh_key, timeout=20)
            records.append(_command_record(name, result, notes=f"switch={host}"))
    status = _result_state(records)
    recommendations = []
    if status != "ok":
        recommendations.append("Confirm NVUE/vtysh access on the designated Cumulus switches to verify RoCE, BGP, and adaptive routing.")
    return records, status, recommendations


def _runtime_family_status(
    *,
    family: str,
    present: bool,
    run_id: str,
    run_dir: Path,
) -> tuple[str, dict[str, Any], list[str]]:
    structured = run_dir / "structured"
    evidence: dict[str, Any] = {}
    refs: list[str] = []
    if not present:
        return "not_present", evidence, refs

    single_nccl = structured / f"{run_id}_node1_nccl.json"
    multi_nccl = structured / f"{run_id}_2nodes_nccl.json"
    nccl_env = structured / f"{run_id}_nccl_env_sensitivity.json"
    connectivity = structured / f"{run_id}_torchrun_connectivity_probe.json"
    alltoall = structured / f"{run_id}_2nodes_alltoall_nccl_alltoall.json"

    if family == "nvlink":
        refs.extend([_relative_to_run_dir(run_dir, p) for p in (single_nccl, nccl_env) if p.exists()])
        evidence["single_nccl_peak_busbw_gbps"] = _max_nccl_metric(single_nccl, "busbw_gbps")
        evidence["single_nccl_peak_algbw_gbps"] = _max_nccl_metric(single_nccl, "algbw_gbps")
        evidence["nccl_env_status"] = (_load_json_optional(nccl_env) or {}).get("status")
        if evidence["single_nccl_peak_algbw_gbps"] > 0:
            return "runtime_verified", evidence, refs
        return "present_unverified", evidence, refs

    connectivity_payload = _load_json_optional(connectivity) or {}
    world_size = int(connectivity_payload.get("world_size") or 0)
    refs.extend([_relative_to_run_dir(run_dir, p) for p in (multi_nccl, connectivity, alltoall) if p.exists()])
    evidence["single_nccl_peak_algbw_gbps"] = _max_nccl_metric(single_nccl, "algbw_gbps")
    evidence["multi_nccl_peak_busbw_gbps"] = _max_nccl_metric(multi_nccl, "busbw_gbps")
    evidence["multi_nccl_peak_algbw_gbps"] = _max_nccl_metric(multi_nccl, "algbw_gbps")
    evidence["world_size"] = world_size
    evidence["alltoall_peak_busbw_gbps"] = _max_nccl_metric(alltoall, "busbw_gbps")
    evidence["alltoall_peak_algbw_gbps"] = _max_nccl_metric(alltoall, "algbw_gbps")
    if evidence["multi_nccl_peak_algbw_gbps"] > 0 or world_size > 1:
        return "runtime_verified", evidence, refs
    return "present_unverified", evidence, refs


def _records_named(records: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("name") or "") == name]


def _any_record_ok(records: list[dict[str, Any]], name: str) -> bool:
    return any(str(record.get("status") or "") == "ok" for record in _records_named(records, name))


def _stdout_line_count(records: list[dict[str, Any]], name: str) -> int:
    total = 0
    for record in _records_named(records, name):
        try:
            total += int(record.get("stdout_line_count") or 0)
        except Exception:
            continue
    return total


def _runtime_ratio(runtime_evidence: dict[str, Any]) -> float:
    try:
        single = float(runtime_evidence.get("single_nccl_peak_algbw_gbps") or 0.0)
        multi = float(runtime_evidence.get("multi_nccl_peak_algbw_gbps") or 0.0)
    except Exception:
        return 0.0
    if single <= 0:
        return 0.0
    return multi / single


def _ib_runtime_interpretation(runtime_status: str, runtime_evidence: dict[str, Any]) -> str:
    ratio = _runtime_ratio(runtime_evidence)
    if runtime_status != "runtime_verified":
        return "Collect multi-node NCCL, all-to-all, and torchrun connectivity artifacts before drawing IB routing conclusions."
    if ratio and ratio < 0.70:
        return "Multi-node NCCL scales below 0.70x of single-node; inspect routing, congestion, subnet state, and HCA/interface binding."
    return "Runtime artifacts show the IB path is alive; compare any residual NCCL or all-to-all anomalies against ibdiagnet, saquery, and path traces."


def _spectrum_runtime_interpretation(runtime_status: str, runtime_evidence: dict[str, Any]) -> str:
    ratio = _runtime_ratio(runtime_evidence)
    if runtime_status != "runtime_verified":
        return "Collect multi-node NCCL, all-to-all, and torchrun connectivity artifacts before drawing Spectrum-X / RoCE conclusions."
    if ratio and ratio < 0.70:
        return "Multi-node NCCL scales below 0.70x of single-node; inspect RoCE QoS, adaptive routing, BGP route visibility, and host RDMA binding."
    return "Runtime artifacts show the Ethernet/RDMA path is alive; correlate any throughput or tail-latency knees with NVUE and BGP evidence."


def _build_ib_scenario_summary(
    *,
    capability: dict[str, Any],
    control: dict[str, Any],
    runtime_status: str,
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    checks = list(control.get("checks") or [])
    routing_checks_ok = [
        name
        for name in ("ibaddr", "ibtracert", "ibroute", "perfquery", "ibdiagnet")
        if _any_record_ok(checks, name)
    ]
    return {
        "capacity_and_path_visibility_ready": any(
            _any_record_ok(checks, name)
            for name in ("ibstat", "ibswitches", "ibhosts", "iblinkinfo", "ibnetdiscover", "saquery")
        ),
        "routing_and_counter_verification_ready": bool(routing_checks_ok),
        "runtime_correlation_ready": runtime_status == "runtime_verified",
        "visible_hca_count": len(capability.get("hcas") or []),
        "visible_host_count": _stdout_line_count(checks, "ibhosts"),
        "visible_switch_count": _stdout_line_count(checks, "ibswitches"),
        "linkinfo_visible": _any_record_ok(checks, "iblinkinfo"),
        "subnet_discovery_visible": _any_record_ok(checks, "ibnetdiscover"),
        "saquery_visible": _any_record_ok(checks, "saquery"),
        "ibdiagnet_visible": _any_record_ok(checks, "ibdiagnet"),
        "routing_checks_ok": routing_checks_ok,
        "single_node_nccl_peak_algbw_gbps": float(runtime_evidence.get("single_nccl_peak_algbw_gbps") or 0.0),
        "multi_node_nccl_peak_algbw_gbps": float(runtime_evidence.get("multi_nccl_peak_algbw_gbps") or 0.0),
        "alltoall_peak_algbw_gbps": float(runtime_evidence.get("alltoall_peak_algbw_gbps") or 0.0),
        "multi_to_single_nccl_ratio": _runtime_ratio(runtime_evidence),
        "world_size": int(runtime_evidence.get("world_size") or 0),
        "runtime_interpretation": _ib_runtime_interpretation(runtime_status, runtime_evidence),
    }


def _build_spectrum_scenario_summary(
    *,
    control: dict[str, Any],
    runtime_status: str,
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    checks = list(control.get("checks") or [])
    def _record_host(record: dict[str, Any]) -> str:
        host = str(record.get("host") or "").strip()
        if host:
            return host
        notes = str(record.get("notes") or "")
        if "switch=" in notes:
            return notes.split("switch=", 1)[1].split()[0].strip()
        return ""

    targeted_hosts = sorted(
        {
            _record_host(record)
            for record in checks
            if _record_host(record)
        }
    )
    adaptive_ok = _any_record_ok(checks, "adaptive_routing")
    roce_ok = _any_record_ok(checks, "roce")
    bgp_neighbors_ok = _any_record_ok(checks, "bgp_neighbors")
    bgp_summary_ok = _any_record_ok(checks, "bgp_summary")
    bgp_routes_ok = _any_record_ok(checks, "bgp_routes")
    return {
        "fabric_readiness_ready": adaptive_ok and roce_ok and (bgp_neighbors_ok or bgp_summary_ok),
        "runtime_correlation_ready": runtime_status == "runtime_verified",
        "switch_count_targeted": len(targeted_hosts),
        "switches_targeted": targeted_hosts,
        "adaptive_routing_visible": adaptive_ok,
        "roce_qos_visible": roce_ok,
        "bgp_neighbor_state_visible": bgp_neighbors_ok,
        "bgp_summary_visible": bgp_summary_ok,
        "bgp_route_visibility": bgp_routes_ok,
        "single_node_nccl_peak_algbw_gbps": float(runtime_evidence.get("single_nccl_peak_algbw_gbps") or 0.0),
        "multi_node_nccl_peak_algbw_gbps": float(runtime_evidence.get("multi_nccl_peak_algbw_gbps") or 0.0),
        "alltoall_peak_algbw_gbps": float(runtime_evidence.get("alltoall_peak_algbw_gbps") or 0.0),
        "multi_to_single_nccl_ratio": _runtime_ratio(runtime_evidence),
        "world_size": int(runtime_evidence.get("world_size") or 0),
        "runtime_interpretation": _spectrum_runtime_interpretation(runtime_status, runtime_evidence),
    }


def _build_ai_correlation(
    *,
    run_id: str,
    run_dir: Path,
    primary_label: str | None,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    structured = run_dir / "structured"
    label = (primary_label or "").strip()
    if not label:
        meta_records = _meta_records(structured, run_id)
        label = str(meta_records[0]["label"]) if meta_records else "localhost"

    single_nccl = _max_nccl_metric(structured / f"{run_id}_node1_nccl.json", "algbw_gbps")
    multi_nccl = _max_nccl_metric(structured / f"{run_id}_2nodes_nccl.json", "algbw_gbps")
    ratio = (multi_nccl / single_nccl) if single_nccl > 0 else 0.0
    vllm_rows = _load_csv_rows(structured / f"{run_id}_{label}_vllm_serve_sweep.csv")
    throughput = [float(row.get("total_token_throughput") or 0.0) for row in vllm_rows]
    p99_ttft = [float(row.get("p99_ttft_ms") or 0.0) for row in vllm_rows]
    findings: list[str] = []
    recommendations: list[str] = []
    if capabilities["families"]["nvlink"]["present"]:
        findings.append(f"NVLink/NVSwitch runtime evidence peaks at {single_nccl:.1f} GB/s algbw on the single-node NCCL path.")
    if capabilities["families"]["infiniband"]["present"] and multi_nccl > 0:
        findings.append(f"Multi-node NCCL algbw peaks at {multi_nccl:.1f} GB/s with a multi-to-single ratio of {ratio:.2f}.")
        if ratio < 0.7:
            recommendations.append("Multi-node NCCL scaling is below 0.70x of single-node; inspect routing, congestion, and HCA/interface binding.")
    if capabilities["families"]["spectrum-x"]["present"] and multi_nccl > 0:
        findings.append("Spectrum-X / RoCE runtime verification is tied to the same multi-node NCCL and connectivity artifacts used for InfiniBand.")
    if throughput:
        findings.append(
            f"vLLM throughput ranges from {min(throughput):.1f} to {max(throughput):.1f} tok/s across {len(throughput)} concurrency points."
        )
    if len(p99_ttft) >= 2 and max(p99_ttft) > min(p99_ttft) * 1.8:
        findings.append("vLLM p99 TTFT shows a pronounced knee across the concurrency sweep.")
        recommendations.append("Correlate vLLM tail latency knees with NCCL env sensitivity and fabric control-plane state before tuning model-side queues.")
    evidence_refs = [
        _relative_to_run_dir(run_dir, path)
        for path in (
            structured / f"{run_id}_node1_nccl.json",
            structured / f"{run_id}_2nodes_nccl.json",
            structured / f"{run_id}_nccl_env_sensitivity.json",
            structured / f"{run_id}_{label}_vllm_serve_sweep.csv",
        )
        if path.exists()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "fabric_family": "all",
        "collection_mode": "artifact_correlation",
        "status": "ok",
        "completeness": capabilities["completeness"],
        "evidence_refs": evidence_refs,
        "recommendations": recommendations,
        "findings": findings,
        "summary": {
            "single_node_nccl_peak_algbw_gbps": single_nccl,
            "multi_node_nccl_peak_algbw_gbps": multi_nccl,
            "multi_to_single_nccl_ratio": ratio,
            "vllm_total_tok_s_mean": mean(throughput) if throughput else 0.0,
            "vllm_p99_ttft_max_ms": max(p99_ttft) if p99_ttft else 0.0,
        },
    }


def _build_fabric_scorecard(
    *,
    run_id: str,
    capabilities: dict[str, Any],
    verification: dict[str, Any],
    ai_correlation: dict[str, Any],
    require_management_plane: bool,
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    recommendations = list(ai_correlation.get("recommendations") or [])
    evidence_refs: list[str] = []
    configured_planes = 0
    full_stack_families = 0
    runtime_families = 0

    for family, capability in (capabilities.get("families") or {}).items():
        verification_family = (verification.get("families") or {}).get(family) or {}
        present = bool(capability.get("present"))
        control_state = str((verification_family.get("control_plane") or {}).get("status") or "not_present")
        runtime_state = str((verification_family.get("runtime") or {}).get("status") or "not_present")
        completeness = str(verification_family.get("completeness") or capability.get("completeness") or "not_present")
        if capability.get("management_plane_configured"):
            configured_planes += 1
        if runtime_state == "runtime_verified":
            runtime_families += 1
        if completeness == "full_stack_verified":
            full_stack_families += 1
        evidence_refs.extend(verification_family.get("evidence_refs") or [])
        if present and not capability.get("management_plane_configured"):
            recommendations.append(f"Configure the {family} management plane to move {family} from runtime-only evidence to full-stack verification.")
        families[family] = {
            "present": present,
            "completeness": completeness,
            "management_plane_configured": bool(capability.get("management_plane_configured")),
            "topology_correctness": "pass" if present else "n/a",
            "control_plane_health": control_state,
            "link_health": runtime_state if present else "n/a",
            "routing_correctness": "pass" if completeness == "full_stack_verified" else ("unknown" if present else "n/a"),
            "congestion_features_enabled": "verified" if family == "spectrum-x" and completeness == "full_stack_verified" else ("unknown" if present else "n/a"),
            "ai_workload_impact": next((finding for finding in ai_correlation.get("findings", []) if family.split("-")[0].lower() in finding.lower()), "No direct AI workload finding."),
        }

    if require_management_plane and configured_planes == 0:
        status = "error"
    elif runtime_families > 0:
        status = "ok" if full_stack_families >= runtime_families else "partial"
    else:
        status = "partial"
    completeness = _overall_completeness(capabilities.get("families") or {})
    if any(str((verification.get("families") or {}).get(name, {}).get("completeness")) == "full_stack_verified" for name in families):
        completeness = "full_stack_verified"
    elif runtime_families > 0:
        completeness = "runtime_verified"

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "fabric_family": "all",
        "collection_mode": "scored",
        "status": status,
        "completeness": completeness,
        "evidence_refs": sorted(set(evidence_refs)),
        "recommendations": sorted(dict.fromkeys(recommendations)),
        "summary": {
            "require_management_plane": require_management_plane,
            "configured_management_planes": configured_planes,
            "runtime_verified_families": runtime_families,
            "full_stack_verified_families": full_stack_families,
            "ai_findings_count": len(ai_correlation.get("findings") or []),
        },
        "families": families,
    }


def _render_scorecard_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Fabric Scorecard",
        "",
        f"Run ID: `{payload['run_id']}`",
        "",
        "| Family | Present | Completeness | Mgmt plane | Link health | Routing | Congestion features |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family, values in (payload.get("families") or {}).items():
        lines.append(
            f"| `{family}` | `{values.get('present')}` | `{values.get('completeness')}` | "
            f"`{values.get('management_plane_configured')}` | `{values.get('link_health')}` | "
            f"`{values.get('routing_correctness')}` | `{values.get('congestion_features_enabled')}` |"
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in payload.get("recommendations") or ["No additional fabric recommendations."]:
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def build_fabric_payloads(
    *,
    run_id: str,
    run_dir: Path,
    primary_label: str | None = None,
    labels: list[str] | None = None,
    management: ManagementConfig | None = None,
    runner: CommandRunner | None = None,
    source_root: Path | None = None,
    require_management_plane: bool = False,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    run_dir = run_dir.resolve()
    structured_dir = run_dir / "structured"
    mgmt = management or make_management_config(ssh_user=ssh_user, ssh_key=ssh_key)
    command_runner = runner or default_command_runner

    catalog = generate_catalog_payload(source_root, run_id=run_id)
    capability_matrix = _build_capability_matrix(
        run_id=run_id,
        structured_dir=structured_dir,
        run_dir=run_dir,
        labels=labels,
        management=mgmt,
    )

    control_plane: dict[str, Any] = {}
    recommendations: list[str] = []
    evidence_refs: list[str] = []

    nvlink_results, nvlink_state, nvlink_recs, nvlink_nmx = _collect_nvlink_management(mgmt)
    control_plane["nvlink"] = {"status": nvlink_state, "checks": nvlink_results}
    if nvlink_nmx:
        control_plane["nvlink"]["nmx"] = nvlink_nmx
    recommendations.extend(nvlink_recs)

    ib_results, ib_state, ib_recs = _collect_ib_control_plane(
        present=bool(capability_matrix["families"]["infiniband"]["present"]),
        management=mgmt,
        runner=command_runner,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
    )
    control_plane["infiniband"] = {"status": ib_state, "checks": ib_results}
    recommendations.extend(ib_recs)

    spectrum_results, spectrum_state, spectrum_recs = _collect_spectrum_control_plane(
        present=bool(capability_matrix["families"]["spectrum-x"]["present"]),
        management=mgmt,
        runner=command_runner,
    )
    control_plane["spectrum-x"] = {"status": spectrum_state, "checks": spectrum_results}
    recommendations.extend(spectrum_recs)

    verification_families: dict[str, Any] = {}
    for family in ("nvlink", "infiniband", "spectrum-x"):
        capability = capability_matrix["families"][family]
        control = control_plane[family]
        runtime_status, runtime_evidence, runtime_refs = _runtime_family_status(
            family=family,
            present=bool(capability.get("present")),
            run_id=run_id,
            run_dir=run_dir,
        )
        evidence_refs.extend(capability.get("evidence_refs") or [])
        evidence_refs.extend(runtime_refs)
        if control["status"] == "ok" and runtime_status == "runtime_verified":
            completeness = "full_stack_verified"
        elif runtime_status == "runtime_verified":
            completeness = "runtime_verified"
        else:
            completeness = capability.get("completeness", "not_present")
        family_payload = {
            "present": bool(capability.get("present")),
            "completeness": completeness,
            "evidence_refs": sorted(set((capability.get("evidence_refs") or []) + runtime_refs)),
            "control_plane": control,
            "runtime": {
                "status": runtime_status,
                "evidence": runtime_evidence,
            },
        }
        if family == "infiniband":
            family_payload["scenario_summary"] = _build_ib_scenario_summary(
                capability=capability,
                control=control,
                runtime_status=runtime_status,
                runtime_evidence=runtime_evidence,
            )
        elif family == "spectrum-x":
            family_payload["scenario_summary"] = _build_spectrum_scenario_summary(
                control=control,
                runtime_status=runtime_status,
                runtime_evidence=runtime_evidence,
            )
        verification_families[family] = family_payload

    verification = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "fabric_family": "all",
        "collection_mode": "verification",
        "status": "ok" if any(f["present"] for f in verification_families.values()) else "partial",
        "completeness": _overall_completeness(verification_families),
        "evidence_refs": sorted(set(evidence_refs)),
        "recommendations": sorted(dict.fromkeys(recommendations)),
        "families": verification_families,
    }

    ai_correlation = _build_ai_correlation(
        run_id=run_id,
        run_dir=run_dir,
        primary_label=primary_label,
        capabilities=capability_matrix,
    )
    fabric_scorecard = _build_fabric_scorecard(
        run_id=run_id,
        capabilities=capability_matrix,
        verification=verification,
        ai_correlation=ai_correlation,
        require_management_plane=require_management_plane,
    )
    if require_management_plane and fabric_scorecard["summary"]["configured_management_planes"] == 0:
        fabric_scorecard["recommendations"] = sorted(
            dict.fromkeys(
                list(fabric_scorecard.get("recommendations") or [])
                + ["Publish-grade fabric validation requires management-plane access; configure IB, NMX, or Cumulus endpoints."]
            )
        )
    return {
        "fabric_command_catalog": catalog,
        "fabric_capability_matrix": capability_matrix,
        "fabric_verification": verification,
        "fabric_ai_correlation": ai_correlation,
        "fabric_scorecard": fabric_scorecard,
        "fabric_scorecard_md": {
            "markdown": _render_scorecard_markdown(fabric_scorecard),
        },
    }
