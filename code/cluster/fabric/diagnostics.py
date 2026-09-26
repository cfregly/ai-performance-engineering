from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.diagnostics.evidence import _atomic_text

SCHEMA_VERSION = "fabric-diagnostics.v1"
CommandRunner = Callable[..., dict[str, Any]]

_UNSUPPORTED_PATTERNS = (
    "command not found",
    "not supported",
    "unsupported",
    "unknown command",
    "invalid command",
    "not a valid command",
    "not a valid option",
    "invalid value",
    "no such file or directory",
    "not installed",
)
_COUNTER_TERMS = (
    "byte",
    "cnp",
    "congestion",
    "discard",
    "drop",
    "ecn",
    "error",
    "mark",
    "octet",
    "packet",
    "pause",
    "pfc",
    "port_rcv",
    "port_xmit",
    "portrcv",
    "portxmit",
    "symbol",
    "vl15",
    "wait",
)
_GAUGE_TERMS = ("occupancy", "temperature", "utilization")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.-])(-?\d+(?:\.\d+)?)")
_KEY_VALUE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_ ./()\[\]-]{1,100}?)\s*(?::|=|\.{2,}:)\s*"
    r"(-?\d+(?:\.\d+)?)\s*([A-Za-z%/]+)?\s*$"
)
_TABLE_SPLIT_RE = re.compile(r"\s{2,}")
_TABLE_NUMBER_RE = re.compile(r"^(-?[\d,]+(?:\.\d+)?)\s*([A-Za-z%/]+)?$")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    family: str
    host: str | None
    command: str
    scope: str
    parser: str
    user: str | None = None
    ssh_key: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collector_clock_domain() -> str:
    host_token = hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:12]
    return f"collector_wall_clock:{host_token}"


def _host_identity(host: str | None) -> dict[str, Any]:
    kernel_hostname = platform.node()
    local_aliases = {
        "",
        "localhost",
        "127.0.0.1",
        "::1",
        kernel_hostname,
        kernel_hostname.split(".")[0],
    }
    requested = host or ""
    if requested in local_aliases:
        token = hashlib.sha256(kernel_hostname.encode("utf-8")).hexdigest()[:12]
        return {
            "host": token,
            "endpoint_alias_hash": None,
            "host_identity_source": "collector_kernel_hostname",
            "kernel_hostname_verified": True,
        }
    alias_hash = hashlib.sha256(requested.encode("utf-8")).hexdigest()[:12]
    return {
        "host": None,
        "endpoint_alias_hash": alias_hash,
        "host_identity_source": "requested_endpoint_alias",
        "kernel_hostname_verified": False,
    }


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return token[:96] or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _default_runner(
    command: str,
    *,
    host: str | None = None,
    user: str | None = None,
    ssh_key: str | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    local_names = {
        "",
        "localhost",
        "127.0.0.1",
        "::1",
        os.uname().nodename,
        os.uname().nodename.split(".")[0],
    }
    if (host or "") in local_names:
        argv = ["bash", "-lc", command]
        process_timeout = timeout
        remote_cancellation = "not_applicable"
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ConnectionAttempts=2",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if ssh_key:
            argv.extend(["-i", ssh_key])
        target = f"{user}@{host}" if user else str(host)
        remote_script = (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout --signal=TERM --kill-after=5s {int(timeout)}s "
            f"bash -lc {shlex.quote(command)}; "
            "else echo 'unsupported: remote timeout utility unavailable' >&2; exit 125; fi"
        )
        argv.extend(["--", target, f"bash -lc {shlex.quote(remote_script)}"])
        process_timeout = timeout + 10
        remote_cancellation = "bounded_by_remote_timeout_wrapper"
    timed_out = False
    term_sent = False
    kill_sent = False
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if host not in local_names and host:
                remote_cancellation = "not_verified_after_ssh_process_group_timeout"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                term_sent = True
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    kill_sent = True
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
    except OSError as exc:
        return {
            "status": "error",
            "host": host or "localhost",
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "timed_out": False,
            "termination": {
                "owned_process_group": True,
                "term_sent": False,
                "kill_sent": False,
                "remote_cancellation": remote_cancellation,
            },
        }
    remote_timeout = bool((host or "") not in local_names and proc.returncode == 124)
    timed_out = timed_out or remote_timeout
    if remote_timeout:
        remote_cancellation = "remote_timeout_wrapper_reported_timeout"
    return {
        "status": "ok" if proc.returncode == 0 and not timed_out else "error",
        "host": host or "localhost",
        "command": command,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error": f"timeout after {timeout}s"
        if timed_out
        else ""
        if proc.returncode == 0
        else f"returncode={proc.returncode}",
        "timed_out": timed_out,
        "termination": {
            "owned_process_group": True,
            "term_sent": term_sent,
            "kill_sent": kill_sent,
            "remote_cancellation": remote_cancellation,
        },
    }


def _result_status(result: dict[str, Any]) -> str:
    if str(result.get("status") or "") == "ok" and int(result.get("returncode") or 0) == 0:
        return "ok"
    combined = "\n".join(
        str(result.get(key) or "") for key in ("stdout", "stderr", "error")
    ).lower()
    if (
        any(pattern in combined for pattern in _UNSUPPORTED_PATTERNS)
        or result.get("returncode") == 127
    ):
        return "unsupported"
    return "error"


def _normalize_key(raw: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    return value


def _number_from_token(token: str) -> int | float:
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric token {token!r}")
    return value


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_numeric_fields(text: str) -> dict[str, dict[str, Any]]:
    """Parse named numeric fields without assigning counter semantics."""
    fields: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = _KEY_VALUE_RE.match(line)
        if not match:
            continue
        key = _normalize_key(match.group(1))
        if not key:
            continue
        number = _number_from_token(match.group(2))
        if key in fields:
            raise ValueError(
                f"duplicate normalized field {key!r} cannot be assigned a unique counter scope"
            )
        fields[key] = {
            "value": number,
            "unit": match.group(3) or None,
            "source_line": line.strip(),
        }
    return fields


def _parse_perfquery(text: str) -> dict[str, Any]:
    fields = parse_numeric_fields(text)
    if not fields:
        for line in text.splitlines():
            if ":" not in line:
                continue
            key_text, value_text = line.split(":", 1)
            number = _NUMBER_RE.search(value_text)
            if number is None:
                continue
            key = _normalize_key(key_text)
            value = _number_from_token(number.group(1))
            if key in fields:
                raise ValueError(
                    f"duplicate normalized field {key!r} cannot be assigned a unique counter scope"
                )
            fields[key] = {
                "value": value,
                "unit": None,
                "source_line": line.strip(),
            }
    return {"fields": fields}


def parse_cumulus_counter_tables(text: str) -> dict[str, dict[str, Any]]:
    """Parse aligned NVUE counter tables with row-qualified field names."""
    fields: dict[str, dict[str, Any]] = {}
    lines = text.splitlines()
    for index in range(len(lines) - 1):
        headers = _TABLE_SPLIT_RE.split(lines[index].strip())
        separators = _TABLE_SPLIT_RE.split(lines[index + 1].strip())
        if (
            len(headers) < 2
            or len(separators) != len(headers)
            or not all(re.fullmatch(r"-+", token) for token in separators)
        ):
            continue
        first_header = _normalize_key(headers[0])
        for row_line in lines[index + 2 :]:
            if not row_line.strip():
                break
            cells = _TABLE_SPLIT_RE.split(row_line.strip())
            if len(cells) != len(headers):
                break
            row_name = _normalize_key(cells[0])
            prefix = (
                row_name
                if first_header in {"counter", "statistic"}
                else f"{first_header}_{row_name}"
            )
            for header, cell in zip(headers[1:], cells[1:], strict=True):
                match = _TABLE_NUMBER_RE.fullmatch(cell.strip())
                if match is None:
                    continue
                key = f"{prefix}_{_normalize_key(header)}"
                if key in fields:
                    raise ValueError(
                        f"duplicate normalized field {key!r} cannot be assigned a unique counter scope"
                    )
                fields[key] = {
                    "value": _number_from_token(match.group(1).replace(",", "")),
                    "unit": match.group(2) or None,
                    "source_line": row_line.strip(),
                }
    scalar_fields = parse_numeric_fields(text)
    for key, details in scalar_fields.items():
        if key in fields:
            raise ValueError(
                f"duplicate normalized field {key!r} cannot be assigned a unique counter scope"
            )
        fields[key] = details
    return fields


def _parse_bgp(text: str) -> dict[str, Any]:
    peers: list[dict[str, Any]] = []
    for line in text.splitlines():
        words = line.split()
        if not words or not re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", words[0]):
            continue
        state = words[-1]
        peers.append(
            {
                "peer": words[0],
                "state": "established" if state.isdigit() else state.lower(),
                "raw": line.strip(),
            }
        )
    route_lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b(?:via|nexthop|next-hop|metric|best)\b", line, re.IGNORECASE)
    ]
    return {"peers": peers, "route_lines": route_lines[:200]}


def _parse_route(text: str) -> dict[str, Any]:
    hops = [line.strip() for line in text.splitlines() if line.strip()]
    return {"path_lines": hops[:400], "path_line_count": len(hops)}


def _parse_command(spec: CommandSpec, stdout: str) -> dict[str, Any]:
    if spec.parser == "counters":
        return _parse_perfquery(stdout)
    if spec.parser == "cumulus_counters":
        return {"fields": parse_cumulus_counter_tables(stdout)}
    if spec.parser == "bgp":
        return _parse_bgp(stdout)
    if spec.parser == "route":
        return _parse_route(stdout)
    return {"fields": parse_numeric_fields(stdout)}


def _counter_records(spec: CommandSpec, parsed: dict[str, Any]) -> list[dict[str, Any]]:
    if spec.parser not in {"counters", "cumulus_counters"}:
        return []
    records: list[dict[str, Any]] = []
    host_identity = _host_identity(spec.host)
    identity_key = str(host_identity.get("host") or f"alias:{host_identity['endpoint_alias_hash']}")
    for key, details in (parsed.get("fields") or {}).items():
        lowered = key.lower()
        kind = (
            "counter"
            if any(term in lowered for term in _COUNTER_TERMS)
            else "gauge"
            if any(term in lowered for term in _GAUGE_TERMS)
            else "field"
        )
        if kind == "field":
            continue
        records.append(
            {
                "id": "|".join((spec.family, identity_key, spec.scope, spec.name, key)),
                "family": spec.family,
                **host_identity,
                "scope": spec.scope,
                "source_command": spec.name,
                "name": key,
                "kind": kind,
                "value": details["value"],
                "unit": details.get("unit"),
            }
        )
    return records


def _signal_kind(name: str) -> str | None:
    lowered = name.lower()
    if "pfc" in lowered or "pause" in lowered:
        return "pfc"
    if any(term in lowered for term in ("ecn", "cnp", "congestion_mark")):
        return "ecn"
    if any(term in lowered for term in ("error", "discard", "drop", "symbol", "vl15")):
        return "link_error"
    if any(term in lowered for term in ("byte", "octet", "packet", "portxmitdata", "portrcvdata")):
        return "throughput"
    return None


def _signal_scope(row: dict[str, Any], workload_id: str | None = None) -> dict[str, Any]:
    scope_text = str(row.get("scope") or "")
    scope: dict[str, Any] = {}
    if row.get("host"):
        scope["host"] = row["host"]
    if row.get("endpoint_alias_hash"):
        scope["endpoint_alias_hash"] = row["endpoint_alias_hash"]
    if row.get("family") == "roce" and scope_text != "global":
        scope["interface"] = scope_text
    if row.get("family") == "infiniband" and scope_text != "global":
        scope["path"] = scope_text
        hca_text, at, endpoint = scope_text.partition("@")
        lid_port = endpoint if at else hca_text
        lid, separator, port = lid_port.partition(":")
        if at:
            scope["hca"] = hca_text
        if separator:
            scope["lid"] = lid
            scope["port"] = port
    if "host" not in scope and "interface" not in scope and "path" not in scope:
        scope["path"] = f"{row.get('family', 'fabric')}:global"
    if workload_id:
        scope["workload_id"] = workload_id
    return scope


def _summarize_path_balance(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or _signal_kind(str(row.get("name") or "")) != "throughput":
            continue
        rate = _finite_number(row.get("rate_per_second"))
        if rate is None or rate < 0 or str(row.get("scope") or "") == "global":
            continue
        key = (
            str(row.get("family") or ""),
            str(row.get("source_command") or ""),
            str(row.get("name") or ""),
            str(row.get("unit") or "count"),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (family, source_command, metric, unit), members in sorted(groups.items()):
        resources = {
            (
                str(row.get("host") or row.get("endpoint_alias_hash") or ""),
                str(row.get("scope") or ""),
            )
            for row in members
        }
        if len(resources) < 2:
            continue
        overlap_start = max(float(row["start_unix_s"]) for row in members)
        overlap_end = min(float(row["end_unix_s"]) for row in members)
        if overlap_end <= overlap_start:
            continue
        rates = [float(row["rate_per_second"]) for row in members]
        minimum = min(rates)
        maximum = max(rates)
        average = sum(rates) / len(rates)
        summaries.append(
            {
                "family": family,
                "source_command": source_command or None,
                "metric": metric,
                "unit": f"{unit}/s",
                "resource_count": len(resources),
                "overlap_start_unix_s": overlap_start,
                "overlap_end_unix_s": overlap_end,
                "min_rate_per_second": minimum,
                "max_rate_per_second": maximum,
                "mean_rate_per_second": average,
                "max_to_min_ratio": maximum / minimum if minimum > 0 else None,
                "observation": "unequal_rates_observed"
                if minimum != maximum
                else "equal_rates_observed",
                "interpretation": "Rate differences describe traffic distribution across sampled ports. They do not prove an ECMP collision or its cause.",
                "ports": [
                    {
                        "scope": _signal_scope(row),
                        "rate_per_second": row["rate_per_second"],
                        "start_unix_s": row["start_unix_s"],
                        "end_unix_s": row["end_unix_s"],
                        "evidence": row.get("evidence") or [],
                    }
                    for row in members
                ],
            }
        )
    return summaries


def _roce_specs(
    hosts: Sequence[str],
    interfaces: Sequence[str],
    *,
    user: str | None,
    ssh_key: str | None,
) -> tuple[list[CommandSpec], list[dict[str, Any]]]:
    specs: list[CommandSpec] = []
    unsupported: list[dict[str, Any]] = []
    if not hosts:
        unsupported.append(
            {
                "family": "roce",
                "capability": "switch_access",
                "status": "unsupported",
                "reason": "no Cumulus switch host was provided",
            }
        )
        return specs, unsupported
    for host in hosts:
        specs.extend(
            [
                CommandSpec(
                    "roce_qos", "roce", host, "nv show qos roce", "global", "fields", user, ssh_key
                ),
                CommandSpec(
                    "adaptive_routing",
                    "roce",
                    host,
                    "nv show router adaptive-routing",
                    "global",
                    "fields",
                    user,
                    ssh_key,
                ),
                CommandSpec(
                    "bgp_summary",
                    "roce",
                    host,
                    'vtysh -c "show bgp ipv4 unicast summary"',
                    "global",
                    "bgp",
                    user,
                    ssh_key,
                ),
                CommandSpec(
                    "bgp_routes",
                    "roce",
                    host,
                    'vtysh -c "show ip route vrf default bgp"',
                    "global",
                    "bgp",
                    user,
                    ssh_key,
                ),
            ]
        )
        if not interfaces:
            unsupported.append(
                {
                    "family": "roce",
                    **_host_identity(host),
                    "capability": "interface_counters",
                    "status": "unsupported",
                    "reason": "no switch interfaces were provided",
                }
            )
        for interface in interfaces:
            iface = shlex.quote(interface)
            specs.extend(
                [
                    CommandSpec(
                        "roce_counters",
                        "roce",
                        host,
                        f"nv show interface {iface} qos roce counters",
                        interface,
                        "cumulus_counters",
                        user,
                        ssh_key,
                    ),
                    CommandSpec(
                        "qos_counters",
                        "roce",
                        host,
                        f"nv show interface {iface} counters qos",
                        interface,
                        "cumulus_counters",
                        user,
                        ssh_key,
                    ),
                    CommandSpec(
                        "qos_egress_queue_counters",
                        "roce",
                        host,
                        f"nv show interface {iface} counters qos egress-queue-stats",
                        interface,
                        "cumulus_counters",
                        user,
                        ssh_key,
                    ),
                    CommandSpec(
                        "interface_counters",
                        "roce",
                        host,
                        f"nv show interface {iface} counters",
                        interface,
                        "cumulus_counters",
                        user,
                        ssh_key,
                    ),
                ]
            )
    return specs, unsupported


def _parse_endpoint(value: str) -> tuple[str | None, str, int]:
    hca_text, at, endpoint = value.partition("@")
    lid_port = endpoint if at else hca_text
    lid, separator, port_text = lid_port.partition(":")
    if not separator or not lid.isdigit() or not port_text.isdigit() or int(port_text) < 1:
        raise ValueError(f"invalid InfiniBand endpoint {value!r}, expected [<hca>@]<lid>:<port>")
    return hca_text if at else None, lid, int(port_text)


def _ib_specs(
    host: str | None,
    endpoints: Sequence[str],
    route_lids: Sequence[str],
    switch_lids: Sequence[str],
    *,
    user: str | None,
    ssh_key: str | None,
) -> tuple[list[CommandSpec], list[dict[str, Any]]]:
    specs: list[CommandSpec] = []
    unsupported: list[dict[str, Any]] = []
    if not host:
        unsupported.append(
            {
                "family": "infiniband",
                "capability": "management_access",
                "status": "unsupported",
                "reason": "no InfiniBand management host was provided",
            }
        )
        return specs, unsupported
    specs.extend(
        [
            CommandSpec("ibstat", "infiniband", host, "ibstat", "global", "fields", user, ssh_key),
            CommandSpec("saquery", "infiniband", host, "saquery", "global", "route", user, ssh_key),
            CommandSpec(
                "ibdiagnet", "infiniband", host, "ibdiagnet -r", "global", "fields", user, ssh_key
            ),
        ]
    )
    for endpoint in endpoints:
        _, lid, port = _parse_endpoint(endpoint)
        specs.append(
            CommandSpec(
                "perfquery",
                "infiniband",
                host,
                f"perfquery -x {lid} {port}",
                endpoint,
                "counters",
                user,
                ssh_key,
            )
        )
    if not endpoints:
        unsupported.append(
            {
                "family": "infiniband",
                **_host_identity(host),
                "capability": "port_counters",
                "status": "unsupported",
                "reason": "no InfiniBand LID and port endpoints were provided",
            }
        )
    if len(route_lids) == 2:
        specs.append(
            CommandSpec(
                "ibtracert",
                "infiniband",
                host,
                f"ibtracert {route_lids[0]} {route_lids[1]}",
                f"{route_lids[0]}->{route_lids[1]}",
                "route",
                user,
                ssh_key,
            )
        )
    elif route_lids:
        raise ValueError("InfiniBand route LIDs require exactly two values")
    else:
        unsupported.append(
            {
                "family": "infiniband",
                **_host_identity(host),
                "capability": "path_trace",
                "status": "unsupported",
                "reason": "no source and destination LIDs were provided",
            }
        )
    for lid in switch_lids:
        if not lid.isdigit():
            raise ValueError(f"invalid InfiniBand switch LID {lid!r}")
        specs.append(
            CommandSpec(
                "ibroute", "infiniband", host, f"ibroute {lid}", lid, "route", user, ssh_key
            )
        )
    return specs, unsupported


def collect_fabric_snapshot(
    *,
    run_id: str,
    phase: str,
    run_dir: Path,
    family: str = "all",
    cumulus_hosts: Sequence[str] = (),
    switch_interfaces: Sequence[str] = (),
    cumulus_user: str | None = None,
    cumulus_ssh_key: str | None = None,
    ib_mgmt_host: str | None = None,
    ib_mgmt_user: str | None = None,
    ib_mgmt_ssh_key: str | None = None,
    ib_endpoints: Sequence[str] = (),
    ib_route_lids: Sequence[str] = (),
    ib_switch_lids: Sequence[str] = (),
    runner: CommandRunner | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    command_runner = runner or _default_runner
    specs: list[CommandSpec] = []
    unsupported: list[dict[str, Any]] = []
    if family in {"all", "roce"}:
        roce_specs, roce_unsupported = _roce_specs(
            cumulus_hosts, switch_interfaces, user=cumulus_user, ssh_key=cumulus_ssh_key
        )
        specs.extend(roce_specs)
        unsupported.extend(roce_unsupported)
    if family in {"all", "infiniband"}:
        ib_specs, ib_unsupported = _ib_specs(
            ib_mgmt_host,
            ib_endpoints,
            ib_route_lids,
            ib_switch_lids,
            user=ib_mgmt_user,
            ssh_key=ib_mgmt_ssh_key,
        )
        specs.extend(ib_specs)
        unsupported.extend(ib_unsupported)

    raw_dir = run_dir / "raw"
    records: list[dict[str, Any]] = []
    counters: list[dict[str, Any]] = []
    captured_epoch_s = time.time()
    captured_at = _utc_now()
    clock_domain = _collector_clock_domain()
    for index, spec in enumerate(specs, start=1):
        host_identity = _host_identity(spec.host)
        started = time.time()
        result = command_runner(
            spec.command,
            host=spec.host,
            user=spec.user,
            ssh_key=spec.ssh_key,
            timeout=timeout_seconds,
        )
        finished = time.time()
        status = _result_status(result)
        stdout = str(result.get("stdout") or "")
        parse_error = ""
        if status == "ok":
            try:
                parsed = _parse_command(spec, stdout)
            except ValueError as exc:
                status = "error"
                parsed = {}
                parse_error = str(exc)
        else:
            parsed = {}
        evidence_name = f"{run_id}_fabric_diagnostics_{_safe_token(phase)}_{index:03d}_{_safe_token(spec.family)}_{_safe_token(spec.name)}.json"
        evidence_path = raw_dir / evidence_name
        raw_payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": phase,
            "captured_at": _utc_now(),
            "duration_seconds": max(0.0, finished - started),
            "spec": {
                "name": spec.name,
                "family": spec.family,
                **host_identity,
                "scope": spec.scope,
                "command": spec.command,
                "parser": spec.parser,
            },
            "result": {
                "status": status,
                "returncode": result.get("returncode"),
                "stdout": stdout,
                "stderr": str(result.get("stderr") or ""),
                "error": parse_error or str(result.get("error") or ""),
                "timed_out": bool(result.get("timed_out")),
                "termination": result.get("termination"),
            },
            "parsed": parsed,
        }
        _write_json(evidence_path, raw_payload)
        counter_rows = _counter_records(spec, parsed) if status == "ok" else []
        for counter_row in counter_rows:
            counter_row["evidence"] = [str(evidence_path.relative_to(run_dir))]
            counter_row["sampled_unix_s"] = finished
            counter_row["clock_domain"] = clock_domain
        counters.extend(counter_rows)
        records.append(
            {
                "name": spec.name,
                "family": spec.family,
                **host_identity,
                "scope": spec.scope,
                "command": spec.command,
                "status": status,
                "returncode": result.get("returncode"),
                "duration_seconds": max(0.0, finished - started),
                "raw_evidence": str(evidence_path.relative_to(run_dir)),
                "parsed": parsed,
                "counter_count": len(counter_rows),
            }
        )
    statuses = [record["status"] for record in records]
    if records and all(status == "ok" for status in statuses):
        status = "ok"
    elif records and any(status == "ok" for status in statuses):
        status = "partial"
    elif records and any(status == "error" for status in statuses):
        status = "error"
    else:
        status = "unsupported"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fabric_snapshot",
        "run_id": run_id,
        "phase": phase,
        "collection_mode": "read_only",
        "canonical": False,
        "captured_at": captured_at,
        "captured_epoch_s": captured_epoch_s,
        "clock_domain": clock_domain,
        "host_identity_note": "Remote host tokens hash the requested endpoint alias. They do not claim that the alias equals the remote kernel hostname.",
        "family": family,
        "status": status,
        "commands": records,
        "counters": counters,
        "unsupported": unsupported,
    }


def analyze_snapshot_pair(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Calculate counter deltas without guessing across resets or wraps."""
    before_run_id = str(before.get("run_id") or "")
    after_run_id = str(after.get("run_id") or "")
    run_id = after_run_id or before_run_id or None
    before_time = float(before.get("captured_epoch_s") or 0.0)
    after_time = float(after.get("captured_epoch_s") or 0.0)
    elapsed = after_time - before_time
    before_clock = str(before.get("clock_domain") or "")
    after_clock = str(after.get("clock_domain") or "")
    if elapsed <= 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": "aisp.diagnostic-signals.v1",
            "artifact_type": "fabric_counter_delta",
            "status": "invalid",
            "run_id": run_id,
            "reason": "after snapshot timestamp must be later than before snapshot timestamp",
            "elapsed_seconds": elapsed,
            "counters": [],
            "signals": [],
            "path_balance": [],
        }
    if not before_clock or before_clock != after_clock:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": "aisp.diagnostic-signals.v1",
            "artifact_type": "fabric_counter_delta",
            "status": "invalid",
            "run_id": run_id,
            "reason": "before and after snapshots must identify the same collector clock domain",
            "elapsed_seconds": elapsed,
            "before_clock_domain": before_clock or None,
            "after_clock_domain": after_clock or None,
            "counters": [],
            "signals": [],
            "path_balance": [],
        }
    if before_run_id and after_run_id and before_run_id != after_run_id:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": "aisp.diagnostic-signals.v1",
            "artifact_type": "fabric_counter_delta",
            "status": "invalid",
            "run_id": None,
            "reason": "before and after snapshots must have the same run_id",
            "elapsed_seconds": elapsed,
            "before_run_id": before_run_id,
            "after_run_id": after_run_id,
            "counters": [],
            "signals": [],
            "path_balance": [],
        }
    before_rows = before.get("counters") or []
    after_rows = after.get("counters") or []
    before_ids = [str(row["id"]) for row in before_rows]
    after_ids = [str(row["id"]) for row in after_rows]
    if len(before_ids) != len(set(before_ids)) or len(after_ids) != len(set(after_ids)):
        return {
            "schema_version": SCHEMA_VERSION,
            "schema": "aisp.diagnostic-signals.v1",
            "artifact_type": "fabric_counter_delta",
            "status": "invalid",
            "run_id": run_id,
            "reason": "snapshot contains duplicate counter ids",
            "elapsed_seconds": elapsed,
            "counters": [],
            "signals": [],
            "path_balance": [],
        }
    before_counters = {str(row["id"]): row for row in before_rows}
    after_counters = {str(row["id"]): row for row in after_rows}
    rows: list[dict[str, Any]] = []
    for counter_id in sorted(set(before_counters) | set(after_counters)):
        left = before_counters.get(counter_id)
        right = after_counters.get(counter_id)
        if left is None or right is None:
            rows.append(
                {
                    "id": counter_id,
                    "status": "unmatched",
                    "before": left,
                    "after": right,
                    "delta": None,
                    "rate_per_second": None,
                    "start_unix_s": before_time,
                    "end_unix_s": after_time,
                    "clock_domain": before_clock,
                }
            )
            continue
        before_value = _finite_number(left.get("value"))
        after_value = _finite_number(right.get("value"))
        start_unix_s = float(left.get("sampled_unix_s") or before_time)
        end_unix_s = float(right.get("sampled_unix_s") or after_time)
        counter_elapsed = end_unix_s - start_unix_s
        evidence = list(
            dict.fromkeys([*(left.get("evidence") or []), *(right.get("evidence") or [])])
        )
        timing = {
            "start_unix_s": start_unix_s,
            "end_unix_s": end_unix_s,
            "clock_domain": before_clock,
            "evidence": evidence,
        }
        if before_value is None or after_value is None:
            rows.append(
                {
                    **right,
                    **timing,
                    "status": "invalid_numeric",
                    "before_value": left.get("value"),
                    "after_value": right.get("value"),
                    "delta": None,
                    "rate_per_second": None,
                }
            )
            continue
        if counter_elapsed <= 0:
            rows.append(
                {
                    **right,
                    **timing,
                    "status": "invalid_sample_order",
                    "before_value": before_value,
                    "after_value": after_value,
                    "delta": None,
                    "rate_per_second": None,
                }
            )
            continue
        if str(left.get("kind")) == "gauge":
            rows.append(
                {
                    **right,
                    **timing,
                    "status": "gauge",
                    "before_value": before_value,
                    "after_value": after_value,
                    "delta": after_value - before_value,
                    "rate_per_second": None,
                }
            )
            continue
        if after_value < before_value:
            rows.append(
                {
                    **right,
                    **timing,
                    "status": "reset_or_wrap",
                    "before_value": before_value,
                    "after_value": after_value,
                    "delta": None,
                    "rate_per_second": None,
                    "reason": "counter width and reset provenance are unknown",
                }
            )
            continue
        delta = after_value - before_value
        rows.append(
            {
                **right,
                **timing,
                "status": "ok",
                "before_value": before_value,
                "after_value": after_value,
                "delta": delta,
                "rate_per_second": delta / counter_elapsed,
            }
        )
    usable = [row for row in rows if row["status"] in {"ok", "gauge"}]
    status = (
        "ok"
        if usable and all(row["status"] in {"ok", "gauge"} for row in rows)
        else "partial"
        if usable
        else "unsupported"
    )
    signals: list[dict[str, Any]] = []
    for row in usable:
        kind = _signal_kind(str(row.get("name") or ""))
        if kind is None:
            continue
        is_gauge = row["status"] == "gauge"
        unit = row.get("unit") or "count"
        signal = {
            "kind": kind,
            "role": "measurement" if is_gauge else "counter",
            "metric": row.get("name"),
            "value": row.get("after_value") if is_gauge else row.get("delta"),
            "unit": unit,
            "start_unix_s": row["start_unix_s"],
            "end_unix_s": row["end_unix_s"],
            "clock_domain": before_clock,
            "scope": _signal_scope(row),
            "scope_identity": {
                "host_identity_source": row.get("host_identity_source"),
                "kernel_hostname_verified": bool(row.get("kernel_hostname_verified")),
            },
            "evidence": row.get("evidence") or [],
        }
        if is_gauge:
            signal["gauge_change"] = row.get("delta")
        else:
            signal["semantics"] = "delta"
            signal["rate_per_second"] = row.get("rate_per_second")
            signal["rate_unit"] = f"{unit}/s"
        signals.append(signal)
    path_balance = _summarize_path_balance(usable)
    return {
        "schema_version": SCHEMA_VERSION,
        "schema": "aisp.diagnostic-signals.v1",
        "artifact_type": "fabric_counter_delta",
        "status": status,
        "run_id": run_id,
        "before_phase": before.get("phase"),
        "after_phase": after.get("phase"),
        "before_captured_at": before.get("captured_at"),
        "after_captured_at": after.get("captured_at"),
        "elapsed_seconds": elapsed,
        "clock_domain": before_clock,
        "counter_count": len(rows),
        "usable_counter_count": len(usable),
        "counters": rows,
        "signals": signals,
        "path_balance": path_balance,
    }


def _run_workload(
    command: str,
    *,
    run_id: str,
    run_dir: Path,
    runner: CommandRunner,
    timeout_seconds: int,
    workload_id: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    result = runner(command, host=None, user=None, ssh_key=None, timeout=timeout_seconds)
    finished = time.time()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "diagnostic_workload",
        "run_id": run_id,
        "workload_id": workload_id or hashlib.sha256(command.encode("utf-8")).hexdigest()[:12],
        "command": command,
        "status": _result_status(result),
        "returncode": result.get("returncode"),
        "duration_seconds": max(0.0, finished - started),
        "stdout": str(result.get("stdout") or ""),
        "stderr": str(result.get("stderr") or ""),
        "error": str(result.get("error") or ""),
        "timed_out": bool(result.get("timed_out")),
        "termination": result.get("termination"),
    }
    path = run_dir / "raw" / f"{run_id}_fabric_diagnostics_workload.json"
    _write_json(path, payload)
    return {**payload, "raw_evidence": str(path.relative_to(run_dir))}


def _tag_delta_workload(
    delta: dict[str, Any],
    workload_id: str,
    *,
    status: str,
    returncode: int | None,
) -> None:
    delta["declared_workload_id"] = workload_id
    for counter in delta.get("counters") or []:
        counter["workload_id"] = workload_id
        counter["workload_status"] = status
        counter["workload_returncode"] = returncode
    for signal_row in delta.get("signals") or []:
        signal_row.setdefault("scope", {})["workload_id"] = workload_id


def _render_report(payload: dict[str, Any]) -> str:
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    delta = payload.get("delta") or {}
    lines = [
        "# Fabric Diagnostics",
        "",
        f"Run ID: `{payload.get('run_id')}`",
        "",
        "This is diagnostic evidence. It is not a canonical performance result.",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
        f"| Before snapshot | `{before.get('status', 'not_collected')}` | `{payload.get('artifacts', {}).get('before', '')}` |",
        f"| Workload | `{(payload.get('workload') or {}).get('status', 'not_run')}` | `{(payload.get('workload') or {}).get('raw_evidence', '')}` |",
        f"| After snapshot | `{after.get('status', 'not_collected')}` | `{payload.get('artifacts', {}).get('after', '')}` |",
        f"| Counter delta | `{delta.get('status', 'not_analyzed')}` | `{payload.get('artifacts', {}).get('delta', '')}` |",
        "",
        "## Counter Deltas",
        "",
        "| Family | Host | Scope | Counter | Status | Delta | Rate per second |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in (delta.get("counters") or [])[:200]:
        lines.append(
            "| {family} | {host} | {scope} | {name} | `{status}` | {delta} | {rate} |".format(
                family=row.get("family", ""),
                host=row.get("host") or row.get("endpoint_alias_hash", ""),
                scope=row.get("scope", ""),
                name=row.get("name", row.get("id", "")),
                status=row.get("status", ""),
                delta="" if row.get("delta") is None else row.get("delta"),
                rate=""
                if row.get("rate_per_second") is None
                else f"{float(row['rate_per_second']):.6f}",
            )
        )
    if not delta.get("counters"):
        lines.append("|  |  |  |  | `unsupported` |  |  |")
    lines.extend(["", "## Path Balance Observations", ""])
    if delta.get("path_balance"):
        lines.extend(
            [
                "| Family | Source command | Counter | Ports | Minimum rate | Maximum rate | Observation |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in delta["path_balance"]:
            lines.append(
                "| {family} | {source_command} | {metric} | {ports} | {minimum:.6f} | {maximum:.6f} | `{observation}` |".format(
                    family=row["family"],
                    source_command=row.get("source_command") or "",
                    metric=row["metric"],
                    ports=row["resource_count"],
                    minimum=row["min_rate_per_second"],
                    maximum=row["max_rate_per_second"],
                    observation=row["observation"],
                )
            )
        lines.extend(
            [
                "",
                "Unequal sampled rates describe traffic distribution. They do not prove an ECMP collision or identify its cause.",
            ]
        )
    else:
        lines.append(
            "No comparable traffic counters were available from at least two sampled ports."
        )
    lines.extend(
        [
            "",
            "A decreasing cumulative counter is marked `reset_or_wrap`. No delta is calculated when the counter width and reset provenance are unknown.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest_fragment(
    run_id: str,
    run_dir: Path,
    artifact_paths: Iterable[Path],
    *,
    kind: str,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    files: list[dict[str, Any]] = []
    for path in sorted(set(artifact_paths)):
        if not path.exists():
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": schema_version,
        "artifact_type": "diagnostic_manifest_fragment",
        "diagnostic_kind": kind,
        "run_id": run_id,
        "created_at": _utc_now(),
        "canonical": False,
        "files": files,
    }
    path = run_dir / "structured" / f"{run_id}_{kind}_manifest.json"
    _write_json(path, payload)
    _merge_run_manifest_tool(
        run_id,
        run_dir,
        kind=kind,
        fragment_path=path,
        schema_version=schema_version,
    )
    return path


def _merge_run_manifest_tool(
    run_id: str,
    run_dir: Path,
    *,
    kind: str,
    fragment_path: Path,
    schema_version: str,
) -> None:
    """Add a tool artifact reference without replacing canonical manifest fields."""
    import fcntl

    manifest_path = run_dir / "manifest.json"
    lock_path = run_dir / ".diagnostic-bundle.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"run manifest must contain a JSON object: {manifest_path}")
        else:
            manifest = {
                "manifest_version": 2,
                "run_id": run_id,
                "artifact_root": str(run_dir),
                "finalized": False,
            }
        tools = manifest.setdefault("tools", {})
        if not isinstance(tools, dict):
            raise ValueError(
                f"run manifest tools field must contain a JSON object: {manifest_path}"
            )
        tools[kind] = {
            "schema_version": schema_version,
            "canonical": False,
            "manifest_fragment": str(fragment_path.relative_to(run_dir)),
            "updated_at": _utc_now(),
        }
        _atomic_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )


def run_fabric_diagnostics(
    *,
    run_id: str,
    run_dir: Path,
    family: str = "all",
    cumulus_hosts: Sequence[str] = (),
    switch_interfaces: Sequence[str] = (),
    cumulus_user: str | None = None,
    cumulus_ssh_key: str | None = None,
    ib_mgmt_host: str | None = None,
    ib_mgmt_user: str | None = None,
    ib_mgmt_ssh_key: str | None = None,
    ib_endpoints: Sequence[str] = (),
    ib_route_lids: Sequence[str] = (),
    ib_switch_lids: Sequence[str] = (),
    interval_seconds: float = 1.0,
    workload_command: str | None = None,
    workload_id: str | None = None,
    runner: CommandRunner | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    if interval_seconds < 0 or interval_seconds > 300:
        raise ValueError("interval_seconds must be between 0 and 300")
    run_dir = Path(run_dir).resolve()
    for subdir in ("raw", "structured", "reports"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    command_runner = runner or _default_runner
    common = {
        "run_id": run_id,
        "run_dir": run_dir,
        "family": family,
        "cumulus_hosts": cumulus_hosts,
        "switch_interfaces": switch_interfaces,
        "cumulus_user": cumulus_user,
        "cumulus_ssh_key": cumulus_ssh_key,
        "ib_mgmt_host": ib_mgmt_host,
        "ib_mgmt_user": ib_mgmt_user,
        "ib_mgmt_ssh_key": ib_mgmt_ssh_key,
        "ib_endpoints": ib_endpoints,
        "ib_route_lids": ib_route_lids,
        "ib_switch_lids": ib_switch_lids,
        "runner": command_runner,
        "timeout_seconds": timeout_seconds,
    }
    before = collect_fabric_snapshot(phase="before", **common)
    workload: dict[str, Any] | None = None
    if workload_command:
        workload = _run_workload(
            workload_command,
            run_id=run_id,
            run_dir=run_dir,
            runner=command_runner,
            timeout_seconds=timeout_seconds,
            workload_id=workload_id,
        )
    elif interval_seconds:
        time.sleep(interval_seconds)
    after = collect_fabric_snapshot(phase="after", **common)
    delta = analyze_snapshot_pair(before, after)
    if workload:
        delta["workload"] = {
            "workload_id": workload["workload_id"],
            "status": workload["status"],
            "returncode": workload["returncode"],
            "duration_seconds": workload["duration_seconds"],
            "raw_evidence": workload["raw_evidence"],
        }
        _tag_delta_workload(
            delta,
            workload["workload_id"],
            status=workload["status"],
            returncode=workload["returncode"],
        )
    elif workload_id:
        _tag_delta_workload(
            delta,
            workload_id,
            status="declared_interval",
            returncode=None,
        )
    status = str(delta.get("status") or "invalid")
    if workload and workload.get("status") != "ok":
        status = "error"
    elif status not in {"invalid", "error"} and any(
        snapshot.get("status") == "error" for snapshot in (before, after)
    ):
        status = "error" if status == "unsupported" else "partial"
    elif status == "ok" and any(
        snapshot.get("status") == "partial" for snapshot in (before, after)
    ):
        status = "partial"
    structured = run_dir / "structured"
    before_path = structured / f"{run_id}_fabric_diagnostics_before.json"
    after_path = structured / f"{run_id}_fabric_diagnostics_after.json"
    delta_path = structured / f"{run_id}_fabric_counter_deltas.json"
    _write_json(before_path, before)
    _write_json(after_path, after)
    _write_json(delta_path, delta)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "schema": "aisp.diagnostic-signals.v1",
        "artifact_type": "fabric_diagnostics",
        "run_id": run_id,
        "created_at": _utc_now(),
        "canonical": False,
        "collection_mode": "read_only_timed_snapshot",
        "status": status,
        "before": before,
        "workload": workload,
        "declared_workload_id": workload["workload_id"] if workload else workload_id,
        "after": after,
        "delta": delta,
        "signals": delta.get("signals") or [],
        "artifacts": {
            "before": str(before_path.relative_to(run_dir)),
            "after": str(after_path.relative_to(run_dir)),
            "delta": str(delta_path.relative_to(run_dir)),
        },
    }
    payload_path = structured / f"{run_id}_fabric_diagnostics.json"
    report_path = run_dir / "reports" / f"{run_id}_fabric_diagnostics.md"
    manifest_path = structured / f"{run_id}_fabric_diagnostics_manifest.json"
    payload["artifacts"].update(
        {
            "summary": str(payload_path.relative_to(run_dir)),
            "report": str(report_path.relative_to(run_dir)),
            "manifest": str(manifest_path.relative_to(run_dir)),
        }
    )
    _write_json(payload_path, payload)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    _write_manifest_fragment(
        run_id,
        run_dir,
        [
            before_path,
            after_path,
            delta_path,
            payload_path,
            report_path,
            *(run_dir / "raw").glob(f"{run_id}_fabric_diagnostics_*.json"),
        ],
        kind="fabric_diagnostics",
    )
    return payload


def analyze_retained_snapshots(
    *,
    run_id: str,
    run_dir: Path,
    before_path: Path,
    after_path: Path,
    workload_id: str | None = None,
) -> dict[str, Any]:
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    delta = analyze_snapshot_pair(before, after)
    delta["source_run_ids"] = sorted(
        {str(snapshot.get("run_id")) for snapshot in (before, after) if snapshot.get("run_id")}
    )
    delta["run_id"] = run_id
    if workload_id:
        _tag_delta_workload(
            delta,
            workload_id,
            status="declared_retained_interval",
            returncode=None,
        )
    run_dir = Path(run_dir).resolve()
    for subdir in ("structured", "reports"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    delta_path = run_dir / "structured" / f"{run_id}_fabric_counter_deltas.json"
    payload_path = run_dir / "structured" / f"{run_id}_fabric_diagnostics.json"
    report_path = run_dir / "reports" / f"{run_id}_fabric_diagnostics.md"
    manifest_path = run_dir / "structured" / f"{run_id}_fabric_diagnostics_manifest.json"
    _write_json(delta_path, delta)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "schema": "aisp.diagnostic-signals.v1",
        "artifact_type": "fabric_diagnostics",
        "run_id": run_id,
        "created_at": _utc_now(),
        "canonical": False,
        "collection_mode": "retained_snapshot_analysis",
        "status": delta["status"],
        "before": before,
        "after": after,
        "delta": delta,
        "signals": delta.get("signals") or [],
        "workload": None,
        "declared_workload_id": workload_id,
        "artifacts": {
            "before": str(before_path),
            "after": str(after_path),
            "delta": str(delta_path.relative_to(run_dir)),
            "summary": str(payload_path.relative_to(run_dir)),
            "report": str(report_path.relative_to(run_dir)),
            "manifest": str(manifest_path.relative_to(run_dir)),
        },
    }
    _write_json(payload_path, payload)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    _write_manifest_fragment(
        run_id, run_dir, [delta_path, payload_path, report_path], kind="fabric_diagnostics"
    )
    return payload


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect or analyze read-only AI fabric diagnostic evidence."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--family", choices=("all", "roce", "infiniband"), default="all")
    parser.add_argument("--cumulus-hosts", default="")
    parser.add_argument("--switch-interfaces", default="")
    parser.add_argument("--cumulus-user", default="")
    parser.add_argument("--cumulus-ssh-key", default="")
    parser.add_argument("--ib-mgmt-host", default="")
    parser.add_argument("--ib-mgmt-user", default="")
    parser.add_argument("--ib-mgmt-ssh-key", default="")
    parser.add_argument("--ib-endpoints", default="", help="Comma-separated <lid>:<port> values")
    parser.add_argument("--ib-route-lids", default="", help="Two comma-separated host LIDs")
    parser.add_argument("--ib-switch-lids", default="")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--workload-command", default="")
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--before", default="", help="Retained before snapshot JSON")
    parser.add_argument("--after", default="", help="Retained after snapshot JSON")
    parser.add_argument("--require-evidence", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.before) != bool(args.after):
        raise SystemExit("--before and --after must be supplied together")
    if args.before:
        payload = analyze_retained_snapshots(
            run_id=args.run_id,
            run_dir=Path(args.run_dir),
            before_path=Path(args.before),
            after_path=Path(args.after),
            workload_id=args.workload_id or None,
        )
    else:
        payload = run_fabric_diagnostics(
            run_id=args.run_id,
            run_dir=Path(args.run_dir),
            family=args.family,
            cumulus_hosts=_csv(args.cumulus_hosts),
            switch_interfaces=_csv(args.switch_interfaces),
            cumulus_user=args.cumulus_user or None,
            cumulus_ssh_key=args.cumulus_ssh_key or None,
            ib_mgmt_host=args.ib_mgmt_host or None,
            ib_mgmt_user=args.ib_mgmt_user or None,
            ib_mgmt_ssh_key=args.ib_mgmt_ssh_key or None,
            ib_endpoints=_csv(args.ib_endpoints),
            ib_route_lids=_csv(args.ib_route_lids),
            ib_switch_lids=_csv(args.ib_switch_lids),
            interval_seconds=args.interval_seconds,
            workload_command=args.workload_command or None,
            workload_id=args.workload_id or None,
            timeout_seconds=args.timeout_seconds,
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_id": payload["run_id"],
                "artifacts": payload["artifacts"],
            },
            sort_keys=True,
        )
    )
    workload_status = str((payload.get("workload") or {}).get("status") or "")
    command_error = any(
        command.get("status") == "error"
        for snapshot_name in ("before", "after")
        for command in (payload.get(snapshot_name) or {}).get("commands") or []
    )
    if (
        payload["status"] in {"invalid", "error"}
        or workload_status in {"invalid", "error"}
        or command_error
    ):
        return 1
    if payload["status"] == "unsupported":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
