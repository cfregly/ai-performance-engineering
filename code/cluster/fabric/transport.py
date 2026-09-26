from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster.fabric.diagnostics import (
    CommandRunner,
    _collector_clock_domain,
    _default_runner,
    _host_identity,
    _result_status,
    _safe_token,
    _utc_now,
    _write_json,
    _write_manifest_fragment,
)

SCHEMA_VERSION = "transport-diagnostics.v1"
_GDR_UNSUPPORTED_PATTERNS = (
    "cuda memory type is not supported",
    "unsupported with no odp mr",
    "no cuda devices",
    "cuda is not available",
    "dmabuf is not supported",
)


@dataclass(frozen=True)
class TransportConfig:
    run_id: str
    run_dir: Path
    cases: tuple[str, ...]
    server_address: str | None
    hca: str | None
    rail: str | None
    ib_port: int
    host_rdma_control_port: int
    gdr_control_port: int
    payload_bytes: int
    iterations: int
    queue_pairs: int
    direction: str
    gpu_id: int | None
    cuda_mem_type: int | None
    use_dmabuf: bool
    workload_id: str | None
    local_p2p_command: str | None
    nccl_command: str | None
    iperf_seconds: int
    iperf_parallel: int
    timeout_seconds: int


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _identity(config: TransportConfig, case: str) -> dict[str, Any]:
    if case == "local_p2p":
        return {
            "case": case,
            "peer": "local_gpu_pairs",
            "hca": None,
            "rail": "local_gpu_fabric",
            "ib_port": None,
            "control_port": None,
            "payload_bytes": config.payload_bytes,
            "payload_regime": "command_defined",
            "direction": "pairwise",
            "iterations": None,
            "queue_pairs": None,
        }
    return {
        "case": case,
        "peer": config.server_address,
        "hca": config.hca,
        "rail": config.rail,
        "ib_port": config.ib_port,
        "control_port": config.gdr_control_port
        if case == "gdr"
        else config.host_rdma_control_port
        if case == "host_rdma"
        else None,
        "payload_bytes": config.payload_bytes,
        "payload_regime": "fixed_message" if case != "iperf" else "bulk_stream",
        "direction": config.direction,
        "iterations": config.iterations if case != "iperf" else None,
        "queue_pairs": config.queue_pairs if case in {"host_rdma", "gdr"} else None,
    }


def _extract_marked_section(text: str, begin: str, end: str) -> str:
    if begin not in text or end not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0]


def _parse_gpu_matrix(text: str, *, access: bool) -> dict[str, Any]:
    # Recent nvidia-smi releases underline headers even in captured output.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    header: list[str] = []
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and all(re.fullmatch(r"GPU\d+", token) for token in tokens[:2]):
            header = [token for token in tokens if re.fullmatch(r"GPU\d+", token)]
            continue
        if header and tokens and re.fullmatch(r"GPU\d+", tokens[0]):
            rows[tokens[0]] = tokens[1 : 1 + len(header)]
    parsed: dict[str, Any] = {}
    for source, cells in rows.items():
        for target, value in zip(header, cells, strict=False):
            if source == target:
                continue
            key = f"{source[3:]}-{target[3:]}"
            if access:
                normalized = value.upper()
                if normalized in {"OK", "Y", "YES", "1"}:
                    parsed[key] = True
                elif normalized in {"CNS", "N", "NO", "0"}:
                    parsed[key] = False
            else:
                parsed[key] = value
    return parsed


def parse_local_p2p_output(text: str) -> dict[str, Any] | None:
    topology_text = _extract_marked_section(
        text,
        "__AISP_TOPOLOGY_MATRIX_BEGIN__",
        "__AISP_TOPOLOGY_MATRIX_END__",
    )
    access_text = _extract_marked_section(
        text,
        "__AISP_P2P_ACCESS_BEGIN__",
        "__AISP_P2P_ACCESS_END__",
    )
    pairs: list[dict[str, Any]] = []
    pair_pattern = re.compile(
        r"GPU\s*(\d+)\s*(?:→|->|to)\s*GPU\s*(\d+)\s*:\s*"
        r"(\d+(?:\.\d+)?)\s*GB/s",
        re.IGNORECASE,
    )
    for match in pair_pattern.finditer(text):
        bandwidth = float(match.group(3))
        if math.isfinite(bandwidth) and bandwidth > 0:
            pairs.append(
                {
                    "src_gpu": int(match.group(1)),
                    "dst_gpu": int(match.group(2)),
                    "bandwidth_gbps": bandwidth,
                    "source": "pairwise_text",
                }
            )
    for line in text.splitlines():
        if not line.startswith("AISP_LOCAL_P2P_JSON="):
            continue
        try:
            payload = json.loads(line.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        for row in payload.get("pairs", []) if isinstance(payload, dict) else []:
            try:
                source = int(row["src_gpu"])
                target = int(row["dst_gpu"])
                bandwidth = float(row["bandwidth_gbps"])
            except (KeyError, TypeError, ValueError):
                continue
            if source != target and math.isfinite(bandwidth) and bandwidth > 0:
                pairs.append(
                    {
                        "src_gpu": source,
                        "dst_gpu": target,
                        "bandwidth_gbps": bandwidth,
                        "source": "aisp_local_p2p_json",
                    }
                )
    if not pairs:
        return None
    deduplicated = {(row["src_gpu"], row["dst_gpu"]): row for row in pairs}
    pairs = [deduplicated[key] for key in sorted(deduplicated)]
    rates = [row["bandwidth_gbps"] for row in pairs]
    return {
        "pair_count": len(pairs),
        "min_bandwidth_gbps": min(rates),
        "mean_bandwidth_gbps": sum(rates) / len(rates),
        "max_bandwidth_gbps": max(rates),
        "unit": "GB/s",
        "pairs": pairs,
        "topology_links": _parse_gpu_matrix(topology_text, access=False),
        "peer_read_access": _parse_gpu_matrix(access_text, access=True),
    }


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _command_env_value(command: str, name: str) -> str | None:
    prefix = f"{name}="
    tokens = _command_tokens(command)
    for index, token in enumerate(tokens):
        candidate = token
        if token in {"-x", "--env"} and index + 1 < len(tokens):
            candidate = tokens[index + 1]
        elif token.startswith("-x") and token != "-x":
            candidate = token[2:]
        if candidate.startswith(prefix):
            return candidate[len(prefix) :]
    return None


def _command_mentions_value(command: str, value: str | None) -> bool:
    if not value:
        return False
    for token in _command_tokens(command):
        if token == value or token.endswith(f"={value}"):
            return True
        if token.startswith("--host") and value in token:
            return True
    return False


def _dimension_evidence(
    config: TransportConfig,
    case: str,
    command: str,
    metric: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    observed: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    if case == "local_p2p":
        pairs = list((metric or {}).get("pairs") or [])
        topology = dict((metric or {}).get("topology_links") or {})
        access = dict((metric or {}).get("peer_read_access") or {})
        observed = {
            "gpu_pairs": [f"{row['src_gpu']}->{row['dst_gpu']}" for row in pairs],
            "topology_links": topology,
            "peer_read_access": access,
        }
        validation = {
            "gpu_pairs": "matched_output" if pairs else "unverified",
            "topology": "matched_nvidia_smi_output" if topology else "unverified",
            "peer_access": "matched_nvidia_smi_output" if access else "unverified",
            "payload": "declared_only_effective_payload_unverified",
            "path": "capability_observed_runtime_path_unverified",
        }
        return observed, validation, False, False
    if case in {"host_rdma", "gdr"}:
        observed = {
            "payload_bytes": (metric or {}).get("message_bytes"),
            "gdr_supported_by_execution": bool(case == "gdr" and metric),
        }
        payload_matches = observed["payload_bytes"] == config.payload_bytes
        validation = {
            "peer": "configured_by_generated_command",
            "hca": "configured_by_generated_command_effective_path_unverified",
            "ib_port": "configured_by_generated_command_effective_path_unverified",
            "payload": "matched"
            if payload_matches
            else "mismatch"
            if observed["payload_bytes"] is not None
            else "configured_by_generated_command_output_unverified",
            "direction": "configured_by_generated_command",
            "rail": "declared_only" if config.rail else "not_declared",
        }
        return observed, validation, bool(payload_matches), False
    if case == "nccl":
        sizes = sorted({int(row["size_bytes"]) for row in (metric or {}).get("results", [])})
        command_hcas = _command_env_value(command, "NCCL_IB_HCA")
        declared_hca = bool(config.hca and command_hcas and config.hca in command_hcas.split(","))
        declared_peer = _command_mentions_value(command, config.server_address)
        declared_rail = _command_mentions_value(command, config.rail)
        runtime_hcas = list((metric or {}).get("runtime_transport_evidence", {}).get("hcas") or [])
        runtime_hca_match = bool(config.hca and config.hca in runtime_hcas)
        payload_match = config.payload_bytes in sizes
        observed = {
            "payload_bytes": sizes,
            "hcas": runtime_hcas,
            "ib_ports": list(
                (metric or {}).get("runtime_transport_evidence", {}).get("ib_ports") or []
            ),
            "direction": "collective",
        }
        validation = {
            "payload": "matched" if payload_match else "mismatch",
            "hca": "matched_runtime_log"
            if runtime_hca_match
            else "declared_in_command_effective_path_unverified"
            if declared_hca
            else "unverified",
            "peer": "declared_in_command_effective_path_unverified"
            if declared_peer
            else "unverified",
            "rail": "declared_in_command_effective_path_unverified"
            if declared_rail
            else "unverified",
            "direction": "not_comparable_to_point_to_point_direction",
        }
        return observed, validation, payload_match, False
    if case == "iperf":
        observed_peer = (metric or {}).get("peer")
        observed_direction = (metric or {}).get("direction")
        observed = {
            "peer": observed_peer,
            "direction": observed_direction,
        }
        validation = {
            "peer": "matched_output"
            if observed_peer == config.server_address
            else "mismatch_output"
            if observed_peer
            else "configured_by_generated_command_runtime_peer_unverified",
            "direction": "matched_output"
            if observed_direction == ("read" if config.direction == "read" else "write")
            else "mismatch_output"
            if observed_direction
            else "configured_by_generated_command_runtime_direction_unverified",
            "payload": "not_comparable_to_fixed_message_payload",
            "hca": "not_applicable",
            "rail": "unverified",
        }
        return observed, validation, False, False
    return observed, validation, False, False


def parse_perftest_output(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        dictionaries: list[dict[str, Any]] = []
        stack = [payload]
        while stack:
            current = stack.pop()
            dictionaries.append(current)
            stack.extend(value for value in current.values() if isinstance(value, dict))
        metric: tuple[float, dict[str, Any]] | None = None
        for results in dictionaries:
            for key, raw_value in results.items():
                normalized = re.sub(r"[^a-z0-9]+", "", str(key).lower())
                if normalized not in {
                    "bwaverage",
                    "bwavggbps",
                    "bwaveragegbps",
                    "averagebwgbps",
                    "bwaveragegbsec",
                }:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    metric = (value, results)
                    break
            if metric:
                break
        if metric:

            def find_positive_integer(keys: set[str]) -> int | None:
                for record in dictionaries:
                    for key, raw_value in record.items():
                        normalized = re.sub(r"[^a-z0-9]+", "", str(key).lower())
                        if normalized not in keys:
                            continue
                        try:
                            value = int(raw_value)
                        except (TypeError, ValueError):
                            continue
                        if value > 0:
                            return value
                return None

            value, _ = metric
            return {
                "message_bytes": find_positive_integer({"size", "bytes", "messagesize", "msgsize"}),
                "iterations": find_positive_integer({"iterations", "iters"}),
                "average_gbps": value,
                "unit": "Gb/s",
                "source": "perftest_json",
            }
    candidates: list[dict[str, Any]] = []
    recognized_header = False
    for line in stripped.splitlines():
        lowered = line.lower()
        if "#bytes" in lowered and "#iterations" in lowered and "bw" in lowered:
            recognized_header = True
            continue
        if not recognized_header:
            continue
        values = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", line)
        if len(values) < 4 or line.lstrip().startswith("#"):
            continue
        try:
            size = int(float(values[0]))
            iterations = int(float(values[1]))
            average_gbps = float(values[3])
        except ValueError:
            continue
        if size > 0 and iterations > 0 and math.isfinite(average_gbps) and average_gbps > 0:
            candidates.append(
                {
                    "message_bytes": size,
                    "iterations": iterations,
                    "average_gbps": average_gbps,
                    "unit": "Gb/s",
                    "source": "perftest_text",
                    "source_line": line.strip(),
                }
            )
    return candidates[-1] if candidates else None


def parse_nccl_output(text: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    runtime_hcas: set[str] = set()
    runtime_ports: set[int] = set()
    recognized_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"NET/IB", stripped, re.IGNORECASE):
            for transport_match in re.finditer(
                r"(mlx5_[A-Za-z0-9_.-]+)(?::(\d+))?",
                stripped,
                re.IGNORECASE,
            ):
                runtime_hcas.add(transport_match.group(1))
                if transport_match.group(2):
                    runtime_ports.add(int(transport_match.group(2)))
        header_tokens = set(re.findall(r"[a-z]+", stripped.lower()))
        if {"size", "count", "time", "algbw", "busbw"}.issubset(header_tokens):
            recognized_header = True
        if not stripped or stripped.startswith("#"):
            continue
        if not recognized_header:
            continue
        tokens = stripped.split()
        if len(tokens) < 8 or not tokens[0].isdigit():
            continue
        try:
            size = int(tokens[0])
            time_us = float(tokens[5])
            algbw = float(tokens[6])
            busbw = float(tokens[7])
        except (ValueError, IndexError):
            continue
        if (
            size > 0
            and all(math.isfinite(value) for value in (time_us, algbw, busbw))
            and time_us > 0
            and algbw >= 0
            and busbw >= 0
        ):
            rows.append(
                {
                    "size_bytes": size,
                    "time_us": time_us,
                    "algbw_gbps": algbw,
                    "busbw_gbps": busbw,
                    "source_line": stripped,
                }
            )
    if not rows:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        for row in payload.get("results", []) if isinstance(payload, dict) else []:
            try:
                parsed = {
                    "size_bytes": int(row["size_bytes"]),
                    "time_us": float(row.get("time_us") or row.get("latency_us")),
                    "algbw_gbps": float(row["algbw_gbps"]),
                    "busbw_gbps": float(row["busbw_gbps"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if (
                parsed["size_bytes"] > 0
                and all(
                    math.isfinite(parsed[key]) for key in ("time_us", "algbw_gbps", "busbw_gbps")
                )
                and parsed["time_us"] > 0
                and parsed["algbw_gbps"] >= 0
                and parsed["busbw_gbps"] >= 0
            ):
                rows.append(parsed)
    if not rows:
        return None
    best = max(rows, key=lambda row: row["busbw_gbps"])
    return {
        "result_count": len(rows),
        "peak_busbw_gbps": best["busbw_gbps"],
        "peak_algbw_gbps": best["algbw_gbps"],
        "peak_size_bytes": best["size_bytes"],
        "unit": "GB/s",
        "results": rows,
        "runtime_transport_evidence": {
            "hcas": sorted(runtime_hcas),
            "ib_ports": sorted(runtime_ports),
        },
    }


def parse_iperf_output(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    end = payload.get("end") if isinstance(payload, dict) else None
    if not isinstance(end, dict):
        return None
    for key in ("sum_received", "sum_sent", "sum"):
        record = end.get(key)
        if not isinstance(record, dict):
            continue
        try:
            bps = float(record["bits_per_second"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(bps) and bps > 0:
            start = payload.get("start")
            start = start if isinstance(start, dict) else {}
            connected = start.get("connected", [])
            peer = None
            if isinstance(connected, list) and connected and isinstance(connected[0], dict):
                peer = connected[0].get("remote_host")
            test_start = start.get("test_start", {})
            reverse = bool(test_start.get("reverse")) if isinstance(test_start, dict) else False
            return {
                "gbps": bps / 1e9,
                "unit": "Gb/s",
                "retransmits": record.get("retransmits"),
                "seconds": record.get("seconds"),
                "source": key,
                "peer": peer,
                "direction": "read" if reverse else "write",
            }
    return None


def _perftest_command(config: TransportConfig, *, gdr: bool) -> str:
    if not config.server_address or not config.hca:
        raise ValueError("server_address and hca are required for perftest")
    tool = f"ib_{config.direction}_bw"
    argv = [
        tool,
        "-d",
        config.hca,
        "-i",
        str(config.ib_port),
        "-q",
        str(config.queue_pairs),
        "-s",
        str(config.payload_bytes),
        "-n",
        str(config.iterations),
        "--report_gbits",
        "--out_json",
        "-p",
        str(config.gdr_control_port if gdr else config.host_rdma_control_port),
    ]
    if gdr:
        if config.gpu_id is None:
            raise ValueError("gpu_id is required for GDR perftest")
        argv.extend(["--use_cuda", str(config.gpu_id)])
        if config.cuda_mem_type is not None:
            argv.extend(["--cuda_mem_type", str(config.cuda_mem_type)])
        if config.use_dmabuf:
            argv.append("--use_cuda_dmabuf")
    argv.append(config.server_address)
    return " ".join(shlex.quote(value) for value in argv)


def _command_for_case(config: TransportConfig, case: str) -> tuple[str | None, str | None]:
    if case == "local_p2p":
        if not config.local_p2p_command:
            return None, "local GPU P2P requires an explicit bounded --local-p2p-command"
        command = (
            "printf '__AISP_TOPOLOGY_MATRIX_BEGIN__\\n'; "
            "nvidia-smi topo -m; topology_rc=$?; "
            "printf '__AISP_TOPOLOGY_MATRIX_END__\\n'; "
            'if [ "$topology_rc" -ne 0 ]; then exit "$topology_rc"; fi; '
            "printf '__AISP_P2P_ACCESS_BEGIN__\\n'; "
            "nvidia-smi topo -p2p r; access_rc=$?; "
            "printf '__AISP_P2P_ACCESS_END__\\n'; "
            'if [ "$access_rc" -ne 0 ]; then '
            "echo 'unsupported: nvidia-smi peer access matrix unavailable' >&2; fi; "
            f"{config.local_p2p_command}"
        )
        return command, None
    if case == "host_rdma":
        if not config.server_address or not config.hca:
            return None, "host RDMA requires --server-address and --hca"
        return _perftest_command(config, gdr=False), None
    if case == "gdr":
        if not config.server_address or not config.hca or config.gpu_id is None:
            return None, "GDR requires --server-address, --hca, and --gpu-id"
        return _perftest_command(config, gdr=True), None
    if case == "nccl":
        if not config.nccl_command:
            return None, "NCCL requires an explicit bounded --nccl-command"
        return config.nccl_command, None
    if case == "iperf":
        if not config.server_address:
            return None, "iperf requires --server-address"
        argv = [
            "iperf3",
            "-c",
            config.server_address,
            "-t",
            str(config.iperf_seconds),
            "-P",
            str(config.iperf_parallel),
            "--json",
        ]
        if config.direction == "read":
            argv.append("-R")
        return " ".join(shlex.quote(value) for value in argv), None
    raise ValueError(f"unknown transport case {case!r}")


def _parse_case(case: str, stdout: str) -> dict[str, Any] | None:
    if case == "local_p2p":
        return parse_local_p2p_output(stdout)
    if case in {"host_rdma", "gdr"}:
        return parse_perftest_output(stdout)
    if case == "nccl":
        return parse_nccl_output(stdout)
    if case == "iperf":
        return parse_iperf_output(stdout)
    return None


def _measurement_signal(
    config: TransportConfig,
    case: str,
    command: str,
    metric: dict[str, Any] | None,
    *,
    start_unix_s: float,
    end_unix_s: float,
    evidence: str,
) -> list[dict[str, Any]]:
    if not metric or end_unix_s <= start_unix_s:
        return []
    metric_fields = {
        "local_p2p": ("mean_bandwidth_gbps", "local_p2p_mean_bandwidth"),
        "host_rdma": ("average_gbps", "host_rdma_average_bandwidth"),
        "gdr": ("average_gbps", "gdr_average_bandwidth"),
        "nccl": ("peak_busbw_gbps", "nccl_peak_bus_bandwidth"),
        "iperf": ("gbps", "tcp_average_bandwidth"),
    }
    field, signal_metric = metric_fields[case]
    value = metric.get(field)
    unit = metric.get("unit")
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        or unit not in {"Gb/s", "GB/s"}
    ):
        return []
    host_token = str(_host_identity(None)["host"])
    scope = {
        "host": host_token,
        "workload_id": config.workload_id
        or hashlib.sha256(command.encode("utf-8")).hexdigest()[:12],
    }
    if case == "local_p2p":
        pairs = ",".join(f"{row['src_gpu']}->{row['dst_gpu']}" for row in metric.get("pairs") or [])
        if pairs:
            scope["path"] = f"observed:local_gpu_pairs:{pairs}"
    elif case in {"host_rdma", "gdr"}:
        scope["path"] = (
            f"configured:rdma:{config.direction}:{config.hca}:{config.ib_port}"
            f"->{config.server_address}"
        )
    elif case == "nccl":
        scope["path"] = (
            f"configured:nccl:hca={config.hca or 'unspecified'}:"
            f"peer={config.server_address or 'unspecified'}"
        )
    else:
        scope["path"] = f"configured:tcp:peer={config.server_address}"
    return [
        {
            "kind": "throughput",
            "role": "measurement",
            "metric": signal_metric,
            "value": value,
            "unit": unit,
            "start_unix_s": start_unix_s,
            "end_unix_s": end_unix_s,
            "clock_domain": _collector_clock_domain(),
            "scope": scope,
            "path_verification": (
                "observed_local_topology_and_peer_read_capability_runtime_route_unverified"
                if metric.get("peer_read_access")
                else "observed_local_pair_rate_peer_access_unverified"
            )
            if case == "local_p2p"
            else "configured_selection_effective_path_unverified",
            "evidence": [evidence],
        }
    ]


def _run_case(config: TransportConfig, case: str, runner: CommandRunner) -> dict[str, Any]:
    command, unsupported_reason = _command_for_case(config, case)
    declared_identity = _identity(config, case)
    if command is None:
        return {
            "case": case,
            "declared_identity": declared_identity,
            "observed_identity": {},
            "dimension_validation": {},
            "rate_comparison_eligible": False,
            "matched_path_eligible": False,
            "status": "unsupported",
            "reason": unsupported_reason,
            "command": None,
            "metric": None,
            "signals": [],
            "raw_evidence": None,
        }
    started = time.time()
    result = runner(command, host=None, user=None, ssh_key=None, timeout=config.timeout_seconds)
    finished = time.time()
    command_status = _result_status(result)
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    combined = f"{stdout}\n{stderr}".lower()
    if case == "gdr" and any(pattern in combined for pattern in _GDR_UNSUPPORTED_PATTERNS):
        command_status = "unsupported"
    parse_text = f"{stdout}\n{stderr}" if case == "nccl" else stdout
    metric = _parse_case(case, parse_text) if command_status == "ok" else None
    status = command_status
    reason = str(result.get("error") or "") or None
    if command_status == "ok" and metric is None:
        status = "invalid"
        reason = "command exited successfully but emitted no recognized performance metric"
    observed_identity, dimension_validation, rate_comparison_eligible, matched_path_eligible = (
        _dimension_evidence(
            config,
            case,
            command,
            metric,
        )
    )
    if status == "ok" and dimension_validation.get("payload") == "mismatch":
        status = "invalid"
        reason = "observed payload does not include the requested payload"
        rate_comparison_eligible = False
        matched_path_eligible = False
    raw_path = config.run_dir / "raw" / f"{config.run_id}_transport_{_safe_token(case)}.json"
    raw_reference = str(raw_path.relative_to(config.run_dir))
    signals = (
        _measurement_signal(
            config,
            case,
            command,
            metric,
            start_unix_s=started,
            end_unix_s=finished,
            evidence=raw_reference,
        )
        if status == "ok"
        else []
    )
    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "transport_command_evidence",
        "run_id": config.run_id,
        "declared_workload_id": config.workload_id,
        "captured_at": _utc_now(),
        "canonical": False,
        "declared_identity": declared_identity,
        "observed_identity": observed_identity,
        "dimension_validation": dimension_validation,
        "rate_comparison_eligible": rate_comparison_eligible,
        "matched_path_eligible": matched_path_eligible,
        "command": command,
        "status": status,
        "returncode": result.get("returncode"),
        "duration_seconds": max(0.0, finished - started),
        "start_unix_s": started,
        "end_unix_s": finished,
        "clock_domain": _collector_clock_domain(),
        "stdout": stdout,
        "stderr": stderr,
        "error": str(result.get("error") or ""),
        "timed_out": bool(result.get("timed_out")),
        "termination": result.get("termination"),
        "metric": metric,
        "signals": signals,
    }
    _write_json(raw_path, raw_payload)
    return {
        "case": case,
        "declared_identity": declared_identity,
        "observed_identity": observed_identity,
        "dimension_validation": dimension_validation,
        "rate_comparison_eligible": rate_comparison_eligible,
        "matched_path_eligible": matched_path_eligible,
        "status": status,
        "reason": reason,
        "command": command,
        "returncode": result.get("returncode"),
        "duration_seconds": max(0.0, finished - started),
        "start_unix_s": started,
        "end_unix_s": finished,
        "clock_domain": _collector_clock_domain(),
        "timed_out": bool(result.get("timed_out")),
        "termination": result.get("termination"),
        "metric": metric,
        "signals": signals,
        "raw_evidence": raw_reference,
    }


def _overall_status(cases: Sequence[dict[str, Any]]) -> str:
    statuses = [str(case.get("status")) for case in cases]
    valid = sum(status == "ok" for status in statuses)
    if statuses and valid == len(statuses):
        return "ok"
    if valid:
        return "partial"
    if any(status in {"error", "invalid"} for status in statuses):
        return "error"
    return "unsupported"


def _host_gdr_comparison(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row.get("case")): row for row in cases}
    host = by_name.get("host_rdma")
    gdr = by_name.get("gdr")
    base = {
        "comparison": "host_rdma_vs_gdr",
        "causal_interpretation_eligible": False,
        "path_verification": "unverified",
        "correctness_verification": "not_collected",
        "interpretation": "The rate ratio is descriptive. Missing effective-path and data-correctness evidence blocks a causal conclusion.",
    }
    if not host or not gdr:
        return {
            **base,
            "status": "unavailable",
            "reason": "both host_rdma and gdr cases are required",
        }
    if host.get("status") != "ok" or gdr.get("status") != "ok":
        return {**base, "status": "unavailable", "reason": "both cases require parsed measurements"}
    host_metric = host.get("metric") or {}
    gdr_metric = gdr.get("metric") or {}
    host_rate = host_metric.get("average_gbps")
    gdr_rate = gdr_metric.get("average_gbps")
    host_size = host_metric.get("message_bytes")
    gdr_size = gdr_metric.get("message_bytes")
    host_declared = host.get("declared_identity") or {}
    gdr_declared = gdr.get("declared_identity") or {}
    setting_keys = (
        "peer",
        "hca",
        "rail",
        "ib_port",
        "payload_bytes",
        "direction",
        "iterations",
        "queue_pairs",
    )
    settings_match = all(host_declared.get(key) == gdr_declared.get(key) for key in setting_keys)
    size_match = host_size is not None and host_size == gdr_size == host_declared.get(
        "payload_bytes"
    )
    if (
        not settings_match
        or not size_match
        or not isinstance(host_rate, int | float)
        or isinstance(host_rate, bool)
        or not isinstance(gdr_rate, int | float)
        or isinstance(gdr_rate, bool)
        or not math.isfinite(float(host_rate))
        or not math.isfinite(float(gdr_rate))
        or float(host_rate) <= 0
        or float(gdr_rate) <= 0
    ):
        return {
            **base,
            "status": "unavailable",
            "reason": "generated settings and parsed message sizes must match with positive finite rates",
        }
    return {
        **base,
        "status": "descriptive",
        "matched_generated_settings": {key: host_declared.get(key) for key in setting_keys},
        "message_bytes": host_size,
        "host_rdma_average_gbps": host_rate,
        "gdr_average_gbps": gdr_rate,
        "gdr_to_host_rate_ratio": float(gdr_rate) / float(host_rate),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Transport Diagnostics",
        "",
        f"Run ID: `{payload.get('run_id')}`",
        "",
        "These measurements are diagnostic and noncanonical. Each successful row comes from an executed command with a parsed metric.",
        "",
        "| Case | Status | Declared | Configured only | Observed | Unknown | Rate comparison | Matched path | Primary metric | Raw evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in payload.get("cases") or []:
        declared = case.get("declared_identity") or {}
        observed = case.get("observed_identity") or {}
        validation = case.get("dimension_validation") or {}
        metric = case.get("metric") or {}
        declared_text = ", ".join(
            f"{key}={value}"
            for key, value in declared.items()
            if value is not None and key not in {"case", "iterations"}
        )
        configured = sorted(
            key
            for key, state in validation.items()
            if str(state).startswith(("configured_", "declared_"))
        )
        observed_text = ", ".join(
            f"{key}={value}"
            for key, value in observed.items()
            if value is not None and value != "" and value != [] and value != {}
        )
        unknown = sorted(
            key
            for key, state in validation.items()
            if "unverified" in str(state) or str(state).startswith(("not_comparable", "mismatch"))
        )
        primary = ""
        for key in (
            "mean_bandwidth_gbps",
            "average_gbps",
            "peak_busbw_gbps",
            "gbps",
        ):
            if key in metric:
                primary = f"{key}={float(metric[key]):.3f} {metric.get('unit', '')}".strip()
                break
        lines.append(
            "| {case} | `{status}` | {declared} | {configured} | {observed} | {unknown} | {rate_eligible} | {path_eligible} | {primary} | `{evidence}` |".format(
                case=case.get("case", ""),
                status=case.get("status", ""),
                declared=declared_text,
                configured=", ".join(configured),
                observed=observed_text,
                unknown=", ".join(unknown),
                rate_eligible=str(bool(case.get("rate_comparison_eligible"))).lower(),
                path_eligible=str(bool(case.get("matched_path_eligible"))).lower(),
                primary=primary,
                evidence=case.get("raw_evidence") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Host RDMA and GDR Rate Ratio",
            "",
        ]
    )
    comparison = (payload.get("comparisons") or {}).get("host_vs_gdr") or {}
    if comparison.get("status") == "descriptive":
        lines.extend(
            [
                f"Host RDMA: `{float(comparison['host_rdma_average_gbps']):.3f} Gbps`",
                "",
                f"GDR: `{float(comparison['gdr_average_gbps']):.3f} Gbps`",
                "",
                f"GDR to host rate ratio: `{float(comparison['gdr_to_host_rate_ratio']):.6f}`",
                "",
                str(comparison["interpretation"]),
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Unavailable: {comparison.get('reason', 'matching measurements were not available')}.",
                "",
            ]
        )
    lines.extend(
        [
            "Perftest servers must be prepared explicitly before the corresponding client case runs. The tool does not change switch, NIC, or host configuration.",
            "",
            "A successful process without a recognized metric is marked `invalid`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_transport_diagnostics(
    config: TransportConfig, runner: CommandRunner | None = None
) -> dict[str, Any]:
    _positive(config.payload_bytes, "payload_bytes")
    _positive(config.iterations, "iterations")
    _positive(config.queue_pairs, "queue_pairs")
    _positive(config.ib_port, "ib_port")
    _positive(config.host_rdma_control_port, "host_rdma_control_port")
    _positive(config.gdr_control_port, "gdr_control_port")
    _positive(config.timeout_seconds, "timeout_seconds")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("raw", "structured", "reports"):
        (config.run_dir / subdir).mkdir(parents=True, exist_ok=True)
    command_runner = runner or _default_runner
    case_rows = [_run_case(config, case, command_runner) for case in config.cases]
    signals = [signal for row in case_rows for signal in row.get("signals") or []]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "schema": "aisp.diagnostic-signals.v1",
        "artifact_type": "transport_diagnostics",
        "run_id": config.run_id,
        "declared_workload_id": config.workload_id,
        "created_at": _utc_now(),
        "canonical": False,
        "collection_mode": "explicit_client_execution",
        "status": _overall_status(case_rows),
        "declared_dimensions": ["peer", "hca", "rail", "ib_port", "payload_bytes", "direction"],
        "comparison_rule": "Rate comparison and matched path verification are separate. Caller labels and command declarations do not prove the effective path.",
        "cases": case_rows,
        "signals": signals,
        "comparisons": {"host_vs_gdr": _host_gdr_comparison(case_rows)},
        "limitations": [
            "The tool does not prepare or mutate the remote server.",
            "A local topology matrix reports capability and connectivity. It does not prove the route taken by every transfer.",
            "iperf measures a TCP stream rather than RDMA message transport.",
            "NCCL command provenance is caller supplied and retained verbatim.",
            "Results remain noncanonical until the repository publication gates are satisfied.",
        ],
        "artifacts": {},
    }
    structured_path = config.run_dir / "structured" / f"{config.run_id}_transport_diagnostics.json"
    report_path = config.run_dir / "reports" / f"{config.run_id}_transport_diagnostics.md"
    manifest_path = (
        config.run_dir / "structured" / f"{config.run_id}_transport_diagnostics_manifest.json"
    )
    payload["artifacts"] = {
        "summary": str(structured_path.relative_to(config.run_dir)),
        "report": str(report_path.relative_to(config.run_dir)),
        "manifest": str(manifest_path.relative_to(config.run_dir)),
    }
    _write_json(structured_path, payload)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    _write_manifest_fragment(
        config.run_id,
        config.run_dir,
        [
            structured_path,
            report_path,
            *(config.run_dir / "raw").glob(f"{config.run_id}_transport_*.json"),
        ],
        kind="transport_diagnostics",
        schema_version=SCHEMA_VERSION,
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded local GPU, host RDMA, GDR, NCCL, and TCP diagnostics."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--cases",
        default="local_p2p,host_rdma,gdr,nccl",
        help="Comma-separated local_p2p,host_rdma,gdr,nccl,iperf",
    )
    parser.add_argument("--server-address", default="")
    parser.add_argument("--hca", default="")
    parser.add_argument("--rail", default="")
    parser.add_argument("--ib-port", type=int, default=1)
    parser.add_argument("--host-rdma-control-port", type=int, default=18515)
    parser.add_argument("--gdr-control-port", type=int, default=18516)
    parser.add_argument("--payload-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--queue-pairs", type=int, default=1)
    parser.add_argument("--direction", choices=("write", "read", "send"), default="write")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--cuda-mem-type", type=int)
    parser.add_argument("--use-dmabuf", action="store_true")
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--local-p2p-command", default="")
    parser.add_argument("--nccl-command", default="")
    parser.add_argument("--iperf-seconds", type=int, default=10)
    parser.add_argument("--iperf-parallel", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    allowed = {"local_p2p", "host_rdma", "gdr", "nccl", "iperf"}
    unknown = sorted(set(cases) - allowed)
    if not cases or unknown:
        raise SystemExit(f"--cases must contain known cases, unknown={','.join(unknown)}")
    config = TransportConfig(
        run_id=args.run_id,
        run_dir=Path(args.run_dir).resolve(),
        cases=cases,
        server_address=args.server_address or None,
        hca=args.hca or None,
        rail=args.rail or None,
        ib_port=args.ib_port,
        host_rdma_control_port=args.host_rdma_control_port,
        gdr_control_port=args.gdr_control_port,
        payload_bytes=args.payload_bytes,
        iterations=args.iterations,
        queue_pairs=args.queue_pairs,
        direction=args.direction,
        gpu_id=args.gpu_id,
        cuda_mem_type=args.cuda_mem_type,
        use_dmabuf=bool(args.use_dmabuf),
        workload_id=args.workload_id or None,
        local_p2p_command=args.local_p2p_command or None,
        nccl_command=args.nccl_command or None,
        iperf_seconds=args.iperf_seconds,
        iperf_parallel=args.iperf_parallel,
        timeout_seconds=args.timeout_seconds,
    )
    payload = run_transport_diagnostics(config)
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
    if payload["status"] == "error" or any(
        row.get("status") in {"error", "invalid"} for row in payload.get("cases") or []
    ):
        return 1
    if payload["status"] == "unsupported":
        return 2
    if args.require_all and payload["status"] != "ok":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
