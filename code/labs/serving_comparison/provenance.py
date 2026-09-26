"""Bind declared P/D roles to the launch file and observed GPU processes."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .process_group_launcher import _load
from .schema import ConfigError

LAUNCHER = "labs.serving_comparison.process_group_launcher"


def _python_command(command: list[str], module: str) -> bool:
    return (
        len(command) >= 3
        and re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(command[0]).name) is not None
        and command[1:3] == ["-m", module]
    )


def _child_entrypoint(command: list[str], engine: str, role: str) -> str:
    """Accept explicit native entrypoints, including a bounded SIF exec wrapper."""
    native = command
    if Path(command[0]).name in {"singularity", "apptainer"}:
        if len(command) < 4 or command[1] != "exec":
            raise ConfigError("P/D container launch requires an explicit exec command")
        index = 2
        while index < len(command) and command[index].startswith("-"):
            flag = command[index]
            if flag == "--nv":
                index += 1
            elif flag in {"--bind", "-B"}:
                index += 2
            elif flag.startswith("--bind="):
                index += 1
            else:
                raise ConfigError(f"Unsupported P/D container launch option: {flag}")
        if index >= len(command) or not command[index].endswith(".sif"):
            raise ConfigError("P/D container launch requires a pinned SIF image path")
        native = command[index + 1 :]
    if not native:
        raise ConfigError("P/D launch has no native entrypoint")
    if role == "router":
        module = (
            "labs.serving_comparison.vllm_pd_proxy"
            if engine == "vllm"
            else "sglang_router.launch_router"
        )
        valid = _python_command(native, module)
        if engine == "sglang" and native.count("--pd-disaggregation") != 1:
            raise ConfigError("SGLang router requires --pd-disaggregation exactly once")
    elif engine == "vllm":
        module = "vllm.entrypoints.openai.api_server"
        valid = _python_command(native, module) or (
            len(native) >= 3 and Path(native[0]).name == "vllm" and native[1] == "serve"
        )
    else:
        module = "sglang.launch_server"
        valid = _python_command(native, module)
    if not valid:
        raise ConfigError(f"P/D {engine} {role} uses an unsupported native entrypoint")
    return module


def _flag(command: list[str], name: str) -> str:
    values = []
    for index, part in enumerate(command):
        if part == name:
            if index + 1 == len(command):
                raise ConfigError(f"Launch argument {name} needs a value")
            values.append(command[index + 1])
        elif part.startswith(name + "="):
            values.append(part.split("=", 1)[1])
    if len(values) != 1:
        raise ConfigError(f"Launch command must declare {name} exactly once")
    return values[0]


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if host == "localhost":
        host = "127.0.0.1"
    return parsed.scheme, host, parsed.port or (443 if parsed.scheme == "https" else 80)


def bind_pd_launch(
    *,
    engine: str,
    endpoint: str,
    lifecycle: dict[str, Any],
    provenance: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Check the exact process-group bytes and every declared role before launch."""
    command = list(lifecycle["start_command"])
    if not _python_command(command, LAUNCHER):
        raise ConfigError("Engine P/D requires the bundled process_group_launcher --spec")
    command[0] = sys.executable
    spec_path = (Path(lifecycle["working_directory"]) / _flag(command, "--spec")).resolve()
    try:
        raw = spec_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        declarations = _load(spec_path, expected_sha256=digest)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot bind P/D launch spec: {exc}") from exc
    if provenance["manifest_digest"] != "sha256:" + digest:
        raise ConfigError("P/D manifest_digest does not match the launch spec bytes")
    if len(declarations) != 3:
        raise ConfigError("P/D launch spec requires exactly prefill, decode, and router processes")
    endpoints = provenance["endpoints"]
    if endpoint.rstrip("/") != endpoints["router"].rstrip("/"):
        raise ConfigError("P/D provenance router endpoint does not match the arm endpoint")
    if len({_origin(endpoints[role]) for role in ("prefill", "decode", "router")}) != 3:
        raise ConfigError("P/D prefill, decode, and router endpoints must be distinct")
    roles: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        argv = declaration["command"]
        host = _flag(argv, "--host")
        try:
            port = int(_flag(argv, "--port"))
        except ValueError as exc:
            raise ConfigError("P/D child --port must be an integer") from exc
        normalized_host = "127.0.0.1" if host == "localhost" else host
        matches = [
            role
            for role in ("prefill", "decode", "router")
            if _origin(endpoints[role])[2] == port
            and (host in {"0.0.0.0", "::"} or _origin(endpoints[role])[1] == normalized_host)
        ]
        if len(matches) != 1 or matches[0] in roles:
            raise ConfigError(f"P/D child {declaration['name']} does not bind one unique endpoint")
        role = matches[0]
        entrypoint = _child_entrypoint(argv, engine, role)
        visible = declaration["environment"].get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            raise ConfigError(f"P/D {role} must explicitly declare CUDA_VISIBLE_DEVICES")
        gpu_ids = visible.split(",") if visible else []
        expected = provenance["gpu_pools"][role] if role != "router" else []
        if set(gpu_ids) != set(expected) or len(gpu_ids) != len(expected):
            raise ConfigError(f"P/D {role} CUDA_VISIBLE_DEVICES does not match its GPU pool")
        if role != "router":
            if _flag(argv, "--served-model-name") != model:
                raise ConfigError(f"P/D {role} served model does not match the workload")
            if engine == "sglang":
                if _flag(argv, "--disaggregation-mode") != role:
                    raise ConfigError(f"SGLang {role} disaggregation mode does not match its role")
                if (
                    _flag(argv, "--disaggregation-transfer-backend").lower()
                    != provenance["connector"]["backend"].lower()
                ):
                    raise ConfigError(f"SGLang {role} transfer backend does not match provenance")
            else:
                try:
                    connector = json.loads(_flag(argv, "--kv-transfer-config"))
                except ValueError as exc:
                    raise ConfigError("vLLM KV transfer config must be JSON") from exc
                required_role = "kv_producer" if role == "prefill" else "kv_consumer"
                if (
                    not isinstance(connector, dict)
                    or connector.get("kv_connector") != provenance["connector"]["name"]
                ):
                    raise ConfigError(f"vLLM {role} connector does not match provenance")
                if connector.get("kv_role") not in {required_role, "kv_both"}:
                    raise ConfigError(f"vLLM {role} KV role does not match provenance")
        else:
            expected_proxy = "aisp_vllm_pd_proxy" if engine == "vllm" else "sglang_model_gateway"
            if provenance["proxy"]["implementation"] != expected_proxy:
                raise ConfigError("P/D proxy implementation does not match its native entrypoint")
            for backend in ("prefill", "decode"):
                flag = f"--{backend}-url" if engine == "vllm" else f"--{backend}"
                if _flag(argv, flag).rstrip("/") != endpoints[backend].rstrip("/"):
                    raise ConfigError(f"P/D router {backend} URL does not match provenance")
        roles[role] = {
            "process_name": declaration["name"],
            "endpoint": endpoints[role],
            "gpu_ids": gpu_ids,
            "entrypoint": entrypoint,
            "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        }
    if set(roles) != {"prefill", "decode", "router"}:
        raise ConfigError("P/D launch spec does not bind all required roles")
    if any(
        part == "--expected-sha256" or part.startswith("--expected-sha256=") for part in command
    ):
        if _flag(command, "--expected-sha256") != digest:
            raise ConfigError("P/D launcher digest argument does not match the launch spec")
    else:
        command += ["--expected-sha256", digest]
    return {
        "spec_path": str(spec_path),
        "sha256": digest,
        "roles": roles,
        "launcher_argv": command,
        "evidence": "verified_launch_configuration",
        "runtime_pool_allocation_verified": False,
    }


def verify_pd_allocation(
    binding: dict[str, Any],
    *,
    stdout_path: Path,
    allocation: dict[str, Any],
    gpu_uuid_map: dict[str, str],
    descendants: Callable[[int], set[int]],
) -> dict[str, Any]:
    """Match launcher child identities and their descendants to actual GPU contexts."""
    starts: dict[str, int] = {}
    for line in stdout_path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") == "process_started":
            name, pid = event.get("name"), event.get("pid")
            if not isinstance(name, str) or type(pid) is not int or pid <= 0 or name in starts:
                raise ConfigError("P/D launcher emitted an invalid or duplicate child identity")
            starts[name] = pid
    observed = {}
    assigned_pids = set()
    for role, declaration in binding["roles"].items():
        name = declaration["process_name"]
        if name not in starts:
            raise ConfigError(f"P/D launcher has no retained child identity for {role}")
        pids = descendants(starts[name]) | {starts[name]}
        rows = [row for row in allocation["compute_processes"] if row["pid"] in pids]
        try:
            expected = {gpu_uuid_map[gpu] for gpu in declaration["gpu_ids"]}
        except KeyError as exc:
            raise ConfigError("P/D GPU pool contains an unresolved device") from exc
        if {row["gpu_uuid"] for row in rows} != expected:
            raise ConfigError(f"P/D {role} observed GPU contexts do not match its declared pool")
        assigned_pids.update(row["pid"] for row in rows)
        observed[role] = {"root_pid": starts[name], "compute_processes": rows}
    if assigned_pids != {row["pid"] for row in allocation["compute_processes"]}:
        raise ConfigError("P/D allocation contains a GPU process outside its declared roles")
    return {**binding, "runtime_pool_allocation_verified": True, "observed_roles": observed}
