"""Launch a declared group of serving processes under one owned process group."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "serving-comparison.process-group.v1"


def _load(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA:
        raise ValueError(f"process group spec must use {SCHEMA}")
    processes = raw.get("processes")
    if not isinstance(processes, list) or not processes:
        raise ValueError("process group spec needs at least one process")
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(processes):
        if not isinstance(item, dict):
            raise ValueError(f"processes[{index}] must be an object")
        name = item.get("name")
        command = item.get("command")
        environment = item.get("environment", {})
        passthrough = item.get("env_passthrough", [])
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"processes[{index}].name must be unique")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ValueError(f"processes[{index}].command must be a nonempty argv list")
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(f"processes[{index}].environment must map strings to strings")
        if not isinstance(passthrough, list) or any(
            not isinstance(name, str) or not name for name in passthrough
        ):
            raise ValueError(f"processes[{index}].env_passthrough must contain names")
        working_directory = item.get("working_directory", ".")
        if not isinstance(working_directory, str) or not working_directory:
            raise ValueError(f"processes[{index}].working_directory must be a path")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "command": command,
                "environment": environment,
                "env_passthrough": passthrough,
                "working_directory": str((path.parent / working_directory).resolve()),
            }
        )
    return normalized


def _stop(children: list[subprocess.Popen[bytes]], timeout_s: float) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + timeout_s
    while any(child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(0.05)
    for child in children:
        if child.poll() is None:
            child.kill()
    for child in children:
        child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--shutdown-timeout-s", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.shutdown_timeout_s <= 0:
        parser.error("--shutdown-timeout-s must be positive")
    try:
        declarations = _load(args.spec.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    children: list[subprocess.Popen[bytes]] = []
    try:
        for declaration in declarations:
            environment = os.environ.copy()
            for name in declaration["env_passthrough"]:
                if name not in os.environ:
                    raise ValueError(f"required environment variable is unset: {name}")
            environment.update(declaration["environment"])
            child = subprocess.Popen(
                declaration["command"],
                cwd=declaration["working_directory"],
                env=environment,
            )
            children.append(child)
            print(
                json.dumps(
                    {
                        "event": "process_started",
                        "name": declaration["name"],
                        "pid": child.pid,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        while not stopping:
            exited = [child for child in children if child.poll() is not None]
            if exited:
                returncode = exited[0].returncode
                print(
                    json.dumps(
                        {"event": "process_exited", "pid": exited[0].pid, "returncode": returncode},
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                return returncode if returncode not in {None, 0} else 1
            time.sleep(0.1)
        return 0
    except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
        print(json.dumps({"event": "launch_failed", "error": str(exc)}), file=sys.stderr)
        return 2
    finally:
        _stop(children, args.shutdown_timeout_s)


if __name__ == "__main__":
    raise SystemExit(main())
