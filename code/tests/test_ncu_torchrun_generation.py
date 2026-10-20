"""Real control-flow coverage for rank-local NCU replay generations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.profiling.ncu_torchrun_generation import (
    GENERATION_CONFIG_SCHEMA,
    GENERATION_MARKER_PREFIX,
    load_generation_config,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


def _generation_payload(output_dir: Path, worker_argv: list[str], **updates):
    payload = {
        "schema": GENERATION_CONFIG_SCHEMA,
        "worker_argv": worker_argv,
        "output_dir": str(output_dir),
        "world_size": 2,
        "expected_passes": 2,
        "master_addr": "127.0.0.1",
        "master_port_base": 30400,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"schema": "wrong"}, "Unsupported generation config schema"),
        ({"worker_argv": []}, "worker_argv must be a non-empty"),
        ({"world_size": 1}, "world_size must be exactly 2"),
        ({"expected_passes": 0}, "expected_passes must be a positive integer"),
        ({"master_addr": "localhost"}, "master_addr must be '127.0.0.1'"),
        ({"master_port_base": 1023}, "port range must stay within"),
        (
            {"master_port_base": 65535, "expected_passes": 2},
            "port range must stay within",
        ),
    ],
)
def test_generation_config_rejects_malformed_values(
    tmp_path: Path, updates: dict[str, object], match: str
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    config_path = tmp_path / "generation-config.json"
    config_path.write_text(
        json.dumps(
            _generation_payload(output_dir, [sys.executable, "-c", "pass"], **updates)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        load_generation_config(config_path)


def test_generation_entrypoint_replays_worker_with_fresh_store_and_bounded_counter(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "capture"
    output_dir.mkdir()
    worker_path = tmp_path / "worker.py"
    worker_path.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        f"output = Path({str(output_dir)!r})\n"
        "payload = {\n"
        "    'master_addr': os.environ.get('MASTER_ADDR'),\n"
        "    'master_port': os.environ.get('MASTER_PORT'),\n"
        "    'restart_count_present': 'TORCHELASTIC_RESTART_COUNT' in os.environ,\n"
        "    'use_agent_store': os.environ.get('TORCHELASTIC_USE_AGENT_STORE'),\n"
        "}\n"
        "(output / f\"worker-{payload['master_port']}.json\").write_text(\n"
        "    json.dumps(payload, sort_keys=True), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "generation-config.json"
    config_path.write_text(
        json.dumps(
            _generation_payload(output_dir, [sys.executable, str(worker_path)])
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "GROUP_RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "RANK": "0",
            "TORCHELASTIC_RESTART_COUNT": "7",
            "WORLD_SIZE": "2",
        }
    )

    completed = [
        subprocess.run(
            [
                sys.executable,
                "-m",
                "core.profiling.ncu_torchrun_generation",
                str(config_path),
            ],
            cwd=CODE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for _ in range(2)
    ]

    assert [item.returncode for item in completed] == [0, 0]
    events = []
    for item in completed:
        lines = item.stdout.splitlines()
        assert len(lines) == 2
        events.append(
            [
                json.loads(line.removeprefix(GENERATION_MARKER_PREFIX))
                for line in lines
                if line.startswith(GENERATION_MARKER_PREFIX)
            ]
        )
    assert [[event["event"] for event in pair] for pair in events] == [
        ["begin", "end"],
        ["begin", "end"],
    ]
    assert [[event["generation"] for event in pair] for pair in events] == [
        [0, 0],
        [1, 1],
    ]
    assert [[event["master_port"] for event in pair] for pair in events] == [
        [30400, 30400],
        [30401, 30401],
    ]
    assert all(pair[0]["pid"] == pair[1]["pid"] for pair in events)

    worker_receipts = [
        json.loads((output_dir / f"worker-{port}.json").read_text(encoding="utf-8"))
        for port in (30400, 30401)
    ]
    assert worker_receipts == [
        {
            "master_addr": "127.0.0.1",
            "master_port": str(port),
            "restart_count_present": False,
            "use_agent_store": "False",
        }
        for port in (30400, 30401)
    ]
    assert (output_dir / "rank-0-generation.counter").read_text(encoding="ascii") == "2\n"

    overflow = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.profiling.ncu_torchrun_generation",
            str(config_path),
        ],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert overflow.returncode != 0
    assert "generation must be in 0..1" in overflow.stderr
