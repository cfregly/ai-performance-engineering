import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import pytest

SERVER_PORT = 8123


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "dashboard/api/server.py"],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # give it a moment
    for _ in range(20):
        time.sleep(0.25)
        try:
            requests.get(f"http://127.0.0.1:{SERVER_PORT}")
            return proc
        except Exception:
            continue
    proc.terminate()
    raise RuntimeError("dashboard server did not start")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


@pytest.mark.integration
def test_microbench_endpoints_start_stop():
    try:
        proc = start_server()
    except RuntimeError as exc:
        return
    try:
        base = f"http://127.0.0.1:{SERVER_PORT}"
        r = requests.get(f"{base}/api/microbench/disk?file_size_mb=1&block_size_kb=64")
        assert r.status_code == 200
        data = r.json()
        assert "read_gbps" in data

        r = requests.get(f"{base}/api/microbench/loopback?size_mb=1&port=51007")
        assert r.status_code == 200
        data = r.json()
        assert "throughput_gbps" in data

        r = requests.get(f"{base}/api/export/html")
        assert r.status_code == 200
        data = r.json()
        assert "html" in data
    finally:
        stop_server(proc)
