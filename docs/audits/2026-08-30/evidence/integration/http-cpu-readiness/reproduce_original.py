"""Reproduce the original real child-startup failure being reported as a pass.

Only port isolation, a bounded request and proxy avoidance differ from the original test:
the reserved ephemeral loopback port cannot belong to an unrelated listener.
The original dashboard child command and test exception handling execute intact.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests


def main() -> None:
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[5]
    code = root / "code"
    source = evidence / "test_http_microbench.before.py.txt"
    module = {"__file__": str(code / "tests/test_http_microbench.py"), "__name__": "original_http_test"}
    exec(compile(source.read_text(), str(source), "exec"), module)
    children = []
    actual_popen = subprocess.Popen
    session = requests.Session()
    session.trust_env = False
    original_get = requests.get
    requests.get = lambda url, **kwargs: session.get(url, timeout=0.1, **kwargs)

    def observed_popen(command, **kwargs):
        kwargs["env"] = {**os.environ, "PYTHONPATH": str(code)}
        process = actual_popen(command, **kwargs)
        children.append((command, process))
        return process

    subprocess.Popen = observed_popen
    started = time.monotonic()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            module["SERVER_PORT"] = reservation.getsockname()[1]
            result = module["test_microbench_endpoints_start_stop"]()
        receipts = []
        for command, process in children:
            stdout, stderr = process.communicate(timeout=5)
            receipts.append({"command": command, "pid": process.pid, "exit_code": process.returncode,
                             "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")})
        print(json.dumps({"status": "REPRODUCED_FALSE_PASS", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "test_returned_normally": result is None, "duration_seconds": time.monotonic() - started,
                          "children": receipts, "limitations": "Original HTTP endpoint bodies did not execute; failed child was silently accepted."}, indent=2))
        assert len(receipts) == 1 and receipts[0]["exit_code"] is not None
    finally:
        subprocess.Popen = actual_popen
        requests.get = original_get
        session.close()
        for _, process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
