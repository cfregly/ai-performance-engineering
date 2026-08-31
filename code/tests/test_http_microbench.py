"""Owned-child HTTP readiness; retired microbenchmark routes remain unverified."""

from dataclasses import dataclass
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATHS = (
    "/api/microbench/disk",
    "/api/microbench/loopback",
    "/api/export/html",
)
STARTUP_TIMEOUT = 20.0
REQUEST_TIMEOUT = (0.3, 1.0)

# The parent retains an exclusively bound socket throughout the child lifetime.
# Uvicorn serves the actual production ASGI app through that inherited socket;
# no port is released for an unrelated server to acquire during readiness checks.
CHILD_SERVER = """
import socket
import sys
import uvicorn
from dashboard.api.server import fastapi_app

listener = socket.socket(fileno=int(sys.argv[1]))
config = uvicorn.Config(fastapi_app, log_level="warning")
uvicorn.Server(config).run(sockets=[listener])
"""


@dataclass
class OwnedDashboard:
    process: subprocess.Popen
    listener: socket.socket
    log_path: Path
    session: requests.Session

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.listener.getsockname()[1]}"

    def diagnostics(self):
        return (f"dashboard child pid={self.process.pid}, exit={self.process.poll()}, "
                f"log={self.log_path}\n{self.log_path.read_text(errors='replace')[-12000:]}")

    def assert_alive(self):
        if self.process.poll() is not None:
            raise RuntimeError(f"Owned {self.diagnostics()}")

    def get(self, path):
        self.assert_alive()
        response = self.session.get(self.base_url + path, timeout=REQUEST_TIMEOUT)
        self.assert_alive()
        return response


def start_server(tmp_path, *, command=None, startup_timeout=STARTUP_TIMEOUT):
    if os.name != "posix":
        pytest.skip("Owned inherited-socket dashboard check requires POSIX pass_fds")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    session = requests.Session()
    session.trust_env = False  # Loopback traffic must never use an external proxy.
    log_path = tmp_path / "dashboard-child.log"
    proc = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        env = {**os.environ, "PYTHONPATH": str(CODE_ROOT)}
        env.pop("AISP_DASHBOARD_CAMPAIGN_ROOT", None)
        env.pop("AISP_DASHBOARD_ALLOWED_ORIGINS", None)
        argv = command or [sys.executable, "-u", "-c", CHILD_SERVER, str(listener.fileno())]
        with log_path.open("xb") as log:
            proc = subprocess.Popen(argv, cwd=CODE_ROOT, env=env, pass_fds=(listener.fileno(),),
                                    stdout=log, stderr=subprocess.STDOUT)
        owned = OwnedDashboard(proc, listener, log_path, session)
        deadline = time.monotonic() + startup_timeout
        last_error = "no response"
        while time.monotonic() < deadline:
            owned.assert_alive()
            try:
                response = owned.get("/openapi.json")
                if response.status_code == 200:
                    schema = response.json()
                    if schema.get("info", {}).get("title") != "AISP Dashboard API":
                        raise RuntimeError(f"Unexpected server identity: {schema.get('info')}; {owned.diagnostics()}")
                    return owned
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except requests.RequestException as exc:
                last_error = repr(exc)
            time.sleep(0.05)
        raise RuntimeError(f"Dashboard readiness timed out: {last_error}; {owned.diagnostics()}")
    except BaseException:
        if proc is not None:
            stop_server(OwnedDashboard(proc, listener, log_path, session))
        else:
            listener.close()
            session.close()
        raise


def stop_server(owned):
    try:
        if owned.process.poll() is None:
            owned.process.terminate()
        try:
            owned.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owned.process.kill()
            owned.process.wait(timeout=5)
    finally:
        owned.listener.close()
        owned.session.close()


@pytest.fixture
def dashboard_server(tmp_path):
    owned = start_server(tmp_path)
    try:
        yield owned
    finally:
        stop_server(owned)


@pytest.mark.integration
def test_microbench_endpoints_start_stop():
    pytest.skip("Legacy routes /api/microbench/disk, /api/microbench/loopback and /api/export/html "
                "are absent from the dashboard API; their microbenchmark/export functionality "
                "is unavailable through these routes and remains unverified.")


@pytest.mark.integration
def test_current_dashboard_http_readiness_and_campaign_guard(dashboard_server):
    response = dashboard_server.get("/openapi.json")
    assert response.status_code == 200, dashboard_server.diagnostics()
    schema = response.json()
    assert schema["info"] == {"title": "AISP Dashboard API", "version": "1.0"}
    assert "/api/optimization/campaign/artifact" in schema["paths"]
    assert "/api/benchmark/data" in schema["paths"]
    # Exercise a real supported route without probing hardware or starting jobs.
    response = dashboard_server.get("/api/optimization/campaign/artifact?workspace=unused&artifact=unused")
    assert response.status_code == 503, response.text
    assert response.json() == {"detail": "Campaign API is disabled until AISP_DASHBOARD_CAMPAIGN_ROOT or --campaign-root is set"}


@pytest.mark.integration
def test_retired_http_routes_explicitly_return_not_found(dashboard_server):
    schema = dashboard_server.get("/openapi.json").json()
    for path in LEGACY_PATHS:
        assert path not in schema["paths"]
        response = dashboard_server.get(path)
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}


def test_dashboard_readiness_rejects_failed_owned_child(tmp_path):
    command = [sys.executable, "-u", "-c", "import sys; print('intentional startup failure'); sys.exit(37)"]
    with pytest.raises(RuntimeError, match=r"exit=37") as caught:
        start_server(tmp_path, command=command)
    assert "intentional startup failure" in str(caught.value)


@pytest.mark.integration
def test_dashboard_requests_reject_killed_owned_child(dashboard_server):
    dashboard_server.process.kill()
    dashboard_server.process.wait(timeout=5)
    with pytest.raises(RuntimeError, match="Owned dashboard child"):
        dashboard_server.get("/openapi.json")


def test_dashboard_readiness_timeout_cleans_up_owned_child(tmp_path):
    command = [sys.executable, "-u", "-c", "import os,time; print('waiting child pid=' + str(os.getpid())); time.sleep(60)"]
    with pytest.raises(RuntimeError, match="readiness timed out") as caught:
        start_server(tmp_path, command=command, startup_timeout=0.1)
    assert "waiting child pid=" in str(caught.value)
    child_pid = int((tmp_path / "dashboard-child.log").read_text().strip().split("=")[-1])
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
