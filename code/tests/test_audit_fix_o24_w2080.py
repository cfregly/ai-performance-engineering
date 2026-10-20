"""W2-080: dashboard routes must keep blocking work off the ASGI event loop.

Each test gates one blocking step on proof that the event loop still runs other
coroutines. If the step ran on the loop thread, the loop could not make that
progress, the gate would time out, and the request would fail.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import core.api.response as response_module
import core.optimization.campaign_dashboard as campaign_dashboard
from core.api.registry import ApiRoute
from core.optimization.campaign import CampaignWorkspace
from dashboard.api import server
from tests.test_optimization_campaign import (
    ARTIFACT_CONTENT,
    _config,
    _record,
    _workspace_record,
)

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

GATE_TIMEOUT_SECONDS = 5.0


class _LoopGate:
    """Blocks a worker step until the event loop proves it is still running."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.loop_ran = threading.Event()
        self.thread_ids: list[int] = []

    def wait_for_loop(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.entered.set()
        if not self.loop_ran.wait(timeout=GATE_TIMEOUT_SECONDS):
            raise AssertionError("the ASGI event loop stayed blocked while dashboard work ran")


async def _release_gate_when_entered(gate: _LoopGate, work: asyncio.Future[Any]) -> Any:
    while not gate.entered.is_set() and not work.done():
        await asyncio.sleep(0.005)
    gate.loop_ran.set()
    return await work


def _asgi_get_while_gated(gate: _LoopGate, url: str, **kwargs: Any) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.fastapi_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=GATE_TIMEOUT_SECONDS * 4,
        ) as client:
            work = asyncio.ensure_future(client.get(url, **kwargs))
            return await _release_gate_when_entered(gate, work)

    return asyncio.run(_run())


@pytest.mark.parametrize(
    ("gated_level", "params"),
    [
        ("summary", {}),
        ("full", {"include_context": "true", "context_level": "full"}),
    ],
)
def test_response_context_refresh_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    gated_level: str,
    params: dict[str, str],
) -> None:
    gate = _LoopGate()

    def gated_build_context(level: str) -> dict[str, Any]:
        if level == gated_level:
            gate.wait_for_loop()
        return {"level": level}

    monkeypatch.setattr(response_module, "_build_context", gated_build_context)
    monkeypatch.setitem(response_module._CONTEXT_CACHE, "summary", None)
    monkeypatch.setitem(response_module._CONTEXT_CACHE, "full", None)
    loop_thread = threading.get_ident()

    response = _asgi_get_while_gated(gate, "/api/benchmark/compare", params=params)

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool"] == "benchmark.compare"
    assert payload["context_summary"] == {"level": "summary"}
    if gated_level == "full":
        assert payload["context"] == {"level": "full"}
        assert payload["context_level"] == "full"
    assert gate.thread_ids
    assert loop_thread not in gate.thread_ids


def test_campaign_artifact_hash_runs_off_the_event_loop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root = tmp_path / "campaigns"
    campaign_root.mkdir()
    workspace = CampaignWorkspace.initialize(campaign_root / "workspace", _config())
    record = _workspace_record(workspace, _record(status="promoted"))
    artifact = record.raw_artifacts[0]

    gate = _LoopGate()
    real_sha256_file = campaign_dashboard.sha256_file

    def gated_sha256_file(path):
        gate.wait_for_loop()
        return real_sha256_file(path)

    monkeypatch.setattr(campaign_dashboard, "sha256_file", gated_sha256_file)
    server._configure_campaign_root(campaign_root)
    loop_thread = threading.get_ident()
    try:
        response = _asgi_get_while_gated(
            gate,
            "/api/optimization/campaign/artifact",
            params={"workspace": "workspace", "artifact": artifact},
        )
    finally:
        server._configure_campaign_root(None)

    assert response.status_code == 200
    assert response.content == ARTIFACT_CONTENT
    assert gate.thread_ids
    assert loop_thread not in gate.thread_ids


def test_registry_handler_and_envelope_run_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _LoopGate()
    handler_threads: list[int] = []
    context_threads: list[int] = []

    def handler(params: dict[str, Any]) -> dict[str, Any]:
        handler_threads.append(threading.get_ident())
        gate.wait_for_loop()
        return {"params": params}

    def build_context(level: str) -> dict[str, Any]:
        context_threads.append(threading.get_ident())
        return {"level": level}

    monkeypatch.setattr(response_module, "_build_context", build_context)
    monkeypatch.setitem(response_module._CONTEXT_CACHE, "summary", None)
    route = ApiRoute("GET", "/test/o24", "test.o24", handler)
    endpoint = server._make_endpoint(route)
    request = SimpleNamespace(method="GET", query_params={"value": "1"})
    loop_thread = threading.get_ident()

    async def _run():
        work = asyncio.ensure_future(endpoint(request))
        return await _release_gate_when_entered(gate, work)

    response = asyncio.run(_run())

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["tool"] == "test.o24"
    assert payload["result"] == {"params": {"value": "1"}}
    assert payload["context_summary"] == {"level": "summary"}
    assert handler_threads and loop_thread not in handler_threads
    assert context_threads and loop_thread not in context_threads
