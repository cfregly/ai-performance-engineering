"""Wave-1 API regressions, using real local paths and no external LLM calls."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core import book, llm
from core.engine import get_engine


CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args", [[], ["tui"], ["bench", "tui"], ["bench", "tui", "--simple"]])
@pytest.mark.parametrize("menu_input", ["q\n", ""])
def test_cli_opens_real_analysis_menu(args, menu_input):
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", *args],
        input=menu_input,
        capture_output=True,
        text=True,
        cwd=CODE_ROOT,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "BENCHMARK ANALYSIS MENU" in result.stdout
    assert "Select option:" in result.stdout
    assert "Traceback" not in result.stderr
    assert "Falling back" not in result.stdout
    assert "requires curses" not in result.stdout


@pytest.fixture
def budget_environment(monkeypatch):
    # Test config parsing after module import has loaded repository dotenv files.
    # Selecting a provider explicitly also avoids local-server autodetection.
    monkeypatch.setenv("PERF_LLM_PROVIDER", "vllm")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("PERF_LLM_MAX_TOKENS", raising=False)


def test_llm_environment_uses_dataclass_default(budget_environment):
    assert llm.LLMConfig.from_env().max_tokens == 4096


@pytest.mark.parametrize(
    "provider_model, generic_model, expected",
    [
        (None, None, "claude-sonnet-4-6"),
        (None, "explicit-generic-model", "explicit-generic-model"),
        ("explicit-anthropic-model", "explicit-generic-model", "explicit-anthropic-model"),
    ],
)
def test_anthropic_model_default_and_explicit_overrides(
    budget_environment, monkeypatch, provider_model, generic_model, expected
):
    monkeypatch.setenv("PERF_LLM_PROVIDER", "anthropic")
    for key, value in (("ANTHROPIC_MODEL", provider_model), ("PERF_LLM_MODEL", generic_model)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert llm.LLMConfig.from_env().model == expected


@pytest.mark.parametrize("explicit_model", [None, "explicit-model"])
@pytest.mark.parametrize("environment_model", [None, "environment-model"])
def test_legacy_anthropic_clients_share_default_and_preserve_overrides(
    monkeypatch, explicit_model, environment_model
):
    from core.analysis.llm_advisor import LLMConfig as AdvisorConfig
    from core.analysis.llm_profile_analyzer import LLMProfileAnalyzer

    monkeypatch.delenv("PERF_LLM_MODEL", raising=False)
    if environment_model is None:
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_MODEL", environment_model)
    expected = explicit_model or environment_model or "claude-sonnet-4-6"
    config = AdvisorConfig(provider="anthropic", model=explicit_model or "", api_key="local-fixture-key")
    analyzer = LLMProfileAnalyzer(provider="anthropic", model=explicit_model, api_key="local-fixture-key")
    assert config.model == analyzer.model == expected


@pytest.mark.parametrize("environment_model", [None, "explicit-anthropic-model"])
def test_parallelism_advisor_sends_supported_default_or_override_locally(
    monkeypatch, local_llm_endpoint, environment_model
):
    pytest.importorskip("anthropic")
    from core.optimization.parallelism_planner.llm_advisor import (
        LLMOptimizationAdvisor,
        OptimizationGoal,
        OptimizationRequest,
        SystemContext,
    )

    base_url, requests = local_llm_endpoint
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "local-fixture-key")
    monkeypatch.delenv("PERF_LLM_MODEL", raising=False)
    if environment_model is None:
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_MODEL", environment_model)
    request = OptimizationRequest(goal=OptimizationGoal.LATENCY, context=SystemContext(model_name="public-fixture"))
    advice = LLMOptimizationAdvisor(llm_provider="anthropic").get_advice(request)
    assert advice.raw_response == "local protocol fixture"
    assert len(requests) == 1
    assert requests[0][1]["model"] == (environment_model or "claude-sonnet-4-6")


@pytest.mark.parametrize(
    "primary, legacy, expected", [("64", "8192", 64), (None, "128", 128)]
)
def test_llm_environment_honors_small_budgets(
    budget_environment, monkeypatch, primary, legacy, expected
):
    if primary is not None:
        monkeypatch.setenv("LLM_MAX_TOKENS", primary)
    monkeypatch.setenv("PERF_LLM_MAX_TOKENS", legacy)
    assert llm.LLMConfig.from_env().max_tokens == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_llm_environment_rejects_invalid_budgets(budget_environment, monkeypatch, value):
    monkeypatch.setenv("LLM_MAX_TOKENS", value)
    with pytest.raises(ValueError):
        llm.LLMConfig.from_env()


@pytest.fixture
def local_llm_endpoint():
    """Capture actual HTTP JSON payloads; this fixture does not evaluate a model."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            response = {
                "choices": [{"message": {"content": "local protocol fixture"}}],
                "response": "local protocol fixture",
                "output": [{"content": [{"type": "output_text", "text": "local protocol fixture"}]}],
                "id": "local-fixture-message",
                "type": "message",
                "role": "assistant",
                "model": "local-fixture",
                "content": [{"type": "text", "text": "local protocol fixture"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("provider", ["vllm", "ollama"])
@pytest.mark.parametrize("override, expected", [(None, 512), (64, 64), (8192, 8192)])
def test_llm_sends_exact_budget_to_local_endpoint(
    monkeypatch, local_llm_endpoint, provider, override, expected
):
    base_url, requests = local_llm_endpoint
    monkeypatch.setattr(
        llm,
        "_config",
        llm.LLMConfig(provider=provider, model="local-fixture", base_url=base_url, max_tokens=512),
    )
    assert llm.llm_call("public synthetic protocol test", max_tokens=override) == "local protocol fixture"
    assert len(requests) == 1
    path, payload = requests[0]
    if provider == "vllm":
        assert path == "/v1/chat/completions"
        assert payload["max_tokens"] == expected
    else:
        assert path == "/api/generate"
        assert payload["options"]["num_predict"] == expected


@pytest.mark.parametrize(
    "provider, model, path, budget_field",
    [
        ("openai", "gpt-4o", "/v1/chat/completions", "max_tokens"),
        ("openai", "gpt-5", "/v1/responses", "max_output_tokens"),
        ("anthropic", "claude-sonnet-4-6", "/v1/messages", "max_tokens"),
    ],
)
@pytest.mark.parametrize("override, expected", [(None, 512), (64, 64), (8192, 8192)])
def test_hosted_provider_payload_budget_without_external_requests(
    monkeypatch, local_llm_endpoint, provider, model, path, budget_field, override, expected
):
    pytest.importorskip(provider)
    base_url, requests = local_llm_endpoint
    monkeypatch.setenv("OPENAI_BASE_URL", base_url + "/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.setattr(
        llm,
        "_config",
        llm.LLMConfig(provider=provider, model=model, api_key="local-fixture-key", max_tokens=512),
    )
    if path == "/v1/responses":
        # The Responses adapter has a fixed production URL. Redirect only its
        # transport to the real loopback server; preserve the request body.
        urlopen = urllib.request.urlopen

        def local_transport(request, **kwargs):
            assert request.full_url == "https://api.openai.com/v1/responses"
            request = urllib.request.Request(
                base_url + path,
                data=request.data,
                headers={"Content-Type": "application/json"},
            )
            return urlopen(request, **kwargs)

        monkeypatch.setattr(urllib.request, "urlopen", local_transport)

    assert llm.llm_call("public synthetic protocol test", max_tokens=override) == "local protocol fixture"
    assert len(requests) == 1
    assert requests[0][0] == path
    assert requests[0][1][budget_field] == expected


def test_llm_status_probe_sends_64_tokens(monkeypatch, local_llm_endpoint):
    base_url, requests = local_llm_endpoint
    monkeypatch.setattr(
        llm,
        "_config",
        llm.LLMConfig(provider="vllm", model="local-fixture", base_url=base_url),
    )
    status = llm.get_llm_status(probe=True)
    assert status["available"] is True
    assert status["probed"] is True
    assert len(requests) == 1
    assert requests[0][1]["max_tokens"] == 64


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "64"])
def test_llm_rejects_invalid_override_before_http(monkeypatch, local_llm_endpoint, value):
    base_url, requests = local_llm_endpoint
    monkeypatch.setattr(
        llm,
        "_config",
        llm.LLMConfig(provider="vllm", model="local-fixture", base_url=base_url),
    )
    with pytest.raises(ValueError, match="positive integer"):
        llm.llm_call("public synthetic protocol test", max_tokens=value)
    assert requests == []


@pytest.fixture
def local_book(monkeypatch, tmp_path):
    content = (
        "Auditwidget introduces the example. Its second sentence explains the contract! "
        "This third sentence must not enter the summary.\n"
        "- First useful detail\n"
        "* Second useful detail\n"
        "1. Third useful detail\n"
        "2. Fourth useful detail\n"
        "3. Fifth useful detail\n"
        "4. Sixth detail stays outside the five-point limit"
    )
    (tmp_path / "ch01.md").write_text(
        "# Chapter 1: Local Test Book\n\n## Test concept\n\n" + content + "\n",
        encoding="utf-8",
    )
    index_type = book.BookIndex
    monkeypatch.setattr(book, "BookIndex", lambda: index_type(book_dir=tmp_path))
    # Citation behavior is useful even when no paid backend is configured.
    monkeypatch.setattr(llm, "_config", llm.LLMConfig(provider="none", model="none"))
    return content


def test_ask_returns_serializable_file_backed_citations_without_llm(local_book):
    result = get_engine().ai.ask("auditwidget")
    assert result["success"] is False
    assert result["error_type"] == "llm_unavailable"
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert citation["chapter"] == "ch01"
    assert citation["content"] == local_book
    assert citation["line_number"] == 5
    assert json.loads(json.dumps(result))["citations"] == result["citations"]


def test_ask_can_omit_citations(local_book):
    result = get_engine().ai.ask("auditwidget", include_citations=False)
    assert "citations" not in result
    assert result["error_type"] == "llm_unavailable"


def test_explain_parses_summary_and_bullets_from_real_book_index(local_book):
    result = get_engine().ai.explain("auditwidget")
    assert result["success"] is True
    assert result["explanation"] == (
        "Auditwidget introduces the example. Its second sentence explains the contract!"
    )
    assert result["key_points"] == [
        "First useful detail",
        "Second useful detail",
        "Third useful detail",
        "Fourth useful detail",
        "Fifth useful detail",
    ]
    assert result["citations"][0]["content"] == local_book
