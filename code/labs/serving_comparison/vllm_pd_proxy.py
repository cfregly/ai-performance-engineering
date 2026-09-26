"""Small vLLM completion proxy that performs a real connector-backed P/D handoff."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .protocol import require_httpx


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: Any
    ):
        super().__init__(address, handler)
        self.config = config


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, message_format: str, *args: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "proxy_access",
                    "client": self.client_address[0],
                    "message": message_format % args,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @property
    def config(self) -> Any:
        return self.server.config  # type: ignore[attr-defined]

    def _upstream_headers(self, request_id: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if request_id:
            headers["X-Request-Id"] = request_id
        if self.config.api_key_env:
            value = os.environ.get(self.config.api_key_env)
            if not value:
                raise RuntimeError(
                    f"required API key environment variable is unset: {self.config.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {value}"
        return headers

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        httpx = require_httpx()
        target = None
        if self.path == "/health":
            targets = [f"{self.config.prefill_url}/health", f"{self.config.decode_url}/health"]
            try:
                with httpx.Client(timeout=self.config.timeout_s, trust_env=False) as client:
                    responses = [
                        client.get(url, headers=self._upstream_headers()) for url in targets
                    ]
                if all(response.status_code < 400 for response in responses):
                    body = b'{"ready":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except httpx.HTTPError:
                pass
            self._json_error(503, "prefill or decode endpoint is not ready")
            return
        if self.path in {"/version", "/v1/models"}:
            target = f"{self.config.decode_url}{self.path}"
        if target is None:
            self._json_error(404, "not found")
            return
        try:
            with httpx.Client(timeout=self.config.timeout_s, trust_env=False) as client:
                response = client.get(target, headers=self._upstream_headers())
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("content-type", "application/json")
            )
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)
        except httpx.HTTPError as exc:
            self._json_error(502, f"upstream identity request failed: {type(exc).__name__}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/completions":
            self._json_error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            prompt = payload.get("prompt")
            if (
                not isinstance(prompt, list)
                or not prompt
                or any(type(token) is not int or token < 0 for token in prompt)
            ):
                raise ValueError("prompt must contain explicit token ids")
            if payload.get("stream") is not True or payload.get("return_token_ids") is not True:
                raise ValueError("stream and return_token_ids must both be true")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(400, str(exc))
            return

        httpx = require_httpx()
        request_id = self.headers.get("X-Request-Id") or str(uuid.uuid4())
        stream_started = False
        prefill_payload = dict(payload)
        prefill_payload.pop("stream_options", None)
        prefill_payload.update(
            {
                "stream": False,
                "max_tokens": 1,
                "min_tokens": 1,
                "return_token_ids": True,
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                },
            }
        )
        try:
            with httpx.Client(timeout=self.config.timeout_s, trust_env=False) as client:
                prefill = client.post(
                    f"{self.config.prefill_url}/v1/completions",
                    headers=self._upstream_headers(request_id),
                    json=prefill_payload,
                )
            if prefill.status_code >= 400:
                self._json_error(502, f"prefill returned HTTP {prefill.status_code}")
                return
            prefill_document = prefill.json()
            if not isinstance(prefill_document, dict):
                self._json_error(502, "prefill response is not a JSON object")
                return
            transfer = prefill_document.get("kv_transfer_params")
            if not isinstance(transfer, dict) or transfer.get("do_remote_prefill") is not True:
                self._json_error(502, "prefill response lacks decode-side KV transfer parameters")
                return
            decode_payload = dict(payload)
            decode_payload["kv_transfer_params"] = transfer
            timeout = httpx.Timeout(self.config.timeout_s)
            with (
                httpx.Client(timeout=timeout, trust_env=False) as client,
                client.stream(
                    "POST",
                    f"{self.config.decode_url}/v1/completions",
                    headers=self._upstream_headers(request_id),
                    json=decode_payload,
                ) as decode,
            ):
                if decode.status_code >= 400:
                    self._json_error(502, f"decode returned HTTP {decode.status_code}")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                stream_started = True
                for chunk in decode.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()
                self.close_connection = True
        except (
            httpx.HTTPError,
            RuntimeError,
            ValueError,
            BrokenPipeError,
            ConnectionResetError,
        ) as exc:
            if not stream_started:
                self._json_error(502, f"P/D proxy request failed: {type(exc).__name__}")
            else:
                self.close_connection = True
            print(
                json.dumps(
                    {"event": "pd_proxy_request_failed", "error": type(exc).__name__},
                    sort_keys=True,
                ),
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    _Server((args.host, args.port), _Handler, args).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
