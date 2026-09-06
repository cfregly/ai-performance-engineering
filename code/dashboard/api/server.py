#!/usr/bin/env python3
"""FastAPI backend for the dashboard."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import typer

from core.api.registry import ApiRoute, get_routes
from core.api.response import build_response

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for minimal test environments
    _FASTAPI_AVAILABLE = False

    class Request:  # type: ignore[no-redef]
        pass

    class HTTPException(Exception):  # type: ignore[no-redef]  # noqa: N818
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class JSONResponse:  # type: ignore[no-redef]
        def __init__(self, content: Any, **_: Any) -> None:
            self.content = content

    class StreamingResponse:  # type: ignore[no-redef]
        def __init__(self, content: Any, **_: Any) -> None:
            self.content = content

    class FileResponse:  # type: ignore[no-redef]
        def __init__(self, path: Any, **_: Any) -> None:
            self.path = path

    class CORSMiddleware:  # type: ignore[no-redef]
        pass

    class _StubRoute:
        def __init__(self, path: str, methods: set[str]) -> None:
            self.path = path
            self.methods = methods

    class FastAPI:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            self.routes: list[_StubRoute] = []

        def add_middleware(self, *_: Any, **__: Any) -> None:
            return None

        def get(self, path: str):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"GET"}))
                return fn

            return decorator

        def post(self, path: str):
            def decorator(fn):
                self.routes.append(_StubRoute(path, {"POST"}))
                return fn

            return decorator


CAMPAIGN_ROOT_ENV = "AISP_DASHBOARD_CAMPAIGN_ROOT"
UI_ORIGINS_ENV = "AISP_DASHBOARD_ALLOWED_ORIGINS"
DEFAULT_UI_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")
MIN_GPU_STREAM_INTERVAL_SECONDS = 0.1
_campaign_root_override: Path | None = None


def _allowed_ui_origins(raw_value: str | None = None) -> list[str]:
    raw = raw_value if raw_value is not None else os.environ.get(UI_ORIGINS_ENV, "")
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not origins:
        origins = list(DEFAULT_UI_ORIGINS)
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"invalid dashboard UI origin: {origin}")
    return list(dict.fromkeys(origins))


def _configure_campaign_root(root: Path | None) -> None:
    global _campaign_root_override
    if root is None:
        _campaign_root_override = None
        return
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Campaign root must be an existing directory: {resolved}")
    _campaign_root_override = resolved


def _configured_campaign_root() -> Path:
    if _campaign_root_override is not None:
        return _campaign_root_override
    raw_root = os.environ.get(CAMPAIGN_ROOT_ENV, "").strip()
    if not raw_root:
        raise RuntimeError(
            f"Campaign API is disabled until {CAMPAIGN_ROOT_ENV} or --campaign-root is set"
        )
    resolved = Path(raw_root).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Configured campaign root is not a directory: {resolved}")
    return resolved


def _resolve_campaign_workspace(workspace: str) -> Path:
    root = _configured_campaign_root()
    raw_workspace = str(workspace).strip()
    if not raw_workspace:
        raise ValueError("workspace is required")
    requested = Path(raw_workspace).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError("campaign workspace is outside the configured campaign root")
    if not resolved.is_dir():
        raise FileNotFoundError(f"campaign workspace not found: {resolved}")
    return resolved


fastapi_app = FastAPI(title="AISP Dashboard API", version="1.0")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_ui_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _configure_engine(data_file: Path | None) -> None:
    if data_file is None:
        return
    path = Path(data_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dashboard data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Dashboard data file must be a file: {path}")
    path = path.resolve()

    import core.engine as engine
    from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data
    from core.perf_core import get_core

    handler = get_core(data_file=path, refresh=True)
    engine._handler_instance = handler
    engine._analyzer_instance = PerformanceAnalyzer(
        lambda: load_benchmark_data(path, handler.bench_roots)
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _collect_params(request: Request, body: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(request.query_params)
    if body:
        params.update(body)
    return params


def _exception_type_name(exc: BaseException) -> str:
    name = type(exc).__name__
    chars: list[str] = []
    for index, char in enumerate(name):
        if index and char.isupper() and (not name[index - 1].isupper()):
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _make_endpoint(route: ApiRoute):
    async def _endpoint(request: Request) -> JSONResponse:
        started = time.time()
        body: dict[str, Any] | None = None
        if request.method in {"POST", "PUT"}:
            try:
                payload = await request.json()
                body = payload if isinstance(payload, dict) else {"body": payload}
            except Exception:
                body = None
        params = _collect_params(request, body)
        if route.name == "optimization.campaign":
            try:
                params["workspace"] = str(
                    _resolve_campaign_workspace(str(params.get("workspace") or ""))
                )
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        include_context = _parse_bool(params.get("include_context"))
        context_level = str(params.get("context_level", "summary"))

        had_exception = False
        try:
            # Registry handlers are synchronous and may perform filesystem,
            # subprocess, or GPU-driver work. Keep them off the ASGI event loop
            # so one slow request cannot stall unrelated routes or the SSE feed.
            result = await asyncio.to_thread(route.handler, params)
        except Exception as exc:
            had_exception = True
            result = {
                "success": False,
                "error": str(exc),
                "error_type": _exception_type_name(exc),
            }
        duration_ms = int((time.time() - started) * 1000)
        payload = build_response(
            route.name,
            params,
            result,
            duration_ms,
            had_exception=had_exception,
            include_context=include_context,
            context_level=context_level,
        )
        return JSONResponse(payload)

    return _endpoint


def _register_routes() -> None:
    for route in get_routes():
        if route.method.upper() == "GET":
            fastapi_app.get(route.path)(_make_endpoint(route))
        elif route.method.upper() == "POST":
            fastapi_app.post(route.path)(_make_endpoint(route))
        else:
            raise RuntimeError(f"Unsupported HTTP method in API registry: {route.method}")


_register_routes()


@fastapi_app.get("/api/optimization/campaign/artifact")
async def campaign_artifact(workspace: str, artifact: str) -> FileResponse:
    """Serve one hash-valid artifact declared by a campaign ledger."""

    from core.optimization.campaign_dashboard import resolve_campaign_artifact

    try:
        resolved_workspace = _resolve_campaign_workspace(workspace)
        path = resolve_campaign_artifact(resolved_workspace, artifact)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name)


@fastapi_app.get("/api/gpu/stream")
async def gpu_stream(
    request: Request,
    interval: float = 5.0,
    max_events: int | None = None,
) -> StreamingResponse:
    if not math.isfinite(interval) or interval < MIN_GPU_STREAM_INTERVAL_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"interval must be finite and at least {MIN_GPU_STREAM_INTERVAL_SECONDS} seconds"
            ),
        )
    if max_events is not None and max_events <= 0:
        raise HTTPException(status_code=422, detail="max_events must be > 0")

    async def _event_stream():
        from core.engine import get_engine

        count = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "gpu": await asyncio.to_thread(get_engine().gpu.info),
                }
                yield f"event: gpu\ndata: {json.dumps(payload)}\n\n"
            except Exception as exc:
                error_payload = {"error": str(exc)}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

            count += 1
            if max_events is not None and count >= max_events:
                break
            await asyncio.sleep(interval)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)


cli = typer.Typer(help="AISP dashboard API server", no_args_is_help=True)


@cli.callback()
def cli_main() -> None:
    """AISP dashboard API server."""
    return None


def serve_dashboard(
    port: int = 6970,
    data_file: Path | None = None,
    campaign_root: Path | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    log_level: str = "info",
) -> None:
    """Start the dashboard API server."""
    _configure_engine(data_file)
    if campaign_root is not None:
        _configure_campaign_root(campaign_root)
    if open_browser:
        browser_host = host
        if host in {"0.0.0.0", "::"}:
            browser_host = "127.0.0.1"
        url = f"http://{browser_host}:{port}"
        if not webbrowser.open(url):
            raise RuntimeError(f"Failed to open browser at {url}")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the dashboard API server") from exc
    uvicorn.run(fastapi_app, host=host, port=port, log_level=log_level)


@cli.command("serve")
def cli_serve(
    port: int = typer.Option(6970, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level"),
    data_file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer 0.12 compatibility
        None,
        "--data",
        "-d",
        help="Path to benchmark_test_results.json",
    ),
    campaign_root: Optional[Path] = typer.Option(  # noqa: UP045 - Typer 0.12 compatibility
        None,
        "--campaign-root",
        help=f"Restrict campaign API access to this directory (or set {CAMPAIGN_ROOT_ENV}).",
    ),
    open_browser: bool = typer.Option(
        False, "--open-browser", help="Open browser to the backend URL"
    ),
) -> None:
    """Start the dashboard API server."""
    serve_dashboard(
        port=port,
        host=host,
        log_level=log_level,
        data_file=data_file,
        campaign_root=campaign_root,
        open_browser=open_browser,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
