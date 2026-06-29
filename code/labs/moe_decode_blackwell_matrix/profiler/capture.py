from __future__ import annotations

import heapq
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ..matrix_types import MatrixScenario
from ..runner import build_decode_batches, instantiate_experts, run_decode_step


def _self_device_time_us(event: Any) -> float:
    return float(
        getattr(
            event,
            "self_cuda_time_total",
            getattr(event, "self_device_time_total", 0.0),
        )
    )


def _device_time_us(event: Any) -> float:
    return float(
        getattr(
            event,
            "cuda_time_total",
            getattr(event, "device_time_total", 0.0),
        )
    )


def _top_ops(events: Iterable[Any], *, top_ops: int) -> list[dict[str, Any]]:
    rows = []
    for event in heapq.nlargest(top_ops, events, key=_self_device_time_us):
        rows.append(
            {
                "name": event.key,
                "self_cuda_time_us": _self_device_time_us(event),
                "cuda_time_us": _device_time_us(event),
                "cpu_time_us": float(event.cpu_time_total),
                "count": int(event.count),
            }
        )
    return rows


def profile_scenario(
    scenario: MatrixScenario,
    *,
    device: torch.device,
    trace_path: Path,
    top_ops: int,
) -> dict[str, Any]:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    batches = build_decode_batches(scenario, device=device)
    experts = instantiate_experts(scenario, device=device)
    if scenario.launch_mode == "cuda_graph":
        run_decode_step(experts, batches[0], scenario=scenario)
        torch.cuda.synchronize(device)

    for _ in range(max(1, scenario.warmup)):
        for batch in batches:
            run_decode_step(experts, batch, scenario=scenario)
    torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        for batch in batches:
            run_decode_step(experts, batch, scenario=scenario)
        torch.cuda.synchronize(device)

    profile.export_chrome_trace(str(trace_path))
    events = profile.key_averages()
    ops = _top_ops(events, top_ops=top_ops)
    total_self_cuda_time_us = 0.0
    total_cuda_time_us = 0.0
    total_cpu_time_us = 0.0
    for event in events:
        total_self_cuda_time_us += _self_device_time_us(event)
        total_cuda_time_us += _device_time_us(event)
        total_cpu_time_us += float(event.cpu_time_total)
    return {
        "config_id": scenario.config_id,
        "trace_path": str(trace_path),
        "top_ops": ops,
        "total_self_cuda_time_us": round(total_self_cuda_time_us, 3),
        "total_cuda_time_us": round(total_cuda_time_us, 3),
        "total_cpu_time_us": round(total_cpu_time_us, 3),
    }
