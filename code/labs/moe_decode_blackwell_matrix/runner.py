from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from labs.moe_optimization_journey.moe_model import MoEExperts, MoEOptimizations

from .matrix_types import MatrixScenario


@dataclass
class DispatchBatch:
    hidden_states: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    routing_entropy_norm: float
    active_expert_fraction: float
    max_tokens_per_expert: int


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise KeyError(f"Unsupported dtype {name!r}")


def _stable_seed(payload: dict[str, Any]) -> int:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _workload_seed(scenario: MatrixScenario) -> int:
    return scenario.seed + _stable_seed(
        {
            "hidden_size": scenario.hidden_size,
            "intermediate_size": scenario.intermediate_size,
            "num_experts": scenario.num_experts,
            "top_k": scenario.top_k,
            "decode_batch": scenario.decode_batch,
            "routing_policy": scenario.routing_policy,
            "steps": scenario.steps,
            "dtype": scenario.dtype,
        }
    )


def _weight_seed(scenario: MatrixScenario) -> int:
    return _workload_seed(scenario) + 97


def _policy_probs(
    policy: str,
    num_experts: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if policy == "balanced":
        probs = torch.ones(num_experts, dtype=torch.float32)
    elif policy == "skewed":
        ranks = torch.arange(1, num_experts + 1, dtype=torch.float32)
        probs = ranks.pow(-1.35)
    elif policy == "sticky":
        hot_count = max(2, math.ceil(num_experts * 0.25))
        hot = torch.randperm(num_experts, generator=generator)[:hot_count]
        probs = torch.full((num_experts,), 0.15 / max(1, num_experts - hot_count), dtype=torch.float32)
        probs[hot] = 0.85 / hot_count
    else:  # pragma: no cover - validated upstream
        raise KeyError(f"Unsupported routing policy {policy!r}")
    return probs / probs.sum()


def _routing_stats(indices: torch.Tensor, *, num_experts: int) -> tuple[float, float, int]:
    counts = torch.bincount(indices.reshape(-1), minlength=num_experts).to(torch.float32)
    total = float(counts.sum().item())
    active = float((counts > 0).sum().item()) / float(num_experts)
    max_tokens = int(counts.max().item()) if total > 0 else 0
    if total <= 0:
        return 0.0, active, max_tokens
    probs = counts / total
    nz = probs[probs > 0]
    entropy = float((-(nz * nz.log()).sum() / math.log(num_experts)).item()) if num_experts > 1 else 0.0
    return entropy, active, max_tokens


def build_decode_batches(
    scenario: MatrixScenario,
    *,
    device: torch.device,
) -> list[DispatchBatch]:
    dtype = dtype_from_name(scenario.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_workload_seed(scenario))
    probs = _policy_probs(scenario.routing_policy, scenario.num_experts, generator)
    batches: list[DispatchBatch] = []
    for _step in range(scenario.steps):
        hidden_states = torch.randn(
            (scenario.decode_batch, scenario.hidden_size),
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        expert_indices = torch.multinomial(
            probs.expand(scenario.decode_batch, -1),
            scenario.top_k,
            replacement=False,
            generator=generator,
        )
        weight_logits = torch.randn(
            (scenario.decode_batch, scenario.top_k),
            generator=generator,
            dtype=torch.float32,
        )
        expert_weights = torch.softmax(weight_logits, dim=-1)
        entropy, active_fraction, max_tokens = _routing_stats(
            expert_indices, num_experts=scenario.num_experts
        )
        batches.append(
            DispatchBatch(
                hidden_states=hidden_states,
                expert_indices=expert_indices.to(device=device, dtype=torch.long),
                expert_weights=expert_weights.to(device=device, dtype=dtype),
                routing_entropy_norm=entropy,
                active_expert_fraction=active_fraction,
                max_tokens_per_expert=max_tokens,
            )
        )
    return batches


def instantiate_experts(
    scenario: MatrixScenario,
    *,
    device: torch.device,
) -> MoEExperts:
    torch.manual_seed(_weight_seed(scenario))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_weight_seed(scenario))
    experts = MoEExperts(
        num_experts=scenario.num_experts,
        hidden_size=scenario.hidden_size,
        intermediate_size=scenario.intermediate_size,
        opts=MoEOptimizations(),
    )
    dtype = dtype_from_name(scenario.dtype)
    experts = experts.to(device=device, dtype=dtype)
    experts.eval()
    return experts


def run_decode_step(
    experts: MoEExperts,
    batch: DispatchBatch,
    *,
    scenario: MatrixScenario,
) -> torch.Tensor:
    with torch.no_grad():
        if scenario.schedule_mode == "dynamic":
            return experts.forward_grouped(
                batch.hidden_states, batch.expert_indices, batch.expert_weights
            )
        if scenario.schedule_mode == "persistent" and scenario.launch_mode == "eager":
            return experts._forward_bmm_fused_graphable(  # noqa: SLF001
                batch.hidden_states, batch.expert_indices, batch.expert_weights
            )
        if scenario.schedule_mode == "persistent" and scenario.launch_mode == "cuda_graph":
            return experts.forward_cuda_graphs(
                batch.hidden_states, batch.expert_indices, batch.expert_weights
            )
    raise RuntimeError(f"Unsupported scenario combination: {scenario}")


def _reference_outputs(
    experts: MoEExperts,
    batches: Sequence[DispatchBatch],
) -> list[torch.Tensor]:
    refs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            refs.append(
                experts.forward_grouped(
                    batch.hidden_states, batch.expert_indices, batch.expert_weights
                ).detach()
            )
    return refs


def _compare_outputs(
    experts: MoEExperts,
    batches: Sequence[DispatchBatch],
    refs: Sequence[torch.Tensor],
    *,
    scenario: MatrixScenario,
) -> float:
    diffs: list[float] = []
    with torch.no_grad():
        for batch, ref in zip(batches, refs, strict=True):
            out = run_decode_step(experts, batch, scenario=scenario)
            diffs.append(float(torch.max(torch.abs(out.float() - ref.float())).item()))
    return max(diffs, default=0.0)


def measure_scenario(
    scenario: MatrixScenario,
    *,
    device: torch.device,
    clock_state: dict[str, Any],
) -> dict[str, Any]:
    row = scenario.to_dict()
    row.update(clock_state)
    if scenario.schedule_mode == "dynamic" and scenario.launch_mode == "cuda_graph":
        row.update(
            {
                "status": "unsupported",
                "note": "dynamic grouped schedule is intentionally non-graphable in this lab",
                "capture_ms": None,
                "step_mean_ms": None,
                "step_stdev_ms": None,
                "step_p95_ms": None,
                "tokens_per_second": None,
                "dispatch_tokens_per_second": None,
                "graph_captured": 0.0,
                "graph_replays": 0.0,
                "max_abs_diff": None,
            }
        )
        return row

    batches = build_decode_batches(scenario, device=device)
    experts = instantiate_experts(scenario, device=device)
    refs = _reference_outputs(experts, batches)

    capture_ms: float | None = None
    graph_captured = 0.0
    graph_replays = 0.0
    note = ""

    if scenario.launch_mode == "cuda_graph":
        start = time.perf_counter()
        run_decode_step(experts, batches[0], scenario=scenario)
        torch.cuda.synchronize(device)
        capture_ms = (time.perf_counter() - start) * 1000.0
        metrics = experts.get_cuda_graph_metrics()
        graph_captured = float(metrics.get("cuda_graph_captured", 0.0))
        graph_replays = float(metrics.get("cuda_graph_replays", 0.0))
        note = str(getattr(experts, "_cuda_graph_last_error", "") or "")
        if graph_captured < 1.0 or metrics.get("cuda_graph_fallback", 0.0) > 0.0:
            row.update(
                {
                    "status": "error",
                    "note": note or "cuda_graph capture did not complete cleanly",
                    "capture_ms": capture_ms,
                    "step_mean_ms": None,
                    "step_stdev_ms": None,
                    "step_p95_ms": None,
                    "tokens_per_second": None,
                    "dispatch_tokens_per_second": None,
                    "graph_captured": graph_captured,
                    "graph_replays": graph_replays,
                    "max_abs_diff": None,
                }
            )
            return row

    for _ in range(scenario.warmup):
        for batch in batches:
            run_decode_step(experts, batch, scenario=scenario)
    torch.cuda.synchronize(device)

    elapsed_per_step_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(scenario.repeats):
        start_event.record()
        for batch in batches:
            run_decode_step(experts, batch, scenario=scenario)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_per_step_ms.append(start_event.elapsed_time(end_event) / len(batches))

    max_abs_diff = _compare_outputs(experts, batches, refs, scenario=scenario)
    entropy_mean = statistics.fmean(batch.routing_entropy_norm for batch in batches)
    active_fraction_mean = statistics.fmean(batch.active_expert_fraction for batch in batches)
    max_tokens_per_expert_mean = statistics.fmean(batch.max_tokens_per_expert for batch in batches)
    step_mean_ms = statistics.fmean(elapsed_per_step_ms)
    step_stdev_ms = (
        statistics.pstdev(elapsed_per_step_ms) if len(elapsed_per_step_ms) > 1 else 0.0
    )
    sorted_latencies = sorted(elapsed_per_step_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    step_p95_ms = sorted_latencies[p95_index]

    metrics = experts.get_cuda_graph_metrics()
    graph_captured = float(metrics.get("cuda_graph_captured", 0.0))
    graph_replays = float(metrics.get("cuda_graph_replays", 0.0))
    if scenario.launch_mode == "cuda_graph":
        note = str(getattr(experts, "_cuda_graph_last_error", "") or "")

    row.update(
        {
            "status": "ok",
            "note": note,
            "capture_ms": round(capture_ms, 6) if capture_ms is not None else None,
            "step_mean_ms": round(step_mean_ms, 6),
            "step_stdev_ms": round(step_stdev_ms, 6),
            "step_p95_ms": round(step_p95_ms, 6),
            "tokens_per_second": round(scenario.decode_batch / (step_mean_ms / 1000.0), 3),
            "dispatch_tokens_per_second": round(
                (scenario.decode_batch * scenario.top_k) / (step_mean_ms / 1000.0),
                3,
            ),
            "routing_entropy_norm_mean": round(entropy_mean, 6),
            "active_expert_fraction_mean": round(active_fraction_mean, 6),
            "max_tokens_per_expert_mean": round(max_tokens_per_expert_mean, 6),
            "graph_captured": graph_captured,
            "graph_replays": graph_replays,
            "max_abs_diff": round(max_abs_diff, 8),
        }
    )
    return row


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "ok_row_count": len(ok_rows),
        "unsupported_row_count": sum(1 for row in rows if row.get("status") == "unsupported"),
        "error_row_count": sum(1 for row in rows if row.get("status") == "error"),
        "best_overall": None,
        "persistent_vs_dynamic": [],
        "graph_vs_eager": [],
    }
    if ok_rows:
        best = min(ok_rows, key=lambda row: float(row["step_mean_ms"]))
        summary["best_overall"] = {
            "config_id": best["config_id"],
            "step_mean_ms": best["step_mean_ms"],
            "tokens_per_second": best["tokens_per_second"],
            "workload_key": best["workload_key"],
        }

    by_config = {
        (
            row["workload_key"],
            row["schedule_mode"],
            row["launch_mode"],
        ): row
        for row in ok_rows
    }
    workload_keys = sorted({row["workload_key"] for row in ok_rows})
    for workload_key in workload_keys:
        dynamic = by_config.get((workload_key, "dynamic", "eager"))
        persistent = by_config.get((workload_key, "persistent", "eager"))
        graph = by_config.get((workload_key, "persistent", "cuda_graph"))
        if dynamic and persistent:
            summary["persistent_vs_dynamic"].append(
                {
                    "workload_key": workload_key,
                    "dynamic_config_id": dynamic["config_id"],
                    "persistent_config_id": persistent["config_id"],
                    "dynamic_step_mean_ms": dynamic["step_mean_ms"],
                    "persistent_step_mean_ms": persistent["step_mean_ms"],
                    "speedup": round(
                        float(dynamic["step_mean_ms"]) / float(persistent["step_mean_ms"]),
                        6,
                    ),
                }
            )
        if persistent and graph:
            summary["graph_vs_eager"].append(
                {
                    "workload_key": workload_key,
                    "eager_config_id": persistent["config_id"],
                    "graph_config_id": graph["config_id"],
                    "eager_step_mean_ms": persistent["step_mean_ms"],
                    "graph_step_mean_ms": graph["step_mean_ms"],
                    "graph_capture_ms": graph["capture_ms"],
                    "speedup": round(
                        float(persistent["step_mean_ms"]) / float(graph["step_mean_ms"]),
                        6,
                    ),
                }
            )
    return summary


def render_console_table(rows: Sequence[dict[str, Any]], *, limit: int = 16) -> str:
    ok_rows = sorted(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: float(row["step_mean_ms"]),
    )[:limit]
    lines = [
        "| config_id | batch | routing | schedule | launch | mean ms | tok/s |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            "| `{config_id}` | `{decode_batch}` | `{routing_policy}` | "
            "`{schedule_mode}` | `{launch_mode}` | `{step_mean_ms}` | "
            "`{tokens_per_second}` |".format(**row)
        )
    return "\n".join(lines)
