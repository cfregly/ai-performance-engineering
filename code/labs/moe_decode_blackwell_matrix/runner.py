from __future__ import annotations

import hashlib
import heapq
import json
import math
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
    total_tensor = counts.sum()
    probs = counts / total_tensor.clamp_min(1.0)
    nz = probs[probs > 0]
    entropy_tensor = (
        -(nz * nz.log()).sum() / math.log(num_experts)
        if num_experts > 1
        else counts.new_zeros(())
    )
    stats = torch.empty(4, device=counts.device, dtype=counts.dtype)
    stats[0].copy_(total_tensor)
    stats[1].copy_((counts > 0).sum().to(counts.dtype))
    stats[2].copy_(counts.max())
    stats[3].copy_(entropy_tensor)
    stats_host = stats.detach().cpu()
    total = float(stats_host[0])
    active_count = float(stats_host[1])
    max_tokens_float = float(stats_host[2])
    entropy = float(stats_host[3])
    active = float(active_count) / float(num_experts)
    max_tokens = int(max_tokens_float) if total > 0 else 0
    if total <= 0:
        return 0.0, active, max_tokens
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
    with torch.inference_mode():
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
    with torch.inference_mode():
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
    max_diff: torch.Tensor | None = None
    with torch.inference_mode():
        for batch, ref in zip(batches, refs, strict=True):
            out = run_decode_step(experts, batch, scenario=scenario)
            diff = torch.abs(out.float() - ref.float()).amax()
            if max_diff is None:
                max_diff = diff.clone()
            else:
                torch.maximum(max_diff, diff, out=max_diff)
    if max_diff is None:
        return 0.0
    return float(max_diff.detach().cpu())


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
    current_stream = torch.cuda.current_stream(device)
    for _ in range(scenario.repeats):
        start_event.record(current_stream)
        for batch in batches:
            run_decode_step(experts, batch, scenario=scenario)
        end_event.record(current_stream)
        end_event.synchronize()
        elapsed_per_step_ms.append(start_event.elapsed_time(end_event) / len(batches))

    batch_count = len(batches)
    entropy_total = 0.0
    active_fraction_total = 0.0
    max_tokens_per_expert_total = 0.0
    for batch in batches:
        entropy_total += batch.routing_entropy_norm
        active_fraction_total += batch.active_expert_fraction
        max_tokens_per_expert_total += float(batch.max_tokens_per_expert)
    entropy_mean = entropy_total / batch_count
    active_fraction_mean = active_fraction_total / batch_count
    max_tokens_per_expert_mean = max_tokens_per_expert_total / batch_count

    sample_count = len(elapsed_per_step_ms)
    latency_total = 0.0
    latency_total_sq = 0.0
    for elapsed_ms in elapsed_per_step_ms:
        latency_total += elapsed_ms
        latency_total_sq += elapsed_ms * elapsed_ms
    step_mean_ms = latency_total / sample_count
    if sample_count > 1:
        latency_variance = (latency_total_sq / sample_count) - step_mean_ms * step_mean_ms
        step_stdev_ms = math.sqrt(max(0.0, latency_variance))
    else:
        step_stdev_ms = 0.0
    p95_index = max(0, math.ceil(0.95 * sample_count) - 1)
    upper_tail_count = sample_count - p95_index
    step_p95_ms = heapq.nlargest(upper_tail_count, elapsed_per_step_ms)[-1]

    max_abs_diff = _compare_outputs(experts, batches, refs, scenario=scenario)

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
    ok_row_count = 0
    unsupported_row_count = 0
    error_row_count = 0
    best: dict[str, Any] | None = None
    by_config: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    workload_keys: set[Any] = set()
    for row in rows:
        status = row.get("status")
        if status == "ok":
            ok_row_count += 1
            if best is None or float(row["step_mean_ms"]) < float(best["step_mean_ms"]):
                best = row
            by_config[(row["workload_key"], row["schedule_mode"], row["launch_mode"])] = row
            workload_keys.add(row["workload_key"])
        elif status == "unsupported":
            unsupported_row_count += 1
        elif status == "error":
            error_row_count += 1

    summary: dict[str, Any] = {
        "row_count": len(rows),
        "ok_row_count": ok_row_count,
        "unsupported_row_count": unsupported_row_count,
        "error_row_count": error_row_count,
        "best_overall": None,
        "persistent_vs_dynamic": [],
        "graph_vs_eager": [],
    }
    if best is not None:
        summary["best_overall"] = {
            "config_id": best["config_id"],
            "step_mean_ms": best["step_mean_ms"],
            "tokens_per_second": best["tokens_per_second"],
            "workload_key": best["workload_key"],
        }

    for workload_key in sorted(workload_keys):
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
    ok_rows = heapq.nsmallest(
        limit,
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: float(row["step_mean_ms"]),
    )
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
