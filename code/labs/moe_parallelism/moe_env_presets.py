"""Shared helpers for MoE communication environment presets."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent
from typing import Dict, Iterable, List, Optional, Tuple

from labs.moe_parallelism.plan import ClusterSpec, ModelSpec, ParallelismPlan, _bytes_to_ms

# Minimal validation commands common to the MoE env presets (format placeholders).
DEFAULT_VALIDATION: List[str] = [
    "NVBandwidth -gpu 0-7 -bidirectional -csv > /tmp/nvbandwidth.csv",
    "mpirun -np {ngpu} --bind-to none all_reduce_perf -b 512 -e 64K -f 2 -g 1 -c 1 -n 100",
    "mpirun -np {ngpu} --bind-to none alltoall_perf  -b 512 -e 64K -f 2 -g 1 -c 1 -n 100",
]


@dataclass(frozen=True)
class EnvPreset:
    name: str
    fabric: str
    efficiency: float
    handshake_ms: float
    env: Dict[str, str]
    notes: List[str]
    validation: List[str]
    flag_prefix: str = "--env"


def bytes_to_ms(value_bytes: float, bandwidth_GBps: float) -> float:
    """Transfer time in ms at a decimal GB/s rate. A zero-byte payload takes 0 ms.

    Uses the same conversion as plan.PlanEvaluator, so both estimates agree.
    """
    return _bytes_to_ms(value_bytes, bandwidth_GBps)


def estimate_payload_bytes(plan: ParallelismPlan, model: ModelSpec) -> float:
    """Rough payload for MoE all-to-all at a given batch/sequence. EP=1 moves no data."""
    tokens_total = model.sequence_length * plan.microbatch_sequences
    moe_tokens_per_gpu = tokens_total * model.router_topk / max(plan.ep, 1)
    hidden_per_gpu = model.hidden_size / max(plan.tp, 1)
    return (
        moe_tokens_per_gpu
        * hidden_per_gpu
        * model.dtype_bytes
        * plan.microbatches
        * max(plan.ep - 1, 0)
    )


def estimate_collective(
    plan: ParallelismPlan,
    model: ModelSpec,
    cluster: ClusterSpec,
    efficiency: float,
    handshake_ms: float,
) -> Tuple[float, float, float, float]:
    """Return (payload_bytes, base_bw_GBps, effective_bw_GBps, collective_ms)."""
    payload_bytes = estimate_payload_bytes(plan, model)
    base_bw = (
        cluster.nic_bandwidth_GBps if plan.cross_node_ep else cluster.nvlink_bandwidth_GBps
    )
    effective_bw = base_bw * efficiency
    collective_ms = bytes_to_ms(payload_bytes, effective_bw)
    # EP=1 runs no collective, so it pays no handshake either.
    if payload_bytes > 0:
        collective_ms += handshake_ms
    return payload_bytes, base_bw, effective_bw, collective_ms


def render_summary(
    plan: ParallelismPlan,
    preset: EnvPreset,
    report: str,
    payload_bytes: float,
    base_bw: float,
    effective_bw: float,
    collective_ms: float,
    step_ms: float,
    throughput: float,
    format_kwargs: Optional[Dict[str, object]] = None,
) -> str:
    format_kwargs = format_kwargs or {}
    formatted_validation = [
        line.format(**format_kwargs) for line in (preset.validation or DEFAULT_VALIDATION)
    ]
    flag_block = render_flag_block(preset.env, preset.flag_prefix)
    payload_gb = payload_bytes / (1024 ** 3)
    lines: List[str] = [
        f"{plan.name} — fabric: {preset.fabric} ({preset.name})",
        report,
        (
            f"  MoE all-to-all payload ~{payload_gb:.2f} GB -> {collective_ms:.2f} ms "
            f"(base bw {base_bw:.1f} GB/s, effective {effective_bw:.1f} GB/s, "
            f"handshake +{preset.handshake_ms:.2f} ms)"
        ),
        f"  Step estimate: {step_ms:.1f} ms -> projected throughput {throughput:,.0f} tokens/s",
    ]
    if flag_block:
        lines.extend(
            [
                "  Launcher flags (pass to torchrun/mpirun):",
                indent(flag_block, "    "),
            ]
        )
    lines.extend(
        [
            "  Validation checks:",
            indent("\n".join(formatted_validation), "    "),
        ]
    )
    if preset.notes:
        lines.extend(["  Notes:", indent("\n".join(preset.notes), "    ")])
    return "\n".join(lines)


def render_flag_block(env: Dict[str, str], flag_prefix: str = "--env") -> str:
    if not env:
        return ""
    return " ".join(f"{flag_prefix} {k}={v}" for k, v in env.items())


def extend_notes(notes: Iterable[str]) -> List[str]:
    return [n for n in notes if n]
