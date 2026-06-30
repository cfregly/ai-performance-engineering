"""
Synthetic driver to compare baseline vs optimized routing.

Usage examples:
    python labs/dynamic_router/driver.py --mode optimized --scenario flagship_vs_mid --ticks 400 --arrival-rate 1.2
    python labs/dynamic_router/driver.py --mode optimized --scenario mig_slices --ticks 400 --arrival-rate 1.0 --burst-factor 2.0 --log-json artifacts/dynamic_router/mig_run.json

What it does:
  - Spawns virtual GPUs with prefill/decode roles and cost metadata.
  - Generates synthetic requests (prompt + decode lengths) with configurable arrivals and burstiness.
  - Runs a short simulation loop, logging TTFT/TPOT estimates and computing goodput-per-dollar.

This is a teaching aid: the virtual GPUs are simple queues with fixed rates.
Swap in real engine hooks (vLLM/SGLang/TRT-LLM) at the INTEGRATION POINTS to
turn this into a live experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from labs.dynamic_router.router_round_robin import BaselineRouter, Request
from labs.dynamic_router.router_policy import Router, SequenceInfo

TICK_SECONDS = 0.05  # wall-clock seconds per simulation tick


# ------------------------------------------------------------
# Virtual GPU model
# ------------------------------------------------------------


@dataclass
class PrefillTask:
    req_id: str
    remaining_time: float


@dataclass
class DecodeTask:
    req_id: str
    remaining_tokens: int
    first_token_emitted: bool = False


@dataclass
class VirtualGPU:
    gpu_id: str
    is_prefill: bool
    is_decode: bool
    prefill_rate: float  # prompt tokens processed per second
    decode_rate: float  # tokens per second
    hourly_cost: float  # dollars per GPU-hour
    kv_transfer_ms: float = 0.0
    tier: str | None = None
    numa_node: Optional[int] = None
    host_kv_local_gb: float = 24.0
    host_kv_remote_gb: float = 4.0
    prefill_q: Deque[PrefillTask] = field(default_factory=deque)
    decode_q: Deque[DecodeTask] = field(default_factory=deque)
    ttft_ema: float = 0.0
    tpot_ema: float = 0.0
    queue_depth_sum: float = 0.0
    queue_depth_max: int = 0
    queue_depth_samples: int = 0

    def enqueue_prefill(self, req: Request) -> None:
        prompt = max(1, req.prompt_tokens)
        remaining = prompt / self.prefill_rate
        self.prefill_q.append(PrefillTask(req.req_id, remaining))

    def enqueue_decode(self, req_id: str, expected_new_tokens: int) -> None:
        tokens = max(1, expected_new_tokens)
        self.decode_q.append(DecodeTask(req_id, tokens))

    def step(self, tick_s: float) -> Tuple[List[str], List[Tuple[str, int, bool]]]:
        """
        Advance one tick.

        Returns:
          - completed_prefills: list of req_ids that finished prefill
          - decode_events: list of (req_id, tokens_emitted, first_token_emitted)
        """
        completed_prefills: List[str] = []
        decode_events: List[Tuple[str, int, bool]] = []

        self._record_queue_depth()

        # Process prefill (one-at-a-time queue)
        if self.prefill_q:
            task = self.prefill_q[0]
            task.remaining_time -= tick_s
            if task.remaining_time <= 0:
                completed_prefills.append(task.req_id)
                self.prefill_q.popleft()

        # Process decode tasks (simple FCFS, one-at-a-time)
        if self.decode_q:
            task = self.decode_q[0]
            tokens = min(task.remaining_tokens, math.ceil(self.decode_rate * tick_s))
            task.remaining_tokens -= tokens
            decode_events.append((task.req_id, tokens, not task.first_token_emitted))
            task.first_token_emitted = True
            if task.remaining_tokens <= 0:
                self.decode_q.popleft()

        return completed_prefills, decode_events

    def update_smoothed_metrics(
        self, ttft_ms_samples: List[float], tokens_emitted: int, alpha: float = 0.3
    ) -> None:
        """Update simple EMAs for TTFT and tokens-per-occupied-second."""
        for sample in ttft_ms_samples:
            self.ttft_ema = alpha * sample + (1.0 - alpha) * self.ttft_ema
        occupied = 1.0 if (self.prefill_q or self.decode_q) else 0.0
        if occupied:
            tpot = tokens_emitted / TICK_SECONDS
            self.tpot_ema = alpha * tpot + (1.0 - alpha) * self.tpot_ema

    def _record_queue_depth(self) -> None:
        depth = len(self.prefill_q) + len(self.decode_q)
        self.queue_depth_samples += 1
        self.queue_depth_sum += depth
        self.queue_depth_max = max(self.queue_depth_max, depth)

    def queue_depth_avg(self) -> float:
        if self.queue_depth_samples == 0:
            return 0.0
        return self.queue_depth_sum / float(self.queue_depth_samples)

    def metrics_snapshot(self) -> Dict[str, float]:
        queue_depth = len(self.prefill_q) + len(self.decode_q)
        mem_free_gb = max(0.0, 40.0 - queue_depth * 0.5)  # toy model
        return {
            "ttft_ms": self.ttft_ema,
            "tpot": self.tpot_ema,
            "queue_depth": float(queue_depth),
            "mem_free_gb": mem_free_gb,
            "kv_hit_rate": 0.0,
            "host_kv_local_gb": self.host_kv_local_gb,
            "host_kv_remote_gb": self.host_kv_remote_gb,
        }


# ------------------------------------------------------------
# Simulation harness
# ------------------------------------------------------------


@dataclass
class RequestState:
    req: Request
    admitted_at: float
    prefill_gpu: Optional[str] = None
    decode_gpu: Optional[str] = None
    ttft_ms: Optional[float] = None
    decode_started_at: Optional[float] = None
    decode_finished_at: Optional[float] = None
    finished: bool = False


def make_virtual_gpus(scenario: str) -> Dict[str, VirtualGPU]:
    """
    Create a small fleet with scenario-specific sizing.

    - flagship_vs_mid: 1 flagship prefill + 1 flagship decode + 3 mid-tier decoders
    - mig_slices: 1 full GPU for prefill + four MIG-like decode slices
    """
    if scenario == "mig_slices":
        return {
            "pf0": VirtualGPU(
                "pf0",
                is_prefill=True,
                is_decode=False,
                prefill_rate=9000,
                decode_rate=2000,
                hourly_cost=5.5,
                kv_transfer_ms=0.5,
                tier="full",
            ),
            "dc0": VirtualGPU("dc0", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=2200, hourly_cost=1.8, kv_transfer_ms=0.8, tier="1g"),
            "dc1": VirtualGPU("dc1", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=2200, hourly_cost=1.8, kv_transfer_ms=0.8, tier="1g"),
            "dc2": VirtualGPU("dc2", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=2600, hourly_cost=2.2, kv_transfer_ms=0.6, tier="2g"),
            "dc3": VirtualGPU("dc3", is_prefill=False, is_decode=True, prefill_rate=4000, decode_rate=2600, hourly_cost=2.2, kv_transfer_ms=0.6, tier="2g"),
        }

    # Default: flagship vs mid-tier pool
    return {
        "pf0": VirtualGPU("pf0", is_prefill=True, is_decode=False, prefill_rate=10000, decode_rate=2400, hourly_cost=7.0, kv_transfer_ms=0.5, tier="flagship"),
        "pf1": VirtualGPU("pf1", is_prefill=True, is_decode=False, prefill_rate=9000, decode_rate=2200, hourly_cost=6.0, kv_transfer_ms=0.5, tier="flagship"),
        "dc_big": VirtualGPU("dc_big", is_prefill=False, is_decode=True, prefill_rate=5000, decode_rate=5200, hourly_cost=7.0, kv_transfer_ms=0.5, tier="flagship"),
        "dc_mid0": VirtualGPU("dc_mid0", is_prefill=False, is_decode=True, prefill_rate=3800, decode_rate=3400, hourly_cost=3.0, kv_transfer_ms=0.8, tier="mid"),
        "dc_mid1": VirtualGPU("dc_mid1", is_prefill=False, is_decode=True, prefill_rate=3800, decode_rate=3400, hourly_cost=3.0, kv_transfer_ms=0.8, tier="mid"),
        "dc_mid2": VirtualGPU("dc_mid2", is_prefill=False, is_decode=True, prefill_rate=3800, decode_rate=3400, hourly_cost=3.0, kv_transfer_ms=0.8, tier="mid"),
    }


def build_optimized_router(gpus: Dict[str, VirtualGPU], queue_urgency: float, cost_penalty: float) -> Router:
    r = Router(queue_urgency=queue_urgency, decode_cost_penalty=cost_penalty)
    for gid, gpu in gpus.items():
        r.register_gpu(
            gid,
            is_prefill=gpu.is_prefill,
            is_decode=gpu.is_decode,
            hourly_cost=gpu.hourly_cost,
        )
    return r


def _percentile_from_ordered(data_sorted: List[float], pct: float) -> float:
    assert 0.0 <= pct <= 100.0
    k = (len(data_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    data.sort()
    return _percentile_from_ordered(data, pct)


def _percentiles(data: List[float], pcts: Tuple[float, ...]) -> Tuple[float, ...]:
    if not data:
        return tuple(0.0 for _ in pcts)
    data.sort()
    return tuple(_percentile_from_ordered(data, pct) for pct in pcts)


def _poisson(lam: float) -> int:
    """Small Poisson sampler to avoid pulling in numpy."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(k - 1, 0)


def simulate(
    mode: str,
    num_ticks: int = 400,
    seed: int = 0,
    scenario: str = "flagship_vs_mid",
    arrival_rate: float = 1.2,
    burst_factor: float = 1.0,
    slo_ttft_ms: float = 350.0,
    slo_tpot_ms: float = 45.0,
    log_json: Optional[Path] = None,
    cost_awareness: float = 0.0,
    queue_urgency: float = 1.0,
    log_interval: Optional[int] = 20,
) -> Dict[str, float]:
    random.seed(seed)
    gpus = make_virtual_gpus(scenario)

    baseline = BaselineRouter(gpus.keys()) if mode == "baseline" else None
    optimized = build_optimized_router(gpus, queue_urgency=queue_urgency, cost_penalty=cost_awareness) if mode == "optimized" else None

    requests: Dict[str, RequestState] = {}
    next_id = 0
    completed_ttfts: List[float] = []
    completed_ttft_total = 0.0
    completed_count = 0
    good_requests = 0
    good_tokens = 0

    for tick in range(num_ticks):
        now = tick * TICK_SECONDS

        # 1) Generate new requests
        lam = arrival_rate * (burst_factor if random.random() < 0.2 else 1.0)
        new_requests = _poisson(lam)
        for _ in range(new_requests):
            req = Request(
                req_id=f"req-{next_id}",
                prompt_tokens=random.randint(100, 1200),
                expected_new_tokens=random.randint(32, 256),
                priority=random.randint(0, 2),
            )
            next_id += 1

            if mode == "baseline":
                gpu_id = baseline.route(req)  # type: ignore[union-attr]
                requests[req.req_id] = RequestState(req=req, admitted_at=now, prefill_gpu=gpu_id, decode_gpu=gpu_id)
                gpus[gpu_id].enqueue_prefill(req)
            else:
                gpu_id = optimized.choose_prefill_gpu()  # type: ignore[union-attr]
                if gpu_id is None:
                    gpu_id = "pf0"
                requests[req.req_id] = RequestState(req=req, admitted_at=now, prefill_gpu=gpu_id)
                gpus[gpu_id].enqueue_prefill(req)

        # 2) Update metrics -> Router for optimized
        if optimized:
            for gid, gpu in gpus.items():
                optimized.update_metrics(gid, gpu.metrics_snapshot())

        # 3) Step GPUs and route inter-stage traffic
        for gid, gpu in gpus.items():
            completed_prefills, decode_events = gpu.step(TICK_SECONDS)

            # Prefill completions -> decode admission
            for rid in completed_prefills:
                state = requests[rid]
                if mode == "baseline":
                    gpu.enqueue_decode(rid, state.req.expected_new_tokens)
                else:
                    seq = SequenceInfo(
                        seq_id=rid,
                        current_gpu=state.prefill_gpu or gid,
                        kv_gpus={gid},
                        expected_tokens_remaining=state.req.expected_new_tokens,
                        priority=state.req.priority,
                    )
                    dst_gpu = optimized.choose_decode_gpu(seq)  # type: ignore[union-attr]
                    dst_gpu = dst_gpu or gid
                    state.decode_gpu = dst_gpu
                    gpus[dst_gpu].enqueue_decode(rid, state.req.expected_new_tokens)

            # Decode events -> TTFT + throughput stats
            ttft_samples: List[float] = []
            tokens_emitted = 0
            for rid, tokens, first in decode_events:
                tokens_emitted += tokens
                state = requests[rid]
                if first and state.ttft_ms is None:
                    state.ttft_ms = (now - state.admitted_at) * 1000.0
                    state.decode_started_at = now
                    completed_ttfts.append(state.ttft_ms)
                    completed_ttft_total += state.ttft_ms
                    ttft_samples.append(state.ttft_ms)
                state.decode_finished_at = now
            gpu.update_smoothed_metrics(ttft_samples, tokens_emitted)

        # 4) Optional: migrate (optimized only)
        if optimized and tick % 5 == 0:
            active_decode_ids_by_gpu = {
                gid: {task.req_id for task in gpu.decode_q}
                for gid, gpu in gpus.items()
                if gpu.decode_q
            }
            active = []
            for state in requests.values():
                if state.ttft_ms is not None and state.decode_gpu is not None:
                    decode_ids = active_decode_ids_by_gpu.get(state.decode_gpu)
                    if decode_ids is not None and state.req.req_id in decode_ids:
                        active.append(
                            SequenceInfo(
                                seq_id=state.req.req_id,
                                current_gpu=state.decode_gpu,
                                kv_gpus={state.decode_gpu},
                                expected_tokens_remaining=None,
                                priority=state.req.priority,
                            )
                        )
            migrations = optimized.plan_migrations(active, max_per_call=2)
            for rid, src, dst in migrations:
                src_gpu = gpus[src]
                dst_gpu = gpus[dst]
                for idx, task in enumerate(src_gpu.decode_q):
                    if task.req_id == rid:
                        dst_gpu.decode_q.append(task)
                        del src_gpu.decode_q[idx]
                        break

        # 5) Clean up finished requests
        active_request_ids = {
            task.req_id
            for gpu in gpus.values()
            for queue in (gpu.prefill_q, gpu.decode_q)
            for task in queue
        }
        finished = [
            rid
            for rid, state in requests.items()
            if state.req.req_id not in active_request_ids
        ]
        for rid in finished:
            if mode == "baseline":
                baseline.complete(rid)  # type: ignore[union-attr]
            state = requests.pop(rid, None)
            if state:
                state.finished = True
                completed_count += 1
                if state.ttft_ms is not None and state.decode_finished_at and state.decode_started_at:
                    per_token_ms = ((state.decode_finished_at - state.decode_started_at) * 1000.0) / max(
                        state.req.expected_new_tokens, 1
                    )
                    if state.ttft_ms <= slo_ttft_ms and per_token_ms <= slo_tpot_ms:
                        good_requests += 1
                        good_tokens += state.req.expected_new_tokens

        # Optional slow logging
        if log_interval and log_interval > 0 and tick % log_interval == 0:
            active_ttft_total = 0.0
            active_ttft_count = 0
            for state in requests.values():
                if state.ttft_ms is not None:
                    active_ttft_total += state.ttft_ms
                    active_ttft_count += 1
            ttft_str = (
                f"{active_ttft_total / active_ttft_count:.1f} ms"
                if active_ttft_count
                else "n/a"
            )
            if optimized:
                scores = {gid: g.queue_depth_avg() for gid, g in gpus.items() if g.is_decode}
                print(
                    f"[tick {tick:03d}] mode={mode} active={len(requests)} "
                    f"avg_ttft={ttft_str} decode_scores={scores}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[tick {tick:03d}] mode={mode} active={len(requests)} avg_ttft={ttft_str}",
                    file=sys.stderr,
                )

        time.sleep(0.0)

    print(
        f"\nDone. Mode={mode} | completed={completed_count} | remaining={len(requests)}",
        file=sys.stderr,
    )

    # Summary metrics
    summary: Dict[str, float] = {
        "mode": mode,
        "seed": seed,
        "ticks": num_ticks,
        "scenario": scenario,
        "completed": completed_count,
        "remaining": len(requests),
        "slo_ttft_ms": slo_ttft_ms,
        "slo_tpot_ms": slo_tpot_ms,
        "good_requests": good_requests,
        "good_tokens": good_tokens,
    }

    if completed_ttfts:
        ttft_p50, ttft_p95 = _percentiles(completed_ttfts, (50.0, 95.0))
        summary.update(
            {
                "ttft_ms_mean": completed_ttft_total / len(completed_ttfts),
                "ttft_ms_p50": ttft_p50,
                "ttft_ms_p95": ttft_p95,
            }
        )
    else:
        summary.update({"ttft_ms_mean": 0.0, "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0})

    decode_tpot_total = 0.0
    decode_tpot_count = 0
    prefill_tpot_total = 0.0
    prefill_tpot_count = 0
    for gpu in gpus.values():
        if gpu.is_decode:
            decode_tpot_total += gpu.tpot_ema
            decode_tpot_count += 1
        if gpu.is_prefill:
            prefill_tpot_total += gpu.tpot_ema
            prefill_tpot_count += 1
    summary["avg_decode_tpot_tok_per_s"] = (
        decode_tpot_total / decode_tpot_count if decode_tpot_count else 0.0
    )
    summary["avg_prefill_tpot_tok_per_s"] = (
        prefill_tpot_total / prefill_tpot_count if prefill_tpot_count else 0.0
    )

    total_hours = num_ticks * TICK_SECONDS / 3600.0
    total_cost = sum(g.hourly_cost for g in gpus.values()) * total_hours
    summary["total_gpu_cost_usd"] = total_cost
    time_seconds = num_ticks * TICK_SECONDS
    summary["goodput_tokens_per_sec"] = (good_tokens / time_seconds) if time_seconds > 0 else 0.0
    summary["goodput_tokens_per_dollar"] = (good_tokens / total_cost) if total_cost > 0 else 0.0

    for gid, gpu in gpus.items():
        summary[f"queue_depth_avg_{gid}"] = gpu.queue_depth_avg()
        summary[f"queue_depth_max_{gid}"] = float(gpu.queue_depth_max)
        summary[f"tpot_tok_per_s_{gid}"] = gpu.tpot_ema
        summary[f"ttft_ms_ema_{gid}"] = gpu.ttft_ema
        summary[f"hourly_cost_{gid}"] = gpu.hourly_cost
        if gpu.tier:
            summary[f"tier_{gid}"] = gpu.tier

    if log_json:
        log_json.parent.mkdir(parents=True, exist_ok=True)
        with log_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {log_json}", file=sys.stderr)

    return summary



# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic routing lab driver")
    p.add_argument("--mode", choices=["baseline", "optimized"], default="baseline", help="routing policy variant")
    p.add_argument("--scenario", choices=["flagship_vs_mid", "mig_slices"], default="flagship_vs_mid", help="GPU sizing mix")
    p.add_argument("--ticks", type=int, default=400, help="simulation ticks")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--arrival-rate", type=float, default=1.2, help="Poisson arrivals per tick (tick=50 ms)")
    p.add_argument("--burst-factor", type=float, default=1.0, help="If >1, 20% of ticks arrive at this multiple of the base rate")
    p.add_argument("--slo-ttft-ms", type=float, default=350.0, help="TTFT SLO in milliseconds")
    p.add_argument("--slo-tpot-ms", type=float, default=45.0, help="Per-token SLO in milliseconds")
    p.add_argument("--log-json", type=Path, default=None, help="Optional JSON summary path for plotting")
    p.add_argument("--decode-cost-penalty", type=float, default=0.0, help="Divide decode scores by cost^penalty (0 disables)")
    p.add_argument("--queue-urgency", type=float, default=1.0, help="Weight for queue-depth penalty in scoring")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulate(
        args.mode,
        num_ticks=args.ticks,
        seed=args.seed,
        scenario=args.scenario,
        arrival_rate=args.arrival_rate,
        burst_factor=args.burst_factor,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_tpot_ms=args.slo_tpot_ms,
        log_json=args.log_json,
        cost_awareness=args.decode_cost_penalty,
        queue_urgency=args.queue_urgency,
    )
