"""
Load testing harness for the multi-GPU B200/B300 inference server.

Run with torchrun:
    torchrun --nproc_per_node=<num_gpus> ch16/inference_server_load_test.py \
        --duration 60 --target-qps 400 --output-json results.json

The harness feeds synthetic requests (random prompt lengths, configurable QPS)
into the `InferenceServerMultiGPU` implementation and collects latency/throughput
statistics. All ranks receive the same request stream via broadcast so the
continuous batching state remains consistent across tensor-parallel workers.
"""


from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import argparse
import json
import os

from core.common.device_utils import resolve_local_rank
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from ch16.capacity_planner import LittleLawCapacityPlanner
from ch16.inference_serving_multigpu import (
    DemoCausalLM,
    InferenceRequest,
    InferenceServerMultiGPU,
    RequestState,
)

QUICK_MODE = os.getenv("BENCHMARK_QUICK", "0") not in ("0", "false", "False", "")
COMPILE_MODES = ("default", "max-autotune", "reduce-overhead")


@dataclass
class CompletionRecord:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float


def _init_distributed() -> None:
    if not dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = resolve_local_rank()
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            dist.init_process_group(backend=backend, device_id=local_rank)
        else:
            backend = "gloo"
            dist.init_process_group(backend=backend)


def _broadcast_requests(requests: Optional[List[Dict]]) -> List[Dict]:
    obj_list = [requests]
    dist.broadcast_object_list(obj_list, src=0)
    result = obj_list[0]
    return result if result is not None else []


def run_load_test(
    server: InferenceServerMultiGPU,
    *,
    duration: float,
    target_qps: float,
    interval: float,
    prompt_range: tuple[int, int],
    max_new_tokens: int,
    temperature: float,
    rng: np.random.Generator,
) -> Dict:
    """Main load execution loop (identical across all ranks)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    generated_counter = 0

    completions: List[CompletionRecord] = []

    max_prompt_len = max(1, min(prompt_range[1], server.max_seq_len - 1))
    min_prompt_len = max(1, min(prompt_range[0], max_prompt_len))

    def _on_complete(state: RequestState, completed_at: float) -> None:
        latency_ms = (completed_at - state.request.arrived_at) * 1000.0
        completions.append(
            CompletionRecord(
                request_id=state.request.request_id,
                prompt_tokens=len(state.request.prompt_tokens),
                generated_tokens=len(state.generated_tokens),
                latency_ms=latency_ms,
            )
        )

    server._completion_callback = _on_complete  # type: ignore[attr-defined]

    # Precompute world size to distribute IDs
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    def _run_warmup_phase() -> None:
        if QUICK_MODE:
            return
        warmup_new_tokens = max(1, min(4, max_new_tokens))
        warmup_prompt_len = max(1, min(256, max_prompt_len))
        if warmup_prompt_len <= 0:
            return

        if rank == 0:
            timestamp = time.time()
            warmup_specs: List[Dict[str, object]] = []
            for idx in range(min(4, server.max_batch_size)):
                warmup_specs.append(
                    {
                        "request_id": f"warmup_{idx}",
                        "prompt_len": warmup_prompt_len,
                        "priority": idx % 4,
                        "arrived_at": timestamp,
                    }
                )
        else:
            warmup_specs = None  # type: ignore

        warmup_specs = _broadcast_requests(warmup_specs)  # type: ignore

        for spec in warmup_specs:
            request = InferenceRequest(
                request_id=str(spec["request_id"]),
                prompt_tokens=list(range(int(spec["prompt_len"]))),
                max_new_tokens=warmup_new_tokens,
                temperature=float(temperature),
                top_k=50,
                priority=int(spec["priority"]),
                arrived_at=float(spec["arrived_at"]),
            )
            server.scheduler.add_request(request)

        step_budget = warmup_new_tokens + 2
        for _ in range(step_budget):
            batch = server.scheduler.get_next_batch(server.kv_cache)
            if not batch:
                break
            server.generate_batch(batch, num_tokens=1)
            server.scheduler.update_completions(
                batch,
                server.kv_cache,
                server._on_request_complete,  # type: ignore[attr-defined]
            )

        drain_guard = 0
        while server.scheduler.active_requests:
            drain_guard += 1
            if drain_guard > warmup_new_tokens * 4:
                break
            batch = server.scheduler.get_next_batch(server.kv_cache)
            if not batch:
                break
            server.generate_batch(batch, num_tokens=1)
            server.scheduler.update_completions(
                batch,
                server.kv_cache,
                server._on_request_complete,  # type: ignore[attr-defined]
            )

        if dist.is_initialized():
            dist.barrier()
        server.scheduler.reset_metrics()
        completions.clear()

    _run_warmup_phase()

    run_start = time.perf_counter()

    while time.perf_counter() - run_start < duration:
        tick_start = time.perf_counter()

        # Generate load on rank 0 and broadcast to others
        if rank == 0:
            expected = max(target_qps * interval, 0.0)
            num_requests = rng.poisson(expected)
            current_time = time.time()
            base_request_id = generated_counter
            request_specs: List[Dict[str, object]] = []
            for _ in range(num_requests):
                prompt_len = int(rng.integers(min_prompt_len, max_prompt_len + 1))
                if prompt_len <= 0 or prompt_len >= server.max_seq_len:
                    continue
                request_idx = len(request_specs)
                request_specs.append(
                    {
                        "request_id": f"req_{base_request_id + request_idx}",
                        "prompt_len": prompt_len,
                        "priority": (base_request_id + request_idx) % 4,
                        "arrived_at": current_time,
                    }
                )
        else:
            request_specs = None  # type: ignore

        request_specs = _broadcast_requests(request_specs)  # type: ignore

        for spec in request_specs:
            prompt_len = int(spec["prompt_len"])
            if prompt_len <= 0:
                continue
            prompt_tokens = list(range(prompt_len))
            request = InferenceRequest(
                request_id=str(spec["request_id"]),
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_k=50,
                priority=int(spec["priority"]),
                arrived_at=float(spec["arrived_at"]),
            )
            server.scheduler.add_request(request)

        generated_counter += len(request_specs)

        # Continuous batching iteration (replicates serve_loop body)
        batch = server.scheduler.get_next_batch(server.kv_cache)
        if batch:
            server.generate_batch(batch, num_tokens=1)
            server.scheduler.update_completions(
                batch,
                server.kv_cache,
                server._on_request_complete,  # type: ignore[attr-defined]
            )

        # Maintain target tick rate
        tick_elapsed = time.perf_counter() - tick_start
        sleep_time = interval - tick_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    dist.barrier()

    stats = server.scheduler.get_stats()
    elapsed = time.perf_counter() - run_start
    return {
        "rank": rank,
        "elapsed": elapsed,
        "stats": stats,
        "completions": [asdict(record) for record in completions],
        "world_size": world_size,
    }


def _summarize_samples(values: List[int], total: Optional[float] = None) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "avg": 0.0}
    values.sort()
    p50, p95 = _percentiles_from_ordered(values, (50.0, 95.0))
    return {
        "avg": float((sum(values) if total is None else total) / len(values)),
        "p50": p50,
        "p95": p95,
    }


def _percentile_from_ordered(values: List[float] | List[int], pct: float) -> float:
    if not values:
        return 0.0
    rank = (len(values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = lower if rank == lower else lower + 1
    if lower == upper:
        return float(values[lower])
    return float(values[lower] * (upper - rank) + values[upper] * (rank - lower))


def _percentiles_from_ordered(values: List[float] | List[int], pcts: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(_percentile_from_ordered(values, pct) for pct in pcts)


def aggregate_results(local_result: Dict) -> Dict:
    gathered: List[Dict] = [None] * dist.get_world_size()  # type: ignore
    dist.all_gather_object(gathered, local_result)

    total_requests = sum(item["stats"]["total_requests"] for item in gathered)
    completed_requests = sum(item["stats"]["completed_requests"] for item in gathered)
    rejected_requests = sum(item["stats"]["rejected_requests"] for item in gathered)
    tokens_generated = sum(item["stats"]["total_tokens_generated"] for item in gathered)
    elapsed = max(item["elapsed"] for item in gathered)
    latencies: List[float] = []
    prompt_lengths: List[int] = []
    generated_lengths: List[int] = []
    prompt_token_total = 0.0
    generated_token_total = 0.0
    for item in gathered:
        for rec in item["completions"]:
            latencies.append(rec["latency_ms"])
            prompt_tokens = rec["prompt_tokens"]
            generated_tokens = rec["generated_tokens"]
            prompt_lengths.append(prompt_tokens)
            generated_lengths.append(generated_tokens)
            prompt_token_total += prompt_tokens
            generated_token_total += generated_tokens

    throughput = tokens_generated / elapsed if elapsed > 0 else 0.0
    if latencies:
        latencies.sort()
        p50, p90, p99 = _percentiles_from_ordered(latencies, (50.0, 90.0, 99.0))
    else:
        p50 = p90 = p99 = 0.0

    prompt_stats = _summarize_samples(prompt_lengths, prompt_token_total)
    generated_stats = _summarize_samples(generated_lengths, generated_token_total)
    world_size = gathered[0]["world_size"] if gathered else 1

    return {
        "elapsed": elapsed,
        "world_size": int(world_size),
        "total_requests": int(total_requests),
        "completed_requests": int(completed_requests),
        "rejected_requests": int(rejected_requests),
        "tokens_generated": int(tokens_generated),
        "throughput_tok_per_s": throughput,
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p99_ms": p99,
        "samples_collected": len(latencies),
        "prompt_token_stats": prompt_stats,
        "generated_token_stats": generated_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Load test for multi-GPU inference server")
    parser.add_argument("--duration", type=float, default=30.0, help="Total test duration in seconds.")
    parser.add_argument("--target-qps", type=float, default=200.0, help="Synthetic request rate.")
    parser.add_argument("--tick-interval", type=float, default=0.02, help="Scheduler tick interval (seconds).")
    parser.add_argument("--prompt-len-min", type=int, default=64, help="Minimum prompt length.")
    parser.add_argument("--prompt-len-max", type=int, default=2048, help="Maximum prompt length.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="New tokens generated per request.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of transformer layers in the demo model.")
    parser.add_argument("--d-model", type=int, default=4096, help="Hidden dimension of the demo model.")
    parser.add_argument("--num-heads", type=int, default=64, help="Number of attention heads in the demo model.")
    parser.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch size for the server/model.")
    parser.add_argument("--max-seq-len", type=int, default=16384, help="Maximum sequence length for KV cache and requests.")
    parser.add_argument(
        "--require-world-size",
        type=int,
        default=None,
        help="Optionally enforce an exact world size when validating deployments.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write aggregated results.")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile on server submodules.")
    parser.add_argument("--disable-prefill-graph", action="store_true", help="Disable CUDA graph capture for prefill phase.")
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default="default",
        help="torch.compile mode for optimized submodules.",
    )
    parser.add_argument(
        "--prefill-tokens-per-s",
        type=float,
        default=40000.0,
        help="Measured prompt tokens/sec per GPU (prefill phase).",
    )
    parser.add_argument(
        "--decode-tokens-per-s",
        type=float,
        default=2000.0,
        help="Measured decode tokens/sec per GPU.",
    )
    parser.add_argument(
        "--tokens-per-gpu",
        type=float,
        default=None,
        help="Override sustained tokens/sec per GPU instead of deriving from the run.",
    )
    parser.add_argument(
        "--capacity-headroom",
        type=float,
        default=0.3,
        help="Extra GPU headroom ratio when applying Little's Law (e.g., 0.3 = 30%%).",
    )
    args = parser.parse_args()

    if QUICK_MODE:
        # Keep the run lightweight so benchmarks can finish quickly.
        args.duration = min(args.duration, 5.0)
        args.target_qps = min(args.target_qps, 20.0)
        args.tick_interval = max(args.tick_interval, 0.05)
        args.prompt_len_min = max(8, min(args.prompt_len_min, 32))
        args.prompt_len_max = max(args.prompt_len_min, min(args.prompt_len_max, 128))
        args.max_new_tokens = min(args.max_new_tokens, 16)
        args.num_layers = min(args.num_layers, 6)
        args.d_model = min(args.d_model, 1024)
        args.num_heads = min(args.num_heads, 16)
        args.max_batch_size = min(args.max_batch_size, 32)
        args.max_seq_len = min(args.max_seq_len, 2048)
        os.environ.setdefault("INFERENCE_SERVER_QUICK_MODE", "1")
    return args


def main() -> None:
    args = parse_args()
    _init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    required_world = args.require_world_size
    if required_world is not None and world_size != required_world and rank == 0:
        raise SystemExit(
            f"World size mismatch: expected {required_world} ranks (found {world_size}). "
            "Adjust --require-world-size or launch configuration."
        )
    if rank == 0 and required_world is None:
        if world_size < 2:
            print(
                f"[info] Running single-rank mode ({world_size} rank). "
                "Use torchrun with >=2 GPUs for multi-GPU load tests."
            )
        else:
            print(
                f"[info] Running benchmark with {world_size} rank(s). "
                "Set --require-world-size to enforce a specific size if desired."
            )

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    # Instantiate demo model
    model = DemoCausalLM(
        vocab_size=50000,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_gpus=world_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
    )

    server = InferenceServerMultiGPU(
        model=model,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        enable_compile=not args.disable_compile,
        compile_mode=args.compile_mode,
    )

    if args.disable_prefill_graph:
        server._prefill_graph = None
        server._prefill_graph_available = False

    local_result = run_load_test(
        server,
        duration=args.duration,
        target_qps=args.target_qps,
        interval=args.tick_interval,
        prompt_range=(args.prompt_len_min, args.prompt_len_max),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        rng=rng,
    )

    aggregate = aggregate_results(local_result)
    if rank == 0:
        print("\n=== Load Test Summary ===")
        print(f"Duration: {aggregate['elapsed']:.1f}s")
        print(f"Throughput: {aggregate['throughput_tok_per_s']:.0f} tokens/s")
        print(
            f"Requests: total={aggregate['total_requests']} "
            f"completed={aggregate['completed_requests']} "
            f"rejected={aggregate['rejected_requests']}"
        )
        print(
            "Latency (ms): "
            f"P50={aggregate['latency_p50_ms']:.2f} "
            f"P90={aggregate['latency_p90_ms']:.2f} "
            f"P99={aggregate['latency_p99_ms']:.2f}"
        )
        prompt_stats = aggregate.get("prompt_token_stats", {})
        generated_stats = aggregate.get("generated_token_stats", {})
        qps_observed = (
            aggregate["completed_requests"] / aggregate["elapsed"]
            if aggregate["elapsed"] > 0
            else 0.0
        )
        tokens_per_gpu = args.tokens_per_gpu
        effective_world = float(aggregate.get("world_size") or world_size or 1)
        if tokens_per_gpu is None and effective_world > 0:
            tokens_per_gpu = (
                aggregate["throughput_tok_per_s"] / effective_world
                if aggregate["throughput_tok_per_s"] > 0
                else 0.0
            )

        if tokens_per_gpu and tokens_per_gpu > 0 and qps_observed > 0:
            planner = LittleLawCapacityPlanner(
                prefill_tokens_per_s=args.prefill_tokens_per_s,
                decode_tokens_per_s=args.decode_tokens_per_s,
                tokens_per_gpu=tokens_per_gpu,
                headroom_ratio=args.capacity_headroom,
            )
            estimates = planner.estimate_all(
                qps=qps_observed,
                prompt_token_stats=prompt_stats,
                generated_token_stats=generated_stats,
            )
            if estimates:
                print(
                    "\nLittle's Law capacity plan | QPS={:.2f} | Prefill={:.0f} tok/s | Decode={:.0f} tok/s | Tokens/GPU={:.0f} tok/s | Headroom={:.0%}".format(
                        qps_observed,
                        args.prefill_tokens_per_s,
                        args.decode_tokens_per_s,
                        tokens_per_gpu,
                        args.capacity_headroom,
                    )
                )
                for est in estimates:
                    print(
                        "  {pct}: service={service_ms:.1f} ms (prefill={prefill_ms:.1f} ms, decode={decode_ms:.1f} ms) → GPUs={gpus:.2f}, GPUs+headroom={gpus_headroom:.2f}".format(
                            pct=est.percentile.upper(),
                            service_ms=est.total_service_time_s * 1000.0,
                            prefill_ms=est.service_time_prefill_s * 1000.0,
                            decode_ms=est.service_time_decode_s * 1000.0,
                            gpus=est.required_gpus,
                            gpus_headroom=est.required_gpus_with_headroom,
                        )
                    )
                aggregate["capacity_plan"] = {
                    "qps_observed": qps_observed,
                    "prefill_tokens_per_s": args.prefill_tokens_per_s,
                    "decode_tokens_per_s": args.decode_tokens_per_s,
                    "tokens_per_gpu": tokens_per_gpu,
                    "headroom_ratio": args.capacity_headroom,
                    "estimates": [est.to_dict() for est in estimates],
                }
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(aggregate, indent=2))
            print(f"Saved aggregate metrics to {args.output_json}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
