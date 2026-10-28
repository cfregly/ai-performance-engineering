#!/usr/bin/env python3
"""Run the frozen constrained-logits screen inside a supervised time window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "plan.json"


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan():
    plan = json.loads(PLAN_PATH.read_text())
    valid = (plan["legal_set_sizes"] == [16, 64, 256]
             and plan["set_modes"] == ["static", "changing"]
             and plan["pair_orders"] == ["AB", "BA", "AB"]
             and plan["matched_pairs"] == len(plan["pair_orders"])
             and 0 < plan["child_window_seconds"] < plan["parent_wait_seconds"]
             and plan["parent_wait_seconds"] + plan["term_grace_seconds"]
             + plan["kill_grace_seconds"] + 5 < plan["hard_window_seconds"] < 180)
    if not valid:
        raise ValueError("The frozen screen dimensions, orders or time windows are invalid")
    return plan


def summarize(samples, plan):
    baseline = [row["baseline_wall_ms"] for row in samples]
    candidate = [row["candidate_wall_ms"] for row in samples]
    reduction = 1 - statistics.median(candidate) / statistics.median(baseline)
    rank = math.ceil(0.95 * len(samples)) - 1
    baseline_p95, candidate_p95 = sorted(baseline)[rank], sorted(candidate)[rank]
    every_pair = all(c < b for b, c in zip(baseline, candidate))
    passed = (len(samples) == plan["matched_pairs"]
              and reduction >= plan["minimum_median_reduction"]
              and candidate_p95 <= baseline_p95 * (1 + plan["maximum_p95_block_regression"])
              and every_pair)
    return {"screen_pass": passed, "median_reduction": reduction,
            "baseline_median_wall_ms": statistics.median(baseline),
            "candidate_median_wall_ms": statistics.median(candidate),
            "baseline_p95_block_wall_ms": baseline_p95,
            "candidate_p95_block_wall_ms": candidate_p95,
            "every_pair_faster": every_pair, "sample_count_per_path": len(samples)}


def self_check(plan):
    good = [{"baseline_wall_ms": 10.0, "candidate_wall_ms": 8.0}] * 3
    bad = good[:2] + [{"baseline_wall_ms": 10.0, "candidate_wall_ms": 10.3}]
    if (not summarize(good, plan)["screen_pass"]
            or summarize(good[:2], plan)["screen_pass"]
            or summarize(bad, plan)["screen_pass"] or "torch" in sys.modules):
        raise RuntimeError("The dependency-free acceptance checks failed")
    print("Source and frozen-plan checks passed. No torch import or GPU work ran.")


def worker(device, output, plan):
    started = time.monotonic()
    deadline = float(os.environ["LOGITS_SCREEN_DEADLINE"])
    result = {"status": "PARTIAL", "started_utc": utc_now(), "cells": []}
    raw = None

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("The frozen child work window expired")

    def append(row):
        raw.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        raw.flush()

    try:
        import torch

        check_deadline()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. No fallback is allowed")
        torch.cuda.set_device(device)
        dev = torch.device(device)
        properties = torch.cuda.get_device_properties(dev)
        if "B200" not in properties.name:
            raise RuntimeError(f"Requested B200, got {properties.name}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        versions = {"python": sys.version, "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda, "python_executable": sys.executable, "device": device,
                    "device_name": properties.name, "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tf32": torch.backends.cuda.matmul.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        atomic_json(output / "versions.json", versions)
        raw = (output / "raw.jsonl").open("x")
        generator = torch.Generator(device=dev).manual_seed(plan["seed"])
        batch, hidden, vocab = (plan[k] for k in ("batch_size", "hidden_size", "vocab_size"))
        steps = plan["steps_per_block"]
        weights = torch.randn(vocab, hidden, device=dev, generator=generator) / math.sqrt(hidden)
        states = torch.randn(steps, batch, hidden, device=dev, generator=generator)
        full_logits = torch.empty(batch, vocab, device=dev)
        torch.cuda.synchronize(dev)

        def make_paths(local_weights, local_states, id_sets, static):
            size = id_sets[0].numel()
            local_batch = local_states.shape[1]
            dense = full_logits if local_weights is weights else torch.empty(local_batch, local_weights.shape[0], device=dev)
            gathered_weights = torch.empty(size, local_weights.shape[1], device=dev)
            buffers = []
            for _ in range(2):
                buffers.append((torch.empty(local_batch, size, device=dev),
                                torch.empty(local_batch, device=dev),
                                torch.empty(local_batch, dtype=torch.long, device=dev),
                                torch.empty(local_batch, dtype=torch.long, device=dev)))
            torch.cuda.synchronize(dev)
            setup_start = time.perf_counter()
            if static:
                torch.index_select(local_weights, 0, id_sets[0], out=gathered_weights)
            torch.cuda.synchronize(dev)
            setup_ms = (time.perf_counter() - setup_start) * 1000

            def run(path, step):
                ids = id_sets[0 if static else step]
                scores, values, positions, tokens = buffers[path]
                if path == 0:
                    torch.mm(local_states[step], local_weights.t(), out=dense)
                    torch.index_select(dense, 1, ids, out=scores)
                else:
                    if not static:
                        torch.index_select(local_weights, 0, ids, out=gathered_weights)
                    torch.mm(local_states[step], gathered_weights.t(), out=scores)
                torch.max(scores, dim=1, out=(values, positions))
                torch.index_select(ids, 0, positions, out=tokens)
                return scores, tokens

            return run, setup_ms

        with torch.inference_mode():
            tie_weights = torch.zeros(17, 8, device=dev)
            tie_weights[1, 0], tie_weights[7, 0], tie_weights[15, 0] = 1, 1, -1
            tie_states = torch.zeros(1, 2, 8, device=dev)
            tie_states[0, 1, 0] = 1
            tie_ids = [torch.tensor([1, 7, 15], device=dev)]
            for static in (True, False):
                tie_run, _ = make_paths(tie_weights, tie_states, tie_ids, static)
                scores_a, tokens_a = tie_run(0, 0)
                scores_b, tokens_b = tie_run(1, 0)
                torch.testing.assert_close(scores_a, scores_b, rtol=0, atol=0)
                if not tokens_a.tolist() == tokens_b.tolist() == [1, 1]:
                    raise AssertionError("Explicit legal-token tie handling differs")
                append({"kind": "explicit_ties", "mode": "static" if static else "changing",
                        "token_ids": tokens_a.tolist(), "all_equal_row": 0, "two_way_maximum_row": 1})

            for size in plan["legal_set_sizes"]:
                for mode in plan["set_modes"]:
                    check_deadline()
                    rng = random.Random(plan["seed"] + size)
                    count = 1 if mode == "static" else steps
                    id_lists = [sorted(rng.sample(range(0, vocab, 2), size)) for _ in range(count)]
                    id_sets = [torch.tensor(ids, dtype=torch.long, device=dev) for ids in id_lists]
                    run, setup_ms = make_paths(weights, states, id_sets, mode == "static")
                    append({"kind": "cell_config", "size": size, "mode": mode,
                            "legal_ids": id_lists, "static_row_setup_wall_ms": setup_ms})
                    for step in range(steps):
                        check_deadline()
                        scores_a, tokens_a = run(0, step)
                        scores_b, tokens_b = run(1, step)
                        ids = id_sets[0 if mode == "static" else step]
                        oracle = states[step].double() @ weights.index_select(0, ids).double().t()
                        for actual in (scores_a, scores_b):
                            torch.testing.assert_close(actual.double(), oracle,
                                                       rtol=plan["correctness_rtol"], atol=plan["correctness_atol"])
                        expected = ids[oracle.argmax(dim=1)].tolist()
                        if not tokens_a.tolist() == tokens_b.tolist() == expected:
                            raise AssertionError("Legal argmax differs from the full projection or FP64 oracle")
                        append({"kind": "correctness", "size": size, "mode": mode, "step": step,
                                "checked_logits": batch * size, "selected_tokens": expected,
                                "baseline_max_abs_error": float((scores_a.double() - oracle).abs().max()),
                                "candidate_max_abs_error": float((scores_b.double() - oracle).abs().max())})

                    def timed(path):
                        check_deadline()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(dev)
                        wall_start = time.perf_counter()
                        begin.record()
                        for step in range(steps):
                            run(path, step)
                        end.record()
                        torch.cuda.synchronize(dev)
                        wall_ms = (time.perf_counter() - wall_start) * 1000
                        return wall_ms, begin.elapsed_time(end)

                    for _ in range(plan["warmup_blocks"]):
                        timed(0)
                        timed(1)
                    samples = []
                    for pair, order in enumerate(plan["pair_orders"]):
                        measured = {}
                        for letter in order:
                            measured[letter] = timed(0 if letter == "A" else 1)
                            append({"kind": "path_sample", "size": size, "mode": mode,
                                    "pair": pair, "path": letter, "wall_ms": measured[letter][0],
                                    "cuda_ms": measured[letter][1]})
                        row = {"kind": "pair", "size": size, "mode": mode, "pair": pair, "order": order,
                               "steps": steps, "baseline_wall_ms": measured["A"][0],
                               "candidate_wall_ms": measured["B"][0],
                               "baseline_cuda_ms": measured["A"][1], "candidate_cuda_ms": measured["B"][1]}
                        samples.append(row)
                        append(row)
                    result["cells"].append({"size": size, "mode": mode, **summarize(samples, plan)})
                    atomic_json(output / "worker_result.json", result)
        check_deadline()
        result["status"] = "COMPLETE"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if raw is not None:
            raw.close()
        result["finished_utc"] = utc_now()
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "worker_result.json", result)
    return 0 if result["status"] == "COMPLETE" else 1


def supervise(device, output, plan):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "PARTIAL", "scope": plan["scope"], "started_utc": utc_now(),
              "device": device, "pid": os.getpid(), "child_pid": None,
              "source_sha256": {name: digest(HERE / name) for name in ("screen.py", "plan.json", "README.md")},
              "plan": plan, "output": str(output), "cleanup": []}
    atomic_json(output / "launch.json", result)
    process = None

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        environment = os.environ.copy()
        environment["LOGITS_SCREEN_DEADLINE"] = str(started + plan["child_window_seconds"])
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            process = subprocess.Popen([sys.executable, str(HERE / "screen.py"), "--worker",
                                        "--device", device, "--output", str(output)],
                                       env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            result["child_pid"] = process.pid
            atomic_json(output / "launch.json", result)
            result["exit_code"] = process.wait(timeout=max(0.1, started + plan["parent_wait_seconds"] - time.monotonic()))
        worker_path = output / "worker_result.json"
        if worker_path.exists():
            result["worker"] = json.loads(worker_path.read_text())
        unchanged = all(digest(HERE / name) == value for name, value in result["source_sha256"].items())
        result["source_unchanged"] = unchanged
        if result["exit_code"] == 0 and result.get("worker", {}).get("status") == "COMPLETE" and unchanged:
            result["status"] = "COMPLETE"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            for sig, grace in ((signal.SIGTERM, plan["term_grace_seconds"]),
                               (signal.SIGKILL, plan["kill_grace_seconds"])):
                try:
                    os.killpg(process.pid, sig)
                    result["cleanup"].append(signal.Signals(sig).name)
                    process.wait(timeout=grace)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        result["child_reaped"] = process is None or process.poll() is not None
        if not result["child_reaped"]:
            result["status"] = "PARTIAL"
        result["finished_utc"] = utc_now()
        result["elapsed_seconds"] = time.monotonic() - started
        result["within_hard_window"] = result["elapsed_seconds"] < plan["hard_window_seconds"]
        result["within_180_seconds"] = result["elapsed_seconds"] < 180
        if not result["within_hard_window"]:
            result["status"] = "PARTIAL"
        result["next_step"] = ("Review raw samples and correctness within the declared stage scope"
                               if result["status"] == "COMPLETE" else "Inspect logs and retain partial evidence before another assignment")
        atomic_json(output / "HANDBACK.json", result)
    print(json.dumps({"status": result["status"], "handback": str(output / "HANDBACK.json"),
                      "elapsed_seconds": result["elapsed_seconds"], "child_reaped": result["child_reaped"]}))
    return 0 if result["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Exact assigned CUDA ordinal, for example cuda:1")
    parser.add_argument("--output", type=Path, help="New output directory. Existing paths are rejected")
    parser.add_argument("--self-check", action="store_true", help="Check the frozen plan and screen rule without torch")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = load_plan()
    if args.self_check:
        self_check(plan)
        return 0
    if args.device is None or not re.fullmatch(r"cuda:[0-9]+", args.device):
        parser.error("--device must name the exact assigned ordinal as cuda:N")
    if args.output is None or not args.output.is_absolute():
        parser.error("--output must be an absolute new directory")
    if args.worker:
        return worker(args.device, args.output, plan)
    return supervise(args.device, args.output, plan)


if __name__ == "__main__":
    raise SystemExit(main())
