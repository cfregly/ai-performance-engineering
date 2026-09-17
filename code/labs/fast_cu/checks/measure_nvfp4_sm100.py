#!/usr/bin/env python3
"""Exploratory paired CUDA-graph timing for the full SM100 NVFP4 GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from core.harness.benchmark_harness import (
    _resolve_physical_device_index,
    lock_gpu_clocks,
)
from core.harness.clock_lock_check import _query_nvml_clocks
from labs.fast_cu import nvfp4_sm100
from labs.fast_cu.baseline_nvfp4_sm100 import FastCuNvfp4Sm100CublasLtBenchmark
from labs.fast_cu.nvfp4 import Nvfp4Workload
from labs.fast_cu.nvfp4_sm100 import require_sm100_runtime
from labs.fast_cu.optimized_nvfp4_sm100 import FastCuNvfp4Sm100KernelBenchmark

WARMUPS, ROUNDS, REPLAYS, SEED = 10, 5, 1000, 20260916
SOURCE_NAMES = (
    "build.py",
    "nvfp4.py",
    "nvfp4_native.cu",
    "nvfp4_sm100.cuh",
    "nvfp4_sm100.py",
    "baseline_nvfp4_sm100.py",
    "optimized_nvfp4_sm100.py",
)


def nvml_clocks(physical_index: int) -> dict[str, int]:
    state = _query_nvml_clocks(physical_index)
    if state.get("error"):
        raise RuntimeError(str(state["error"]))
    return {name: int(value) for name, value in state.items() if value is not None}


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_stdev": statistics.stdev(values),
    }


def measure_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(replays):
            graph.replay()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) / replays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device_index = 0
    torch.cuda.set_device(device_index)
    require_sm100_runtime()
    physical_index = int(_resolve_physical_device_index(device_index))
    properties = torch.cuda.get_device_properties(device_index)
    workload = Nvfp4Workload()
    baseline = FastCuNvfp4Sm100CublasLtBenchmark(workload)
    candidate = FastCuNvfp4Sm100KernelBenchmark(workload)
    device = torch.device("cuda", device_index)
    baseline.device = candidate.device = device
    stream = torch.cuda.Stream(device=device)
    rounds: list[dict] = []

    try:
        with lock_gpu_clocks(device_index) as peaks:
            locked_start = nvml_clocks(physical_index)
            torch.manual_seed(SEED)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                baseline.setup()
                candidate.setup()
                baseline.benchmark_fn()
                candidate.benchmark_fn()
            stream.synchronize()

            if baseline._inputs.keys() != candidate._inputs.keys():
                raise AssertionError("baseline and candidate input sets differ")
            for name in baseline._inputs:
                torch.testing.assert_close(
                    baseline._inputs[name], candidate._inputs[name], rtol=0, atol=0
                )
            torch.testing.assert_close(candidate.output, baseline.output, rtol=0.002, atol=0.5)

            graphs = {}
            for name, benchmark in (("baseline", baseline), ("candidate", candidate)):
                context, output = benchmark._native_context, benchmark._output_buffer
                if context is None or output is None:
                    raise RuntimeError("benchmark setup did not create native state")
                launch = context.launch_fast if benchmark.optimized else context.launch_cublaslt
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    launch(output)
                graphs[name] = graph
            with torch.cuda.stream(stream):
                for _ in range(WARMUPS):
                    graphs["baseline"].replay()
                    graphs["candidate"].replay()
            stream.synchronize()

            for round_index in range(ROUNDS):
                order = (
                    ["baseline", "candidate"] if round_index % 2 == 0 else ["candidate", "baseline"]
                )
                row = {"round": round_index + 1, "order": order}
                for arm in order:
                    row[f"{arm}_ms_per_launch"] = measure_graph(graphs[arm], stream, REPLAYS)
                row["baseline_over_candidate"] = (
                    row["baseline_ms_per_launch"] / row["candidate_ms_per_launch"]
                )
                rounds.append(row)
            locked_end = nvml_clocks(physical_index)

        baseline_ms = [row["baseline_ms_per_launch"] for row in rounds]
        candidate_ms = [row["candidate_ms_per_launch"] for row in rounds]
        ratios = [row["baseline_over_candidate"] for row in rounds]
        lab_dir = Path(nvfp4_sm100.__file__).resolve().parent
        sources = {name: lab_dir / name for name in SOURCE_NAMES}
        sources[Path(__file__).name] = Path(__file__).resolve()
        payload = {
            "evidence_label": "virtualized_exploratory",
            "speed_claim": "none",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "shape": {"m": workload.m, "n": workload.n, "k": workload.k},
            "seed": SEED,
            "measurement": {
                "timer": "CUDA events",
                "graph_capture": "one native launch per arm",
                "warmups_per_arm": WARMUPS,
                "paired_rounds": ROUNDS,
                "replays_per_arm_per_round": REPLAYS,
                "rounds": rounds,
                "baseline_ms_per_launch": summarize(baseline_ms),
                "candidate_ms_per_launch": summarize(candidate_ms),
                "baseline_over_candidate": summarize(ratios),
            },
            "environment": {
                "torch_version": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "cuda_device_index": device_index,
                "nvml_physical_index": physical_index,
                "locked_start": locked_start,
                "locked_end": locked_end,
                "theoretical_fp16_tflops": peaks[0],
                "theoretical_memory_gbps": peaks[1],
            },
            "source_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in sources.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote exploratory evidence to {args.output}")
        print(
            "median ms/launch "
            f"baseline={statistics.median(baseline_ms):.6f} "
            f"candidate={statistics.median(candidate_ms):.6f} "
            f"paired_ratio={statistics.median(ratios):.6f}"
        )
        print("No stable speed claim is made from this virtualized exploratory run.")
        return 0
    finally:
        candidate.teardown()
        baseline.teardown()


if __name__ == "__main__":
    raise SystemExit(main())
