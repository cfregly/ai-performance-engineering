"""Retain actual original-source CPU failures; never simulate CUDA success.

Run with --source pointing at the immutable original module capture. The one
adapter selects CPU storage for the original count exchange, because the
original module's property otherwise hardcodes CUDA. Original route/control
flow and real Gloo collectives are unchanged; this is not a CUDA reproduction.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_original(path):
    spec = importlib.util.spec_from_file_location("hybrid_ep_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, world, source, directory):
    torch.set_num_threads(1)
    original = load_original(source)
    report = {"rank": rank, "backend": "actual_gloo", "cuda_used": False}
    dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=rank,
                            world_size=world, timeout=timedelta(seconds=5))
    try:
        class OriginalWithCpuCountStorage(original.DeepSeekHybridEPModule):
            @property
            def cuda_device(self):
                return torch.device("cpu")

        topology = original.TopologyInfo(rank, world, rank % 2, 2, rank // 2, 2, True, None)
        torch.manual_seed(1000 + rank)
        model = OriginalWithCpuCountStorage(4, world, 1, 1, topology, route_mode="uniform", optimized=False)
        parameters = [None] * world
        dist.all_gather_object(parameters, next(model.replicated_parameters()).detach().clone())
        report["replicas_equal_before_any_update"] = all(torch.equal(parameters[0], p) for p in parameters[1:])
        try:
            model._apply_local_experts(torch.ones(2, 4), torch.zeros(2, dtype=torch.long), torch.ones(2, 1))
        except RuntimeError as exc:
            report["original_autograd_error"] = str(exc)
        count = 2 if rank == 1 else 1 if rank == 3 else 0
        destination = 2 if rank == 1 else 0
        try:
            result, events = model._roundtrip_routes(
                tokens=torch.ones(count, 4), weights=torch.ones(count, 1),
                dest_ranks=torch.full((count,), destination, dtype=torch.long),
                token_indices=torch.arange(count), local_expert_ids=torch.zeros(count, dtype=torch.long),
                group=None, group_size=world, group_rank=rank, use_single=True, reuse=False,
                event_label="original_zero_send",
            )
            report["route_outcome"] = "returned_without_collectives"
            report["output_shape"] = list(result.shape)
            assert count == 0 and events is None
            # Keep this process group alive while the other ranks reach the real
            # five-second collective deadline. No communication is faked.
            time.sleep(6)
        except RuntimeError as exc:
            report["route_outcome"] = "collective_failed"
            report["route_error"] = str(exc)
    finally:
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    context = mp.spawn(worker, args=(4, str(Path(args.source).resolve()), str(destination)), nprocs=4, join=False)
    deadline = time.monotonic() + 30
    try:
        while not context.join(timeout=1, grace_period=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("original-source Gloo probe exceeded 30 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    reports = [json.loads((destination / f"rank-{rank}.json").read_text()) for rank in range(4)]
    for report in reports:
        assert report["replicas_equal_before_any_update"] is False
        assert "out=... arguments don't support automatic differentiation" in report["original_autograd_error"]
        assert report["route_outcome"] == ("returned_without_collectives" if report["rank"] % 2 == 0 else "collective_failed")
    print(json.dumps({"status": "EXPECTED_ORIGINAL_FAILURES_REPRODUCED", "ranks": reports}, indent=2))


if __name__ == "__main__":
    main()
