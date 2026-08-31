"""Actual two-rank CPU production-hook probe; no replacement collectives."""
import json
from datetime import timedelta
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist


def worker(rank, directory):
    from labs.train_distributed.zero2_common import build_model, build_training_components, training_step
    output = Path(directory)
    record = {"rank": rank, "backend": "gloo", "device": "cpu", "status": "RUNNING"}
    try:
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=(output / "rendezvous").as_uri(),
                                rank=rank, world_size=2, timeout=timedelta(seconds=15))
        torch.manual_seed(42)
        model = build_model(8, torch.device("cpu"))
        ddp, optimizer = build_training_components(model, .01, optimized=True)
        generator = torch.Generator().manual_seed(43 + rank)
        before = [p.detach().clone() for p in model.parameters()]
        x, y = torch.empty(3, 8), torch.empty(3, 8)
        for _ in range(3):
            training_step(ddp, optimizer, x, y, generator, 2)
        record.update(status="EXECUTED", updates=3,
                      delta=sum(float((p.detach() - q).abs().sum()) for p, q in zip(model.parameters(), before)))
    except Exception as exc:
        record.update(status="UNSUPPORTED_OR_FAILED", error_type=type(exc).__name__, error=str(exc))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        (output / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    context = torch.multiprocessing.spawn(worker, args=(str(destination),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("Actual two-rank production-hook probe timed out")
    finally:
        for child in context.processes:
            if child.is_alive():
                child.terminate()
            child.join(timeout=3)
            if child.is_alive():
                child.kill()
                child.join(timeout=3)
    records = [json.loads((destination / f"rank-{rank}.json").read_text()) for rank in range(2)]
    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "production_hook_replaced": False, "records": records,
              "status": "EXECUTED_NOT_NUMERICAL_QUALIFICATION" if all(r["status"] == "EXECUTED" for r in records) else "UNSUPPORTED_OR_FAILED"}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.exit(0 if all(r["status"] == "EXECUTED" for r in records) else 3)
