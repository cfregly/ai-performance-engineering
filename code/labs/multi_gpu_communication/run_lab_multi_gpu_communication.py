"""Run native MPI, NCCL, or NVSHMEM Jacobi communication diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labs.multi_gpu_communication.native_common import (
    LAB_DIR,
    SCHEMA,
    NativeUnavailable,
    NativeWorkload,
    invoke_native,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mpi", "nccl", "nvshmem"), default="nccl")
    parser.add_argument("--schedule", choices=("serialized", "overlap"), default="overlap")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=257)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ranks", type=int, default=2, help="Explicit resource allocation to use")
    parser.add_argument("--registered-buffers", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--assert-cuda-aware-mpi", action="store_true")
    parser.add_argument("--binary", type=Path, default=LAB_DIR / "jacobi_native")
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dump-output", type=Path, help="Retain the complete timed row-major float32 mesh")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    workload = NativeWorkload(
        backend=args.backend, schedule=args.schedule, nx=args.nx, ny=args.ny,
        iterations=args.iterations, warmup=args.warmup, ranks=args.ranks,
        registered_buffers=args.registered_buffers, graph=args.graph,
        assert_cuda_aware_mpi=args.assert_cuda_aware_mpi,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="multi_gpu_communication_") as directory:
            result_path = args.json_out or Path(directory) / "result.json"
            result = invoke_native(workload, binary=args.binary, json_out=result_path,
                                   dump_output=args.dump_output, launcher=args.launcher,
                                   launcher_args=tuple(args.launcher_arg))
    except (NativeUnavailable, ValueError, RuntimeError, OSError) as error:
        skipped = isinstance(error, NativeUnavailable)
        result = {"schema": SCHEMA, "status": "skipped" if skipped else "failed", "reason": str(error)}
        if args.json_out is not None and not args.json_out.exists():
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result) if args.json else str(error))
        return 3 if skipped else 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"PASS: {args.backend} {args.schedule}, {args.ranks} ranks, {args.iterations} fixed iterations")
        print(f"Maximum-rank solve: {result['steady_ms']:.6f} ms")
        print(f"Complete-output maximum error: {result['max_abs_error']:.3g}")
        print("Diagnostic timing. Performance claims require the repository benchmark and profiling gates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
