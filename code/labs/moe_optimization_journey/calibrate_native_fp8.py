"""Collect full-output FP8 errors on actual CUDA; never grant acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from labs.moe_optimization_journey.level6_native_fp8 import NativeFP8MoE
from labs.moe_optimization_journey.native_fp8_math import full_output_errors, reference_moe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; never overwrite evidence")
    for name in ("hidden-size", "intermediate-size", "num-experts", "top-k", "batch-size", "seq-len"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evidence already exists: {args.output}")
    config = {name.upper(): getattr(args, name) if getattr(args, name) is not None
              else getattr(NativeFP8MoE, name.upper()) for name in (
                  "hidden_size", "intermediate_size", "num_experts", "top_k", "batch_size", "seq_len")}
    report = {
        "status": "STARTED_NOT_ACCEPTED", "schema_version": 1,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "config": config, "seed": args.seed,
        "errors": [],
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("level6_native_fp8.py", "native_fp8_math.py", "calibrate_native_fp8.py")},
        "limits": "Fixed seed/routing and three input amplitudes only. No latency, speedup or acceptance inferred.",
    }
    bench = None
    exit_code = 1
    try:
        if not torch.cuda.is_available():
            report.update(status="HOLD", reason="Actual CUDA GPU required; no FP8 numerical evidence collected")
            exit_code = 3
        else:
            report.update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
            bench = NativeFP8MoE()
            for name, value in config.items():
                setattr(bench, name, value)
            # A private CPU generator consumes this initial seed without resetting CUDA RNG.
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.random.default_generator.manual_seed(args.seed)
                bench._setup_workload()
                base = bench.x.clone()
                for amplitude in (1, 4, 16):
                    report["attempted_input_amplitude"] = amplitude
                    bench.x.copy_(base * amplitude)
                    bench.benchmark_fn()
                    expected = reference_moe(bench.x, bench._reference_weights_cpu,
                                             bench._reference_ids_cpu, bench._reference_routing_cpu)
                    report["errors"].append({"input_amplitude": amplitude,
                                             **full_output_errors(bench.output, expected)})
            report["status"] = "CALIBRATION_ONLY_NOT_ACCEPTED"
            exit_code = 0
    except Exception as error:
        report.update(status="FAILURE_NOT_ACCEPTED", error_type=type(error).__name__, error=str(error))
    finally:
        if bench is not None:
            try:
                bench.teardown()
            except Exception as error:
                report.update(status="FAILURE_NOT_ACCEPTED", teardown_error=str(error))
                exit_code = 1
        report["exit_code"] = exit_code
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    print(f"{report['status']}: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
