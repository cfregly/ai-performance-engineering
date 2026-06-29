from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from core.harness.benchmark_harness import lock_gpu_clocks

from .artifact_io import build_manifest, write_csv, write_json, write_jsonl
from .matrix_catalog import available_playbooks, load_playbook
from .matrix_types import MatrixPlaybook
from .preflight import profiler_tool_status, query_app_clock_state, require_blackwell_cuda
from .runner import measure_scenario, render_console_table, summarize_rows


def _default_run_id(playbook_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}__{playbook_name}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--playbook",
        default="deck_matrix",
        help=f"Playbook name or YAML path. Available: {', '.join(available_playbooks())}",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/moe_decode_blackwell_matrix/runs"),
    )
    parser.add_argument("--expert-counts", nargs="*", type=int, default=None)
    parser.add_argument("--top-k-values", nargs="*", type=int, default=None)
    parser.add_argument("--decode-batches", nargs="*", type=int, default=None)
    parser.add_argument("--routing-policies", nargs="*", default=None)
    parser.add_argument("--schedule-modes", nargs="*", default=None)
    parser.add_argument("--launch-modes", nargs="*", default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sm-clock-mhz", type=int, default=None)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--allow-non-blackwell", action="store_true")
    return parser.parse_args(argv)


def _resolve_playbook(args: argparse.Namespace) -> MatrixPlaybook:
    playbook = load_playbook(args.playbook)
    return playbook.with_overrides(
        expert_counts=args.expert_counts,
        top_k_values=args.top_k_values,
        decode_batches=args.decode_batches,
        routing_policies=args.routing_policies,
        schedule_modes=args.schedule_modes,
        launch_modes=args.launch_modes,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        dtype=args.dtype,
        seed=args.seed,
        sm_clock_mhz=args.sm_clock_mhz,
        mem_clock_mhz=args.mem_clock_mhz,
        allow_non_blackwell=args.allow_non_blackwell or None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    playbook = _resolve_playbook(args)
    run_id = args.run_id or _default_run_id(playbook.name)
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device_meta = require_blackwell_cuda(allow_non_blackwell=playbook.allow_non_blackwell)
    profiler_meta = profiler_tool_status()
    scenarios = list(playbook.iter_scenarios())
    device = torch.device("cuda")

    with lock_gpu_clocks(
        device=0,
        sm_clock_mhz=playbook.sm_clock_mhz,
        mem_clock_mhz=playbook.mem_clock_mhz,
    ) as (theoretical_tflops, theoretical_gbps):
        clock_state = query_app_clock_state(0)
        if not clock_state.get("app_clock_sm_mhz"):
            raise RuntimeError("Application-clock lock did not report an SM clock")
        if not clock_state.get("app_clock_mem_mhz"):
            raise RuntimeError("Application-clock lock did not report a memory clock")
        locked_state = {
            **clock_state,
            "theoretical_tflops": theoretical_tflops,
            "theoretical_gbps": theoretical_gbps,
        }
        rows = [measure_scenario(scenario, device=device, clock_state=locked_state) for scenario in scenarios]

    summary = summarize_rows(rows)
    sys_meta = {
        "run_id": run_id,
        "playbook": playbook.to_dict(),
        "device": device_meta,
        "profiler_tools": profiler_meta,
        "command": sys.argv,
        "output_dir": str(run_dir),
        "clock_state": locked_state,
        "deck_duplication_note": (
            "This lab exists because the source PDF is duplicated; the repo gains more from "
            "a matrix/playbook surface than from cloning a second benchmark-pair lab."
        ),
    }
    write_json(run_dir / "sys_meta.json", sys_meta)
    write_jsonl(run_dir / "matrix.jsonl", rows)
    write_csv(run_dir / "matrix.csv", rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", build_manifest(run_dir))

    print(f"Run directory: {run_dir}")
    print(render_console_table(rows))

    return 0 if int(summary["error_row_count"]) == 0 else 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
